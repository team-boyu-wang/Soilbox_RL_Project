# -*- coding: utf-8 -*-
"""为有文献物理支撑的 SoilGuard 土箱环境训练 PPO 智能体(训练管线主模块)。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from experiment_configs import EXPERIMENTS
from evaluation import evaluate_v2, format_battery_table, idle_policy
from soilbox_env import (
    DEFAULT_CONFIG,
    SEED,
    SoilBoxEnv,
    max_electrode_voltage,
    rule_policy,
)


def ppo_policy_of(model) -> Callable:
    """把 Stable-Baselines3 模型包装成 policy_fn(obs) -> action,供考卷评估调用。"""

    return lambda obs: model.predict(obs, deterministic=True)[0]


class ClearMLTrainingDashboardWriter:
    """把选定的 Stable-Baselines3 训练标量转发到 ClearML 面板(挂进 SB3 Logger 的输出格式)。"""

    SCALARS = {
        "rollout/ep_rew_mean": ("Training Outcome", "episode_reward", 1.0),
        "rollout/ep_len_mean": ("Training Outcome", "episode_length", 1.0),
        "train/value_loss": ("Critic Diagnostics", "value_loss", 1.0),
        "train/explained_variance": ("Critic Diagnostics", "explained_variance", 1.0),
        "train/entropy_loss": ("Policy Diagnostics", "policy_entropy", -1.0),
        "train/kl_to_bc": ("Policy Diagnostics", "kl_to_bc", 1.0),
    }

    def __init__(self, task, series_prefix: str) -> None:
        self.task = task
        self.series_prefix = series_prefix

    def write(
        self,
        key_values: dict[str, Any],
        key_excluded: dict[str, tuple[str, ...]],
        step: int = 0,
    ) -> None:
        del key_excluded
        if self.task is None:
            return

        logger = self.task.get_logger()
        for raw_key, (title, series, scale) in self.SCALARS.items():
            if raw_key not in key_values:
                continue

            try:
                value = float(key_values[raw_key]) * scale
            except (TypeError, ValueError):
                continue
            logger.report_scalar(
                title=title,
                series=f"{self.series_prefix}/{series}",
                value=value,
                iteration=int(step),
            )

    def close(self) -> None:
        return None


def make_training_dashboard_callback(
    task,
    series_prefix: str,
    *,
    dashboard_dir: Path | None = None,
):
    if task is None and dashboard_dir is None:
        return None

    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.utils import explained_variance
    import torch

    class ClearMLTrainingDashboardCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.episode_rewards = deque(maxlen=100)
            self.episode_lengths = deque(maxlen=100)
            self.rollout_values: dict[str, list[float]] = {}
            self.history: list[dict[str, float]] = []
            self.metrics_path = None
            self.plot_path = None
            # 按环境序号分开保存:并行环境下若共用一个值,
            # 会拿不同环境的动作互相比较。
            self.prev_action: dict[int, np.ndarray] = {}
            if dashboard_dir is not None:
                dashboard_dir.mkdir(parents=True, exist_ok=True)
                self.metrics_path = dashboard_dir / f"{series_prefix}_training_dashboard.jsonl"
                self.plot_path = dashboard_dir / f"{series_prefix}_training_dashboard.png"

        def _append_rollout_value(self, key: str, value: float) -> None:
            self.rollout_values.setdefault(key, []).append(float(value))

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                episode = info.get("episode")
                if episode is None:
                    continue
                self.episode_rewards.append(float(episode["r"]))
                self.episode_lengths.append(float(episode["l"]))

            for env_idx, info in enumerate(self.locals.get("infos", [])):
                terms = info.get("terms") or {}
                cfg = info.get("config")
                if cfg is not None and terms:
                    if "reward_u" in terms:
                        # v5.0 奖励:考卷分数按步分期支付,直接读取。
                        self._append_rollout_value("reward_uniformity_component", terms["reward_u"])
                        self._append_rollout_value("reward_ph_component", terms["reward_ph"])
                        self._append_rollout_value("reward_energy_component", terms["reward_e"])
                        self._append_rollout_value("step_quality_mean", terms["step_quality"])
                        self._append_rollout_value("raw_power_phys", terms["power_phys"])
                        self._append_rollout_value("raw_ph_violation_mixed", terms["ph_violation_mixed"])
                    else:
                        # 旧版(v5.0 之前)加权惩罚式奖励结构。
                        self._append_rollout_value("reward_flux_component", cfg.w_flux * terms["flux"])
                        self._append_rollout_value(
                            "reward_uniformity_goal_component",
                            cfg.w_uniformity_goal
                            * terms.get("uniformity_excess", terms["uniformity_improvement"]),
                        )
                        self._append_rollout_value(
                            "reward_uniformity_penalty",
                            -cfg.w_uniformity * terms.get("moisture_cv_gap", terms["moisture_cv"]),
                        )
                        self._append_rollout_value(
                            "reward_energy_penalty",
                            -cfg.w_energy * terms["energy"],
                        )
                        self._append_rollout_value(
                            "reward_ph_penalty",
                            -cfg.w_ph * terms.get("ph_penalty_charge", terms["ph_violation"]),
                        )
                        self._append_rollout_value(
                            "reward_local_dry_penalty",
                            -cfg.w_local_dry
                            * terms.get("local_dry_violation", terms["anode_dry_violation"]),
                        )
                    self._append_rollout_value("raw_flux", terms["flux"])
                    self._append_rollout_value("raw_energy", terms["energy"])
                    self._append_rollout_value("raw_ph_violation", terms["ph_violation"])
                    self._append_rollout_value("raw_moisture_cv", terms["moisture_cv"])
                    self._append_rollout_value(
                        "raw_local_dry_violation",
                        terms.get("local_dry_violation", terms["anode_dry_violation"]),
                    )

                if "mean_moisture" in info:
                    self._append_rollout_value("outcome_mean_moisture", info["mean_moisture"])
                if "mean_pH" in info:
                    self._append_rollout_value("outcome_mean_ph", info["mean_pH"])

                action = info.get("electrode_action")
                max_voltage = float(info.get("max_electrode_voltage_v", 1.0))
                nominal_voltage = float(info.get("nominal_applied_voltage_v", 1.0))
                if action is not None and max_voltage > 0:
                    action_arr = np.asarray(action, dtype=np.float64)
                    normalized_abs = np.abs(action_arr) / max_voltage
                    self._append_rollout_value("action_abs_mean", float(np.mean(normalized_abs)))
                    self._append_rollout_value(
                        "action_saturation_frac",
                        float(np.mean(normalized_abs >= 0.95)),
                    )
                    prev = self.prev_action.get(env_idx)
                    if prev is not None and prev.shape == action_arr.shape:
                        delta = np.mean(np.abs(action_arr - prev)) / (2.0 * max_voltage)
                        self._append_rollout_value("action_delta_mean", float(delta))
                    self.prev_action[env_idx] = action_arr
                if nominal_voltage > 0 and "voltage_drop_v" in info:
                    self._append_rollout_value(
                        "action_voltage_drop_ratio",
                        abs(float(info["voltage_drop_v"])) / nominal_voltage,
                    )
            return True

        def _on_rollout_end(self) -> None:
            step = int(self.num_timesteps)
            metrics: dict[str, float] = {"step": float(step)}
            if self.episode_rewards:
                metrics["episode_reward"] = sum(self.episode_rewards) / len(self.episode_rewards)
            if self.episode_lengths:
                metrics["episode_length"] = sum(self.episode_lengths) / len(self.episode_lengths)

            rollout_buffer = self.model.rollout_buffer
            values = rollout_buffer.values.flatten()
            returns = rollout_buffer.returns.flatten()
            metrics["explained_variance"] = float(explained_variance(values, returns))

            observations = rollout_buffer.observations.reshape(
                (-1,) + self.model.observation_space.shape
            )
            obs_tensor = torch.as_tensor(observations, device=self.model.device)
            with torch.no_grad():
                distribution = self.model.policy.get_distribution(obs_tensor)
                metrics["policy_entropy"] = float(distribution.entropy().mean().item())

            # 最近一次 PPO 更新的 KL(pi || pi_BC)(仅 KLToBCPPO 有)。
            # _on_rollout_end 在下一次 train() 之前触发,所以数值滞后
            # 一个迭代 —— 用来看趋势足够。
            kl_to_bc = getattr(self.model, "last_kl_to_bc", None)
            if kl_to_bc is not None and np.isfinite(kl_to_bc):
                metrics["kl_to_bc"] = float(kl_to_bc)

            # 课程学习回调发布的课程状态(v27)。
            difficulty = getattr(self.model, "curriculum_difficulty_current", None)
            if difficulty is not None:
                metrics["curriculum_difficulty"] = float(difficulty)
            rolling = getattr(self.model, "curriculum_rolling_success", None)
            if rolling is not None:
                metrics["curriculum_rolling_success"] = float(rolling)

            # 探索噪声水平(v28):实际的 log_std,以及退火回调激活时
            # 其逐步下降的上限。
            if hasattr(self.model.policy, "log_std"):
                metrics["log_std_mean"] = float(
                    self.model.policy.log_std.detach().mean().item()
                )
            log_std_cap = getattr(self.model, "log_std_cap_current", None)
            if log_std_cap is not None:
                metrics["log_std_cap"] = float(log_std_cap)

            for key, values in sorted(self.rollout_values.items()):
                if values:
                    metrics[key] = float(sum(values) / len(values))
            self.rollout_values.clear()

            self.history.append(metrics)
            if self.metrics_path is not None:
                with self.metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(metrics, sort_keys=True) + "\n")

            tensorboard_groups = {
                "reward_terms/uniformity_component": "reward_uniformity_component",
                "reward_terms/ph_component": "reward_ph_component",
                "reward_terms/energy_component": "reward_energy_component",
                "environment/step_quality": "step_quality_mean",
                "environment/power_phys": "raw_power_phys",
                "environment/ph_violation_mixed": "raw_ph_violation_mixed",
                "reward_terms/flux_component": "reward_flux_component",
                "reward_terms/uniformity_goal_component": "reward_uniformity_goal_component",
                "reward_terms/uniformity_penalty": "reward_uniformity_penalty",
                "reward_terms/energy_penalty": "reward_energy_penalty",
                "reward_terms/ph_penalty": "reward_ph_penalty",
                "reward_terms/local_dry_penalty": "reward_local_dry_penalty",
                "environment/raw_flux": "raw_flux",
                "environment/raw_energy": "raw_energy",
                "environment/raw_ph_violation": "raw_ph_violation",
                "environment/raw_local_dry_violation": "raw_local_dry_violation",
                "environment/raw_moisture_cv": "raw_moisture_cv",
                "environment/mean_moisture": "outcome_mean_moisture",
                "environment/mean_ph": "outcome_mean_ph",
                "action/abs_mean": "action_abs_mean",
                "action/saturation_frac": "action_saturation_frac",
                "action/voltage_drop_ratio": "action_voltage_drop_ratio",
                "action/delta_mean": "action_delta_mean",
                "critic/explained_variance_rollout": "explained_variance",
                "policy/entropy": "policy_entropy",
                "policy/kl_to_bc": "kl_to_bc",
                "policy/log_std_mean": "log_std_mean",
                "policy/log_std_cap": "log_std_cap",
                "curriculum/difficulty": "curriculum_difficulty",
                "curriculum/rolling_success": "curriculum_rolling_success",
            }
            for tensorboard_key, metric_key in tensorboard_groups.items():
                if metric_key in metrics:
                    self.logger.record(tensorboard_key, metrics[metric_key])

            if task is not None:
                logger = task.get_logger()
                clearml_groups = {
                    "Reward Components": [
                        "reward_uniformity_component",
                        "reward_ph_component",
                        "reward_energy_component",
                        "reward_flux_component",
                        "reward_uniformity_goal_component",
                        "reward_uniformity_penalty",
                        "reward_energy_penalty",
                        "reward_ph_penalty",
                        "reward_local_dry_penalty",
                    ],
                    "Environment Outcomes": [
                        "raw_flux",
                        "outcome_mean_moisture",
                        "outcome_mean_ph",
                        "raw_ph_violation",
                        "raw_ph_violation_mixed",
                        "raw_power_phys",
                        "step_quality_mean",
                        "raw_local_dry_violation",
                        "raw_moisture_cv",
                    ],
                    "Action Diagnostics": [
                        "action_abs_mean",
                        "action_saturation_frac",
                        "action_voltage_drop_ratio",
                        "action_delta_mean",
                    ],
                    "Critic Diagnostics": ["explained_variance"],
                    "Policy Diagnostics": ["policy_entropy", "kl_to_bc", "log_std_mean", "log_std_cap"],
                    "Curriculum": ["curriculum_difficulty", "curriculum_rolling_success"],
                    "Training Outcome": ["episode_reward", "episode_length"],
                }
                for title, keys in clearml_groups.items():
                    for key in keys:
                        if key not in metrics:
                            continue
                        logger.report_scalar(
                            title=title,
                            series=f"{series_prefix}/{key}",
                            value=metrics[key],
                            iteration=step,
                        )

        def _on_training_end(self) -> None:
            if self.plot_path is None or not self.history:
                return
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                return

            series = [
                (
                    [
                        ("reward_uniformity_goal_component", "CV improvement"),
                        ("reward_uniformity_penalty", "Uniformity"),
                        ("reward_energy_penalty", "Energy"),
                        ("reward_ph_penalty", "pH"),
                        ("reward_local_dry_penalty", "Local dry"),
                    ],
                    "Reward Components",
                ),
                (
                    [
                        ("explained_variance", "Explained variance"),
                    ],
                    "Critic Fit",
                ),
                (
                    [
                        ("action_abs_mean", "Abs action"),
                        ("action_saturation_frac", "Saturation"),
                        ("action_voltage_drop_ratio", "Voltage drop"),
                    ],
                    "Action Diagnostics",
                ),
                (
                    [
                        ("episode_reward", "Reward"),
                        ("episode_length", "Length"),
                    ],
                    "Training Outcome",
                ),
            ]
            fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
            steps = [row["step"] for row in self.history]
            for ax, (lines, title) in zip(axes.ravel(), series):
                for key, label in lines:
                    xs = []
                    ys = []
                    for step, row in zip(steps, self.history):
                        if key in row:
                            xs.append(step)
                            ys.append(row[key])
                    if xs:
                        ax.plot(xs, ys, marker="o", markersize=2.5, linewidth=1.5, label=label)
                ax.set_title(title)
                ax.set_xlabel("Environment steps")
                ax.grid(True, alpha=0.25)
                ax.legend(fontsize=8)
            fig.savefig(self.plot_path, dpi=150)
            plt.close(fig)

    return ClearMLTrainingDashboardCallback()


def attach_training_logger(
    model,
    *,
    task,
    tensorboard_log: Path | None,
    run_name: str,
    verbose: int,
) -> None:
    """给模型挂上 ClearML/TensorBoard 记录器,不改动 PPO 训练逻辑(train_ppo 装配时调用)。"""

    if task is None and tensorboard_log is None:
        return

    from stable_baselines3.common.logger import (
        HumanOutputFormat,
        Logger,
        TensorBoardOutputFormat,
    )

    output_formats = []
    folder = None
    if verbose > 0:
        output_formats.append(HumanOutputFormat(sys.stdout))
    if tensorboard_log is not None:
        folder = str(tensorboard_log / run_name)
        Path(folder).mkdir(parents=True, exist_ok=True)
        output_formats.append(TensorBoardOutputFormat(folder))
    if task is not None:
        output_formats.append(ClearMLTrainingDashboardWriter(task, run_name))

    model.set_logger(Logger(folder=folder, output_formats=output_formats))


def training_dashboard_spec() -> dict[str, dict[str, str]]:
    return {
        "Reward Components": {
            "source": "Environment info['terms'] accumulated by rollout callback.",
            "meaning": (
                "Weighted reward sub-terms: uniformity-improvement reward and "
                "uniformity/energy/pH/local-dry penalties. Use this to check "
                "whether one term dominates or whether penalties are actually active."
            ),
        },
        "Critic Diagnostics": {
            "source": "Stable-Baselines3 train/value_loss and rollout explained variance.",
            "meaning": (
                "Value loss and explained variance show whether the critic can predict "
                "returns from the current observation. Persistent high value loss or low "
                "explained variance suggests the state space may be missing important variables."
            ),
        },
        "Action Diagnostics": {
            "source": "Environment info['electrode_action'] accumulated by rollout callback.",
            "meaning": (
                "Mean normalized voltage, voltage-drop ratio, and saturation fraction. "
                "Saturation is useful for checking action range; action smoothness is "
                "logged as a secondary diagnostic, not a primary objective."
            ),
        },
        "Environment Outcomes": {
            "source": "Environment info and reward terms.",
            "meaning": "Raw flux, pH violation, local drying, moisture CV, and mean moisture during training.",
        },
        "Training Outcome": {
            "source": "Stable-Baselines3 rollout/ep_rew_mean and rollout/ep_len_mean.",
            "meaning": "Overall reward and episode length; useful as a summary but not enough by itself.",
        },
    }


ROLLOUT_STEPS_TOTAL = 2048


def pretrain_policy_bc(
    model,
    make_env: Callable[[int], SoilBoxEnv],
    *,
    episodes: int = 300,
    epochs: int = 20,
    seed: int = SEED,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    teacher_fn: Callable | None = None,
) -> dict[str, float]:
    """把比例控制器教师行为克隆进 ``model.policy``(BC 热启动)。

    针对 v15 诊断的热启动:初始场随机时,方向性梯度在权重空间里互相抵消,
    而"少动"的梯度是相干的,于是从零开始的 PPO 会塌缩成不作为,永远采样
    不到保持奖金这份收入。监督模仿没有这些问题 —— 每个 (obs, action) 对
    都是独立的干净标签 —— 先把"读场、往干侧推"的映射刻进网络,PPO 随后
    直接从工资区内部起步,此时收入和惩罚两侧的梯度都是活的。

    只训练动作均值(MSE);价值头和 log_std 不接收梯度,PPO 自身的优化器
    状态也不被触碰。
    由 train_ppo 在 bc_pretrain=True 时调用;teacher_fn 为 None 时默认用
    2D 比例控制器,3D 运行由调用方传入 proportional_policy_3d。
    """

    import torch

    if teacher_fn is None:
        # 默认教师:2D 比例控制器(v17 谱系)。
        # 3D 运行通过 teacher_fn 参数传入 proportional_policy_3d。
        from reward_audit import proportional_policy as teacher_fn

    # 示教数据在控制器达标后不久即截断:控制器几十步内完成均匀化然后闲置,
    # 完整的 180 步回合约有 2/3 是"已均匀 -> 动作 ~ 0"的样本。这种不平衡
    # 会主导 MSE,克隆出一个方向正确却过于胆小、实际到不了目标的策略
    # (实测:标准场上仅 ~0.6 V)。保留输运阶段外加一小段保持尾巴,
    # 既教"用力推",也教"均匀后收力"。
    hold_tail_steps = 20
    env = make_env(seed)
    obs_list: list[np.ndarray] = []
    act_list: list[np.ndarray] = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        below_target_steps = 0
        for _ in range(env.cfg.max_steps):
            action = teacher_fn(obs, env.cfg)
            obs_list.append(np.asarray(obs, dtype=np.float32))
            act_list.append(np.asarray(action, dtype=np.float32))
            obs, _, terminated, truncated, info = env.step(action)
            if info["success"]:
                below_target_steps += 1
                if below_target_steps >= hold_tail_steps:
                    break
            if terminated or truncated:
                break

    policy = model.policy
    obs_tensor = torch.as_tensor(np.stack(obs_list), device=model.device)
    act_tensor = torch.as_tensor(np.stack(act_list), device=model.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    n_pairs = len(obs_list)

    policy.set_training_mode(True)
    first_epoch_loss = last_epoch_loss = float("nan")
    for epoch in range(epochs):
        permutation = rng.permutation(n_pairs)
        losses = []
        for start in range(0, n_pairs, batch_size):
            idx = torch.as_tensor(permutation[start : start + batch_size], device=model.device)
            distribution = policy.get_distribution(obs_tensor[idx])
            predicted_mean = distribution.distribution.mean
            loss = torch.nn.functional.mse_loss(predicted_mean, act_tensor[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        epoch_loss = float(np.mean(losses))
        if epoch == 0:
            first_epoch_loss = epoch_loss
        last_epoch_loss = epoch_loss
    policy.set_training_mode(False)

    return {
        "bc_pairs": float(n_pairs),
        "bc_loss_first_epoch": first_epoch_loss,
        "bc_loss_last_epoch": last_epoch_loss,
    }


# 下方 KLToBCPPO.train() 的函数体抄自该 SB3 版本的 PPO.train(),只有一处
# 带标记的新增;升级 SB3 时请连同 requirements 一起更新此版本号。
KL_BC_SB3_BASE_VERSION = "2.9.0"


def _make_kl_to_bc_ppo():
    """构建 KLToBCPPO 类(惰性导入,与本模块其余 SB3 用法一致;train_ppo 在 kl_bc_coef>0 时调用)。

    v25 抗侵蚀系绳:在 v19/v20/v21/v24 中,PPO 微调总是从 BC 锚点滑走 ——
    在锚点处,保持工资在每个 batch 内近似常数(所有回合都达标),对策略
    梯度不可见;而 batch 内的方差全在成本侧(pH/干燥随场地难度变化)。
    唯一相干的梯度方向就是"省成本、少推":一个滑出工资区的单向棘轮。
    系绳在损失里加 ``kl_bc_coef * KL(pi_theta || pi_BC)``,其中 pi_BC 是
    冻结的 BC 后策略。高斯 KL 含方差项,因此还能压制 v24 观察到的 log_std
    回涨(ent_coef=0 时 sigma 0.067 -> 0.149)。
    """

    import torch as th
    import torch.nn.functional as F
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.utils import explained_variance

    class KLToBCPPO(PPO):
        def __init__(self, *args: Any, kl_bc_coef: float = 0.1, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.kl_bc_coef = float(kl_bc_coef)
            self.bc_policy = None
            # 最近一次更新的平均 KL,由面板回调读取。
            self.last_kl_to_bc = float("nan")

        def set_bc_reference(self) -> None:
            """把当前(BC 后)策略冻结一份副本作为 KL 锚点(BC 热启动完成后由 train_ppo 调用)。

            用"新建实例 + load_state_dict"而不是 deepcopy:BC 训练之后,
            在线策略缓存着连在计算图上的张量(``action_dist``),
            torch 拒绝对其 deepcopy。
            """

            bc_policy = self.policy_class(
                self.observation_space,
                self.action_space,
                self.lr_schedule,
                use_sde=self.use_sde,
                **self.policy_kwargs,
            ).to(self.device)
            bc_policy.load_state_dict(self.policy.state_dict())
            for param in bc_policy.parameters():
                param.requires_grad_(False)
            bc_policy.set_training_mode(False)
            self.bc_policy = bc_policy

        def _excluded_save_params(self) -> list[str]:
            # 冻结参考只是训练期状态;排除掉它,保存的 zip 才能被
            # 普通 PPO.load 加载用于评估。
            return super()._excluded_save_params() + ["bc_policy"]

        def train(self) -> None:
            """stable-baselines3 2.9.0 的 PPO.train() 原文 + KL-to-BC 项。

            新增只有标记为 ``# >>> KL-to-BC`` 的代码块;其余逐字照抄,
            因此 kl_bc_coef=0 时可复现原版 PPO。
            """

            self.policy.set_training_mode(True)
            self._update_learning_rate(self.policy.optimizer)
            clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
            if self.clip_range_vf is not None:
                clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

            entropy_losses = []
            pg_losses, value_losses = [], []
            clip_fractions = []
            kl_bc_values = []

            continue_training = True
            for epoch in range(self.n_epochs):
                approx_kl_divs = []
                for rollout_data in self.rollout_buffer.get(self.batch_size):
                    actions = rollout_data.actions
                    if isinstance(self.action_space, spaces.Discrete):
                        actions = rollout_data.actions.long().flatten()

                    values, log_prob, entropy = self.policy.evaluate_actions(
                        rollout_data.observations, actions
                    )
                    values = values.flatten()
                    advantages = rollout_data.advantages
                    if self.normalize_advantage and len(advantages) > 1:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    ratio = th.exp(log_prob - rollout_data.old_log_prob)
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                    pg_losses.append(policy_loss.item())
                    clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                    clip_fractions.append(clip_fraction)

                    if self.clip_range_vf is None:
                        values_pred = values
                    else:
                        values_pred = rollout_data.old_values + th.clamp(
                            values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                        )
                    value_loss = F.mse_loss(rollout_data.returns, values_pred)
                    value_losses.append(value_loss.item())

                    if entropy is None:
                        entropy_loss = -th.mean(-log_prob)
                    else:
                        entropy_loss = -th.mean(entropy)
                    entropy_losses.append(entropy_loss.item())

                    # >>> KL-to-BC:在本 minibatch 的观测上,计算当前策略
                    # 与冻结的 BC 后参考策略之间的解析高斯 KL
                    # (额外一次很小的前向传播)。
                    if self.bc_policy is not None and self.kl_bc_coef > 0.0:
                        current_dist = self.policy.get_distribution(rollout_data.observations)
                        with th.no_grad():
                            bc_dist = self.bc_policy.get_distribution(rollout_data.observations)
                        kl_bc = th.distributions.kl_divergence(
                            current_dist.distribution, bc_dist.distribution
                        ).sum(dim=-1).mean()
                    else:
                        kl_bc = th.zeros((), device=values.device)
                    kl_bc_values.append(float(kl_bc.item()))
                    # <<< KL-to-BC

                    loss = (
                        policy_loss
                        + self.ent_coef * entropy_loss
                        + self.vf_coef * value_loss
                        + self.kl_bc_coef * kl_bc  # >>> KL-to-BC
                    )

                    with th.no_grad():
                        log_ratio = log_prob - rollout_data.old_log_prob
                        approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                        approx_kl_divs.append(approx_kl_div)

                    if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                        continue_training = False
                        if self.verbose >= 1:
                            print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                        break

                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()

                self._n_updates += 1
                if not continue_training:
                    break

            explained_var = explained_variance(
                self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
            )

            self.logger.record("train/entropy_loss", np.mean(entropy_losses))
            self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
            self.logger.record("train/value_loss", np.mean(value_losses))
            self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
            self.logger.record("train/clip_fraction", np.mean(clip_fractions))
            self.logger.record("train/loss", loss.item())
            self.logger.record("train/explained_variance", explained_var)
            if hasattr(self.policy, "log_std"):
                self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
            # >>> KL-to-BC
            self.last_kl_to_bc = float(np.mean(kl_bc_values)) if kl_bc_values else float("nan")
            self.logger.record("train/kl_to_bc", self.last_kl_to_bc)
            # <<< KL-to-BC

            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            self.logger.record("train/clip_range", clip_range)
            if self.clip_range_vf is not None:
                self.logger.record("train/clip_range_vf", clip_range_vf)

    return KLToBCPPO


def make_best_checkpoint_callback(
    *,
    run_dir: Path,
    cfg,
    seed0: int,
    every_steps: int,
    task=None,
    eval_fn: Callable | None = None,
    subset_indices=None,
):
    """周期性用考卷子集小测当前策略,保留得分最高的 zip(最优检查点回调)。

    只收割、不转向:保存快照不触碰训练动力学。动机来自 v24 —— 当时训练
    峰值(143k 步 ep_rew +142,行为接近锚点)因只保存最终模型而丢失。
    子集为每族一个变体(s0)—— 20 个场景,每次小测约几秒 —— 在标准环境
    上按官方逐场景种子做确定性评估。第一次小测立即触发,先给纯 BC 后策略
    打分,使后续检查点能在同一子集上与锚点对比。训练结束后,最终模型和
    最优模型仍会各自产出官方完整考卷分数。
    由 run_train_standard 构建,作为 extra_callbacks 传入 train_ppo。
    """

    if every_steps <= 0:
        return None

    from stable_baselines3.common.callbacks import BaseCallback

    if eval_fn is None:
        eval_fn = evaluate_v2
    if subset_indices is None:
        from evaluation import SCENARIOS, VARIANTS_PER_FAMILY

        subset_indices = tuple(range(0, len(SCENARIOS), VARIANTS_PER_FAMILY))
    subset = tuple(subset_indices)

    class BestCheckpointCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.next_eval_at = 0
            self.best_subset_score = float("-inf")
            self.best_step = -1
            self.history: list[dict[str, float]] = []
            self.model_path = run_dir / "soilguard_ppo_best.zip"

        def _evaluate_now(self) -> None:
            result = eval_fn(
                ppo_policy_of(self.model), config=cfg, seed0=seed0, scenario_indices=subset
            )
            agg = result["aggregate"]
            score = float(agg["final_score"])
            self.history.append(
                {
                    "step": float(self.num_timesteps),
                    "subset_score": score,
                    "success": float(agg["success"]),
                    "ph_exposure": float(agg["ph_exposure"]),
                    "energy": float(agg["energy"]),
                }
            )
            if score > self.best_subset_score:
                self.best_subset_score = score
                self.best_step = int(self.num_timesteps)
                self.model.save(self.model_path)
                marker = " <- new best, saved"
            else:
                marker = f" (best {self.best_subset_score:.2f} @ step {self.best_step})"
            print(
                f"[checkpoint] step={self.num_timesteps} subset_score={score:.2f} "
                f"succ={agg['success']:.2f}{marker}"
            )
            if task is not None:
                task.get_logger().report_scalar(
                    title="Checkpoint Battery",
                    series="subset_score",
                    value=score,
                    iteration=int(self.num_timesteps),
                )

        def _on_step(self) -> bool:
            if self.num_timesteps >= self.next_eval_at:
                while self.next_eval_at <= self.num_timesteps:
                    self.next_eval_at += every_steps
                self._evaluate_now()
            return True

    return BestCheckpointCallback()


def make_curriculum_callback(
    *,
    threshold: float,
    window: int,
    step: float,
    start: float = 0.0,
    task=None,
):
    """自适应课程学习(v27):策略挣够了资格才提升难度(课程学习回调)。

    在滚动窗口内统计回合结束时的成功(最后一步仍均匀 —— 达标且守住,
    比只碰到过目标更严格)。当滚动成功率在当前难度下以满窗清过
    ``threshold``,难度提升 ``step``,并经 VecEnv.env_method 广播给所有
    并行 worker;窗口随之清零,让每级难度只凭自己的回合受评。若策略始终
    过不了线,难度原地不动 —— 卡在 0 级本身就是诊断(最简单的任务都
    从零学不会)。
    """

    from stable_baselines3.common.callbacks import BaseCallback

    class CurriculumCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.difficulty = float(start)
            self.successes: deque = deque(maxlen=window)
            self.episodes_at_level = 0
            self.advance_history: list[dict[str, float]] = []

        def _publish(self) -> None:
            # 面板回调在 rollout 结束时读取(与 last_kl_to_bc 同一套模式),
            # 使难度进入 jsonl/ClearML 曲线。
            self.model.curriculum_difficulty_current = float(self.difficulty)
            if self.successes:
                self.model.curriculum_rolling_success = float(
                    sum(self.successes) / len(self.successes)
                )

        def _on_training_start(self) -> None:
            self._publish()

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                if info.get("episode") is None:
                    continue
                self.successes.append(1.0 if info.get("success") else 0.0)
                self.episodes_at_level += 1
            if (
                self.difficulty < 1.0
                and self.episodes_at_level >= window
                and len(self.successes) == window
            ):
                rate = float(sum(self.successes) / len(self.successes))
                if rate >= threshold:
                    self.difficulty = float(min(1.0, self.difficulty + step))
                    self.training_env.env_method(
                        "set_curriculum_difficulty", self.difficulty
                    )
                    self.advance_history.append(
                        {
                            "step": float(self.num_timesteps),
                            "difficulty": self.difficulty,
                            "success_rate": rate,
                        }
                    )
                    print(
                        f"[curriculum] step={self.num_timesteps} difficulty -> "
                        f"{self.difficulty:.2f} (rolling succ {rate:.2f} over {window} eps)"
                    )
                    if task is not None:
                        task.get_logger().report_scalar(
                            title="Curriculum",
                            series="difficulty",
                            value=self.difficulty,
                            iteration=int(self.num_timesteps),
                        )
                    self.successes.clear()
                    self.episodes_at_level = 0
            self._publish()
            return True

    return CurriculumCallback()


def make_log_std_anneal_callback(
    *,
    anneal_to: float,
    anneal_steps: int,
    curriculum_cb=None,
):
    """v28:课程学习到顶之后收缩探索噪声(噪声退火回调)。

    v27 暴露了从零训练的结构性冲突:起步需要大噪声(正是 sigma = 1.0
    才让随机推动碰进工资区、开始学习),但 PPO 自己收缩 log_std 极慢
    (3M 步熵 17 -> 11.7,sigma 仍 0.64),于是策略一直在为"带噪声的
    自己"优化 —— 均值饱和,确定性考卷策略退化成满电压轰炸(能耗
    42 -> 79,考分下滑而训练成功率保持 0.9+)。

    机制:一旦 ``curriculum_cb.difficulty`` 达到 1.0(未提供课程回调则
    立即生效),对 log_std 施加逐步的上限,从策略当前最大值经
    ``anneal_steps`` 步线性降到 ``anneal_to``:``log_std.data.clamp_(max=cap)``。
    是上限而非赋值 —— PPO 可以比日程更快地收缩噪声,只是不能高于上限。
    -2.7(sigma ~ 半量程的 6.7% ~ 1.3 V)是 v24/v25 验证过的精细操作
    噪声水平。
    """

    if anneal_steps <= 0:
        return None

    import torch as th
    from stable_baselines3.common.callbacks import BaseCallback

    class LogStdAnnealCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.start_step: int | None = None
            self.anneal_from = 0.0
            self.completed = False

        def _apply(self) -> None:
            if self.start_step is None:
                if curriculum_cb is not None and curriculum_cb.difficulty < 1.0:
                    return
                self.start_step = int(self.num_timesteps)
                self.anneal_from = float(self.model.policy.log_std.detach().max().item())
                print(
                    f"[anneal] step={self.num_timesteps} noise annealing armed: "
                    f"log_std cap {self.anneal_from:.2f} -> {anneal_to:g} "
                    f"over {anneal_steps} steps"
                )
            progress = min(1.0, (self.num_timesteps - self.start_step) / float(anneal_steps))
            cap = self.anneal_from + progress * (anneal_to - self.anneal_from)
            with th.no_grad():
                self.model.policy.log_std.data.clamp_(max=cap)
            # 面板回调在 rollout 结束时读取。
            self.model.log_std_cap_current = float(cap)
            if not self.completed and progress >= 1.0:
                self.completed = True
                print(f"[anneal] step={self.num_timesteps} annealing complete: log_std cap = {anneal_to:g}")

        def _on_rollout_start(self) -> None:
            self._apply()

        def _on_step(self) -> bool:
            self._apply()
            return True

        def _on_training_end(self) -> None:
            # 最后一次 train() 更新可能把 log_std 顶回上限之上,而后面
            # 没有 rollout 再来钳制;保存下来的模型必须遵守退火日程。
            self._apply()

    return LogStdAnnealCallback()


def train_ppo(
    make_env: Callable[[int], SoilBoxEnv],
    steps: int,
    *,
    n_envs: int = 1,
    seed: int = SEED,
    verbose: int = 0,
    tensorboard_log: Path | None = None,
    run_name: str = "ppo",
    task=None,
    dashboard_dir: Path | None = None,
    log_std_init: float = 0.0, # 自 v21 起动作空间为 [-1,1]，σ=e^log_std 按半量程计：
                               # 0.0 意味着 σ=100% 半量程（v21 教训）；BC 热启动应配
                               # v19 等效噪声比例 ≈6.7%，即 log_std_init≈-2.7
    kl_bc_coef: float = 0.0,   # >0 启用 KL-to-BC 系绳（v25）；需要 bc_pretrain=True
    bc_pretrain: bool = False,
    bc_episodes: int = 300,
    bc_epochs: int = 20,
    bc_teacher_fn: Callable | None = None,   # None = 用 2D 比例控制器教师
    after_bc: Callable | None = None,
    extra_callbacks: list | None = None,
):
    """在 ``make_env(seed)`` 构建的 SoilBoxEnv 实例上训练一个 PPO 模型。

    由 run_train_standard 调用;BC 热启动、KL 系绳与各回调都在这里装配。
    ``n_envs > 1`` 用同样数量的环境进程并行收集 rollout(SubprocVecEnv),
    这就是多核加速点:环境步进是纯单线程 Python,n_envs=1 时主导墙钟时间。
    ``n_steps`` 缩放为 ``ROLLOUT_STEPS_TOTAL // n_envs``,每次更新仍使用
    与串行协议相同的总批量(2048 条转移)。注意:并行 rollout 会改变数据
    顺序,同种子下与串行运行不能逐位复现。

    历史教训:曾用 ``log_std_init=2.0``(初始动作 std ~7.4 V,即 +/-15 V
    量程的一半)替代 SB3 默认 0.0(std 1 V,量程的 7%):均匀化任务需要
    多电极协同推动,默认探索下 PPO 卡死在不作为的局部最优(实测:默认
    std 在 50 万步后停在 CV~0.31;加大 std 后 20 万步即达 CV~0.18)。
    """

    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv

    n_envs = max(int(n_envs), 1)
    if n_envs == 1:
        vec_env = Monitor(make_env(seed))
    else:
        def worker(rank: int):
            def _init():
                return Monitor(make_env(seed + rank))

            return _init

        vec_env = SubprocVecEnv([worker(rank) for rank in range(n_envs)])

    model_kwargs = dict(
        seed=seed,
        verbose=verbose,
        device="cpu",
        n_steps=max(ROLLOUT_STEPS_TOTAL // n_envs, 16),
        policy_kwargs=dict(log_std_init=log_std_init),
        tensorboard_log=str(tensorboard_log) if tensorboard_log else None,
    )
    if kl_bc_coef > 0.0:
        if not bc_pretrain:
            raise ValueError("kl_bc_coef > 0 requires bc_pretrain=True (the KL reference is the post-BC policy).")
        import stable_baselines3 as sb3_pkg

        if sb3_pkg.__version__ != KL_BC_SB3_BASE_VERSION:
            print(
                f"WARNING: KLToBCPPO.train() was copied from stable-baselines3 "
                f"{KL_BC_SB3_BASE_VERSION} but {sb3_pkg.__version__} is installed — "
                "verify PPO.train() still matches before trusting this run."
            )
        model = _make_kl_to_bc_ppo()("MlpPolicy", vec_env, kl_bc_coef=kl_bc_coef, **model_kwargs)
    else:
        # 原版 PPO,保持 v19-v24 运行逐位可比(不走抄来的 train())。
        model = PPO("MlpPolicy", vec_env, **model_kwargs)
    # 启动自检：打印策略网络里"实际生效"的探索噪声，端到端穿透
    # 配置->args->透传->SB3 整条链（v23 静默失效事故的硬闸）。
    actual_log_std = float(model.policy.log_std.detach().mean())
    print(
        f"self-check: policy log_std = {actual_log_std:.2f} "
        f"(sigma = {float(np.exp(actual_log_std)):.3f} half-range), "
        f"kl_bc_coef = {kl_bc_coef:g}, algo = {type(model).__name__}"
    )
    if bc_pretrain:
        bc_stats = pretrain_policy_bc(
            model, make_env, episodes=bc_episodes, epochs=bc_epochs, seed=seed,
            teacher_fn=bc_teacher_fn,
        )
        print(
            f"BC pretrain: {int(bc_stats['bc_pairs'])} pairs, "
            f"loss {bc_stats['bc_loss_first_epoch']:.3f} -> {bc_stats['bc_loss_last_epoch']:.3f}"
        )
        if kl_bc_coef > 0.0:
            model.set_bc_reference()
            print("self-check: KL-to-BC tether armed (reference = frozen post-BC policy)")
        if after_bc is not None:
            # 留给调用方的钩子:在 PPO 微调触碰之前,对纯 BC 策略
            # 做快照/评估(即锚点分数)。
            after_bc(model)
    attach_training_logger(
        model,
        task=task,
        tensorboard_log=tensorboard_log,
        run_name=run_name,
        verbose=verbose,
    )
    callbacks = [make_training_dashboard_callback(task, run_name, dashboard_dir=dashboard_dir)]
    callbacks.extend(extra_callbacks or [])
    callbacks = [cb for cb in callbacks if cb is not None]
    model.learn(
        total_timesteps=steps,
        tb_log_name=run_name,
        callback=callbacks or None,
    )
    return model


def load_ppo(model_path: Path):
    """加载已保存的 Stable-Baselines3 PPO 模型,用于纯评估类实验。"""

    if not model_path.exists():
        raise FileNotFoundError(f"PPO model not found: {model_path}")
    from stable_baselines3 import PPO

    return PPO.load(model_path, device="cpu")


def _jsonable_metrics(data: Any) -> Any:
    if isinstance(data, dict):
        return {str(key): _jsonable_metrics(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [_jsonable_metrics(value) for value in data]
    if isinstance(data, (bool, str, int, float)) or data is None:
        return data
    return str(data)


def safe_name(raw: str) -> str:
    cleaned = []
    for char in raw:
        if char.isalnum() or char in ("-", "_"):
            cleaned.append(char)
        else:
            cleaned.append("_")
    name = "".join(cleaned).strip("_")
    return name or "run"


def create_run_dir(output_root: Path, task_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = output_root / f"{timestamp}_{safe_name(task_name)}"
    run_dir = base
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = output_root / f"{base.name}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def training_config(args: argparse.Namespace):
    overrides = getattr(args, "reward_overrides", None) or {}
    if str(getattr(args, "env", "2d")) == "3d":
        from soilbox3d_env import DEFAULT_3D_CONFIG

        return DEFAULT_3D_CONFIG.with_overrides(**overrides)
    return DEFAULT_CONFIG.with_overrides(**overrides)


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    values = vars(args).copy()
    for key, value in list(values.items()):
        if isinstance(value, Path):
            values[key] = str(value)
    return values


def init_clearml(args: argparse.Namespace):
    if not args.clearml:
        return None

    try:
        from clearml import Task
    except ImportError as exc:
        raise ImportError(
            "ClearML tracking was requested, but `clearml` is not installed. "
            "Install dependencies with `pip install -r soilguard-ppo/requirements.txt`."
        ) from exc

    if args.clearml and not args.clearml_online:
        Task.set_offline(offline_mode=True)

    task = Task.init(
        project_name=args.clearml_project,
        task_name=args.clearml_task_name,
        output_uri=args.clearml_output_uri or None,
        reuse_last_task_id=False,
        auto_connect_arg_parser=False,
        auto_connect_frameworks=False,
    )
    task.connect(serializable_args(args), name="Args")
    task.connect(asdict(training_config(args)), name="SoilBoxConfig")
    return task


def report_metrics(task, title: str, metrics: dict[str, float], iteration: int = 0) -> None:
    if task is None:
        return
    logger = task.get_logger()
    for key, value in metrics.items():
        logger.report_scalar(title=title, series=key, value=value, iteration=iteration)


def report_model(task, model_path: Path, name: str) -> None:
    if task is None:
        return
    task.upload_artifact(name=name, artifact_object=str(model_path))


def write_diagnostics(args: argparse.Namespace, task, diagnostics: dict[str, Any]) -> Path:
    diagnostics_path = args.run_dir / "soilguard_ppo_diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(_jsonable_metrics(diagnostics), indent=2),
        encoding="utf-8",
    )
    if not args.clearml_no_artifacts:
        report_model(task, diagnostics_path, "soilguard_ppo_diagnostics")
    return diagnostics_path


def add_bool_arg(
    parser: argparse.ArgumentParser,
    name: str,
    *,
    default: bool | None = None,
    help: str = "",
) -> None:
    parser.add_argument(name, action=argparse.BooleanOptionalAction, default=default, help=help)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        default=None,
        choices=sorted(EXPERIMENTS),
        help="Named experiment configuration from experiment_configs.py.",
    )
    parser.add_argument(
        "--list-experiments",
        action="store_true",
        help="List available named experiment configurations and exit.",
    )
    parser.add_argument("--steps", type=int, default=None, help="Training steps for the PPO model.")
    parser.add_argument(
        "--n-envs",
        type=int,
        default=None,
        help=(
            "Parallel environment processes for rollout collection. >1 uses "
            "SubprocVecEnv (multi-core); per-update batch stays 2048 transitions."
        ),
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--verbose", type=int, default=None, help="Stable-Baselines3 verbosity.")
    add_bool_arg(
        parser,
        "--randomize-physics",
        default=None,
        help="Domain randomization (training only): resample soil physics per episode.",
    )
    add_bool_arg(
        parser,
        "--randomize-obs-noise",
        default=None,
        help="Domain randomization (training only): resample observation-noise level per episode.",
    )
    add_bool_arg(
        parser,
        "--randomize-initial-field",
        default=None,
        help="Training only: draw a fresh random initial moisture field every episode.",
    )
    parser.add_argument(
        "--eval-noise-levels",
        default=None,
        help=(
            "Comma-separated obs-noise levels; after training, the model retakes "
            "the scenario battery at each level. Empty = no noise evaluation."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Root directory for per-run output folders.",
    )
    parser.add_argument(
        "--tensorboard-log-dir",
        type=Path,
        default=None,
        help="Optional TensorBoard directory.",
    )
    add_bool_arg(parser, "--clearml", default=None, help="Enable ClearML experiment tracking.")
    parser.add_argument("--clearml-project", default=None, help="ClearML project name.")
    parser.add_argument("--clearml-task-name", default=None, help="ClearML task name.")
    parser.add_argument(
        "--clearml-output-uri",
        default=None,
        help="Optional ClearML model/artifact upload URI, e.g. s3://bucket/path.",
    )
    add_bool_arg(
        parser,
        "--clearml-online",
        default=None,
        help="Send ClearML logs directly to the server instead of storing an offline session.",
    )
    add_bool_arg(
        parser,
        "--clearml-no-artifacts",
        default=None,
        help="Log ClearML scalars only and skip artifact uploads. Useful when files.clear.ml is slow.",
    )
    return parser


def base_defaults() -> dict[str, Any]:
    return {
        "steps": 500_000,
        "n_envs": 1,
        "seed": SEED,
        "verbose": 0,
        "randomize_physics": False,
        "randomize_obs_noise": False,
        "randomize_initial_field": False,
        "eval_noise_levels": "",
        "output_dir": Path(__file__).resolve().parent / "artifacts",
        "tensorboard_log_dir": None,
        "clearml": False,
        "clearml_project": "SoilGuard/RL",
        "clearml_task_name": "soilbox-ppo",
        "clearml_output_uri": "",
        "clearml_online": False,
        "clearml_no_artifacts": False,
        # 行为克隆热启动(v16):PPO 微调前先模仿比例控制器。默认关闭。
        "bc_pretrain": False,
        "bc_episodes": 300,
        "bc_epochs": 20,
        # 初始探索噪声,自 v21 起以归一化动作单位计([-1, 1] 动作空间):
        # sigma = e^log_std_init,按半量程计。
        "log_std_init": 0.0,
        # v25 抗侵蚀系绳:loss += kl_bc_coef * KL(pi || 冻结的 BC 后策略)。
        # 0 = 关闭(原版 PPO,保持 v19-v24 语义)。
        "kl_bc_coef": 0.0,
        # v25 收割:每 N 步用 20 场景的考卷子集小测策略并保留最优快照。
        # 0 = 关闭。
        "checkpoint_every": 0,
        # v27 自适应课程(从零 PPO):把初始场骰子从易到难重塑;当
        # `curriculum_window` 个回合内的滚动回合末成功率清过
        # `curriculum_threshold` 时提升难度。min_contrast 是易端对比度
        # 下限 —— 用 `reward_audit.py --curriculum-audit` 标定,保证每个
        # 难度上"比例控制器 > idle"都成立(防"扩散替易场解题"陷阱)。
        "curriculum": False,
        "curriculum_threshold": 0.8,
        "curriculum_window": 200,
        "curriculum_step": 0.1,
        "curriculum_min_contrast": 0.6,
        # v28 探索噪声退火:课程达到 d=1.0 后(无课程则从第 0 步起),
        # 对 log_std 施加线性下降的上限,经 `log_std_anneal_steps` 步降到
        # `log_std_anneal_to`。0 = 关闭。
        "log_std_anneal_to": -2.7,
        "log_std_anneal_steps": 0,
        # 环境类型:"2d"(旧版 3x3 均匀场土箱)或 "3d"(真实几何 4x2x2,
        # 带电位场求解器,v30+)。
        "env": "2d",
    }


def list_experiments() -> None:
    print("Available experiments:")
    for name in sorted(EXPERIMENTS):
        config = EXPERIMENTS[name]
        print(f"  {name}:")
        for key in sorted(config):
            print(f"    {key}: {config[key]}")


def resolve_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    cli_args = parser.parse_args()
    if cli_args.list_experiments:
        list_experiments()
        raise SystemExit(0)

    if cli_args.experiment is None:
        parser.error("--experiment is required (use --list-experiments to see options)")
    experiment = cli_args.experiment
    values = base_defaults()
    values.update(EXPERIMENTS[experiment])
    values["experiment"] = experiment
    values["list_experiments"] = False

    for key, value in vars(cli_args).items():
        if key in ("experiment", "list_experiments"):
            continue
        if value is not None:
            values[key] = value

    return argparse.Namespace(**values)


def main() -> None:
    parser = build_parser()
    args = resolve_args(parser)
    run_experiment(args)


def run_named_experiment(
    experiment: str,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> argparse.Namespace:
    values = base_defaults()
    values.update(EXPERIMENTS[experiment])
    values["experiment"] = experiment
    values["list_experiments"] = False
    if overrides:
        values.update(dict(overrides))
    args = argparse.Namespace(**values)
    run_experiment(args)
    return args


def print_run_header(args: argparse.Namespace, cfg) -> None:
    print("SoilGuard PPO experiment")
    print(f"Experiment: {args.experiment}")
    print(f"Environment: {getattr(args, 'env', '2d')}")
    print(f"Config: {cfg}")
    print(f"Parallel envs: {args.n_envs}")
    print(
        "Domain randomization: physics="
        f"{args.randomize_physics} obs_noise={args.randomize_obs_noise} "
        f"initial_field={args.randomize_initial_field}"
    )
    print(f"Eval noise levels: {args.eval_noise_levels or '(none)'}")
    bc_on = getattr(args, "bc_pretrain", False)
    print(
        "BC warm start: "
        + (
            f"ON (episodes={getattr(args, 'bc_episodes', 300)}, "
            f"epochs={getattr(args, 'bc_epochs', 10)})"
            if bc_on
            else "off"
        )
    )
    print(f"log_std_init: {getattr(args, 'log_std_init', 0.0)}")
    kl_coef = float(getattr(args, "kl_bc_coef", 0.0))
    print("KL-to-BC tether: " + (f"ON (coef={kl_coef:g})" if kl_coef > 0 else "off"))
    ckpt_every = int(getattr(args, "checkpoint_every", 0))
    print(
        "Best-checkpoint battery: "
        + (f"every {ckpt_every} steps (20-scenario subset)" if ckpt_every > 0 else "off")
    )
    if bool(getattr(args, "curriculum", False)):
        print(
            "Curriculum: ON (adaptive, threshold="
            f"{getattr(args, 'curriculum_threshold', 0.8):g} over "
            f"{getattr(args, 'curriculum_window', 200)} eps, step="
            f"{getattr(args, 'curriculum_step', 0.1):g}, min_contrast="
            f"{getattr(args, 'curriculum_min_contrast', 0.6):g})"
        )
    else:
        print("Curriculum: off")
    anneal_steps = int(getattr(args, "log_std_anneal_steps", 0))
    print(
        "Noise annealing: "
        + (
            f"ON (log_std cap -> {getattr(args, 'log_std_anneal_to', -2.7):g} "
            f"over {anneal_steps} steps after curriculum tops out)"
            if anneal_steps > 0
            else "off"
        )
    )
    print(f"Run output directory: {args.run_dir}")


def run_train_standard(args: argparse.Namespace, cfg, task, tensorboard_log: Path | None) -> None:
    # BC 锚点:PPO 微调之前纯 BC 策略的考卷分数。微调后的模型不得低于
    # 此分 —— 低于即是"PPO 把热启动磨回不作为"的警报。
    bc_anchor_battery: dict[str, Any] = {}

    # ---- 环境类型:"2d"(旧版 3x3)或 "3d"(真实几何 4x2x2,带电位场
    # 求解器)。在这一处统一选定环境类、冻结考卷、基线策略和 BC 教师。
    env_kind = str(getattr(args, "env", "2d"))
    if env_kind == "3d":
        from evaluation3d import SCENARIOS_3D, VARIANTS_PER_FAMILY_3D, evaluate_v3d
        from evaluation3d import idle_policy_3d as _idle3d
        from soilbox3d_env import SoilBox3DEnv, proportional_policy_3d, rule_policy_3d

        if bool(getattr(args, "curriculum", False)):
            raise ValueError("curriculum learning is not implemented for the 3D env yet.")
        if args.randomize_physics or args.randomize_obs_noise:
            raise ValueError("physics/obs-noise domain randomization is not implemented for the 3D env yet.")
        eval_fn = evaluate_v3d
        subset_indices = tuple(range(0, len(SCENARIOS_3D), VARIANTS_PER_FAMILY_3D))
        idle_fn = _idle3d
        rule_fn = lambda obs: rule_policy_3d(obs, cfg)
        teacher_fn = proportional_policy_3d

        def make_base_env(env_seed: int):
            return SoilBox3DEnv(cfg, seed=env_seed, randomize_initial=args.randomize_initial_field)
    else:
        eval_fn = evaluate_v2
        subset_indices = None          # 检查点回调自行推导 2D 默认子集
        idle_fn = idle_policy
        rule_fn = lambda obs: rule_policy(obs, cfg.n_cells, max_electrode_voltage(cfg))
        teacher_fn = None              # pretrain_policy_bc 默认用 2D 教师
        make_base_env = None           # 2D 沿用下方参数更全的旧版环境工厂

    def evaluate_bc_anchor(model) -> None:
        anchor = eval_fn(ppo_policy_of(model), config=cfg, seed0=args.seed + 1000)
        print("== ppo_bc_anchor (pure BC, before PPO fine-tune) ==")
        print(format_battery_table(anchor))
        report_metrics(task, "ppo_bc_anchor", anchor["aggregate"], iteration=0)
        bc_anchor_battery.update(anchor)
        # 锚点是本项目迄今(v19-v24)最强的策略 —— 除分数外也把它
        # 存成可加载的模型文件。
        anchor_path = args.run_dir / "soilguard_ppo_bc_anchor.zip"
        model.save(anchor_path)
        print(f"saved: {anchor_path}")
        if not args.clearml_no_artifacts:
            report_model(task, anchor_path, "soilguard_ppo_bc_anchor")

    # 随机化开关只影响训练环境;下方的评估考卷始终在标称环境上注入
    # 冻结场景运行,保证分数跨实验可比。
    bc_on = bool(getattr(args, "bc_pretrain", False))
    curriculum_on = bool(getattr(args, "curriculum", False))
    if curriculum_on and not args.randomize_initial_field:
        raise ValueError("curriculum=True requires randomize_initial_field=True (the curriculum reshapes the initial-field dice).")
    checkpoint_cb = make_best_checkpoint_callback(
        run_dir=args.run_dir,
        cfg=cfg,
        seed0=args.seed + 1000,
        every_steps=int(getattr(args, "checkpoint_every", 0)),
        task=task,
        eval_fn=eval_fn,
        subset_indices=subset_indices,
    )
    curriculum_cb = None
    if curriculum_on:
        curriculum_cb = make_curriculum_callback(
            threshold=float(getattr(args, "curriculum_threshold", 0.8)),
            window=int(getattr(args, "curriculum_window", 200)),
            step=float(getattr(args, "curriculum_step", 0.1)),
            start=0.0,
            task=task,
        )
    anneal_cb = make_log_std_anneal_callback(
        anneal_to=float(getattr(args, "log_std_anneal_to", -2.7)),
        anneal_steps=int(getattr(args, "log_std_anneal_steps", 0)),
        curriculum_cb=curriculum_cb,
    )
    extra_callbacks = [cb for cb in (checkpoint_cb, curriculum_cb, anneal_cb) if cb is not None]
    curriculum_min_contrast = float(getattr(args, "curriculum_min_contrast", 0.6))
    standard_model = train_ppo(
        make_base_env
        if make_base_env is not None
        else (lambda env_seed: SoilBoxEnv(
            config=cfg,
            seed=env_seed,
            randomize=args.randomize_physics,
            randomize_noise=args.randomize_obs_noise,
            randomize_initial=args.randomize_initial_field,
            curriculum_difficulty=0.0 if curriculum_on else None,
            curriculum_min_contrast=curriculum_min_contrast,
        )),
        args.steps,
        n_envs=args.n_envs,
        seed=args.seed,
        verbose=args.verbose,
        tensorboard_log=tensorboard_log,
        run_name="ppo_standard",
        task=task,
        dashboard_dir=args.run_dir,
        log_std_init=float(getattr(args, "log_std_init", 0.0)),
        kl_bc_coef=float(getattr(args, "kl_bc_coef", 0.0)),
        bc_pretrain=bc_on,
        bc_episodes=int(getattr(args, "bc_episodes", 300)),
        bc_epochs=int(getattr(args, "bc_epochs", 20)),
        bc_teacher_fn=teacher_fn,
        after_bc=evaluate_bc_anchor if bc_on else None,
        extra_callbacks=extra_callbacks or None,
    )
    standard_path = args.run_dir / "soilguard_ppo_standard.zip"
    standard_model.save(standard_path)

    # 固定冻结考卷、全时程回放、分级计分,并让 idle(扩散地板)与 rule
    # 基线考同一份卷子。
    standard_result = eval_fn(ppo_policy_of(standard_model), config=cfg, seed0=args.seed + 1000)
    idle_result = eval_fn(idle_fn, config=cfg, seed0=args.seed + 1000)
    rule_result = eval_fn(rule_fn, config=cfg, seed0=args.seed + 1000)
    print("== ppo_standard ==")
    print(format_battery_table(standard_result))
    print("== baseline: idle (diffusion only) ==")
    print(format_battery_table(idle_result))
    print("== baseline: rule policy ==")
    print(format_battery_table(rule_result))

    standard_metrics = standard_result["aggregate"]
    report_metrics(task, "ppo_standard", standard_metrics, iteration=args.steps)
    report_metrics(task, "baseline_idle", idle_result["aggregate"], iteration=args.steps)
    report_metrics(task, "baseline_rule", rule_result["aggregate"], iteration=args.steps)
    if not args.clearml_no_artifacts:
        report_model(task, standard_path, "soilguard_ppo_standard")

    # 收割:训练中(按考卷子集)最优的检查点重考完整冻结考卷,
    # 使其官方分数与最终模型/锚点并排落账。
    best_battery = None
    best_checkpoint_info = None
    if checkpoint_cb is not None and checkpoint_cb.best_step >= 0:
        best_model = load_ppo(checkpoint_cb.model_path)
        best_battery = eval_fn(ppo_policy_of(best_model), config=cfg, seed0=args.seed + 1000)
        best_checkpoint_info = {
            "step": checkpoint_cb.best_step,
            "subset_score": checkpoint_cb.best_subset_score,
            "path": str(checkpoint_cb.model_path),
        }
        print(
            f"== ppo_best (checkpoint @ step {checkpoint_cb.best_step}, "
            f"subset score {checkpoint_cb.best_subset_score:.2f}) =="
        )
        print(format_battery_table(best_battery))
        report_metrics(task, "ppo_best", best_battery["aggregate"], iteration=checkpoint_cb.best_step)
        if not args.clearml_no_artifacts:
            report_model(task, checkpoint_cb.model_path, "soilguard_ppo_best")

    # 可选噪声评估:训练好的模型在每个指定的观测噪声水平下重考考卷
    # (传感器鲁棒性探针)。
    noise_levels = [
        float(part) for part in str(args.eval_noise_levels or "").split(",") if part.strip()
    ]
    noise_evaluation = {}
    for level in noise_levels:
        noise_result = eval_fn(
            ppo_policy_of(standard_model), config=cfg, obs_noise=level, seed0=args.seed + 1000
        )
        print(f"== ppo_standard @ obs_noise={level:g} ==")
        print(format_battery_table(noise_result))
        report_metrics(
            task, f"ppo_standard_noise_{level:g}", noise_result["aggregate"], iteration=args.steps
        )
        noise_evaluation[f"{level:g}"] = noise_result

    diagnostics = {
        "args": serializable_args(args),
        "config": asdict(cfg),
        "training_dashboard": training_dashboard_spec(),
        "evaluation_protocol": "scenario_battery_" + standard_result["battery_version"],
        "ppo_standard": standard_metrics,
        "ppo_standard_battery": standard_result,
        "ppo_bc_anchor_battery": bc_anchor_battery or None,
        "ppo_best_battery": best_battery,
        "best_checkpoint": best_checkpoint_info,
        "checkpoint_history": checkpoint_cb.history if checkpoint_cb is not None else None,
        "curriculum_advance_history": curriculum_cb.advance_history if curriculum_cb is not None else None,
        "curriculum_final_difficulty": curriculum_cb.difficulty if curriculum_cb is not None else None,
        "log_std_anneal": (
            {
                "start_step": anneal_cb.start_step,
                "completed": anneal_cb.completed,
                "anneal_to": float(getattr(args, "log_std_anneal_to", -2.7)),
                "anneal_steps": int(getattr(args, "log_std_anneal_steps", 0)),
            }
            if anneal_cb is not None
            else None
        ),
        "baseline_idle_battery": idle_result,
        "baseline_rule_battery": rule_result,
        "noise_evaluation": noise_evaluation,
        "saved_models": {"standard": str(standard_path)},
    }
    diagnostics_path = write_diagnostics(args, task, diagnostics)
    print(f"saved: {standard_path}")
    print(f"saved: {diagnostics_path}")


def run_experiment(args: argparse.Namespace) -> None:
    if args.steps <= 0:
        raise ValueError("`steps` must be > 0.")
    args.run_dir = create_run_dir(args.output_dir, args.clearml_task_name)
    cfg = training_config(args)
    task = init_clearml(args)

    tensorboard_log = args.tensorboard_log_dir
    if tensorboard_log is not None:
        tensorboard_log.mkdir(parents=True, exist_ok=True)

    print_run_header(args, cfg)
    try:
        run_train_standard(args, cfg, task, tensorboard_log)
    finally:
        if task is not None:
            task.close()


if __name__ == "__main__":
    main()
