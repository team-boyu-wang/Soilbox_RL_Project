# -*- coding: utf-8 -*-
"""Train PPO agents for the literature-backed SoilGuard soil-box environment."""

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
    """Wrap a Stable-Baselines3 model as policy_fn(obs) -> action."""

    return lambda obs: model.predict(obs, deterministic=True)[0]


class ClearMLTrainingDashboardWriter:
    """Forward selected Stable-Baselines3 training scalars to ClearML panels."""

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
            # Keyed by env index: with parallel envs a single shared value
            # would compare actions across different environments.
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
                        # What is actually subtracted from reward (stock vs
                        # acid-increment); falls back to stock for old envs.
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

            # KL(pi || pi_BC) of the most recent PPO update (KLToBCPPO only).
            # _on_rollout_end fires before the upcoming train(), so this lags
            # one iteration — fine for trend monitoring.
            kl_to_bc = getattr(self.model, "last_kl_to_bc", None)
            if kl_to_bc is not None and np.isfinite(kl_to_bc):
                metrics["kl_to_bc"] = float(kl_to_bc)

            for key, values in sorted(self.rollout_values.items()):
                if values:
                    metrics[key] = float(sum(values) / len(values))
            self.rollout_values.clear()

            self.history.append(metrics)
            if self.metrics_path is not None:
                with self.metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(metrics, sort_keys=True) + "\n")

            tensorboard_groups = {
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
            }
            for tensorboard_key, metric_key in tensorboard_groups.items():
                if metric_key in metrics:
                    self.logger.record(tensorboard_key, metrics[metric_key])

            if task is not None:
                logger = task.get_logger()
                clearml_groups = {
                    "Reward Components": [
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
                    "Policy Diagnostics": ["policy_entropy", "kl_to_bc"],
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
    """Attach ClearML/TensorBoard writers without changing PPO training logic."""

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
) -> dict[str, float]:
    """Behavior-clone the proportional controller into ``model.policy``.

    Warm start for the v15 diagnosis: with random initial fields the
    directional gradients cancel in weight space while the "act less" gradient
    is coherent, so PPO from scratch collapses into idling and never samples
    the hold-bonus income. Supervised imitation has none of those problems —
    each (obs, action) pair is an independent clean label — so the
    "read the field, push toward the dry side" mapping is carved in first and
    PPO then starts INSIDE the wage region, where both income and penalty
    gradients are alive.

    Trains only the action mean (MSE); the value head and log_std receive no
    gradient, and PPO's own optimizer state is untouched.
    """

    import torch

    from reward_audit import proportional_policy

    # Demonstrations are truncated shortly after the controller reaches the
    # target: the controller uniformizes in a few dozen steps and then idles,
    # so full 180-step episodes are ~2/3 "already uniform -> action ~ 0"
    # pairs. That imbalance dominates the MSE and clones a policy that points
    # the right way but is far too timid to actually reach (measured: ~0.6 V
    # on the canonical field). Keeping the transport phase plus a short
    # holding tail teaches both "push hard" and "ease off once uniform".
    hold_tail_steps = 20
    env = make_env(seed)
    obs_list: list[np.ndarray] = []
    act_list: list[np.ndarray] = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        below_target_steps = 0
        for _ in range(env.cfg.max_steps):
            action = proportional_policy(obs, env.cfg)
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


# The KLToBCPPO.train() body below is copied from this SB3 version's PPO.train()
# with one marked addition; bump this together with requirements when upgrading.
KL_BC_SB3_BASE_VERSION = "2.9.0"


def _make_kl_to_bc_ppo():
    """Build the KLToBCPPO class (lazy import, mirrors the module's SB3 usage).

    v25 anti-erosion tether: across v19/v20/v21/v24 PPO fine-tuning always
    slid AWAY from the BC anchor — at the anchor the hold wage is a
    near-constant within every batch (all episodes reach), so it is invisible
    to the policy gradient, while the batch variance lives on the cost side
    (pH/dry vary with field difficulty). The one coherent gradient direction
    is "save costs, push less": a one-way ratchet out of the wage region.
    The tether adds ``kl_bc_coef * KL(pi_theta || pi_BC)`` to the loss, where
    pi_BC is the FROZEN post-BC policy. The Gaussian KL includes variance
    terms, so it also suppresses the log_std regrowth observed in v24
    (sigma 0.067 -> 0.149 with ent_coef=0).
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
            # Mean KL of the most recent update, read by the dashboard callback.
            self.last_kl_to_bc = float("nan")

        def set_bc_reference(self) -> None:
            """Freeze a copy of the CURRENT (post-BC) policy as the KL anchor.

            Built as a fresh instance + load_state_dict instead of deepcopy:
            after BC training the live policy caches graph-connected tensors
            (``action_dist``) that torch refuses to deepcopy.
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
            # The frozen reference is training-time state; keep saved zips
            # loadable by plain PPO.load for evaluation.
            return super()._excluded_save_params() + ["bc_policy"]

        def train(self) -> None:
            """PPO.train() from stable-baselines3 2.9.0 + the KL-to-BC term.

            The addition is the block marked ``# >>> KL-to-BC``; everything
            else is verbatim so kl_bc_coef=0 reproduces vanilla PPO.
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

                    # >>> KL-to-BC: analytic Gaussian KL between the current
                    # policy and the frozen post-BC reference on this
                    # minibatch's observations (extra tiny forward pass).
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
):
    """Periodic battery-subset exam of the current policy; keep the best zip.

    Harvest, not steering: saving snapshots does not touch training dynamics.
    Motivated by v24, where the training peak (ep_rew +142 at 143k steps,
    near-anchor behaviour) was lost because only the final model was saved.
    The subset is one variant (s0) per family — 20 scenarios, ~seconds per
    exam — evaluated deterministically on the standard environment with the
    official per-scenario seeds. The first exam fires immediately, scoring
    the pure post-BC policy so later checkpoints are comparable to the anchor
    on the same subset. Official full-battery scores are still produced after
    training for both the final and the best model.
    """

    if every_steps <= 0:
        return None

    from stable_baselines3.common.callbacks import BaseCallback

    from evaluation import SCENARIOS, VARIANTS_PER_FAMILY

    subset = tuple(range(0, len(SCENARIOS), VARIANTS_PER_FAMILY))

    class BestCheckpointCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.next_eval_at = 0
            self.best_subset_score = float("-inf")
            self.best_step = -1
            self.history: list[dict[str, float]] = []
            self.model_path = run_dir / "soilguard_ppo_best.zip"

        def _evaluate_now(self) -> None:
            result = evaluate_v2(
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
    after_bc: Callable | None = None,
    extra_callbacks: list | None = None,
):
    """Train one PPO model on SoilBoxEnv instances built by ``make_env(seed)``.

    ``n_envs > 1`` collects rollouts from that many environment processes in
    parallel (SubprocVecEnv), which is the multi-core speed-up: the env stepping
    is pure single-threaded Python and dominates wall-clock at n_envs=1.
    ``n_steps`` is scaled to ``ROLLOUT_STEPS_TOTAL // n_envs`` so every update
    still uses the same total batch (2048 transitions) as the serial protocol.
    Note: parallel rollouts change data ordering, so runs are not bit-for-bit
    reproducible against serial ones with the same seed.

    ``log_std_init=2.0`` (initial action std ~7.4 V, half the +/-15 V range)
    instead of the SB3 default 0.0 (std 1 V, 7% of range): the uniformity task
    needs coordinated multi-electrode pushes, and with default exploration PPO
    stalls in the do-nothing local optimum (verified: default std plateaued at
    CV~0.31 after 500k steps; boosted std reached CV~0.18 by 200k).
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
        # Vanilla PPO keeps v19-v24 runs bit-comparable (no copied train()).
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
            model, make_env, episodes=bc_episodes, epochs=bc_epochs, seed=seed
        )
        print(
            f"BC pretrain: {int(bc_stats['bc_pairs'])} pairs, "
            f"loss {bc_stats['bc_loss_first_epoch']:.3f} -> {bc_stats['bc_loss_last_epoch']:.3f}"
        )
        if kl_bc_coef > 0.0:
            model.set_bc_reference()
            print("self-check: KL-to-BC tether armed (reference = frozen post-BC policy)")
        if after_bc is not None:
            # Hook for the caller to snapshot/evaluate the pure-BC policy
            # before PPO fine-tuning touches it (the anchor score).
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
    """Load a saved Stable-Baselines3 PPO model for eval-only experiments."""

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
        # Behavior-cloning warm start (v16): imitate the proportional
        # controller before PPO fine-tuning. Off by default.
        "bc_pretrain": False,
        "bc_episodes": 300,
        "bc_epochs": 20,
        # Initial exploration noise, in NORMALIZED action units since v21
        # ([-1, 1] action space): sigma = e^log_std_init of the half-range.
        "log_std_init": 0.0,
        # v25 anti-erosion tether: loss += kl_bc_coef * KL(pi || frozen post-BC
        # policy). 0 = off (vanilla PPO, keeps v19-v24 semantics).
        "kl_bc_coef": 0.0,
        # v25 harvest: every N steps exam the policy on a 20-scenario battery
        # subset and keep the best snapshot. 0 = off.
        "checkpoint_every": 0,
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
    print(f"Run output directory: {args.run_dir}")


def run_train_standard(args: argparse.Namespace, cfg, task, tensorboard_log: Path | None) -> None:
    # BC anchor: battery score of the pure-BC policy BEFORE PPO fine-tuning.
    # The fine-tuned model must not score below this — that is the "PPO ground
    # the warm start back into idling" alarm.
    bc_anchor_battery: dict[str, Any] = {}

    def evaluate_bc_anchor(model) -> None:
        anchor = evaluate_v2(ppo_policy_of(model), config=cfg, seed0=args.seed + 1000)
        print("== ppo_bc_anchor (pure BC, before PPO fine-tune) ==")
        print(format_battery_table(anchor))
        report_metrics(task, "ppo_bc_anchor", anchor["aggregate"], iteration=0)
        bc_anchor_battery.update(anchor)
        # The anchor has been the strongest policy of the project so far
        # (v19-v24) — keep it as a loadable artifact, not just a score.
        anchor_path = args.run_dir / "soilguard_ppo_bc_anchor.zip"
        model.save(anchor_path)
        print(f"saved: {anchor_path}")
        if not args.clearml_no_artifacts:
            report_model(task, anchor_path, "soilguard_ppo_bc_anchor")

    # Randomization switches affect TRAINING envs only; the evaluation battery
    # below always runs on the nominal environment with injected frozen
    # scenarios, so scores stay comparable across experiments.
    bc_on = bool(getattr(args, "bc_pretrain", False))
    checkpoint_cb = make_best_checkpoint_callback(
        run_dir=args.run_dir,
        cfg=cfg,
        seed0=args.seed + 1000,
        every_steps=int(getattr(args, "checkpoint_every", 0)),
        task=task,
    )
    standard_model = train_ppo(
        lambda env_seed: SoilBoxEnv(
            config=cfg,
            seed=env_seed,
            randomize=args.randomize_physics,
            randomize_noise=args.randomize_obs_noise,
            randomize_initial=args.randomize_initial_field,
        ),
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
        after_bc=evaluate_bc_anchor if bc_on else None,
        extra_callbacks=[checkpoint_cb] if checkpoint_cb is not None else None,
    )
    standard_path = args.run_dir / "soilguard_ppo_standard.zip"
    standard_model.save(standard_path)

    # Protocol v2: fixed scenario battery, full-horizon rollouts, graded
    # scoring, with idle (diffusion floor) and rule baselines on the same exam.
    standard_result = evaluate_v2(ppo_policy_of(standard_model), config=cfg, seed0=args.seed + 1000)
    idle_result = evaluate_v2(idle_policy, config=cfg, seed0=args.seed + 1000)
    rule_result = evaluate_v2(
        lambda obs: rule_policy(obs, cfg.n_cells, max_electrode_voltage(cfg)),
        config=cfg,
        seed0=args.seed + 1000,
    )
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

    # Harvest: the best in-training checkpoint (by battery subset) retakes the
    # FULL frozen battery, so its official score lands next to final/anchor.
    best_battery = None
    best_checkpoint_info = None
    if checkpoint_cb is not None and checkpoint_cb.best_step >= 0:
        best_model = load_ppo(checkpoint_cb.model_path)
        best_battery = evaluate_v2(ppo_policy_of(best_model), config=cfg, seed0=args.seed + 1000)
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

    # Optional noise evaluation: the trained model retakes the battery with
    # observation noise at each requested level (sensor-robustness probe).
    noise_levels = [
        float(part) for part in str(args.eval_noise_levels or "").split(",") if part.strip()
    ]
    noise_evaluation = {}
    for level in noise_levels:
        noise_result = evaluate_v2(
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
