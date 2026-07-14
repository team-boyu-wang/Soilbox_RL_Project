# -*- coding: utf-8 -*-
"""SoilGuard 土箱的场景考卷评估(协议 v2)。

取代旧的"单场景、成功即终止"报告协议:

- **冻结场景考卷**:一份带版本号、按序编号的初始湿度场清单。旋转变体用来
  暴露死记硬背(真正会均匀化的策略应当左湿、右湿一视同仁);幅值变体和
  固定种子的随机场检验泛化能力。考卷冻结不变,每个模型版本考同一份卷子。
- **全时程回放**:评估忽略环境的"达标即终止"标志,始终跑满 ``max_steps``,
  以便测量达标后能否守住均匀,以及整条轨迹的副作用。
- **分级计分**:达标只是场景分数中的一个加权分量,不再是乘法闸门 ——
  险些达标(CV 0.19)现在会比差得远(CV 0.29)得分更高,而不是双双归零。
- **基线锚定**:调用方应在同一份考卷上同时跑 ``idle_policy``(纯扩散,
  地板线)和 ``rule_policy``。

汇总指标按场景均值 +/- 标准差报告,并附最差场景 —— 后者才是诚实的
工程数字("最坏能坏到什么程度")。
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from soilbox_env import (
    DEFAULT_CONFIG,
    SoilBoxConfig,
    SoilBoxEnv,
    _clip01,
    _coerce_config,
)

BATTERY_VERSION = "v5.0"

# v5.0:逐步积分计分。每一步按 q_t = 0.60*U + 0.20*P + 0.20*E 打分(用环境
# 内的分段线性水平曲线,见 SoilBoxConfig),场景分 = 100 * 全时程 mean(q_t)。
# 训练奖励按步分期支付同一个 q_t,因此"训练回报 == 考卷分数"是构造上成立的。
# (v4.0 曾短暂使用只看终态的三轴计分;v3.0 有 0.40 达标闸门 + local_dry 轴
# —— 均已废弃。``reached``/hold_frac/dry 自 v4.0 起仅报告、不计分。)

# 暴露量归一化基准线(工程取值,待用真实土箱数据重新标定):exposure =
# 整个回合的逐步平均违规量。峰值一并报告但不计分 —— 旧的按峰值定零分线
# 被证明对本任务结构性不可达(见 v6-v9 分析)。
PH_EXPOSURE_BAD = 1.0
DRY_EXPOSURE_BAD = 0.08


def _base(cfg: SoilBoxConfig, fill: float = 0.10) -> np.ndarray:
    return np.full((cfg.grid_rows, cfg.grid_cols), fill)


def _cells(positions: list[tuple[int, int]], wet: float) -> Callable[[SoilBoxConfig], np.ndarray]:
    def build(cfg: SoilBoxConfig) -> np.ndarray:
        grid = _base(cfg)
        for i, j in positions:
            grid[i, j] = wet
        return grid

    return build


def _column_wet(col: int, wet: float) -> Callable[[SoilBoxConfig], np.ndarray]:
    def build(cfg: SoilBoxConfig) -> np.ndarray:
        grid = _base(cfg)
        grid[:, col] = wet
        return grid

    return build


def _row_wet(row: int, wet: float) -> Callable[[SoilBoxConfig], np.ndarray]:
    def build(cfg: SoilBoxConfig) -> np.ndarray:
        grid = _base(cfg)
        grid[row, :] = wet
        return grid

    return build


def _half_wet(axis: str, wet: float) -> Callable[[SoilBoxConfig], np.ndarray]:
    def build(cfg: SoilBoxConfig) -> np.ndarray:
        grid = _base(cfg)
        if axis == "left":
            grid[:, :2] = wet
        else:  # 即 axis == "top"
            grid[:2, :] = wet
        return grid

    return build


def _border_ring(wet: float) -> Callable[[SoilBoxConfig], np.ndarray]:
    def build(cfg: SoilBoxConfig) -> np.ndarray:
        grid = _base(cfg, wet)
        grid[1, 1] = 0.10
        return grid

    return build


def _gradient(axis: str) -> Callable[[SoilBoxConfig], np.ndarray]:
    def build(cfg: SoilBoxConfig) -> np.ndarray:
        levels = np.array([0.35, 0.225, 0.10])
        grid = np.tile(levels, (3, 1))
        return grid if axis == "x" else grid.T

    return build


# 20 个分布族。每族是一个确定性的基准图样;考卷对每族实例化 5 个扰动变体
# (在基准上加固定种子的 +/- 小噪声),构成一份冻结的 100 场景考卷。
FAMILIES: list[tuple[str, Callable[[SoilBoxConfig], np.ndarray]]] = [
    ("left_col", _column_wet(0, 0.35)),          # 训练标准场景
    ("right_col", _column_wet(2, 0.35)),         # 镜像:查死记硬背
    ("back_row", _row_wet(0, 0.35)),             # 90 度旋转
    ("front_row", _row_wet(2, 0.35)),
    ("middle_col", _column_wet(1, 0.35)),        # 内部湿:需向外摊平
    ("middle_row", _row_wet(1, 0.35)),
    ("left_col_low", _column_wet(0, 0.30)),      # 幅值变体
    ("left_col_high", _column_wet(0, 0.40)),
    ("corner_tl", _cells([(0, 0)], 0.40)),       # 单个湿格的各种拓扑
    ("corner_br", _cells([(2, 2)], 0.40)),
    ("center_cell", _cells([(1, 1)], 0.40)),
    ("diag_main", _cells([(0, 0), (1, 1), (2, 2)], 0.35)),
    ("diag_anti", _cells([(0, 2), (1, 1), (2, 0)], 0.35)),
    ("opposite_corners", _cells([(0, 0), (2, 2)], 0.35)),
    ("adjacent_corners", _cells([(0, 0), (0, 2)], 0.35)),
    ("cross_wet", _cells([(1, 1), (0, 1), (1, 0), (1, 2), (2, 1)], 0.30)),
    ("left_half", _half_wet("left", 0.28)),
    ("top_half", _half_wet("top", 0.28)),
    ("border_ring", _border_ring(0.30)),         # 湿环、干中心
    ("gradient_x", _gradient("x")),              # 平滑坡度
]

VARIANTS_PER_FAMILY = 5
# 种子变体的扰动幅度:在基准图样上加 +/- 小噪声,使每族检验的是一个邻域,
# 而不是单个点。
PERTURBATION = 0.02


def _perturbed(
    build: Callable[[SoilBoxConfig], np.ndarray], seed: int
) -> Callable[[SoilBoxConfig], np.ndarray]:
    def build_variant(cfg: SoilBoxConfig) -> np.ndarray:
        grid = build(cfg)
        rng = np.random.default_rng(seed)
        noise = rng.uniform(-PERTURBATION, PERTURBATION, size=grid.shape)
        return np.clip(grid + noise, 0.02, 0.5)

    return build_variant


def _build_battery() -> list[tuple[str, Callable[[SoilBoxConfig], np.ndarray]]]:
    scenarios = []
    for fam_idx, (name, build) in enumerate(FAMILIES):
        for k in range(VARIANTS_PER_FAMILY):
            seed = 3000 + fam_idx * 100 + k
            scenarios.append((f"{name}_s{k}", _perturbed(build, seed)))
    return scenarios


# 冻结考卷(20 族 x 5 个种子变体 = 100 个场景)。
# 不要就地重排或改动条目 —— 新增分布族时同步升级 BATTERY_VERSION,
# 保证分数跨模型版本可比。
SCENARIOS: list[tuple[str, Callable[[SoilBoxConfig], np.ndarray]]] = _build_battery()


def idle_policy(obs: np.ndarray) -> np.ndarray:
    """什么都不做的基线:仅靠扩散给出考卷的地板分。"""

    del obs
    return np.zeros(DEFAULT_CONFIG.total_electrode_count, dtype=np.float32)


def _rollout_scenario(
    policy_fn: Callable[[np.ndarray], np.ndarray],
    cfg: SoilBoxConfig,
    initial_moisture: np.ndarray,
    *,
    obs_noise: float,
    seed: int,
) -> dict[str, float]:
    env = SoilBoxEnv(cfg, obs_noise=obs_noise, seed=seed)
    obs, _ = env.reset(seed=seed, options={"initial_moisture": initial_moisture})

    target = cfg.target_moisture_cv
    initial_cv = env.moisture_cv()
    t_reach = 0
    min_cv = initial_cv
    below_target_steps = 0
    ph_sum = 0.0
    ph_mixed_sum = 0.0
    ph_peak = 0.0
    dry_sum = 0.0
    dry_peak = 0.0
    energy = 0.0
    power_sum = 0.0
    energy_to_target = 0.0
    quality_sum = 0.0
    u_sum = 0.0
    ph_score_sum = 0.0
    energy_score_sum = 0.0
    final_cv = initial_cv

    for t in range(1, cfg.max_steps + 1):
        action = policy_fn(obs)
        # 全时程协议:环境可能在达标时置 terminated,但评估继续走步,
        # 以测量稳定性与总暴露量。
        obs, _, _, _, info = env.step(action)
        terms = info["terms"]
        final_cv = float(info["moisture_cv"])
        min_cv = min(min_cv, final_cv)
        ph_sum += float(terms["ph_violation"])
        ph_mixed_sum += float(terms["ph_violation_mixed"])
        ph_peak = max(ph_peak, float(terms["ph_violation"]))
        dry_sum += float(terms["local_dry_violation"])
        dry_peak = max(dry_peak, float(terms["local_dry_violation"]))
        energy += float(terms["energy"])
        power_sum += float(terms["power_phys"])
        # v5.0:逐步小测验,只在环境内部计算一次,并原样共享给训练奖励 ——
        # 两边的计分不可能漂移分叉。
        quality_sum += float(terms["step_quality"])
        u_sum += float(terms["u_score"])
        ph_score_sum += float(terms["ph_score"])
        energy_score_sum += float(terms["energy_score"])
        if t_reach == 0:
            energy_to_target = energy
            if final_cv <= target:
                t_reach = t
        if final_cv <= target:
            below_target_steps += 1

    steps = float(cfg.max_steps)
    reached = t_reach > 0
    hold_frac = below_target_steps / max(steps - t_reach + 1, 1.0) if reached else 0.0
    return {
        "initial_cv": float(initial_cv),
        "reached": float(reached),
        "t_reach": float(t_reach if reached else -1),
        "final_cv": final_cv,
        "min_cv": float(min_cv),
        "hold_frac": float(hold_frac),
        "ph_exposure": ph_sum / steps,
        "ph_exposure_mixed": ph_mixed_sum / steps,
        "ph_peak": ph_peak,
        "dry_exposure": dry_sum / steps,
        "dry_peak": dry_peak,
        "energy": energy,
        "power_mean": power_sum / steps,
        "energy_to_target": energy_to_target if reached else energy,
        "quality_mean": quality_sum / steps,
        "uniformity_level_mean": u_sum / steps,
        "ph_level_mean": ph_score_sum / steps,
        "energy_level_mean": energy_score_sum / steps,
        "final_mean_pH": float(np.mean(env.pH)),
        "final_mean_moisture": float(np.mean(env.m)),
    }


def score_scenario(metrics: Mapping[str, float]) -> dict[str, float]:
    """把回放指标打包成 [0, 100] 的单场景分数(被 3D 考卷 evaluation3d 复用)。

    考卷 v5.0:每一步按 q_t = 0.60*U(cv_t) + 0.20*P(viol_t) + 0.20*E(power_t)
    打分(分段线性水平曲线,在环境内部计算并原样共享给训练奖励),场景分
    就是全时程的 100 * mean(q_t)。由此:
      - 早达标和守得住都有回报(时间积分)—— v3.0 的"时间不计分/保持
        不计分"缺陷不复存在;
      - 训练回报(无折扣、钩子为 0)与本分数构造上相等 —— 不可能再出现
        训练与考卷的对齐缝隙。
    ``reached`` 与 ``dry_score`` 仍为仅报告的监控指标。
    """

    dry_score = _clip01(1.0 - metrics["dry_exposure"] / DRY_EXPOSURE_BAD)  # 仅报告、不计分
    return {
        "uniformity_score": float(metrics["uniformity_level_mean"]),
        "ph_score": float(metrics["ph_level_mean"]),
        "dry_score": float(dry_score),
        "energy_score": float(metrics["energy_level_mean"]),
        "quality_score": float(metrics["quality_mean"]),
        "scenario_score": float(100.0 * metrics["quality_mean"]),
    }


def evaluate_v2(
    policy_fn: Callable[[np.ndarray], np.ndarray],
    *,
    config: SoilBoxConfig | Mapping[str, Any] | None = None,
    obs_noise: float = 0.0,
    seed0: int = 1000,
    scenario_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """跑冻结场景考卷,返回逐场景 + 汇总结果(训练管线的 2D 官方考试入口)。

    ``scenario_indices`` 按原始场景序号选取考卷的一个子集(每场景种子仍为
    ``seed0 + 原序号``,故子集指标与完整考卷中相同场景直接可比)。默认
    None 跑完整冻结考卷 —— 官方协议不受影响。子集运行只供训练中低成本的
    检查点排名;官方分数必须来自完整考卷。
    """

    cfg = _coerce_config(config)
    indices = list(range(len(SCENARIOS))) if scenario_indices is None else list(scenario_indices)
    per_scenario: list[dict[str, Any]] = []
    for idx in indices:
        name, build = SCENARIOS[idx]
        metrics = _rollout_scenario(
            policy_fn,
            cfg,
            build(cfg),
            obs_noise=obs_noise,
            seed=seed0 + idx,
        )
        metrics.update(score_scenario(metrics))
        per_scenario.append({"scenario": name, **metrics})

    scores = np.array([row["scenario_score"] for row in per_scenario])
    worst_idx = int(np.argmin(scores))
    aggregate = {
        "success": float(np.mean([row["reached"] for row in per_scenario])),
        "final_score": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "worst_scenario_score": float(scores[worst_idx]),
        "moisture_cv": float(np.mean([row["final_cv"] for row in per_scenario])),
        "min_cv": float(np.mean([row["min_cv"] for row in per_scenario])),
        "hold_frac": float(np.mean([row["hold_frac"] for row in per_scenario])),
        "ph_exposure": float(np.mean([row["ph_exposure"] for row in per_scenario])),
        "ph_peak": float(np.max([row["ph_peak"] for row in per_scenario])),
        "dry_exposure": float(np.mean([row["dry_exposure"] for row in per_scenario])),
        "energy": float(np.mean([row["energy"] for row in per_scenario])),
        "quality_score": float(np.mean([row["quality_score"] for row in per_scenario])),
    }
    return {
        "battery_version": BATTERY_VERSION,
        "worst_scenario": per_scenario[worst_idx]["scenario"],
        "aggregate": aggregate,
        "per_scenario": per_scenario,
    }


def format_battery_table(result: Mapping[str, Any]) -> str:
    """生成按分布族汇总的控制台表格(完整的逐场景数据保留在返回的
    dict / diagnostics json 里)。"""

    families: dict[str, list[Mapping[str, float]]] = {}
    for row in result["per_scenario"]:
        family = row["scenario"].rsplit("_s", 1)[0]
        families.setdefault(family, []).append(row)

    lines = [
        f"battery {result['battery_version']}  worst={result['worst_scenario']}",
        f"{'family':<20}{'n':>3}{'cv0':>6}{'reach':>7}{'final_cv':>10}{'phExp':>8}{'dryExp':>8}{'energy':>8}{'score':>8}",
    ]
    for family, rows in families.items():
        mean = lambda key: float(np.mean([r[key] for r in rows]))
        lines.append(
            f"{family:<20}{len(rows):>3}{mean('initial_cv'):>6.2f}{mean('reached'):>7.2f}"
            f"{mean('final_cv'):>10.3f}{mean('ph_exposure'):>8.3f}"
            f"{mean('dry_exposure'):>8.3f}{mean('energy'):>8.1f}{mean('scenario_score'):>8.1f}"
        )
    agg = result["aggregate"]
    lines.append(
        f"{'AGGREGATE':<20}{len(result['per_scenario']):>3}{'':>6}{agg['success']:>7.2f}"
        f"{agg['moisture_cv']:>10.3f}{agg['ph_exposure']:>8.3f}{agg['dry_exposure']:>8.3f}"
        f"{agg['energy']:>8.1f}{agg['final_score']:>8.1f}"
    )
    return "\n".join(lines)
