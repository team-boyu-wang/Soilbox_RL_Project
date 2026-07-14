# -*- coding: utf-8 -*-
"""3D 土箱的场景考卷（BATTERY v6.0-3d,已冻结）。

在 4x2x2 真实几何土箱上:32 个分布族 × 5 个扰动变体 = 160 题。
族分类（用户拍板的丙方案,2026-07-10）:
  A. 单面板块(8)   —— 各轴单向搬运,含 60 cm 长轴难题
  B. 斜坡(5)       —— 连续场,直到三轴全对角
  C. 角块(6)       —— 多轴协同
  D. 结构盲区(4)   —— 含 double_end/mid_block（比例教师半箱差恒零,原理性失明）
  E. 随机(3)       —— 固定种子的确定性基场
  F. 幅值/水量(6)  —— 鲁棒性与导电率极端工况

计分用 BATTERY v5.0 每步小测验,在环境内部计算一次、与训练奖励逐字共用
—— "训练回报 ≡ 考卷分"在 3D 同样成立。
不要就地重排或修改族条目;新增请追加并升级版本号,保证分数跨版本可比。

对接：train_core 经 env="3d" 分支调用 evaluate_v3d 出成绩单与检查点子卷;
score_scenario / format_battery_table 复用 2D 模块（evaluation.py）。
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from evaluation import score_scenario, format_battery_table  # 复用 v5.0 计分打包
from soilbox3d_env import DEFAULT_3D_CONFIG, SoilBox3DConfig, SoilBox3DEnv

BATTERY_VERSION_3D = "v6.0-3d"

DRY = 0.10   # 族基场的默认干侧含水率
WET = 0.35   # 族基场的默认湿侧含水率


def _base(cfg: SoilBox3DConfig, fill: float = DRY) -> np.ndarray:
    """生成整箱同值的基场(默认全干)。"""
    return np.full((cfg.nx, cfg.ny, cfg.nz), fill)


def _fixed_rng_field(seed: int, low: float, high: float) -> Callable[[SoilBox3DConfig], np.ndarray]:
    """固定种子的 iid 均匀随机场构造器（种子固定 → 基场确定,考卷可冻结）。"""
    def build(cfg: SoilBox3DConfig) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.uniform(low, high, size=(cfg.nx, cfg.ny, cfg.nz))
    return build


def _families() -> list[tuple[str, Callable[[SoilBox3DConfig], np.ndarray]]]:
    """构造 32 个族的 (名称, 基场构造函数) 清单——考卷的题库定义。"""
    F: list[tuple[str, Callable[[SoilBox3DConfig], np.ndarray]]] = []

    def add(name: str, fn: Callable[[SoilBox3DConfig], np.ndarray]) -> None:
        F.append((name, fn))

    # ---- A. 单面板块系(8) ----
    def slab(axis: str, side: int, wet: float):
        # 单面/半箱湿板块构造器:axis 指定湿区所在的轴与形态
        def build(cfg):
            f = _base(cfg)
            if axis == "x":
                f[side, :, :] = wet
            elif axis == "xhalf":
                f[: cfg.nx // 2, :, :] = wet if side == 0 else f[: cfg.nx // 2, :, :]
                if side == 1:
                    f = _base(cfg)
                    f[cfg.nx // 2:, :, :] = wet
            elif axis == "y":
                f[:, side, :] = wet
            elif axis == "z":
                f[:, :, side] = wet
            return f
        return build
    add("x_end_left", slab("x", 0, WET))
    add("x_end_right", lambda cfg: (lambda f: (f.__setitem__((cfg.nx - 1, slice(None), slice(None)), WET), f)[1])(_base(cfg)))
    add("x_half_left", slab("xhalf", 0, 0.30))
    add("x_half_right", slab("xhalf", 1, 0.30))
    add("top_layer", slab("z", 1, 0.32))
    add("bottom_layer", slab("z", 0, 0.32))
    add("front_wall", slab("y", 0, 0.32))
    add("back_wall", slab("y", 1, 0.32))

    # ---- B. 斜坡系(5) ----
    def ramp(wx: float, wy: float, wz: float, lo: float = 0.08, hi: float = WET):
        # 沿 (wx,wy,wz) 方向的线性湿度斜坡构造器
        def build(cfg):
            xs = np.linspace(0, 1, cfg.nx)[:, None, None]
            ys = np.linspace(0, 1, cfg.ny)[None, :, None]
            zs = np.linspace(0, 1, cfg.nz)[None, None, :]
            r = wx * xs + wy * ys + wz * zs
            r = (r - r.min()) / max(float(r.max() - r.min()), 1e-9)
            return lo + (hi - lo) * (1.0 - r)
        return build
    add("x_ramp_desc", ramp(1, 0, 0))
    add("x_ramp_asc", lambda cfg: ramp(1, 0, 0)(cfg)[::-1, :, :].copy())
    add("xz_ramp", ramp(1, 0, 1))
    add("xy_ramp", ramp(1, 1, 0))
    add("xyz_ramp", ramp(1, 1, 1))

    # ---- C. 角块系(6) ----
    def cells(cell_list, wet):
        # 指定格子置湿的构造器
        def build(cfg):
            f = _base(cfg)
            for c in cell_list:
                f[c] = wet
            return f
        return build
    add("corner_top", cells([(0, 0, 1)], 0.45))
    add("corner_bottom_opp", cells([(3, 1, 0)], 0.45))
    add("opposite_corners", cells([(0, 0, 1), (3, 1, 0)], WET))
    add("adjacent_corners", cells([(0, 0, 1), (0, 1, 1)], WET))
    add("mid_block", lambda cfg: (lambda f: (f.__setitem__((slice(1, 3), slice(None), slice(None)), 0.30), f)[1])(_base(cfg)))
    add("x_bar", cells([(i, 0, 1) for i in range(4)], WET))

    # ---- D. 结构盲区系(4):比例教师的传感零空间 ----
    add("double_end", cells([(0, j, k) for j in range(2) for k in range(2)]
                            + [(3, j, k) for j in range(2) for k in range(2)], 0.30))
    def checker(parity_axis: str, wet: float = 0.28):
        # 棋盘交错湿场构造器:"all"=三维奇偶棋盘,"x"=x 切片交错
        def build(cfg):
            f = _base(cfg)
            for i in range(cfg.nx):
                for j in range(cfg.ny):
                    for k in range(cfg.nz):
                        val = (i + j + k) if parity_axis == "all" else i
                        if val % 2 == 0:
                            f[i, j, k] = wet
            return f
        return build
    add("checker_z", checker("all"))
    add("checker_x", checker("x"))
    add("column_wet", cells([(1, 0, 0), (1, 0, 1)], 0.40))

    # ---- E. 随机系(3):固定种子,基场确定可冻结 ----
    add("iid_random", _fixed_rng_field(9001, 0.08, 0.42))
    def smooth_random(cfg):
        # 平滑相关随机场:固定种子的随机方向斜坡 + 小噪声
        rng = np.random.default_rng(9002)
        d = rng.normal(size=3)
        d /= max(np.linalg.norm(d), 1e-9)
        base = ramp(d[0], d[1], d[2])(cfg)
        return np.clip(base + rng.uniform(-0.04, 0.04, size=base.shape), 0.02, 0.5)
    add("smooth_random", smooth_random)
    def sparse_wet(cfg):
        # 稀疏湿点:固定种子选 3 个格子置为很湿
        rng = np.random.default_rng(9003)
        f = _base(cfg, 0.09)
        cells_ = rng.choice(cfg.n_cells, size=3, replace=False)
        f.reshape(-1)[cells_] = 0.42
        return f
    add("sparse_wet", sparse_wet)

    # ---- F. 幅值/水量系(6):鲁棒性与导电率极端 ----
    add("x_end_low", cells([(0, j, k) for j in range(2) for k in range(2)], 0.30))
    add("x_end_high", cells([(0, j, k) for j in range(2) for k in range(2)], 0.40))
    add("top_low", lambda cfg: (lambda f: (f.__setitem__((slice(None), slice(None), 1), 0.28), f)[1])(_base(cfg)))
    add("top_high", lambda cfg: (lambda f: (f.__setitem__((slice(None), slice(None), 1), 0.40), f)[1])(_base(cfg)))
    def dry_box(cfg):
        # 干箱:总水量少,低导电重灾区
        f = _base(cfg, 0.06)
        f[0, :, :] = 0.22
        return f
    add("dry_box", dry_box)
    def wet_box(cfg):
        # 湿箱:总水量多,近饱和工况
        f = _base(cfg, 0.30)
        f[0, :, :] = 0.45
        return f
    add("wet_box", wet_box)
    return F


FAMILIES_3D = _families()
VARIANTS_PER_FAMILY_3D = 5   # 每族的扰动变体数
PERTURBATION_3D = 0.02       # 变体扰动幅度(基场上 ± 均匀小噪声)


def _perturbed(build: Callable, seed: int) -> Callable:
    """把族基场包装成"基场+固定种子小扰动"的变体构造器(每族探一个邻域)。"""
    def build_variant(cfg: SoilBox3DConfig) -> np.ndarray:
        grid = np.asarray(build(cfg), dtype=np.float64)
        rng = np.random.default_rng(seed)
        noise = rng.uniform(-PERTURBATION_3D, PERTURBATION_3D, size=grid.shape)
        return np.clip(grid + noise, 0.02, 0.5)
    return build_variant


def _build_battery():
    """展开 32 族 × 5 变体 = 160 题;变体种子 = 3000 + 族序号*100 + k。"""
    scenarios = []
    for fam_idx, (name, build) in enumerate(FAMILIES_3D):
        for k in range(VARIANTS_PER_FAMILY_3D):
            seed = 3000 + fam_idx * 100 + k
            scenarios.append((f"{name}_s{k}", _perturbed(build, seed)))
    return scenarios


SCENARIOS_3D = _build_battery()          # 冻结考卷:32 × 5 = 160 题


def idle_policy_3d(obs: np.ndarray) -> np.ndarray:
    """躺平基线:10 台电源全零输出(考卷地板分,陪考用)。"""
    del obs
    return np.zeros(DEFAULT_3D_CONFIG.ny + 2 * DEFAULT_3D_CONFIG.nx, dtype=np.float32)  # 10 台电源


def _rollout_scenario_3d(
    policy_fn: Callable[[np.ndarray], np.ndarray],
    cfg: SoilBox3DConfig,
    initial_moisture: np.ndarray,
    *,
    obs_noise: float,   # 观测噪声档(0=标准考卷)
    seed: int,          # 该题的固定种子
) -> dict[str, float]:
    """单题全时程回放:注入初始场跑满 max_steps,逐步累计计分分项。

    每步小测验分(step_quality/u/ph/e)由环境在 info["terms"] 里算好,
    这里只做时间积分——保证与训练奖励用的是同一份数字,口径不可能漂移。
    返回的指标字典供 score_scenario 打包成场景分。
    """
    env = SoilBox3DEnv(cfg, obs_noise=obs_noise, seed=seed)
    obs, _ = env.reset(seed=seed, options={"initial_moisture": initial_moisture})

    target = cfg.target_moisture_cv
    initial_cv = env.moisture_cv()
    t_reach = 0
    min_cv = initial_cv
    below = 0
    ph_sum = ph_mixed_sum = ph_peak = 0.0
    dry_sum = dry_peak = 0.0
    energy = power_sum = 0.0
    q_sum = u_sum = p_sum = e_sum = 0.0
    final_cv = initial_cv

    for t in range(1, cfg.max_steps + 1):
        obs, _, _, _, info = env.step(policy_fn(obs))
        tm = info["terms"]
        final_cv = float(info["moisture_cv"])
        min_cv = min(min_cv, final_cv)
        ph_sum += tm["ph_violation"]
        ph_mixed_sum += tm["ph_violation_mixed"]
        ph_peak = max(ph_peak, tm["ph_violation"])
        dry_sum += tm["local_dry_violation"]
        dry_peak = max(dry_peak, tm["local_dry_violation"])
        energy += tm["energy"]
        power_sum += tm["power_phys"]
        q_sum += tm["step_quality"]
        u_sum += tm["u_score"]
        p_sum += tm["ph_score"]
        e_sum += tm["energy_score"]
        if t_reach == 0 and final_cv <= target:
            t_reach = t
        if final_cv <= target:
            below += 1

    steps = float(cfg.max_steps)
    reached = t_reach > 0
    hold_frac = below / max(steps - t_reach + 1, 1.0) if reached else 0.0
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
        "energy_to_target": energy,
        "quality_mean": q_sum / steps,
        "uniformity_level_mean": u_sum / steps,
        "ph_level_mean": p_sum / steps,
        "energy_level_mean": e_sum / steps,
        "final_mean_pH": float(np.mean(env.pH)),
        "final_mean_moisture": float(np.mean(env.m)),
    }


def evaluate_v3d(
    policy_fn: Callable[[np.ndarray], np.ndarray],
    *,
    config: SoilBox3DConfig | Mapping[str, Any] | None = None,
    obs_noise: float = 0.0,
    seed0: int = 1000,
    scenario_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """跑冻结的 3D 考卷,输出结构与 2D 的 evaluate_v2 完全一致。

    scenario_indices 按原始题号选子集(种子仍 = seed0+原始题号,子集分
    与全卷同题严格可比)——检查点回调用它抽每族 s0 共 32 题的子卷;
    None = 全卷 160 题(官方成绩)。返回 {battery_version, worst_scenario,
    aggregate, per_scenario},供 train_core 写诊断与 ClearML。
    """

    cfg = config if isinstance(config, SoilBox3DConfig) else (
        DEFAULT_3D_CONFIG if config is None else SoilBox3DConfig(**dict(config)))
    indices = list(range(len(SCENARIOS_3D))) if scenario_indices is None else list(scenario_indices)
    per_scenario: list[dict[str, Any]] = []
    for idx in indices:
        name, build = SCENARIOS_3D[idx]
        metrics = _rollout_scenario_3d(
            policy_fn, cfg, build(cfg), obs_noise=obs_noise, seed=seed0 + idx)
        metrics.update(score_scenario(metrics))
        per_scenario.append({"scenario": name, **metrics})

    scores = np.array([row["scenario_score"] for row in per_scenario])
    worst_idx = int(np.argmin(scores))
    aggregate = {
        "success": float(np.mean([r["reached"] for r in per_scenario])),
        "final_score": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "worst_scenario_score": float(scores[worst_idx]),
        "moisture_cv": float(np.mean([r["final_cv"] for r in per_scenario])),
        "min_cv": float(np.mean([r["min_cv"] for r in per_scenario])),
        "hold_frac": float(np.mean([r["hold_frac"] for r in per_scenario])),
        "ph_exposure": float(np.mean([r["ph_exposure"] for r in per_scenario])),
        "ph_peak": float(np.max([r["ph_peak"] for r in per_scenario])),
        "dry_exposure": float(np.mean([r["dry_exposure"] for r in per_scenario])),
        "energy": float(np.mean([r["energy"] for r in per_scenario])),
        "quality_score": float(np.mean([r["quality_score"] for r in per_scenario])),
    }
    return {
        "battery_version": BATTERY_VERSION_3D,
        "worst_scenario": per_scenario[worst_idx]["scenario"],
        "aggregate": aggregate,
        "per_scenario": per_scenario,
    }
