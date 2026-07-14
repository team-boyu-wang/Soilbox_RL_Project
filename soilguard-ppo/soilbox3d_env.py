# -*- coding: utf-8 -*-
"""SoilGuard 3D 电渗土箱环境（真电位场求解器版）。

几何（与真实土箱一致）：60 x 30 x 30 cm，15 cm 单元 → 网格
(nx, ny, nz) = (4, 2, 2) = 16 格。电极为 20 根 30 cm 石墨棒（用户硬件规格,
2026-07-10）：

    x 轴：左右两个 30x30 端面各竖插 2 根（每 y 行一根,贯穿全高） →  4 根
    y 轴：前后两个 60x30 侧面各竖插 4 根（每 x 切片一根,贯穿全高） →  8 根
    z 轴：顶面横躺 4 根 + 底部预埋 4 根（PCPSS 式表面/底部电极,
          文献 s41598-2024）                                        →  8 根

相对 2D 模块的物理升级（"A 方案",用户决策 2026-07-10）：不再假设"每线均匀
场",而是每步在格点图上求解离散导电方程 div(sigma grad phi) = 0,电极作为
外部定压节点经接触电导与接触格耦合。以下真实效应全部自然涌现（非手写）：
鼓形偶极电流路径、切片间串场、湿层承载更多电流、按界面电流符号决定产酸/
产碱位置（Asadi 2013 电解机理）、能耗 = 精确的 sum(V_e * I_e)。

重力：默认 0。文献依据（toxics-12-00758）："鉴于土样渗透系数低
(4.0e-7 cm/s),重力作用下电解液的日渗漏损失可忽略。"该参数仅为
粗颗粒土域随机化预留。

计分：BATTERY v5.0 每步小测验,与 2D 模块逐字共用同一套 level 折线/权重/
峰均混合口径,因此"训练回报 ≡ 考卷分"在 3D 同样由构造成立。

对接关系：evaluation3d.py 用本环境跑 160 题考卷；train_core.py 经
env="3d" 配置键实例化本环境并用 proportional_policy_3d 做 BC 教师。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError("soilbox3d_env requires gymnasium.") from exc

from soilbox_env import (
    FINAL_SCORE_WEIGHTS,
    _clip01,
    _level_score,
)

SEED = 42


@dataclass(frozen=True)
class SoilBox3DConfig:
    """4x2x2 真实几何土箱的全部物理与计分参数（不可变数据类）。

    被 SoilBox3DEnv / evaluation3d / 教师策略共用；实验配置里的
    reward_overrides 经 with_overrides 覆盖字段。
    """

    nx: int = 4                # x 方向格数（60cm 长轴）
    ny: int = 2                # y 方向格数（30cm 宽）
    nz: int = 2                # z 方向格数（30cm 高）
    cell_cm: float = 15.0      # 单元边长（cm）
    # 协议（用户决策 2026-07-11/13）：72 小时处理窗 × 1 小时控制节奏
    # → 72 步 × 3600 s。窗口依据（实测）：当前电压上限下 24h 物理退化、
    # 48h 边缘,72h 才恢复教师全面优于躺平。节奏依据：72 步回合恰好落进
    # gamma=0.99 的信用分配视野,算力比 15min 节奏省 4 倍,且小时级策略
    # 可直接部署到 15min 硬件。每秒制动力学（EO 输运、pH）随 time_step_s
    # 自动缩放；下方每步制常数已按 dt=3600 重述。
    max_steps: int = 72        # 每局步数（72 步 = 72 小时）
    time_step_s: float = 3600.0  # 每步物理时长（秒）
    # 每对电源上限（用户硬件规格 2026-07-11）：每对 = 一台 0..V_max 的
    # 实验室电源 + 极性开关——动作是对电压差 ∈ [-1,1]×上限
    # （|a| = 电源旋钮读数,符号 = 接线极性）。按跨度配平的"等梯度设计"：
    # x 对跨 60cm → 0-60V,y/z 对跨 30cm → 0-30V,各轴名义 1 V/cm。
    x_pair_max_v: float = 60.0   # x 轴电极对电压上限（V）
    yz_pair_max_v: float = 30.0  # y/z 轴电极对电压上限（V）
    # --- 土壤水分 ---
    m_wet: float = 0.35        # 经典初始场的湿侧含水率
    m_dry: float = 0.10        # 经典初始场的干侧含水率
    m_threshold: float = 0.12  # 孔隙水导电因子的半饱和点（见 _sigma）
    electroosmotic_permeability_cm2_per_s_v: float = 3.775e-5  # 电渗渗透系数 ke（cm²/(s·V)）
    # 基质吸力扩散（每步制,15cm 单元 + 1h 步长）：
    # k = D_w * dt / dx² = 8.3e-5 cm²/s * 3600 s / (15 cm)² ≈ 0.00133
    # （与历来相同的湿软黏土 D_w；数值随 dt 线性缩放）。
    moisture_diffusion_per_step: float = 0.00133
    # 重力排水（每步制,上格水量向下移动的比例）。默认 0：低渗透软黏土在
    # 小时级步长下可忽略（toxics-12-00758 原文）；仅为粗颗粒土域随机化
    # 预留。注意：这是每步制常数——若启用,改 time_step_s 时必须手动重标
    # （它不像每秒制速率那样自动缩放）。
    gravity_per_step: float = 0.0
    # 按格流出预算：一格单步的总流出（EO+重力一算,扩散另算）不得超过
    # 其水量的该比例。修复 2D 逐边限幅在多条出边同时流出时的超支隐患。
    outflow_budget_fraction: float = 0.8
    # --- 电位场求解器 ---
    # 边电导 g = sigma_avg(m) ×（几何因子 A/dx,已吸收进单位使 g 为 O(1)）。
    # 接触电导把电极棒与接触格耦合：g_c = contact_scale * sigma(m_cell)。
    # 取值来自圆柱扩散电阻估算：直径 ~1cm 的棒在 15cm 格内,
    # g_rod ~ 2πL/ln(R/r) ≈ 35,对比边电导 A/dx=15 → 比值 ≈ 2。
    # （标定前的猜测值 0.5 会让接触电阻吃掉 ~57% 对电压,饿死 60cm 长轴。）
    contact_scale: float = 2.0
    # 微小对角正则,防止格子干透（sigma→0 断连）时矩阵奇异。
    solver_regularization: float = 1e-9
    # --- pH 电解/迁移（每秒制动力学,Asadi 2013）---
    k_acid_gen_per_s: float = 0.10 / 3600.0   # 阳极产酸速率（每秒·单位归一电流）
    k_base_gen_per_s: float = 0.06 / 3600.0   # 阴极产碱速率（酸>碱=净酸化不对称）
    k_acid_mig_per_s: float = 0.06 / 3600.0   # 酸随电流迁移速率（酸锋更快）
    k_base_mig_per_s: float = 0.03 / 3600.0   # 碱随电流迁移速率
    ph_half_amount: float = 0.5   # 净酸/碱量到 pH≈4.5/9.5 的饱和映射半量
    ph_tol: float = 1.0           # pH 容差带（7±1 内不算越界）
    # 界面/边电流喂给 pH 动力学前的归一化参考（I_norm = I / current_ref）。
    # 按 72h 协议 + 60/30V 电源上限标定：教师酸峰 0.70~0.85——pH 科有
    # 区分度又不至于压死搬运。
    current_ref: float = 60.0
    # --- v5.0 计分（与 2D 模块结构完全一致）---
    # 均匀度 level 折线拐点：(CV 值, 得分),分段线性,越匀越高分。
    uniformity_level_points: tuple[tuple[float, float], ...] = (
        (0.10, 1.0), (0.15, 0.8), (0.25, 0.4), (0.40, 0.0))
    # pH level 折线拐点：(混合越界量, 得分),0 越界满分,2.0 越界零分。
    ph_level_points: tuple[tuple[float, float], ...] = (
        (0.0, 1.0), (0.5, 0.8), (1.0, 0.45), (2.0, 0.0))
    ph_peak_weight: float = 0.7   # pH 越界的峰值格/全格均值混合权重（用户定 0.7 峰）
    # 能耗科 E = clip01(1 - P/energy_power_ref)；P = sum(V_e*I_e) 经
    # (x_pair_max_v × current_ref) 归一。按"教师功率 ≈ 0.9 分"标定
    # （72h 协议实测教师平均功率 0.1026）。
    energy_power_ref: float = 1.026
    target_moisture_cv: float = 0.15  # 达标线（仅用于 success/reached 报告,不参与计分）
    # 可选钩子,默认全 0（保持"奖励 ≡ 考卷"恒等式）。
    w_uniformity_goal: float = 0.0    # 塑形项权重（若启用须用超越扩散的 excess 口径）
    w_local_dry: float = 0.0          # 电极接触格过干的安全罚权重
    local_dry_limiting_moisture: float = 0.22  # 过干判定阈值（文献接触区下限）

    def with_overrides(self, **overrides: Any) -> "SoilBox3DConfig":
        """返回替换指定字段后的新配置（原配置不可变）。"""
        return replace(self, **overrides)

    @property
    def n_cells(self) -> int:
        """总格数 = nx*ny*nz。"""
        return self.nx * self.ny * self.nz


DEFAULT_3D_CONFIG = SoilBox3DConfig()


def _coerce_config(config: SoilBox3DConfig | Mapping[str, Any] | None) -> SoilBox3DConfig:
    """把 None/字典/配置对象统一转换成 SoilBox3DConfig。"""
    if config is None:
        return DEFAULT_3D_CONFIG
    if isinstance(config, SoilBox3DConfig):
        return config
    return SoilBox3DConfig(**dict(config))


def build_electrodes(cfg: SoilBox3DConfig) -> list[dict[str, Any]]:
    """生成 20 根电极棒的布局记录（名称 + 接触格列表）。

    电极顺序即内部电压向量的顺序：
      [0:2]   xL 棒（x=0 端面,每 y 行一根,贯穿全高）
      [2:4]   xR 棒（x=nx-1 端面）
      [4:8]   yF 棒（y=0 侧面,每 x 切片一根,贯穿全高）
      [8:12]  yB 棒（y=ny-1 侧面）
      [12:16] zT 棒（顶面横躺,每 x 切片一根,沿 y 跨越）
      [16:20] zB 棒（底部预埋）
    每根棒恰好接触它所跨越的格子（本几何下各 2 格）。
    供 SoilBox3DEnv.__init__ 建立电极-格接触链路。
    """

    nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
    electrodes: list[dict[str, Any]] = []
    for j in range(ny):
        electrodes.append({"name": f"xL{j}", "cells": [(0, j, k) for k in range(nz)]})
    for j in range(ny):
        electrodes.append({"name": f"xR{j}", "cells": [(nx - 1, j, k) for k in range(nz)]})
    for i in range(nx):
        electrodes.append({"name": f"yF{i}", "cells": [(i, 0, k) for k in range(nz)]})
    for i in range(nx):
        electrodes.append({"name": f"yB{i}", "cells": [(i, ny - 1, k) for k in range(nz)]})
    for i in range(nx):
        electrodes.append({"name": f"zT{i}", "cells": [(i, j, nz - 1) for j in range(ny)]})
    for i in range(nx):
        electrodes.append({"name": f"zB{i}", "cells": [(i, j, 0) for j in range(ny)]})
    return electrodes


def electrode_pairs(cfg: SoilBox3DConfig) -> list[tuple[int, int]]:
    """生成 10 对电极的 (阳极候选, 阴极候选) 索引对。

    顺序 = 动作向量顺序：先 x 对（每 y 行）、再 y 对（每 x 切片）、
    最后 z 对（每 x 切片）。供环境的对隔离求解与教师/规则策略共用。
    """
    ny, nx = cfg.ny, cfg.nx
    pairs = []
    for j in range(ny):                      # x 对：xL[j] 对 xR[j]
        pairs.append((j, ny + j))
    base = 2 * ny
    for i in range(nx):                      # y 对：yF[i] 对 yB[i]
        pairs.append((base + i, base + nx + i))
    base = 2 * ny + 2 * nx
    for i in range(nx):                      # z 对：zT[i] 对 zB[i]
        pairs.append((base + i, base + nx + i))
    return pairs


class SoilBox3DEnv(gym.Env):
    """封闭 4x2x2 土箱 Gym 环境：总水量守恒,目标是三维均匀化。

    观测：[16 格含水率, 16 格 (pH-7)/7] = 32 维。
    动作：10 维归一化"对电压差" ∈ [-1,1],按对上限缩放
    （x 对 ×60V,y/z 对 ×30V）；|a| = 电源旋钮,符号 = 极性——
    一个动作分量对应一台物理电源,无共模冗余。
    奖励：(100/max_steps) × 每步 v5.0 小测验分（回报 ≡ 考卷分）。

    对接：train_core 经 make_env 工厂创建；evaluation3d 注入
    initial_moisture 跑冻结考卷;教师/规则/躺平策略消费其观测。
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: SoilBox3DConfig | Mapping[str, Any] | None = None,
        *,
        obs_noise: float = 0.0,          # 观测高斯噪声标准差（0=无噪声）
        randomize_initial: bool = False, # True=每局从训练骰子抽初始场
        seed: int = SEED,                # 随机数种子
    ) -> None:
        super().__init__()
        self.cfg = _coerce_config(config)
        cfg = self.cfg
        self.n = cfg.n_cells
        self.obs_noise = float(obs_noise)
        self.randomize_initial = bool(randomize_initial)
        self._rng = np.random.default_rng(seed)

        self.electrodes = build_electrodes(cfg)
        self.n_electrodes = len(self.electrodes)
        self.pairs = electrode_pairs(cfg)
        # 电极索引 → 所属对索引（对隔离电源求解要用）
        self._pair_of_electrode = np.zeros(self.n_electrodes, dtype=np.int64)
        for p_idx, (a, b) in enumerate(self.pairs):
            self._pair_of_electrode[a] = p_idx
            self._pair_of_electrode[b] = p_idx
        self.n_pairs = len(self.pairs)
        # 每对电源上限,顺序与 pairs 一致：[x 对 ×ny, y 对 ×nx, z 对 ×nx]
        self._pair_max_v = np.array(
            [cfg.x_pair_max_v] * cfg.ny
            + [cfg.yz_pair_max_v] * cfg.nx
            + [cfg.yz_pair_max_v] * cfg.nx
        )
        # (i,j,k) → 扁平索引
        self._idx = lambda i, j, k: (i * cfg.ny + j) * cfg.nz + k
        # 预计算内部边：(格a, 格b, 轴向),b 恒为 a 沿该轴 +1 的邻格
        edges = []
        for i in range(cfg.nx):
            for j in range(cfg.ny):
                for k in range(cfg.nz):
                    a = self._idx(i, j, k)
                    if i + 1 < cfg.nx:
                        edges.append((a, self._idx(i + 1, j, k), 0))
                    if j + 1 < cfg.ny:
                        edges.append((a, self._idx(i, j + 1, k), 1))
                    if k + 1 < cfg.nz:
                        edges.append((a, self._idx(i, j, k + 1), 2))
        self._edges = np.array(edges, dtype=np.int64)          # 形状 (边数, 3)
        # 电极-格接触链路的扁平数组：(电极索引, 格索引)
        links = []
        for e_idx, e in enumerate(self.electrodes):
            for (i, j, k) in e["cells"]:
                links.append((e_idx, self._idx(i, j, k)))
        self._links = np.array(links, dtype=np.int64)          # 形状 (链路数, 2)

        self.m = np.zeros(self.n)            # 各格含水率
        self.acid = np.zeros(self.n)         # 各格酸量（H+ 存量账本）
        self.base = np.zeros(self.n)         # 各格碱量（OH- 存量账本）
        self.pH = np.full(self.n, 7.0)       # 各格 pH（由净酸碱量映射）
        self.steps = 0
        self._reached_target = False
        self.prev_moisture_cv = 0.0

        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.n_pairs,), dtype=np.float32)
        obs_low = np.array([0.0] * self.n + [-1.0] * self.n, dtype=np.float32)
        obs_high = np.array([0.5] * self.n + [1.0] * self.n, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

    # ---------- 工具方法 ----------
    def grid(self, values: np.ndarray) -> np.ndarray:
        """把扁平的 16 维数组还原成 (nx,ny,nz) 三维网格视图。"""
        return values.reshape(self.cfg.nx, self.cfg.ny, self.cfg.nz)

    def moisture_cv(self) -> float:
        """当前含水率变异系数 CV = std/mean（均匀化程度的核心指标）。"""
        return float(np.std(self.m) / (np.mean(self.m) + 1e-9))

    def _sigma(self, m: np.ndarray) -> np.ndarray:
        """孔隙水导电因子 m/(m+m_threshold) ∈ [0,1)：越湿越导电。

        水运移、pH 电流、电位求解共用这一个湿度依赖,保持物理一致。
        """

        return m / (m + self.cfg.m_threshold)

    def _get_obs(self) -> np.ndarray:
        """拼装观测向量 [含水率, 归一化pH],可选叠加高斯观测噪声。"""
        obs = np.concatenate([self.m, (self.pH - 7.0) / 7.0])
        if self.obs_noise > 0.0:
            obs = obs + self._rng.normal(0.0, self.obs_noise, size=obs.shape)
        return np.clip(obs, self.observation_space.low, self.observation_space.high).astype(np.float32)

    def _sample_initial_field(self) -> np.ndarray:
        """3D 训练骰子：结构化湿场/斜坡/iid 随机三类,CV≥0.25 拒绝采样。

        刻意宽于且独立于冻结考卷(evaluation3d),防背题;
        randomize_initial=True 时每局 reset 调用。
        """

        cfg = self.cfg
        rng = self._rng
        shape = (cfg.nx, cfg.ny, cfg.nz)
        for _ in range(20):
            wet = float(rng.uniform(0.25, 0.45))
            dry = float(rng.uniform(0.06, 0.14))
            mode = float(rng.random())
            grid = np.full(shape, dry)
            if mode < 0.45:
                pattern = int(rng.integers(0, 6))
                if pattern == 0:      # x 端面板块
                    grid[rng.integers(0, 2) * (cfg.nx - 1), :, :] = wet
                elif pattern == 1:    # 顶层或底层
                    grid[:, :, rng.integers(0, cfg.nz)] = wet
                elif pattern == 2:    # 前墙或后墙
                    grid[:, rng.integers(0, cfg.ny), :] = wet
                elif pattern == 3:    # x 半箱
                    grid[: cfg.nx // 2, :, :] = wet
                elif pattern == 4:    # 角块
                    grid[rng.integers(0, 2) * (cfg.nx - 1),
                         rng.integers(0, cfg.ny),
                         rng.integers(0, cfg.nz)] = wet
                else:                 # 随机散格
                    cells = rng.choice(self.n, size=int(rng.integers(2, 5)), replace=False)
                    grid.reshape(-1)[cells] = wet
            elif mode < 0.75:         # 随机方向的三维线性斜坡
                xs, ys, zs = np.meshgrid(
                    np.linspace(0, 1, cfg.nx), np.linspace(0, 1, cfg.ny),
                    np.linspace(0, 1, cfg.nz), indexing="ij")
                d = rng.normal(size=3)
                d /= max(np.linalg.norm(d), 1e-9)
                ramp = d[0] * xs + d[1] * ys + d[2] * zs
                ramp = (ramp - ramp.min()) / max(float(ramp.max() - ramp.min()), 1e-9)
                grid = dry + (wet - dry) * ramp
            else:
                grid = rng.uniform(0.08, 0.42, size=shape)
            grid = np.clip(grid, 0.02, 0.5)
            if float(np.std(grid) / (np.mean(grid) + 1e-9)) >= 0.25:
                return grid
        grid = np.full(shape, cfg.m_dry)
        grid[0, :, :] = cfg.m_wet
        return grid

    # ---------- Gym 接口 ----------
    def reset(self, seed: int | None = None, options: dict | None = None):
        """开新局。初始场优先级：options 注入（考卷用）> 训练骰子 > 经典左湿场。"""
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        cfg = self.cfg
        if options and "initial_moisture" in options:
            grid = np.asarray(options["initial_moisture"], dtype=np.float64).reshape(
                cfg.nx, cfg.ny, cfg.nz)
            if np.any(grid < 0.0) or np.any(grid > 0.5):
                raise ValueError("initial_moisture values must lie in [0, 0.5]")
        elif self.randomize_initial:
            grid = self._sample_initial_field()
        else:
            grid = np.full((cfg.nx, cfg.ny, cfg.nz), cfg.m_dry)
            grid[0, :, :] = cfg.m_wet
        self.m = grid.reshape(-1).astype(np.float64)
        self.acid = np.zeros(self.n)
        self.base = np.zeros(self.n)
        self.pH = np.full(self.n, 7.0)
        self.steps = 0
        self._reached_target = False
        self.prev_moisture_cv = self.moisture_cv()
        return self._get_obs(), {}

    def _solve_potential(self, voltages: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """求解对隔离电源下的电位场,返回 (格电位, 边电流, 界面电流)。

        接线模型（对应 2D 模块的按对解耦传统,以及 s41598-2024 的 DSAV
        多独立电源接法）：每对电极是一台隔离的二端电源——对内电压差由
        指令给定,对的公共电位 o_k 浮动到该对净注入电流为零。若不这样
        建模,未通电的棒会成为 0V 硬接地并短路电场（实测:持续 x 向
        通电时水堆积在每根闲棒周围,CV 发散到 1.16）。

        未知量：16 格电位 + 10 对浮动偏置 = 26 维对称线性系统。
        每条链路（对 k 的电极 e,接触格 c）：棒电位 = o_k + v_e,
        注入电流 = g_l × (o_k + v_e − phi_c)。格行 = 基尔霍夫电流定律,
        对行 = 对净电流为零。被 step() 每步调用一次。
        """

        cfg = self.cfg
        sigma = self._sigma(self.m)
        ea, eb = self._edges[:, 0], self._edges[:, 1]
        g_edge = 0.5 * (sigma[ea] + sigma[eb])
        le, lc = self._links[:, 0], self._links[:, 1]
        g_link = cfg.contact_scale * sigma[lc]
        pair_of = self._pair_of_electrode
        lp = pair_of[le]                              # 每条链路所属的对索引
        n, npair = self.n, len(self.pairs)
        size = n + npair

        A = np.zeros((size, size))
        b = np.zeros(size)
        # 格行：基尔霍夫电流定律
        np.add.at(A, (ea, ea), g_edge)
        np.add.at(A, (eb, eb), g_edge)
        np.add.at(A, (ea, eb), -g_edge)
        np.add.at(A, (eb, ea), -g_edge)
        np.add.at(A, (lc, lc), g_link)
        np.add.at(A, (lc, n + lp), -g_link)
        np.add.at(b, lc, g_link * voltages[le])
        # 对行：净电流为零（与格行对称）
        np.add.at(A, (n + lp, n + lp), g_link)
        np.add.at(A, (n + lp, lc), -g_link)
        np.add.at(b, n + lp, -g_link * voltages[le])
        A[np.arange(size), np.arange(size)] += cfg.solver_regularization
        x = np.linalg.solve(A, b)
        phi, offsets = x[:n], x[n:]

        i_edge = g_edge * (phi[ea] - phi[eb])                       # a → b 为正
        i_link = g_link * (offsets[lp] + voltages[le] - phi[lc])    # 棒 → 土 为正
        return phi, i_edge, i_link

    def step(self, action):
        """推进一步：解电位场 → EO 输运+重力 → 扩散 → 电解/迁移/中和 → 计分。

        Gym 标准接口,返回 (obs, reward, terminated, truncated, info)；
        info["terms"] 携带每步小测验分项,供考卷积分与训练 dashboard 共用。
        """
        cfg = self.cfg
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if a.shape != (self.n_pairs,):
            raise ValueError(f"Expected action shape ({self.n_pairs},), got {a.shape}")
        # 对指令 → 对内两根棒各取 ±差值的一半（对隔离电源下只有差值有物理意义）
        pair_diff = a * self._pair_max_v
        voltages = np.zeros(self.n_electrodes)
        for p_idx, (ea_idx, eb_idx) in enumerate(self.pairs):
            voltages[ea_idx] = 0.5 * pair_diff[p_idx]
            voltages[eb_idx] = -0.5 * pair_diff[p_idx]

        phi, i_edge, i_link = self._solve_potential(voltages)
        sigma = self._sigma(self.m)
        ea, eb = self._edges[:, 0], self._edges[:, 1]
        le, lc = self._links[:, 0], self._links[:, 1]

        # --- 电渗水运移（沿边,阳极 → 阴极方向）---
        e_field = (phi[ea] - phi[eb]) / cfg.cell_cm                     # 场强 V/cm
        area = cfg.cell_cm * cfg.cell_cm
        volume = cfg.cell_cm ** 3
        pore = 0.5 * (sigma[ea] + sigma[eb])
        frac = (cfg.electroosmotic_permeability_cm2_per_s_v * area * e_field
                * cfg.time_step_s / volume) * pore                       # 正值 = a → b

        # --- 重力（仅 z 边；边约定上格在 eb,向下流即 b → a,取负号）---
        if cfg.gravity_per_step > 0.0:
            z_mask = self._edges[:, 2] == 2
            frac = frac - np.where(z_mask, cfg.gravity_per_step * self.m[eb], 0.0)

        # 逐边供体量 → 按格流出总预算缩放（防多边同时流出超支）
        take_from_a = np.where(frac > 0, frac, 0.0)
        take_from_b = np.where(frac < 0, -frac, 0.0)
        out_a = np.zeros(self.n)
        np.add.at(out_a, ea, take_from_a)
        np.add.at(out_a, eb, take_from_b)
        budget = cfg.outflow_budget_fraction * self.m
        scale = np.ones(self.n)
        over = out_a > budget
        scale[over] = budget[over] / (out_a[over] + 1e-12)
        eo_flux = np.where(frac > 0, frac * scale[ea], frac * scale[eb])

        dm = np.zeros(self.n)
        np.add.at(dm, ea, -eo_flux)
        np.add.at(dm, eb, eo_flux)

        # --- 基质吸力扩散（6 邻域）,独立的流出预算 ---
        diff = cfg.moisture_diffusion_per_step * (self.m[ea] - self.m[eb])
        take_a = np.where(diff > 0, diff, 0.0)
        take_b = np.where(diff < 0, -diff, 0.0)
        out_d = np.zeros(self.n)
        np.add.at(out_d, ea, take_a)
        np.add.at(out_d, eb, take_b)
        scale_d = np.ones(self.n)
        over_d = out_d > budget
        scale_d[over_d] = budget[over_d] / (out_d[over_d] + 1e-12)
        diff_flux = np.where(diff > 0, diff * scale_d[ea], diff * scale_d[eb])
        d_diff = np.zeros(self.n)
        np.add.at(d_diff, ea, -diff_flux)
        np.add.at(d_diff, eb, diff_flux)

        # 纯扩散反事实 CV（供可选塑形钩子 w_uniformity_goal 使用）
        m_diff_only = np.clip(self.m + d_diff, 0.0, 0.5)
        cv_diff_only = float(np.std(m_diff_only) / (np.mean(m_diff_only) + 1e-9))

        self.m = np.clip(self.m + dm + d_diff, 0.0, 0.5)

        # --- 电解：按界面电流符号定位产酸/产碱（流入土=阳极=产酸）---
        i_norm_link = i_link / max(cfg.current_ref, 1e-9)
        d_acid = np.zeros(self.n)
        d_base = np.zeros(self.n)
        dt = cfg.time_step_s
        gen_acid = np.where(i_norm_link > 0, i_norm_link, 0.0) * cfg.k_acid_gen_per_s * dt
        gen_base = np.where(i_norm_link < 0, -i_norm_link, 0.0) * cfg.k_base_gen_per_s * dt
        np.add.at(d_acid, lc, gen_acid)
        np.add.at(d_base, lc, gen_base)

        # --- 酸碱沿边电流迁移（酸顺电流走,碱逆流;酸锋快于碱锋）---
        i_norm_edge = np.abs(i_edge) / max(cfg.current_ref, 1e-9)
        up = np.where(i_edge >= 0, ea, eb)      # 电流上游节点
        dn = np.where(i_edge >= 0, eb, ea)      # 电流下游节点
        acid_moved = np.minimum(cfg.k_acid_mig_per_s * dt * i_norm_edge * self.acid[up], self.acid[up])
        base_moved = np.minimum(cfg.k_base_mig_per_s * dt * i_norm_edge * self.base[dn], self.base[dn])
        np.add.at(d_acid, up, -acid_moved)
        np.add.at(d_acid, dn, acid_moved)
        np.add.at(d_base, dn, -base_moved)
        np.add.at(d_base, up, base_moved)

        acid = np.maximum(0.0, self.acid + d_acid)
        base = np.maximum(0.0, self.base + d_base)
        neutral = np.minimum(acid, base)
        self.acid = acid - neutral
        self.base = base - neutral
        net = self.acid - self.base
        self.pH = np.clip(7.0 - 5.0 * net / (np.abs(net) + cfg.ph_half_amount), 0.0, 14.0)

        # --- v5.0 每步小测验（与 2D 结构完全一致）---
        moisture_cv = self.moisture_cv()
        cell_viol = np.maximum(0.0, np.abs(self.pH - 7.0) - cfg.ph_tol)
        ph_peak = float(np.max(cell_viol))
        ph_mean = float(np.mean(cell_viol))
        ph_mixed = cfg.ph_peak_weight * ph_peak + (1.0 - cfg.ph_peak_weight) * ph_mean
        # 精确电功率：对全部电极-格链路求和 V_e × I_link（对偏置因净电流
        # 为零而抵消）,再按最大电源上限与电流参考归一。
        power_phys = float(np.sum(voltages[le] * i_link) /
                           (cfg.x_pair_max_v * max(cfg.current_ref, 1e-9)))
        u_score = _level_score(moisture_cv, cfg.uniformity_level_points)
        ph_score = _level_score(ph_mixed, cfg.ph_level_points)
        energy_score = _clip01(1.0 - power_phys / max(cfg.energy_power_ref, 1e-9))
        step_quality = (FINAL_SCORE_WEIGHTS["uniformity_score"] * u_score
                        + FINAL_SCORE_WEIGHTS["ph_score"] * ph_score
                        + FINAL_SCORE_WEIGHTS["energy_score"] * energy_score)
        scale_q = 100.0 / max(cfg.max_steps, 1)
        reward_u = scale_q * FINAL_SCORE_WEIGHTS["uniformity_score"] * u_score
        reward_ph = scale_q * FINAL_SCORE_WEIGHTS["ph_score"] * ph_score
        reward_e = scale_q * FINAL_SCORE_WEIGHTS["energy_score"] * energy_score
        uniformity_excess = cv_diff_only - moisture_cv
        # 过干安全钩子：被驱动电极的接触格干燥缺口（按驱动强度加权）
        drive = np.abs(voltages[le]) / (0.5 * cfg.x_pair_max_v)
        dry_deficit = np.maximum(0.0, cfg.local_dry_limiting_moisture - self.m[lc])
        local_dry = float(np.average(dry_deficit, weights=drive + 1e-12)) if np.any(drive > 1e-9) else 0.0
        reward = (reward_u + reward_ph + reward_e
                  + cfg.w_uniformity_goal * uniformity_excess
                  - cfg.w_local_dry * local_dry)

        self.steps += 1
        success = moisture_cv <= cfg.target_moisture_cv
        terminated = False
        truncated = bool(self.steps >= cfg.max_steps)
        self.prev_moisture_cv = moisture_cv

        terms = {
            "moisture_cv": moisture_cv,
            "uniformity_excess": float(uniformity_excess),
            "uniformity_improvement": float(uniformity_excess),  # dashboard 兼容别名
            "ph_violation": ph_peak,
            "ph_violation_mean": ph_mean,
            "ph_violation_mixed": ph_mixed,
            "power_phys": power_phys,
            "energy": power_phys,                 # 考卷报告兼容别名
            "u_score": float(u_score),
            "ph_score": float(ph_score),
            "energy_score": float(energy_score),
            "step_quality": float(step_quality),
            "reward_u": float(reward_u),
            "reward_ph": float(reward_ph),
            "reward_e": float(reward_e),
            "local_dry_violation": local_dry,
            "anode_dry_violation": local_dry,
            "flux": float(np.sum(np.abs(eo_flux))),
            "moisture_cv_gap": float(max(0.0, moisture_cv - cfg.target_moisture_cv)),
        }
        info = {
            "terms": terms,
            "success": success,
            "moisture_cv": moisture_cv,
            "mean_moisture": float(np.mean(self.m)),
            "mean_pH": float(np.mean(self.pH)),
            "config": self.cfg,
            "obs_noise": self.obs_noise,
            "pair_max_v": self._pair_max_v.astype(float).tolist(),
            "max_electrode_voltage_v": float(cfg.x_pair_max_v),  # 旧 dashboard 兼容键
            "electrode_action": voltages.astype(float).tolist(),
            "normalized_action": a.astype(float).tolist(),
            "potentials": phi.astype(float).tolist(),
            "interface_currents": i_link.astype(float).tolist(),
            "voltage_drop_v": float(np.max(voltages) - np.min(voltages)),
        }
        return self._get_obs(), float(reward), terminated, truncated, info


def make_soilbox3d(**overrides: Any) -> SoilBox3DEnv:
    """便捷工厂：用若干配置覆盖创建标准 3D 土箱环境。"""
    return SoilBox3DEnv(DEFAULT_3D_CONFIG.with_overrides(**overrides))


def idle_policy_3d(obs: np.ndarray) -> np.ndarray:
    """躺平基线：10 台电源全部输出 0（纯扩散地板,考卷陪考用）。"""
    del obs
    n_pairs = DEFAULT_3D_CONFIG.ny + 2 * DEFAULT_3D_CONFIG.nx  # 10 台电源
    return np.zeros(n_pairs, dtype=np.float32)


def rule_policy_3d(
    obs: np.ndarray,
    cfg: SoilBox3DConfig = DEFAULT_3D_CONFIG,
) -> np.ndarray:
    """bang-bang 规则基线（考卷陪考）：每对按湿度差的符号给固定半旋钮
    电压,不按幅度调制;任一格 pH 越界时全体减半。2D rule_policy 的 3D 版。
    """

    n = cfg.n_cells
    m = obs[:n].reshape(cfg.nx, cfg.ny, cfg.nz)
    ph = obs[n:] * 7.0 + 7.0
    v = np.zeros(cfg.ny + 2 * cfg.nx, dtype=np.float64)
    level = 0.5
    if float(np.max(np.abs(ph - 7.0))) > cfg.ph_tol:
        level *= 0.5
    for j in range(cfg.ny):                       # x 对
        v[j] = level * np.sign(float(np.mean(m[0, j, :]) - np.mean(m[-1, j, :])))
    base = cfg.ny
    for i in range(cfg.nx):                       # y 对
        v[base + i] = level * np.sign(float(np.mean(m[i, 0, :]) - np.mean(m[i, -1, :])))
    base = cfg.ny + cfg.nx
    for i in range(cfg.nx):                       # z 对
        v[base + i] = level * np.sign(float(np.mean(m[i, :, -1]) - np.mean(m[i, :, 0])))
    return v.astype(np.float32)


def proportional_policy_3d(
    obs: np.ndarray,
    cfg: SoilBox3DConfig = DEFAULT_3D_CONFIG,
    gain_x: float = 480.0,     # x 对增益（伏特/单位湿度差,半箱差~0.125 时恰满 60V）
    gain_yz: float = 120.0,    # y/z 对增益（湿度差~0.25 时恰满 30V）
) -> np.ndarray:
    """比例控制器教师：每对电极朝各自湿侧成比例供电。

    感知用"半箱差"（该对轴线上前半均值 − 后半均值）而非两端格差：
    4 格 x 轴上端差感知会在两端收敛后失明（内部不均看不见,实测卡在
    CV≈0.31）。增益单位是伏特/单位湿度差,并按各对电源上限饱和截断。
    任一格 pH 越界时全体电压减半（与 2D 教师同款保护）。

    对接：BC 热启动的示范数据源（train_core.pretrain_policy_bc 经
    bc_teacher_fn 注入）;考卷陪考与审计的"主动策略"参考。
    输出 10 维归一化对指令。
    """

    n = cfg.n_cells
    m = obs[:n].reshape(cfg.nx, cfg.ny, cfg.nz)
    ph = obs[n:] * 7.0 + 7.0
    v = np.zeros(cfg.ny + 2 * cfg.nx, dtype=np.float64)
    hx = cfg.nx // 2
    for j in range(cfg.ny):                       # x 对：该行的半箱差
        diff = float(np.mean(m[:hx, j, :]) - np.mean(m[hx:, j, :]))
        v[j] = np.clip(gain_x * diff, -cfg.x_pair_max_v, cfg.x_pair_max_v) / cfg.x_pair_max_v
    base = cfg.ny
    hy = cfg.ny // 2
    for i in range(cfg.nx):                       # y 对
        diff = float(np.mean(m[i, :hy, :]) - np.mean(m[i, hy:, :]))
        v[base + i] = np.clip(gain_yz * diff, -cfg.yz_pair_max_v, cfg.yz_pair_max_v) / cfg.yz_pair_max_v
    base = cfg.ny + cfg.nx
    hz = cfg.nz // 2
    for i in range(cfg.nx):                       # z 对（上半层 vs 下半层）
        diff = float(np.mean(m[i, :, hz:]) - np.mean(m[i, :, :hz]))
        v[base + i] = np.clip(gain_yz * diff, -cfg.yz_pair_max_v, cfg.yz_pair_max_v) / cfg.yz_pair_max_v
    if float(np.max(np.abs(ph - 7.0))) > cfg.ph_tol:
        v *= 0.5
    return v.astype(np.float32)
