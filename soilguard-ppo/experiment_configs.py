# -*- coding: utf-8 -*-
"""Named experiment configurations for SoilGuard PPO training.

每个实验条目把全部参数写全，直接在字典里改即可。
评估 = 固定考卷（2D: evaluation.py 百题 / 3D: evaluation3d.py 160 题），
计分 v5.0：每步小测验 q=0.6*U+0.2*P+0.2*E 全程积分，训练奖励=(100/max_steps)*q
分期支付 → 训练回报≡考卷分。2D/3D 及 v5.0 前后的分数互不可比。

实验演进历史（每版一两行；v1-v5 旧环境时代，条目已删，记录在 artifacts/）：
# v1  基线
# v2  增大 w_ph 惩罚权重
# v3  初始水分布改为左湿右干
# v4  增大电压动作幅度
# v5  每局步数 60→120
# v6  重写环境与动作交互
# v7  增大 w_ph/w_local 惩罚
# v8  回调 w_ph/w_local 惩罚
# v9  增大均匀化奖励系数
# v10 每局步数 120→180
# v11 开启多核训练，评估协议改写为百题考卷
# v12 初始场随机化 → 塌缩成躺平
# v13 CV 奖励扣除扩散白给+跑满全程 → 仍躺平（奖励排序错）
# v14 部署远程服务器，进程 8→16
# v15 新增 hold_bonus 达标工资 → 排序修对但 PPO 仍躺平
# v16 总步数 150w
# v17 BC 热启动模仿比例控制器 → 锚点 84.7，首次打赢规则基线
# v18 总步数 200w
# v19 log_std_init 2.0→0.0 → 72.9，但从锚点持续侵蚀
# v20 pH 罚改净酸增量 → 64.4，侵蚀未止
# v21 动作空间归一化等四处对齐 → σ=1.0 满量程噪声毒害 BC，56.7
# v22 3M 步对照 → 65.9，治标不治本
# v23 log_std=-2.7 但 train_core 漏传 = 静默空跑（与 v21 逐位相同）
# v24 同步修复重跑 → 侵蚀最纯复现（52.1），σ 自发回涨
# v25 KL-to-BC 系绳 λ=0.1 + 最优检查点 → 侵蚀止住，best 85.13≈锚点（守成）
# v26 松绳 λ=0.02 → 与 v25 无差异；旧尺子下 PPO 微调无增量价值
# v27 从零 PPO+自适应课程 → 课程破解启动（best 71.3），但 σ 不退火错位病
# v28 课程+噪声递减 → 失败 65.9（σ 骤缩失稳）；从零路线封存
# v29 v5.0 尺子首训（BC+KL）→ 三度守成（锚点 69.57/best 69.75）；2D 结题：教师≈物理最优
# ========== 3D（2026-07-10 起，真实土箱 60x30x30）==========
# 3D 环境：4x2x2=16 格（15cm 单元），电位场求解器（配对隔离电源，每步解 div(σ∇φ)=0），
#     考卷 v6.0-3d：32 族 x5 种子 = 160 题，含教师盲区族（double_end/mid_block）。
# 协议终版（2026-07-13）：72h = 72 步 x1h；电压 x 对 0-60V / yz 对 0-30V（等梯度设计）；
#     动作 = 10 维对指令（每对一台电源）；基线 teacher 52.06 / rule 45.72 / idle 42.95。
# v30 3D 首训 BC+KL（λ=0.1, σ=-2.7）→ BC 克隆完美（锚点 52.02）但 PPO 四度守成
#     （best 52.20）；盲区族钉死 40=躺平分——紧绳+微探索够不着 3D 的真实高地。
# v31 减绳放探索（本轮）：λ 0.1→0.01（约束力≈0，保留 kl_to_bc 观测曲线）、
#     σ -2.7→-1.5（x 轴等效 ±13V 抖动，盲区战术才可能被采样到）；检查点收割兜底。
#     目标：攻下盲区族（40→50+）= 学习者首次做到教师原理上做不到的事；
#     预期全卷 53~55（盲区+x端块的现实上限），若仍守成 → 下一手训练骰子盲区富集。
# v32 把当前的总步数从200w步增加到400w步
# v33 把当前的总步数从400w步增加到800w步
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


COMMON_CONFIG: dict[str, Any] = {}


EXPERIMENT_OVERRIDES: dict[str, dict[str, Any]] = {
    "v33_reward_full": {
        "env": "3d",                    # 真实几何 4x2x2 + 电位场求解器
        "steps": 8000000,
        "n_envs": 16,
        "seed": 42,
        "verbose": 0,
        # v31 变量①：探索噪声放大 3 倍（σ≈0.22，x 轴等效 ±13V 抖动），
        # 让"给盲区场型加实质电压序列"这类大偏离战术进入采样范围。
        "log_std_init": -1.5,
        # v31 变量②：系绳降到象征值——约束力≈0，但保留 kl_to_bc 仪表
        # （置 0 会走原生 PPO 路径，漂移距离就看不见了）。
        "kl_bc_coef": 0.01,
        "bc_pretrain": True,            # 教师 = proportional_policy_3d（自动路由）
        "bc_episodes": 300,
        "bc_epochs": 20,
        "checkpoint_every": 100000,     # 每 10w 步考 32 题子卷存最优快照（下行保险）
        "curriculum": False,
        "randomize_physics": False,
        "randomize_obs_noise": False,
        "randomize_initial_field": True,
        "eval_noise_levels": "",
        "clearml": True,
        "clearml_online": True,
        "clearml_no_artifacts": False,
        "clearml_project": "soilbox_ppo_train",
        "clearml_task_name": "v33_reward_full",
        # 3D 奖励/考卷全部走 soilbox3d_env.py 的标定默认值。
    },
}


def resolve_experiments() -> dict[str, dict[str, Any]]:
    resolved = {}
    for name, overrides in EXPERIMENT_OVERRIDES.items():
        config = deepcopy(COMMON_CONFIG)
        config.update(deepcopy(overrides))
        resolved[name] = config
    return resolved


EXPERIMENTS = resolve_experiments()
