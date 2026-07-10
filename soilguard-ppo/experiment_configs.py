# -*- coding: utf-8 -*-
"""Named experiment configurations for SoilGuard PPO training.

每个实验条目把全部参数写全，直接在字典里改即可。
评估 = 固定百题考卷（evaluation.py，BATTERY v5.0：每步小测验 q=0.6*U+0.2*P+0.2*E 全程积分，
训练奖励=(100/max_steps)*q 分期支付 → 训练回报≡考卷分；口径细节见 soilbox_env.py 配置注释）。
注意：v5.0 前后的分数不可比（旧 85 锚点 ≈ 新口径 70）。

实验演进历史（每版一两行；v1-v5 属旧环境时代，配置条目已删，运行记录在 artifacts/）：
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
# v12 初始场随机化、总步数 100w → 塌缩成躺平
# v13 CV 奖励扣除扩散白给份额+跑满全程不提前终止 → 仍躺平（奖励排序错）
# v14 部署远程服务器，进程 8→16
# v15 新增 hold_bonus 达标工资 → 排序修对但 PPO 仍躺平（优化爬不出谷地）
# v16 总步数 150w
# v17 BC 热启动模仿比例控制器 → 锚点 84.7，首次打赢规则基线
# v18 总步数 200w
# v19 log_std_init 2.0→0.0 → 72.9，但从锚点 84.7 持续侵蚀
# v20 pH 罚"存量"→"净酸增量" → 64.4，侵蚀未止（假说被证伪一半）
# v21 动作空间归一化[-1,1]等四处对齐修订 → σ=1.0 满量程噪声毒害 BC 均值，56.7
# v22 3M 步对照 → 65.9，加步数缓解不治本
# v23 log_std=-2.7 但 train_core 漏传 = 静默空跑（结果与 v21 逐位相同）
# v24 同步修复后重跑 → 侵蚀以最纯形态复现（52.1），σ 还自发回涨
# v25 KL-to-BC 系绳 λ=0.1 + 最优检查点存档 → 侵蚀首次止住，best 85.13≈锚点（守成不超越）
# v26 松绳 λ=0.02 → 与 v25 无差异；判定：旧尺子下 PPO 微调无增量价值
# v27 移除 BC，从零 PPO+自适应课程学习 → 课程破解启动难题（best 71.3），但 σ 不退火错位病复发
# v28 课程+噪声递减（满档后压 σ 至 -2.7） → 失败 65.9（σ 骤缩→梯度放大失稳）；从零路线封存
# v29 v5.0 新尺子首训：BC+KL 系绳（复用 v25 管线）跑"奖励≡考卷"的新奖励，
#     看 PPO 能否在教师 69.9 之上找到增量（pH/能耗科在新尺子下首次有活梯度）
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


COMMON_CONFIG: dict[str, Any] = {}


EXPERIMENT_OVERRIDES: dict[str, dict[str, Any]] = {
    "v29_reward_full": {
        "steps": 2000000,               # BC+KL 微调基准步数（同 v25）
        "n_envs": 16,                   # 并行采样环境数
        "seed": 42,                     # 随机种子
        "verbose": 0,                   # SB3 日志级别
        "log_std_init": -2.7,           # BC 热启动配套探索噪声（σ≈6.7% 半量程≈1.3V）
        "kl_bc_coef": 0.1,              # KL-to-BC 系绳（v25 验证的防侵蚀锚）
        "bc_pretrain": True,            # BC 热启动模仿比例控制器
        "bc_episodes": 300,
        "bc_epochs": 20,
        "checkpoint_every": 100000,     # 每 10w 步考 20 题子卷存最优快照
        "curriculum": False,            # BC 路线不需要课程学习
        "randomize_physics": False,
        "randomize_obs_noise": False,
        "randomize_initial_field": True,
        "eval_noise_levels": "",
        "clearml": True,
        "clearml_online": True,
        "clearml_no_artifacts": False,
        "clearml_project": "soilbox_ppo_train",
        "clearml_task_name": "v29_reward_full",
        # v5.0 奖励/考卷全部走 soilbox_env.py 的默认值，无需 reward_overrides。
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
