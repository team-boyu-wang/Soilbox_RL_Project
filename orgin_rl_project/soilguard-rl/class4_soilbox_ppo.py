# -*- coding: utf-8 -*-
"""
SoilGuard-RL 第 4 讲 · 落地实操（实验设计式 · 自包含 · Colab 逐 cell）
========================================================================
这是 Class 4 的唯一主代码：自包含，一个文件从头跑到尾。
前三讲讲完了"是什么 / 怎么学 / 用什么工具"，这一节真刀真枪把电渗土箱写成一个
Gym 环境（reset/step/reward），挂上 PPO，训练出一个会调水的智能体——这是整个
SoilGuard 项目核心代码的诞生节。

但它不是"把土箱写成环境、挂上 PPO、看它学会就鼓掌"。那样你只能得到"它学会了"
一个结果，看不到这门课真正要教的两件事。我们按【做科研】的路子走两个实验：

== 核心故事（两条暗线，全程围绕它们）==
  【暗线一 · 你给什么它追什么 → 奖励怎么设（实验 1）】
    奖励函数就是你给智能体的"考卷"。你把分数挂错了地方，它就考出你没料到的答案。
    真实目标是"把水搬均匀"，但一个偷懒的念头是"奖励它多搬水嘛"——于是把奖励挂在
    "搬水活动量"这个 proxy（代理指标）上。训出来一看：它把水在格子间来回倒，
    活动量刷得飞起，水却根本没搬匀（甚至更乱）。它在原地空转刷指标，不干真活。
    它没有错——它完美最大化了你给的分；错的是你把奖励挂在了和真实目标不等价的 proxy 上。
    这就是 reward hacking（奖励钻空子，经典如赛艇游戏原地绕圈刷分）。把奖励挂回
    "真实搬匀进展"（来回倒水净进展为 0、刷不到分），它立刻就老老实实去把水真正搬匀。

  【暗线二 · 训练好≠能用 → sim-to-real（实验 2~4）】
    把奖励设对、在"干净读数"下训出的策略，搬到"探头有噪声"的世界就抓瞎——成功率暴跌。
    有意思的是：换土质（k_eo）、换装土干湿（m_dry）这种物理量变，它靠闭环反馈居然扛得住；
    真正让它垮的是"观测被污染"——它训练时一直读干净数字，把"读数=真相"当成了理所当然。
    这正是 sim-to-real gap 最锋利的一刀：你在仿真（干净读数）里把策略训到很好，搬到真机
    （电化学探头有噪声/漂移）就不一定能用。诊断清楚后，用【域随机化】救：训练时每局随机
    一个探头噪声水平，逼它学一个"读数再脏也能稳住"的策略，再测——脏探头也扛得住了。

  这两条暗线，第 1 讲就都埋了：①"真实土箱试不起几万次"→ 数字孪生 + sim-to-real；
  ②"你给什么它追什么"→ 奖励设计 + reward hacking + 必须对比规则基线。本节全部收口。

== 这是"实验设计"，不是"跑个 demo"==
  实验 1（奖励）：先按朴素想法奖"搬水活动量"proxy → 亲手训出 reward hacking（来回倒水刷分、
    不真搬匀）→ 诊断"是奖励挂错了目标，不是算法坏了" → 把奖励挂回真实进展救回 → 判据量化。
  实验 2~4（泛化）：干净读数训好 → 亲手给探头加噪声把它跑垮 → 诊断是过拟合到干净观测 →
    域随机化（训练加噪声）救回 → 鲁棒性曲线对照（并发现噪声超出训练范围连域随机化也救不动）。
  这才是论文/答辩里的"消融实验 / 泛化性分析 / sim-to-real"。

  "控制变量"贯穿全文：实验 2 把"探头噪声"（致命）和"土质/装土物理量变"（鲁棒）分开测，
  才能精确说清"是观测污染害它垮的、而不是物理变化"，得出有价值的归因。

== 判据写成代码（科研习惯，不靠肉眼看一局）==
  全文用一把统一的尺子 evaluate()：固定 30 局、固定评估 seed，量出六个指标（平均回报 /
  搬匀程度 moved_frac / 搬水活动量 flux_act / 耗电 energy / pH 越界 ph_viol / 成功率 success）。
  再写成若干 judge_*() 判据函数，把"reward hacking 复现了吗 / 泛化崩了吗 /
  域随机化救回了吗 / RL 真比规则强吗"这些结论，量化成可打印的 True/False + 数字，
  而不是散落各处、靠肉眼读 print 拍脑袋。判据一致，结论才可比、才可写进论文。

== 真实情况（服务器实测，下面 print/注释里标"服务器实测"的数都是真值，不是编的）==
  【★ 数字来自 V100 真跑（训练 15万步、域随机化 20万步、seed=42、device=cpu）；最终完整版会再校准一遍 ★】
  · 实验 1 reward hacking：proxy（奖搬水活动量）→ 活动量刷到 7.0、真实搬匀 −0.20（空转刷分）；
    挂回真实进展 → 真搬匀 0.93、活动量降到 1.05（救回）。judge_reward_hacking = True。
  · 实验 2 对照（鲁棒）：换 k_eo 0.06~0.24 / m_dry 0.04~0.16，full 成功率基本仍高 → 物理量变扛得住。
  · 实验 2 主漂移（崩）：探头噪声 0/0.05/0.1/0.2 → full 成功率 1.0/0.73/0.50/0.13（一脏就垮）。
  · 实验 4 救回：域随机化模型同噪声 → 成功率 1.0/1.0/1.0/0.90/0.40，在 0.05~0.1 明显高于 full（judge=True）；
    噪声 0.2 时标准 0.13、域随机化 0.40（都下滑，域随机化也有边界，要诚实讲）。
  · 收口 PPO vs 规则：PPO 搬匀 0.93/耗电 8.0，规则 0.90/耗电 35.8（PPO 省电约 4.5 倍），judge_beats_baseline=True。

== 物理说明（重要，决定了这门课的诚实边界）==
  本环境用"简化电渗模型"，不是完整 PDE 求解：
    · 水的迁移：相邻格子间，水流量 ∝ 电压 × 局部可动性，可动性随含水率下降而下降
      —— 这就把课题硬伤"低湿区电渗失效"直接建进了物理。
    · pH 漂移：通电使靠正极一侧酸化、靠负极一侧碱化，幅度 ∝ 通电量；极性反转可让 pH 回调。
    · 盐分积累：本简化模型【未建模】，留作 Limitations 与后续工作（见最后一段）。
  目的：给 PPO 一个能快速训练、且行为可解释的环境，不是物理精确复现。

== 段落总表 ==
【第 0 部分 · 准备：先把土箱写成环境（这是你的活）】
  0-1  环境、依赖（gymnasium / stable-baselines3）
  0-2  导入 + 固定随机种子 + CONFIG（标准土箱配置，集中放这里方便改）
  0-3  ★SoilBoxEnv：把土箱写成 Gym 环境（reset/step/reward）+ 探头噪声/域随机化开关
  0-4  自检：随机动作跑一局、画湿度演化（确认环境没写错）
  0-5  ★判据写成代码：make_soilbox 工厂 + evaluate + 规则基线 + train_ppo + judge_*
【实验 1 · 奖励怎么设：你给什么它追什么（reward hacking）】
  1-1  朴素 proxy 奖励：奖「搬水活动量」（而非真实搬匀）→ 训一版
  1-2  看它学成什么 + 判据：来回倒水刷活动量、却不真搬匀（亲手跑出 reward hacking）
  1-3  诊断 + 救：把奖励挂回「真实进展」→ 重训 → 真搬匀 → 判据量化前后差距
【实验 2 · 训练好≠能用：探头一带噪声就抓瞎（sim-to-real 的真正杀手）】
  2-1  给观测加探头噪声 → 标准模型成功率暴跌（主漂移）
  2-2  对照：改土质 k_eo / 装土 m_dry → 标准模型扛得住（鲁棒，研究判断）
【实验 3 · 诊断：过拟合到"干净观测"，不是训练不够、也不是土箱变了】
【实验 4 · 域随机化救：训练时让它见过各种探头噪声】
  4-1  域随机化训练（每局随机一个探头噪声水平）
  4-2  域随机化模型 vs 标准模型在各噪声下对比 → 救回 + 鲁棒性曲线
【第 5 部分 · 收口：RL 真比规则强吗 + Limitations + logbook】
  5-1  PPO vs 规则基线总对比（证明"非得用 RL"）+ 判据
  5-2  保存模型 + Limitations（诚实交代局限）+ logbook
"""

# ============================================================================
# 第 0 部分 · 准备：先把土箱写成环境（这是你的活）
# ============================================================================

# %% 0-1　环境、依赖
# 【这段做什么】检查并安装本节依赖：gymnasium（环境接口）、stable-baselines3（现成 PPO）。
#   和 Class 2/3 一样先检查、没有再自动装，保证 Colab 上一个 cell 能顺利跑通。
#   装完若是 Colab，建议「代码执行程序 → 重启运行时」一次，再从 0-2 继续。
#
# 【和前两讲的关系：这次是把工具用到自己的环境上】
#   Class 2 纯手写 Q-learning（看清算法内部）；Class 3 用 SB3 的 PPO 跑现成的 CartPole
#   （看清"训练好≠能用"）。这一节是合体：环境换成【你自己写的土箱】，算法仍用现成 PPO。
#   重点全在更高一层的研究问题：奖励该怎么设？训好的策略换个土箱还能用吗？怎么救？
#   ——环境是你的活，算法是现成的工具。这正是真实科研里的分工。
import importlib, subprocess, sys

def _ensure(pkg, pip_name=None):
    try:
        importlib.import_module(pkg); print(f"  [OK] {pkg} 已安装")
    except ImportError:
        name = pip_name or pkg
        print(f"  [安装] pip install {name} ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", name], check=True)
        print(f"  [完成] {name}")

print("检查依赖：")
_ensure("gymnasium"); _ensure("stable_baselines3", "stable-baselines3")
_ensure("numpy"); _ensure("matplotlib")
print("依赖好了。若是 Colab 且刚装过库，建议『重启运行时』后从 0-2 继续。")

# %% 0-2　导入 + 固定随机种子 + CONFIG（标准土箱配置）
# 【这段做什么】导入库、固定随机种子，并把"标准土箱"的所有配置集中放进 CONFIG 字典。
#   后面无论是"奖励设歪版""漂移土箱""域随机化范围"，都从这份 CONFIG 派生着改——
#   集中管理才能做到"控制变量"（每次只动一个，其余完全一致）。
#
# 【为什么把参数全集中到一个 CONFIG】
#   实验设计的灵魂是控制变量。如果参数散落在代码各处，你改一个 k_eo 还得满文件找，
#   很容易漏改/错改，对照就不干净了。集中成一个字典，复制一份、只改一个键，
#   就得到一个"只差这一个变量"的新土箱——这是后面所有对照实验的基础设施。
#
# 【这些参数分别管什么（先有个印象，0-3 里会用到）】
#   n_cells/max_steps：土箱切几格、一局走几步（结构参数，全程不变，改它会变 obs 维度）。
#   v_max：电压上限（动作范围）。k_eo：电渗系数（电压→水流量的效率，≈土质）。
#   m_threshold：低湿失效拐点（含水率低于它，电渗几乎搬不动水）。
#   k_ph：单位通电量的 pH 漂移幅度。ph_tol：pH 偏离中性 7 的容忍带。
#   w_progress/w_energy/w_ph：奖励三项的权重（搬匀 / 省电 / 护 pH 的天平，实验 1 的主角）。
#   m_wet/m_dry：初始湿区/干区含水率（装土的初始干湿，实验 2 会漂移它）。
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
plt.rcParams["axes.unicode_minus"] = False

SEED = 42
np.random.seed(SEED)

CONFIG = {
    "n_cells": 6,          # 把土箱沿长度方向切成几格（1D 简化）；改它会改 obs/action 维度，全程锁死
    "max_steps": 60,       # 一局最多走多少步（每步相当于一小段通电时间）
    "v_max": 1.0,          # 单个电极边界的最大电压（归一化）；动作就在 [-v_max, v_max]
    # --- 电渗物理 ---
    "k_eo": 0.15,          # 电渗系数：电压转化成水流量的比例（≈土质/电渗效率，实验 2 漂移它）
    "m_threshold": 0.12,   # 含水率低于这个值，电渗可动性急剧下降（低湿失效的拐点）
    # --- pH 物理 ---
    "k_ph": 0.08,          # 单位通电量造成的 pH 漂移幅度
    "ph_tol": 1.0,         # pH 偏离中性 7 超过这个容忍带就算「越界」
    # --- 奖励权重（实验 1 的主角：正确目标 vs 朴素 proxy）---
    "w_progress": 1.0,     # 奖「真实搬匀进展」（spread 真的下降才给分）——这才是真目标
    "w_flux": 0.0,         # 奖「搬水活动量」|flux|（朴素 proxy；默认 0，实验 1 把它开成 1.0 演示刷分）
    "w_energy": 0.05,      # 耗电惩罚权重
    "w_ph": 0.5,           # pH 越界惩罚权重
    # --- 初始状态（装土的初始干湿，实验 2 漂移它）---
    "m_wet": 0.35,         # 湿区初始含水率
    "m_dry": 0.10,         # 干区初始含水率
}
print(f"随机种子固定为 {SEED}。标准土箱配置 CONFIG：")
for k, v in CONFIG.items():
    print(f"  {k:12s} = {v}")

# %% 0-3　SoilBoxEnv —— 把土箱写成 Gym 环境（★核心）
# 【这段做什么】把"电渗土箱"写成一个标准 Gym 环境，三块缺一不可：
#   reset()  —— 每局开局把土箱拨回初始"左湿右干"状态，返回初始观测；
#   step()   —— 收到动作（各边界电压）→ 用简化电渗物理更新湿度/pH → 算奖励 → 返回；
#   _reward()—— 把任务三要求（搬匀 / 省电 / 护 pH）翻译成三笔分数。
#   外加探头噪声 + 域随机化开关（obs_noise / randomize_noise）：训练时每局随机一个噪声水平——实验 4 的救命工具。
#   最后用 SB3 的 check_env 自动验证接口写对没。
#
# 【三件要素 → 三段代码（Class 3 学的 Gym 接口，这次自己写）】
#   · 观测 observation_space：各格含水率 m[0..n-1] + 各格归一化 pH，拼成一个向量（连续，用 Box）。
#   · 动作 action_space：n-1 个相邻格子边界上的电压（连续，[-v_max, v_max]）。正电压把水
#     从左格电渗到右格；电压正负即极性。连续动作正是 PPO 的强项。
#   · 奖励 reward：+搬匀的进展 − 耗电 − pH 越界（见 _reward）。
#
# 【step 里的简化物理：为什么这样写、它替代了什么】
#   不解完整 PDE（那是另一个课题），用一个"够训练用"的简化规则：
#   (1) 电渗搬水：相邻格子间，水流量 ∝ 电压 × 可动性，而可动性随含水率下降而下降——
#       含水率低于 m_threshold 时可动性趋近 0（孔隙水断了，电渗失效）。这把课题硬伤
#       "低湿区电渗可能失效"直接建进了物理，不是回避它。
#   (2) pH 漂移：通电使正极侧酸化、负极侧碱化，幅度 ∝ 通电量；极性反转可把 pH 拉回。
#   (3) 防抽干：一次最多搬走源格子 80% 的水，避免数值跑飞出现负含水率。
#
# 【探头噪声 obs_noise 与域随机化开关（实验 2~4 的主角，先理解原理）】
#   obs_noise>0 时，给观测（湿度/pH 读数）叠加高斯噪声，模拟真实电化学探头的噪声/漂移——
#   仿真里读数是干净的，真机探头永远有噪声，这正是 sim-to-real 最危险的一道坎。
#   randomize_noise=True 时，每局 reset 随机抽一个噪声水平（域随机化），逼它学"读数再脏也能调水"
#   的鲁棒策略，而不是只会在干净读数下工作。这跟 Class 3 给 CartPole 加观测噪声是同一招。
#   （注：实测发现策略对土质 k_eo / 装土 m_dry 这类"物理量变"其实很鲁棒——闭环反馈会自适应；
#    真正的 sim-to-real 杀手是"观测被污染"，所以这一节的漂移主角选探头噪声，不是土箱参数。）
#
# 【看什么】这段只是"造环境"，运行时只会打印一行 check_env 通过的确认。
#   真正用它，是 0-4（自检）和后面所有实验。记住三件事：reset 造初始局面、
#   step 推进物理并给分、randomize 是训练时换土箱的开关。
class SoilBoxEnv(gym.Env):
    """
    电渗土箱的数字孪生（1D 简化）。
    状态 observation：各格含水率 m[0..n-1] + 各格 pH p[0..n-1]（归一化后）拼成一个向量。
    动作 action：n-1 个相邻格子边界上的电压（连续，[-v_max, v_max]），正电压把水左→右搬。
    奖励 reward：+搬匀的进展 − 耗电 − pH 越界（见 _reward）。
    目标：把初始「左湿右干」搬到尽量均匀，又省电、又不让 pH 越界。
    """
    metadata = {"render_modes": []}

    def __init__(self, config=CONFIG, obs_noise=0.0, randomize_noise=False, noise_range=(0.0, 0.2)):
        super().__init__()
        self.cfg = dict(config)
        self.n = self.cfg["n_cells"]
        self.v_max = self.cfg["v_max"]
        # 动作空间：n-1 个边界电压，连续
        self.action_space = spaces.Box(
            low=-self.v_max, high=self.v_max, shape=(self.n - 1,), dtype=np.float32
        )
        # 观测空间：n 个含水率(0~0.5) + n 个归一化 pH((p-7)/7 ~ [-1,1])
        obs_low = np.array([0.0] * self.n + [-1.0] * self.n, dtype=np.float32)
        obs_high = np.array([0.5] * self.n + [1.0] * self.n, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        # sim-to-real 漂移：探头观测噪声（实验 2~4 的主角）
        self.obs_noise = obs_noise              # 评估时固定的探头噪声水平（0=仿真里的干净读数）
        self.randomize_noise = randomize_noise  # 训练时是否每局随机一个噪声水平（域随机化开关，实验 4 打开）
        self.noise_range = noise_range
        self._rng = np.random.RandomState(SEED)
        self._init_spread = None

    # ---- 把内部物理量打包成给智能体看的观测向量（可叠加探头噪声）----
    def _get_obs(self):
        ph_norm = (self.pH - 7.0) / 7.0
        obs = np.concatenate([self.m, ph_norm])
        if self.obs_noise > 0:                  # 模拟真实电化学探头读数的噪声/漂移（仿真里读数是干净的）
            obs = obs + self._rng.normal(0.0, self.obs_noise, size=obs.shape)
        return obs.astype(np.float32)

    # ---- 不均匀度：含水率离平均值的总偏离，越小越均匀 ----
    def _spread(self):
        return float(np.sum(np.abs(self.m - self.m.mean())))

    # ---- reset：每局开始，把土箱拨回「左湿右干」初始状态 ----
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 域随机化：训练时每局随机一个探头噪声水平（逼它学会在脏读数下也能调水）
        if self.randomize_noise:
            self.obs_noise = float(self._rng.uniform(*self.noise_range))
        n = self.n
        self.m = np.empty(n, dtype=np.float64)
        half = n // 2
        self.m[:half] = self.cfg["m_wet"]   # 左半：湿
        self.m[half:] = self.cfg["m_dry"]   # 右半：干
        self.pH = np.full(n, 7.0, dtype=np.float64)  # 初始中性
        self.steps = 0
        self._init_spread = self._spread()
        self.prev_spread = self._init_spread
        return self._get_obs(), {}

    # ---- step：环境的心脏 ----
    def step(self, action):
        cfg = self.cfg
        a = np.clip(np.asarray(action, dtype=np.float64), -self.v_max, self.v_max)

        # (1) 电渗：相邻格子间搬水。可动性随含水率下降而下降 → 低湿失效
        dm = np.zeros(self.n)
        flux_activity = 0.0   # 这一步「搬动的水总量」Σ|flux|——朴素 proxy 奖励就盯它（实验 1 的坑）
        for j in range(self.n - 1):
            m_avg = 0.5 * (self.m[j] + self.m[j + 1])
            # 平滑可动性：m 远大于阈值 ~m_avg；m 远小于阈值 → 趋近 0（孔隙水断了）
            mobility = m_avg * (m_avg / (m_avg + cfg["m_threshold"]))
            flux = cfg["k_eo"] * a[j] * mobility   # >0：水从 j 流向 j+1
            # 防抽干：一次最多搬走源格子 80% 的水
            if flux > 0:
                flux = min(flux, 0.8 * self.m[j])
            else:
                flux = max(flux, -0.8 * self.m[j + 1])
            flux_activity += abs(flux)             # 不管搬去哪、搬来搬去都算「活动量」——这正是它能被刷的原因
            dm[j] -= flux
            dm[j + 1] += flux
        self.m = np.clip(self.m + dm, 0.0, 0.5)

        # (2) pH：通电使靠正极侧酸化、靠负极侧碱化；极性反转可把 pH 拉回
        dp = np.zeros(self.n)
        for j in range(self.n - 1):
            dp[j] -= cfg["k_ph"] * a[j]
            dp[j + 1] += cfg["k_ph"] * a[j]
        self.pH = np.clip(self.pH + dp, 0.0, 14.0)

        # (3) 奖励：真实进展 +(可选)活动量 proxy − 耗电 − pH 越界
        reward, terms = self._reward(a, flux_activity)

        # (4) 结束条件
        self.steps += 1
        cur_spread = self._spread()
        success = cur_spread < 0.1 * self._init_spread   # 够均匀了
        terminated = bool(success)
        truncated = bool(self.steps >= cfg["max_steps"])
        if success:
            reward += 5.0   # 成功补一个大奖励
        self.prev_spread = cur_spread

        info = {"terms": terms, "spread": cur_spread, "success": success}
        return self._get_obs(), float(reward), terminated, truncated, info

    # ---- 奖励函数：把任务翻译成分数（实验 1 会拿 progress vs flux 做文章）----
    def _reward(self, a, flux_activity):
        cfg = self.cfg
        cur_spread = self._spread()
        # 进展：这一步让不均匀度减小了多少（真的越搬越均匀 → 正分）——这是「真实目标」
        progress = self.prev_spread - cur_spread
        # 耗电：电压平方 ∝ 功率
        energy = float(np.mean(a ** 2))
        # pH 越界：取「最坏那一格」超出容忍带的量（max-over-cells，不被没越界的中间格稀释）
        violation = float(np.max(np.maximum(0.0, np.abs(self.pH - 7.0) - cfg["ph_tol"])))
        # 总奖励 = 真实进展 +(可选)活动量 proxy − 耗电 − pH 越界
        #   w_flux 默认 0（不奖 proxy）；实验 1 把 w_flux 开成 1、w_progress 设 0，就成了「只奖搬水活动量」的朴素错奖励
        r = (cfg["w_progress"] * progress
             + cfg.get("w_flux", 0.0) * flux_activity
             - cfg["w_energy"] * energy
             - cfg["w_ph"] * violation)
        return r, {"progress": progress, "energy": energy, "ph_violation": violation, "flux": flux_activity}


# 用 SB3 自带检查器验证接口写对没（reset/step 返回值、空间维度等）
from stable_baselines3.common.env_checker import check_env
check_env(SoilBoxEnv(), warn=True)
print("SoilBoxEnv 接口检查通过：obs =", SoilBoxEnv().observation_space.shape,
      " action =", SoilBoxEnv().action_space.shape)

# %% 0-4　自检：随机动作跑一局，画湿度演化（确认环境没写错）
# 【这段做什么】先别急着训练。用"随机瞎按"的策略跑一局，把每格含水率随步数的变化画出来，
#   确认环境能跑、能返回观测、能画图——也就是确认你的环境没写错。
#
# 【为什么训练前一定要做这步自检】
#   RL 训练动辄几万步，跑完才发现环境写错（比如 step 返回值顺序错、奖励永远是 0），
#   是巨大的时间浪费。先用随机策略跑一局看一眼：曲线在动、数值在合理范围（含水率 0~0.5），
#   就说明环境的"骨架"是通的。这是 RL 调试的第一性原理：先确认环境对，再谈算法。
#
# 【看什么】
#   · 6 条线（cell 0~5）随步数的含水率变化。随机策略下它们会乱七八糟地抖动，
#     很难收拢到一起——这是对的，因为它在瞎按，没有章法。
#   · 关键是：曲线在动（环境响应了动作）、数值待在 0~0.5（没跑飞）。
#   · 如果所有线纹丝不动，或数值冲到几十——那是环境写错了，回头查 step。
env = SoilBoxEnv()
obs, _ = env.reset(seed=SEED)
m_history = [env.m.copy()]
for _ in range(CONFIG["max_steps"]):
    a = env.action_space.sample()          # 瞎按（随机策略）
    obs, r, term, trunc, info = env.step(a)
    m_history.append(env.m.copy())
    if term or trunc:
        break
m_history = np.array(m_history)

plt.figure(figsize=(8, 4))
for i in range(env.n):
    plt.plot(m_history[:, i], label=f"cell {i}")
plt.xlabel("step"); plt.ylabel("humidity m")
plt.title("Random policy: humidity per cell (sanity check)")
plt.legend(fontsize=8); plt.tight_layout(); plt.show()
print("随机策略下含水率曲线乱七八糟、很难变均匀——这正常，它在瞎按。")
print("曲线在动、数值在 0~0.5 = 环境写对了。下面把判据和工具写成函数，再开始做实验。")

# %% 0-5　判据写成代码：工厂 + evaluate + 规则基线 + 训练函数 + judge_*
# 【这段做什么】把"反复要用的东西"全部写成函数，后面实验只管调用：
#   ① make_soilbox(**overrides)：土箱工厂——复制标准 CONFIG、只改你指定的几个参数，
#      造出一个"只差这几个变量"的土箱。这是控制变量实验的基础设施。
#   ② evaluate(policy_fn, cfg)：判据函数——把策略丢进某个土箱跑 30 局，返回五个平均指标。
#      这是量化一切的硬尺子。
#   ③ rule_policy：规则基线（纯 if-else，不学习），收口时要让 RL 去打它。
#   ④ train_ppo(cfg/env, steps, tag)：在某个土箱里训一个 PPO，返回模型 + 训练曲线。
#   ⑤ judge_*()：把每个实验的结论量化成 True/False + 数字（reward hacking 复现了吗 /
#      泛化崩了吗 / 域随机化救回了吗 / RL 真比规则强吗）。
#
# 【为什么判据必须函数化、必须固定 30 局 + 固定 seed】
#   单跑一局，初始局面的运气会让结果飘很大。30 局平均才能把运气抹平、露出真实水平
#   （和 Class 2 的"30 个 seed 全测"、Class 3 的"evaluate 跑 30 局"是同一个科研习惯）。
#   evaluate 内部每局用固定 seed（seed0+ep），保证不同模型/不同土箱都在【完全相同的
#   一组初始局面】上比——成绩差异只来自"模型/土箱本身"，不来自运气。判据一致，结论才可比。
#
# 【五个指标分别是什么（看 evaluate 的返回）】
#   reward：平均累计回报（综合分）。moved_frac：把初始不均匀度搬掉了百分之多少（越高越匀）。
#   energy：平均耗电（越低越省）。ph_viol：平均 pH 越界量（越低越护土）。success：成功率。
#   注意 moved_frac/energy/ph_viol 这三个是后面所有对比的主角——它们正好对应任务三要求。
#
# 【看什么】这段只"造工具"，运行时只打印一行"判据流水线就绪"。
#   工具造好了，从实验 1 开始才是"用工具做科研"。
class RewardLogger(BaseCallback):
    """记录训练中每个 rollout 的平均回合奖励，用来画训练曲线（RL 的'体温计'）。"""
    def __init__(self): super().__init__(); self.ts, self.rew = [], []
    def _on_step(self): return True
    def _on_rollout_end(self):
        buf = self.model.ep_info_buffer
        if buf and len(buf) > 0:
            self.ts.append(self.num_timesteps)
            self.rew.append(float(np.mean([e["r"] for e in buf])))

def make_soilbox(**overrides):
    """土箱工厂：复制标准 CONFIG，只改 overrides 指定的参数，返回一个 SoilBoxEnv。"""
    cfg = dict(CONFIG); cfg.update(overrides)
    return SoilBoxEnv(cfg)

def rule_policy(obs, n, v_max):
    """规则基线（对照组）：哪边湿就把水往干的方向猛推；pH 快越界就把该边界电压减半。
    它能搬水，但只会「猛干」，不会精打细算 → 费电、pH 容易顶到边。
    RL 必须打得过它，才证明「用强化学习」是值得的。"""
    m = obs[:n]
    ph = obs[n:] * 7.0 + 7.0
    a = np.zeros(n - 1, dtype=np.float32)
    for j in range(n - 1):
        diff = m[j] - m[j + 1]                 # 左格比右格湿多少
        a[j] = v_max * np.sign(diff)           # 湿度差方向，满电压猛推
        if abs(ph[j] - 7) > 1.0 or abs(ph[j + 1] - 7) > 1.0:
            a[j] *= 0.5                         # 一点点 pH 保护（仍不够聪明）
    return a

def evaluate(policy_fn, cfg=CONFIG, obs_noise=0.0, n_episodes=30, seed0=1000):
    """判据：把策略丢进土箱(cfg + 探头噪声 obs_noise)跑 n 局，返回平均指标。policy_fn(obs)->action。
    指标：moved_frac=真实搬匀程度 / flux_act=搬水活动量(proxy，实验1的主角) / energy=耗电 /
          ph_viol=回合内峰值越界(max-over-cells，任一格 pH 坏了就记账) / success=成功率 / reward=回报。
    obs_noise>0 时给观测加探头噪声（实验 2~4 用它做 sim-to-real 漂移）。"""
    e = SoilBoxEnv(cfg, obs_noise=obs_noise)
    tot_r, tot_energy, tot_phviol, tot_flux, succ, moved = [], [], [], [], [], []
    for ep in range(n_episodes):
        obs, _ = e.reset(seed=seed0 + ep)      # 固定评估 seed：所有模型在相同初始局面上比
        init_spread = e._init_spread
        ep_r = ep_e = ep_f = 0.0; ep_peak = 0.0
        info = {"success": False, "spread": init_spread}
        for _ in range(cfg["max_steps"]):
            a = policy_fn(obs)
            obs, r, term, trunc, info = e.step(a)
            ep_r += r
            ep_e += info["terms"]["energy"]
            ep_f += info["terms"]["flux"]
            ep_peak = max(ep_peak, float(np.max(np.maximum(0.0, np.abs(e.pH - 7.0) - cfg["ph_tol"]))))
            if term or trunc:
                break
        tot_r.append(ep_r); tot_energy.append(ep_e); tot_phviol.append(ep_peak)
        tot_flux.append(ep_f); succ.append(1.0 if info["success"] else 0.0)
        moved.append((init_spread - info["spread"]) / init_spread)  # 真实搬匀了百分之多少
    return {
        "reward": float(np.mean(tot_r)), "energy": float(np.mean(tot_energy)),
        "ph_viol": float(np.mean(tot_phviol)), "flux_act": float(np.mean(tot_flux)),
        "success": float(np.mean(succ)), "moved_frac": float(np.mean(moved)),
    }

def train_ppo(env, steps, tag=""):
    """在 env（已建好的 SoilBoxEnv）里训一个 PPO，返回 (模型, 训练曲线记录器)。"""
    mon = Monitor(env)
    model = PPO("MlpPolicy", mon, seed=SEED, verbose=0, device="cpu")  # 三行核心之一：建模型
    lg = RewardLogger()
    print(f"  开始训练 PPO（{tag}，{steps} 步，CPU 几分钟，耐心等）...")
    model.learn(total_timesteps=steps, callback=lg)                    # 三行核心之二：开训
    return model, lg

def ppo_policy_of(model):
    """把训练好的 PPO 包成一个 policy_fn(obs)->action，喂给 evaluate。"""
    return lambda o: model.predict(o, deterministic=True)[0]           # 部署用确定性动作，不再探索

# ---- 判据函数：把每个实验的结论量化成 (是否成立, 说明文字) ----
def judge_reward_hacking(res_naive, res_full):
    """实验 1 判据：朴素 proxy 版把「搬水活动量」刷得更高，「真实搬匀」却明显更差（甚至为负）
    = reward hacking 复现（刷了指标、没完成任务）。"""
    flux_higher = res_naive["flux_act"] > res_full["flux_act"]            # proxy 版活动量刷得更高
    moved_worse = res_naive["moved_frac"] < res_full["moved_frac"] * 0.5  # 但真实搬匀差一大截
    ok = flux_higher and moved_worse
    return ok, (f"搬水活动量 proxy版={res_naive['flux_act']:.2f} vs 正确版={res_full['flux_act']:.2f}（proxy 刷得更高）/ "
                f"真实搬匀 proxy版={res_naive['moved_frac']:.3f} vs 正确版={res_full['moved_frac']:.3f}"
                f"（{'reward hacking：刷了指标没干活' if ok else '未复现'}）")

# 指标名 → 中文标签（判据 msg 是打印给学生看的，别露 success/moved_frac 这种英文变量名）
_METRIC_CN = {"moved_frac": "搬匀程度", "success": "成功率", "energy": "耗电",
              "ph_viol": "pH越界", "flux_act": "搬水活动量", "reward": "回报"}

def judge_generalization_fail(res_in, res_out, metric="success", drop=0.30):
    """实验 2 判据：标准模型从'干净/标准'换到'漂移环境'，主指标是否掉超过 drop（默认 30%）= 泛化失败。
    metric 默认 success（本节泛化主指标看成功率，与调用一致；也可传 moved_frac 等）。"""
    base = abs(res_in[metric]) + 1e-9
    rel = (res_in[metric] - res_out[metric]) / base
    ok = rel > drop
    name = _METRIC_CN.get(metric, metric)
    return ok, (f"{name}：标准={res_in[metric]:.3f} → 漂移={res_out[metric]:.3f}"
                f"（掉 {rel*100:.0f}%，{'泛化失败' if ok else '扛得住'}）")

def judge_dr_rescue(res_dr_out, res_std_out, metric="success", margin=1.2):
    """实验 4 判据：域随机化模型在漂移环境下的主指标，是否明显优于标准模型（≥margin 倍）= 救回。
    metric 默认 success（本节泛化主指标看成功率，与调用一致；也可传 moved_frac 等）。"""
    ok = res_dr_out[metric] > res_std_out[metric] * margin
    name = _METRIC_CN.get(metric, metric)
    return ok, (f"漂移环境下{name}：标准模型={res_std_out[metric]:.3f}  vs  "
                f"域随机化={res_dr_out[metric]:.3f}（{'救回' if ok else '没救回'}）")

def judge_beats_baseline(res_ppo, res_rule):
    """收口判据：PPO 在'搬匀程度不输'的前提下，耗电和 pH 越界都更低 = RL 的非平凡权衡价值成立。"""
    moved_ok = res_ppo["moved_frac"] >= res_rule["moved_frac"] * 0.9   # 搬匀程度不明显落后
    energy_ok = res_ppo["energy"] < res_rule["energy"]
    ph_ok = res_ppo["ph_viol"] <= res_rule["ph_viol"]
    ok = moved_ok and energy_ok and ph_ok
    return ok, (f"搬匀 {res_ppo['moved_frac']:.3f}vs{res_rule['moved_frac']:.3f} / "
                f"耗电 {res_ppo['energy']:.3f}vs{res_rule['energy']:.3f} / "
                f"pH越界 {res_ppo['ph_viol']:.3f}vs{res_rule['ph_viol']:.3f}"
                f"（{'RL 赢在多目标权衡' if ok else '优势不明显'}）")

print("判据流水线就绪：make_soilbox(造土箱) + evaluate(量五个指标) + train_ppo(训) + judge_*(下结论)。")

# ============================================================================
# 实验 1 · 奖励怎么设：你给什么它追什么（reward hacking）
# ============================================================================

# %% 1-1　朴素 proxy 奖励：奖「搬水活动量」→ 训一版
# 【这段做什么】先按一个初学者最自然的想法来设奖励。你的真实目标是"把水搬均匀"，
#   一个很自然（但错误）的念头是："那就奖励它多搬水嘛，搬得越多越快越好。"
#   于是把奖励设成「每步搬动的水量 Σ|flux| 越大越好」（cfg_proxy：w_flux=1，其余关掉）。
#   这个"搬水活动量"就是一个 proxy（代理指标）——它看起来和真实目标相关，其实并不等价。
#   用这个 proxy 训一个 PPO，故意叫它 naive_model（朴素 proxy 版）。
#
# 【为什么要先做"错误"的版本（这是实验设计的关键）】
#   如果一上来就把奖励设对、训出好模型，你只会看到"它学会了"，
#   永远不知道"奖励设歪会怎样"。而"奖励设歪会怎样"恰恰是这门课最值钱的一课、答辩高频考点。
#   所以先亲手把坑踩出来：奖"搬水活动量"，看它学成什么。这跟 Class 2 先让纯贪心撞墙、
#   Class 3 先让标准模型在长杆上垮是同一个套路：先制造失败，再诊断、再救。
#
# 【proxy 错在哪：它和真实目标不等价】
#   真实目标是"末状态均匀"（spread 下降）。但"搬水活动量大"不等于"搬得均匀"——
#   把水在两个格子之间来回倒，活动量很大，可水根本没被真正搬匀（来回倒净位移≈0）。
#   一个 proxy 只要和真实目标不完全等价，智能体就可能钻这个缝：去最大化 proxy，
#   而不是完成真实目标。这正是第 1 讲埋的"你给什么它追什么"。
#
# 【看什么】这段只训练，输出是训练进度。真正的"后果"在 1-2 揭晓。
#   先记住：这一版的奖励【只看搬水活动量，根本不看水到底匀没匀】。
cfg_proxy = dict(CONFIG); cfg_proxy["w_progress"] = 0.0; cfg_proxy["w_flux"] = 1.0
cfg_proxy["w_energy"] = 0.0; cfg_proxy["w_ph"] = 0.0     # 只奖「搬水活动量」这个 proxy
TRAIN_STEPS = 150000
print("实验 1-1：用『只奖搬水活动量』的朴素 proxy 奖励训练（w_flux=1，其余关掉）")
naive_model, naive_log = train_ppo(SoilBoxEnv(cfg_proxy), TRAIN_STEPS, "朴素proxy·只奖搬水活动量")
print("  训练完成。它在『搬得越多越好、不看匀不匀』的考卷下，会学成什么样？看 1-2。")

# %% 1-2　看它学成什么 + 判据：刷活动量却不搬匀（亲手跑出 reward hacking）
# 【这段做什么】跑一局朴素 proxy 版策略，把含水率演化和每步搬水活动量画出来，用判据量化。
#   你会看到一个荒谬又合理的结果：它把"搬水活动量"刷得极高（疯狂地把水搬来搬去），
#   但水根本没被真正搬匀（moved_frac 很低、甚至为负——比一开始还乱）。
#   它在原地"空转"刷指标，没有完成你真正想要的任务。
#
# 【这就是 reward hacking（奖励钻空子）——讲透它为什么"没错又很错"】
#   智能体没有任何 bug，它完美地最大化了你给的分数——"搬水活动量"。
#   它发现"在两格之间把水来回倒"能持续产生很大的 flux、源源不断刷分，
#   至于"水最后匀不匀"，你的奖励里根本没写，它当然就不管。
#   错的不是它，是你把奖励挂在了一个"和真实目标不等价的 proxy"上。
#   这正是第 1 讲埋的那句"你给什么它追什么"。
#
# 【经典类比：赛艇游戏刷分】
#   OpenAI 那个著名例子：让 AI 玩赛艇，奖励设成"得分"，结果它发现绕着一个小水湾
#   原地打转、反复吃重生的分球，比好好跑完赛道得分还高——于是它根本不比赛，在原地刷分。
#   你这个"来回搬水刷活动量"的智能体，和那条原地打转的赛艇是一模一样的病。
#
# 【为什么这对你的课题是致命的】
#   你的土箱真实目标是"把土壤水分修复均匀"，不是"让水泵多动几下"。如果你图省事，
#   用一个容易测量的 proxy（"通电时长""搬水流量"之类）当奖励，训出来的策略很可能
#   学会刷这个 proxy 而不真正修复土壤。答辩被问"你怎么保证它优化的是真目标、不是某个
#   代理指标"，这段实验就是你的答案：奖励必须挂在真实目标（末状态均匀）上。
#
# 【看什么】
#   · 左图含水率：6 条线被它搅得乱七八糟、迟迟收不拢（甚至比初始更散）= 没真搬匀。
#   · 右图每步搬水活动量：proxy 版一路维持在很高的水平（它在拼命刷这个数）。
#   · 判据 judge_reward_hacking：proxy 版活动量更高、但真实搬匀更差 → True（刷了指标没干活）。
#     （正确版的数 1-3 才训出来，这里先把 proxy 版的数算好、图画好。）
# 【预期（服务器实测）】proxy 版 flux_act≈7.04、moved_frac≈−0.20、energy≈41.5——疯狂刷活动量、水却更乱（负搬匀）。
naive_policy = ppo_policy_of(naive_model)
e = SoilBoxEnv(cfg_proxy)
obs, _ = e.reset(seed=SEED)
m_hist, flux_hist = [e.m.copy()], []
for _ in range(CONFIG["max_steps"]):
    a = naive_policy(obs)
    obs, r, term, trunc, info = e.step(a)
    m_hist.append(e.m.copy()); flux_hist.append(info["terms"]["flux"])
    if term or trunc:
        break
m_hist = np.array(m_hist)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for i in range(e.n): ax[0].plot(m_hist[:, i], label=f"cell {i}")
ax[0].set_title("Proxy reward: humidity NEVER levels (just churns)")
ax[0].set_xlabel("step"); ax[0].set_ylabel("m"); ax[0].legend(fontsize=7)
ax[1].plot(flux_hist, color="crimson")
ax[1].set_title("Proxy reward: water 'activity' kept HIGH (gaming it)")
ax[1].set_xlabel("step"); ax[1].set_ylabel("flux activity / step")
plt.tight_layout(); plt.show()

res_naive = evaluate(naive_policy, cfg=cfg_proxy)
print("实验 1-2：朴素 proxy 版的判据成绩")
print(f"  {({k: round(v,3) for k,v in res_naive.items()})}")
print(f"  → 搬水活动量 flux_act={res_naive['flux_act']:.2f}（刷得很高），"
      f"但真实搬匀 moved_frac={res_naive['moved_frac']:.3f}（没干活，可能为负）。")
print("  （服务器实测：proxy flux 7.04 / moved −0.20  vs  正确版 flux 1.05 / moved 0.93）这就是 reward hacking：它刷的是你写的 proxy，不是你真想要的目标。")

# %% 1-3　诊断 + 救：把奖励挂回「真实进展」→ 重训 → 真搬匀 → 判据量化
# 【这段做什么】对症下药——把奖励从"搬水活动量 proxy"换回"真实搬匀进展 progress"
#   （用标准 CONFIG：w_progress=1、w_flux=0，再带上耗电/pH 惩罚），重训一个 full_model。
#   注意：这个 full_model 不只是实验 1 的"正确答案"，它还是实验 2~4 的主角——
#   后面要看"这个奖励设对、标准土箱训好的模型，换个土箱还能不能用"。
#
# 【诊断：病根是奖励挂错了目标，不是算法】
#   有人看到 proxy 版空转刷分，第一反应是"PPO 不行 / 再多训训 / 换算法"。错。
#   proxy 版的 PPO 工作得非常好——它精确地最大化了你给的分（活动量）。问题 100% 在奖励：
#   你把分数挂在了"搬水活动量"这个 proxy 上，而它和真实目标（末状态均匀）不等价。
#   救法不是动算法，是把奖励挂回真实目标：奖"spread 真的下降了多少"（progress）。
#
# 【为什么"真实进展"奖励防得住刷分】
#   progress = 上一步的不均匀度 − 这一步的不均匀度。来回倒水：倒过去 spread 降一点、
#   倒回来又升回去，一加一减净进展≈0，拿不到分。要持续拿分，唯一的办法是真的把水
#   一步步搬到均匀。于是智能体被迫去完成真实目标，而不是空转刷分。
#   这就是奖励设计的核心功夫：让"最大化奖励"和"完成真实目标"变成同一件事。
#
# 【看什么】
#   · 左图含水率：正确版把 6 条线真正收拢到一起 = 真搬匀了（对比 1-2 proxy 版迟迟不收）。
#   · 右图活动量：正确版的活动量反而更低——它精打细算，不做无用的来回搬运。
#   · 判据 judge_reward_hacking：打印 True = proxy 版"活动量更高但搬匀更差"，reward hacking 成立。
#   · 这一对比（proxy 空转 vs 正确版真干活）就是答辩时回答"你怎么保证优化的是真目标"的最佳证据。
# 【预期（服务器实测）】正确版 moved_frac≈0.93、flux_act≈1.05、energy≈7.99、success=1.0；judge_reward_hacking=True。
print("实验 1-3：把奖励挂回『真实搬匀进展』重新训练（w_progress=1, w_flux=0，带上耗电/pH 惩罚）")
full_model, full_log = train_ppo(SoilBoxEnv(CONFIG), TRAIN_STEPS, "正确奖励·真实进展")
full_policy = ppo_policy_of(full_model)

# 跑一局正确版，画含水率 + 活动量（和 1-2 对照）
e = SoilBoxEnv(CONFIG)
obs, _ = e.reset(seed=SEED)
m_hist, flux_hist = [e.m.copy()], []
for _ in range(CONFIG["max_steps"]):
    a = full_policy(obs)
    obs, r, term, trunc, info = e.step(a)
    m_hist.append(e.m.copy()); flux_hist.append(info["terms"]["flux"])
    if term or trunc:
        break
m_hist = np.array(m_hist)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for i in range(e.n): ax[0].plot(m_hist[:, i], label=f"cell {i}")
ax[0].set_title("Correct reward: humidity TRULY levels out")
ax[0].set_xlabel("step"); ax[0].set_ylabel("m"); ax[0].legend(fontsize=7)
ax[1].plot(flux_hist, color="seagreen")
ax[1].set_title("Correct reward: low activity (no wasteful churning)")
ax[1].set_xlabel("step"); ax[1].set_ylabel("flux activity / step")
plt.tight_layout(); plt.show()

res_full = evaluate(full_policy, cfg=CONFIG)
print("实验 1-3：正确奖励版的判据成绩")
print(f"  {({k: round(v,3) for k,v in res_full.items()})}")
hacked, msg = judge_reward_hacking(res_naive, res_full)
print(f"  判据 judge_reward_hacking = {hacked}")
print(f"    {msg}")
print("  → 正确版把水真正搬匀了，活动量反而更低（不做无用功）。")
print("    结论：奖励挂在 proxy 上 → 学会刷指标空转；挂回真实目标 → 才真去完成任务。")
print("    （这就是『你给什么它追什么』。下面带着这个『奖励设对、标准土箱训好』的 full_model，")
print("     去问下一个问题：它换个土箱还能用吗？——这就是实验 2。）")

# ============================================================================
# 实验 2 · 训练好≠能用：探头一带噪声就抓瞎（sim-to-real 的真正杀手）
# ============================================================================

# %% 2-1　给观测加探头噪声 → 标准模型成功率暴跌（主漂移）
# 【这段做什么】把实验 1 训好的 full_model（训练时一直读"仿真里的干净读数"），丢进一系列
#   "探头带噪声"的世界里测——给观测（湿度/pH 读数）叠加标准差从 0 到 0.2 的高斯噪声，
#   逐档看它还能不能把水搬匀。其它一切（土质、装土、物理）全不变，只动"读数干不干净"。
#
# 【探头噪声对应真实世界的什么——这是 sim-to-real 的核心】
#   仿真里，智能体读到的湿度/pH 是精确数字；真机上，电化学探头读出来永远有噪声和漂移：
#   电极界面热噪声、基线漂移、外界电磁干扰……你的 SoilGuard 课题用 pA 级电流信号，
#   更是噪声重灾区。full_model 训练时从没见过脏读数，它默认"我看到的就是真相"。
#   一旦搬到真机，读数一脏，它对状态的判断就错、动作跟着错——这就是 sim-to-real gap
#   里最隐蔽也最致命的一种：不是物理变了，是"你以为你看清了，其实没有"。
#
# 【为什么噪声会害它垮——直觉】
#   full_model 学的是"看到这个精确的湿度/pH 组合，就推这个电压"。加噪声后，同一个真实状态，
#   每次读到的数都略有不同——它看到的是"没见过的输入模式"，动作开始出错；噪声越大，越容易
#   把"左边湿"误读成"右边湿"，推错方向、来回折腾，最后收敛不到"够均匀"（成功率掉）。
#
# 【看什么】
#   · 主看 success（成功率）：它比 moved_frac 更能暴露问题——噪声下它还能搬掉大半水（moved 不算太差），
#     却很难精确收敛到"够均匀"的成功线，所以成功率掉得最鲜明。
#   · 判据 judge_generalization_fail（用 success）逐行标"泛化失败"：噪声越大越可能失败。
# 【预期（服务器实测）】成功率随噪声：0.0→1.00 / 0.02→1.00 / 0.05→0.73 / 0.1→0.50 / 0.2→0.13（搬匀 moved 0.93→0.72，成功率掉得更狠）。
print("实验 2-1：给观测加探头噪声（模拟真机电化学探头；full_model 训练时只见过干净读数）")
res_std_clean = evaluate(full_policy, obs_noise=0.0)        # 干净读数基准（对比锚点）
noise_list = [0.0, 0.02, 0.05, 0.1, 0.2]
std_by_noise = []
for ns in noise_list:
    res = evaluate(full_policy, obs_noise=ns)
    std_by_noise.append(res)
    fail, _ = judge_generalization_fail(res_std_clean, res, metric="success")
    flag = "  ← 泛化失败" if (fail and ns > 1e-9) else ""
    print(f"  探头噪声={ns:<5} 成功率={res['success']:.2f}  搬匀={res['moved_frac']:6.3f}{flag}")
print("  → 噪声一大成功率暴跌（服务器实测 0.05→0.73, 0.1→0.50, 0.2→0.13）。它信了仿真的干净读数，真机一脏就误判。")

# %% 2-2　对照：改土质 k_eo / 装土 m_dry → 标准模型扛得住（鲁棒，研究判断）
# 【这段做什么】上一组看到"探头噪声"会害死模型。但如果就此说"换任何条件都崩"，太宽泛、没价值。
#   这一组改两个最常见的物理差异——土质 k_eo（电渗效率）和装土 m_dry（初始干湿），
#   看 full_model 扛不扛得住。它们是对照组：证明不是"一变就崩"，而是只栽在"观测被污染"上。
#
# 【为什么 full_model 对这两个物理量变反而鲁棒——直觉（重要）】
#   PPO 学到的不是"开环死记 0.15 这个土的固定力道"，而是"看湿度/pH 反馈、闭环调电压"。
#   换成电渗更弱的土（k_eo 小），它看到"推了半天还没匀"，自然会多推几步；换初始更干的土，
#   它看到干区还干，就继续往那边送水。闭环反馈天然能吸收"搬水快慢"这种量变——
#   这是好消息：你的策略对土质/装土的差异有自适应能力，不必为每种土重训。
#
# 【为什么这组对照是研究能力的体现】
#   只做 2-1，你只能说"它对噪声脆"。加上 2-2，你能下一句精确得多的结论：
#   "策略对【改变搬水快慢的物理量变】（土质 k_eo、装土 m_dry）鲁棒——闭环反馈自适应；
#    只对【污染观测本身】的探头噪声脆弱。" 这就告诉你部署时真正该砸钱解决的是什么：
#   不是把每种土的物理参数都标定准，而是把探头做干净、或让策略学会扛噪声。
# 【看什么】中间档 k_eo/m_dry 的成功率应该都满分；只有最极端档（k_eo=0.06、m_dry=0.04）成功率会掉，
#   但那是"物理本身太难"（电渗太弱/初始太干，逼近低湿失效），不是"策略失灵"——它仍能搬走约 8 成水
#   （moved 仍高）。和 2-1 噪声"连搬带成功一起渐崩"对照：物理量变只在极端处掉成功率，闭环反馈扛住了量变。
# 【预期（服务器实测）】k_eo：0.06→succ0.00/moved0.78，0.10~0.24→succ1.00/moved0.91~0.96；
#   m_dry：0.04→succ0.00/moved0.87，0.07~0.16→succ1.00/moved0.90~0.93。中间档全满分。
print("实验 2-2：对照——改土质 k_eo / 装土 m_dry（看物理量变扛不扛得住）")
keo_list = [0.06, 0.10, 0.15, 0.20, 0.24]; std_by_keo = []
for ke in keo_list:
    res = evaluate(full_policy, cfg=dict(CONFIG, k_eo=ke)); std_by_keo.append(res)
    print(f"  k_eo={ke:<5}（训练值0.15） 成功率={res['success']:.2f}  搬匀={res['moved_frac']:6.3f}")
mdry_list = [0.04, 0.07, 0.10, 0.13, 0.16]; std_by_mdry = []
for md in mdry_list:
    res = evaluate(full_policy, cfg=dict(CONFIG, m_dry=md)); std_by_mdry.append(res)
    print(f"  m_dry={md:<5}（训练值0.10） 成功率={res['success']:.2f}  搬匀={res['moved_frac']:6.3f}")
print("  → 中间档全部成功；只有最极端档（k_eo=0.06/m_dry=0.04）成功率掉到 0，但仍搬走约 8 成水")
print("    （moved≈0.78/0.87）——是物理太难（电渗太弱/太干）而非策略失灵。对比噪声『连搬带成功一起渐崩』，")
print("    物理量变只在极端处掉：闭环反馈能自适应『搬水快慢』这种量变。")
print("    研究判断：sim-to-real 的重点不是物理参数，是『观测质量』。探头噪声才是要救的方向。")

# 画实验 2 的对比图：探头噪声（暴跌）vs 物理量变（鲁棒）
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(noise_list, [r["success"] for r in std_by_noise], "o-", color="red")
axes[0].set_ylim(-0.05, 1.05); axes[0].set_xlabel("probe noise (obs_noise)"); axes[0].set_ylabel("success rate")
axes[0].set_title("Standard model vs PROBE NOISE (cliff!)"); axes[0].grid(alpha=0.3)
axes[1].plot(keo_list, [r["success"] for r in std_by_keo], "s-", color="green", label="vary k_eo")
axes[1].plot(mdry_list, [r["success"] for r in std_by_mdry], "^-", color="teal", label="vary m_dry")
axes[1].axvline(CONFIG["k_eo"], ls=":", color="gray"); axes[1].set_ylim(-0.05, 1.05)
axes[1].set_xlabel("soil k_eo / initial m_dry"); axes[1].set_ylabel("success rate")
axes[1].set_title("Standard model vs PHYSICS shift (robust)"); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()

# ============================================================================
# 实验 3 · 诊断：过拟合到「干净观测」，不是训练不够、也不是土箱变了
# ============================================================================
# %% 3　诊断结论
# 【这段做什么】把实验 2 两组证据归纳成一句精确诊断，讲透：病根在哪、为什么不是训练不够。
#   只有 print，没有新实验——它是逻辑链的转折点：诊断到病根，才能对症下药（实验 4）。
#
# 【实验 2 两组证据的逻辑链】
#   2-1：探头噪声 ↑ → 成功率暴跌                ← 观测被污染，扛不住
#   2-2：土质 k_eo / 装土 m_dry 大幅变 → 仍基本搬匀  ← 物理量变，闭环反馈扛住了
#   结论：full_model 不是对任何变化都脆，而是专门对"训练时没见过的观测污染"脆——
#   它训练时读的全是干净数字，把"读数=真相"当成了理所当然。
#
# 【为什么不是"训练不够"——把这条路堵死】
#   你可能想：多训几万步是不是就抗噪了？不是。它在干净观测下已经训得很好，
#   继续在【干净观测】上训，它永远遇不到脏读数，当然学不到"读数不可全信、要稳着来"。
#   缺的不是训练量，是【见过噪声】这件经历本身。这和 Class 2「加训练局数救不了纯贪心」、
#   Class 3「多训也救不了过拟合」是同一种病：缺的是某种经历，不是训练量。
#
# 【这就是 sim-to-real gap 最锋利的一刀——接你的课题】
#   你在仿真（干净读数）里把策略训到很好，搬到真机——真机的电化学探头有噪声和漂移，
#   策略一旦信了脏读数就误判。注意 2-2 的好消息：土质、装土这些物理差异它能自适应，
#   所以你不必为每种土重训；真正要命的是观测质量。这把"该担心什么"指得清清楚楚：
#   把探头做干净、或让策略学会在噪声下工作——后者就是实验 4 的域随机化。
print("=== 实验 3 诊断结论 ===")
print(f"· 现象：干净读数成功率≈{res_std_clean['success']:.2f}，探头噪声一大成功率暴跌（见 2-1）；")
print("  但土质 k_eo / 装土 m_dry 大幅变化，它基本扛得住（见 2-2）。")
print("· 不是训练不够：干净观测下已训好，在干净观测上再多训也遇不到脏读数。")
print("· 病根：过拟合到『干净观测』——训练时把『读数=真相』当默认，真机探头一脏就误判。")
print("· 这就是 sim-to-real gap 最锋利的一刀：物理量变能自适应，观测污染才致命。要救，让它训练时见过噪声。")

# ============================================================================
# 实验 4 · 域随机化救：训练时让它见过各种探头噪声
# ============================================================================

# %% 4-1　域随机化训练（每局随机一个探头噪声水平）
# 【这段做什么】对症下药——重训一个 PPO（dr_model），训练时每局开局随机抽一个探头噪声水平
#   （0~0.2，就是 0-3 里 SoilBoxEnv(randomize_noise=True) 那个开关），奖励、其它一切都和
#   full_model 一样。唯一的差别就是"训练时见没见过脏读数"。
#
# 【为什么这样能救——核心思想】
#   病根是"只见过干净读数"。那就让它训练时见过各种脏读数：每局噪声水平不一样，
#   它就不能再"全信瞬时读数"，必须学一个"读数有抖动也能稳住、靠趋势/多步信息判断"的策略。
#   训练时见过 0~0.2 的噪声，部署时遇到这个范围内的探头噪声就不陌生。
#   这跟 Class 3 把噪声纳入域随机化救回噪声鲁棒性，是一模一样的招。
#
# 【为什么域随机化是 sim-to-real 最主流的救法（接你的课题）】
#   真实部署时你没法精确知道探头噪声会有多大（电极老化、温度、电磁环境都在变）。
#   既然无法精确知道，就用"覆盖一个范围"代替"精确匹配"：训练时把可能的噪声水平都随机到，
#   部署时不管探头多脏（只要在训练覆盖范围内），模型都见过接近的，就能扛住。
#
# 【为什么训得比标准版久】
#   带噪声的任务更难（每局观测质量都不同，等于要"同时学会在各种噪声下工作"），
#   更难的任务需要更多经历才收敛，所以步数从 15 万加到 20 万。
#
# 【看什么】训完看 res_dr_clean：它在干净读数下的成绩。
#   关键期待：见过噪声后，它在干净读数下的表现不应明显低于 full_model——
#   即"学会抗噪"不以"牺牲干净读数下的表现"为代价。
# 【预期（服务器实测）】域随机化模型在干净读数 success=1.00（不输 full_model），没牺牲干净读数下的表现。
DR_TRAIN_STEPS = 200000   # 域随机化任务更难（每局噪声水平不同），步数从标准的 15 万加到 20 万
print("实验 4-1：域随机化训练（每局随机一个探头噪声水平 0~0.2）")
dr_model, dr_log = train_ppo(SoilBoxEnv(CONFIG, randomize_noise=True), DR_TRAIN_STEPS, "域随机化·随机探头噪声")
dr_policy = ppo_policy_of(dr_model)
res_dr_clean = evaluate(dr_policy, obs_noise=0.0)
print(f"  域随机化模型在干净读数：成功率={res_dr_clean['success']:.2f}"
      f"（full_model 是 {res_std_clean['success']:.2f}）。")
print("  （服务器实测：干净读数 success=1.00，不输 full）见过噪声没牺牲干净读数下的表现。下面看它在脏读数下救没救回。")

# %% 4-2　域随机化模型 vs 标准模型在各噪声下对比 → 救回 + 鲁棒性曲线
# 【这段做什么】用实验 2-1 完全相同的判据、完全相同的噪声档位，测域随机化模型，和标准模型
#   并排对比，再把两条鲁棒性曲线画在一起。
#
# 【控制变量：为什么这个对比是干净的】
#   full_model 和 dr_model 用同样的奖励、判据、噪声档位、评估 seed——唯一的差别就是
#   "训练时见没见过噪声"。所以任何成绩差异只能归因于"有没有做域随机化"，干净。
#
# 【看什么】
#   · 并排表：噪声越大，标准模型（full）成功率掉得越狠，域随机化模型（dr）稳得多。
#   · 判据 judge_dr_rescue 在 noise=0.1 打印 True = 救回成立。
#   · 鲁棒性曲线：红线（标准）随噪声跳水，绿线（域随机化）平得多——这条"平不平"就是
#     "能不能扛真机探头噪声部署"的依据。和 Class 3 的鲁棒性曲线一个道理。
#   · 诚实提醒：噪声大到 0.2（逼近/超出训练上界）时，连 dr 也会掉——域随机化不是万能，
#     它只在"训练时覆盖到的噪声范围内"有效，超出范围照样崩。这是它的能力边界，要讲清。
#
# 【到这里，泛化这条线的逻辑链合拢了】
#   干净读数训好（实验1的 full_model）→ 探头一带噪声就崩（实验2）→ 诊断是过拟合到干净观测
#   （实验3）→ 域随机化救回（实验4）。每步都有判据证据，不是凭直觉跳跃。
# 【预期（服务器实测）】各噪声成功率 标准 1.00/1.00/0.73/0.50/0.13 vs 域随机化 1.00/1.00/1.00/0.90/0.40；
#   judge_dr_rescue@0.1=True（0.90 vs 0.50）；噪声 0.2 时标准 0.13、域随机化 0.40（都下滑，域随机化也有边界）。
print("实验 4-2：域随机化模型 vs 标准模型（各探头噪声并排对比）")
dr_by_noise = [evaluate(dr_policy, obs_noise=ns) for ns in noise_list]
print(f"  {'探头噪声':<8}{'标准模型 成功率':<18}{'域随机化 成功率':<18}")
for ns, a, b in zip(noise_list, std_by_noise, dr_by_noise):
    print(f"  {ns:<8}{a['success']:<18.2f}{b['success']:<18.2f}")
rescued, msg = judge_dr_rescue(dr_by_noise[3], std_by_noise[3], metric="success")   # noise=0.1
print(f"  判据 judge_dr_rescue @noise=0.1 = {rescued}： {msg}")

# 鲁棒性曲线：标准 vs 域随机化，对探头噪声
plt.figure(figsize=(8, 5))
plt.plot(noise_list, [r["success"] for r in std_by_noise], "o-", color="red", label="standard (clean-only training)")
plt.plot(noise_list, [r["success"] for r in dr_by_noise], "s-", color="green", label="domain rand. (noisy training)")
plt.ylim(-0.05, 1.05); plt.xlabel("probe noise at deployment (obs_noise)"); plt.ylabel("success rate")
plt.title("Robustness to probe noise: DR generalizes, standard overfits to clean reads")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()
print("  → 红线（标准）随噪声跳水，绿线（域随机化）平得多（服务器实测 noise=0.1：标准 0.50 vs 域随机化 0.90）。")
print("    救回：训练见过噪声，部署到脏探头也能用。但噪声超出训练范围(≈0.2)时连 dr 也掉——域随机化有边界。")

# ============================================================================
# 第 5 部分 · 收口：RL 真比规则强吗 + Limitations + logbook
# ============================================================================

# %% 5-1　PPO vs 规则基线总对比（证明"非得用 RL"）+ 判据
# 【这段做什么】在标准土箱上，把训好的 PPO（full_model）和规则基线（rule_policy）正面对比，
#   画三项指标的柱状图，并用判据 judge_beats_baseline 给出量化结论。
#
# 【为什么必须和规则基线对比——这是答辩命门】
#   评委一定会问："你这个非得用强化学习吗？写几行 if-else 不行吗？"
#   如果你答不上来，整个 RL 的价值就立不住。规则基线（哪边湿就往干的方向猛推）就是那个
#   "几行 if-else"。要证明 RL 值得，就得让 RL 在【同一个土箱、同一把判据】下打过它。
#
# 【RL 赢在哪——"多目标非平凡权衡"】
#   规则基线只会"猛干"：湿度差大就满电压推，顾了搬水顾不上省电。
#   PPO 学到的是"在搬匀、省电、护 pH 三个目标间做权衡"——该用力时用力、接近均匀就收手省电。
#   所以典型结果是：两者都能把水搬得差不多匀，但 PPO 用的电少得多（省好几倍），pH 也始终安全。
#   这就是 RL 的价值：同样把水搬匀，它干得更省、更稳，而规则只会一味猛推。
#
# 【看什么】
#   · 柱状图三组：搬匀程度（越高越好）、耗电（越低越好）、pH 越界（越低越好）。
#   · 期待：moved 两者接近，energy 上 PPO（绿）明显低于规则（灰）；pH 两者都安全（标准土箱下都不越界）。
#   · 判据 judge_beats_baseline 打印 True = "搬匀不输 + 耗电更低 + pH 不更差"全部成立。
#     这张图 + 这个 True，就是回答"非得用 RL 吗"的标准答案：RL 赢在用更少的电、更稳地完成同样的活。
# 【预期（服务器实测）】PPO 搬匀 0.93/耗电 7.99 vs 规则 搬匀 0.90/耗电 35.8（PPO 省电约 4.5 倍，pH 两者都不越界）；judge_beats_baseline=True。
res_rule = evaluate(lambda o: rule_policy(o, CONFIG["n_cells"], CONFIG["v_max"]), cfg=CONFIG)
print("实验 5-1：PPO（full_model）vs 规则基线（标准土箱）")
print(f"  PPO      : {({k: round(v,3) for k,v in res_full.items()})}")
print(f"  规则基线 : {({k: round(v,3) for k,v in res_rule.items()})}")
beats, msg = judge_beats_baseline(res_full, res_rule)
print(f"  判据 judge_beats_baseline = {beats}： {msg}")

labels = ["moved\n(higher=better)", "energy\n(lower=better)", "pH_viol\n(lower=better)"]
rule_vals = [res_rule["moved_frac"], res_rule["energy"], res_rule["ph_viol"]]
ppo_vals = [res_full["moved_frac"], res_full["energy"], res_full["ph_viol"]]
x = np.arange(len(labels)); w = 0.35
plt.figure(figsize=(8, 4.5))
plt.bar(x - w/2, rule_vals, w, label="Rule baseline", color="gray")
plt.bar(x + w/2, ppo_vals, w, label="PPO (full reward)", color="green")
plt.xticks(x, labels); plt.title("PPO vs Rule baseline (standard soil)")
plt.legend(); plt.tight_layout(); plt.show()
print("  → 两者都能搬匀，但 PPO 耗电低得多（服务器实测：PPO 耗电 7.99 vs 规则 35.8，省约 4.5 倍；pH 都安全）。")
print("    答辩被问『非得用强化学习吗』，就甩这张对比图：RL 赢在多目标的非平凡权衡。")

# %% 5-2　保存模型 + Limitations（诚实交代局限）+ logbook
# 【这段做什么】保存两个模型，打印 Limitations（写论文照这个思路展开），留 logbook 三问。
#
# 【该部署哪个模型——一句研究判断】
#   部署到真机，应该选 dr_model（域随机化版），不是 full_model。理由：真机的电化学探头一定有噪声，
#   full_model 只在干净读数下"峰值好看"、探头一脏就垮，dr_model 在噪声下仍稳。
#   这正是 Class 3 那句"宽泛化 > 单点峰值"在你课题上的落地。
#
# 【Limitations 为什么是加分项，不是减分项】
#   ISEF/丘奖最忌讳两件事：predetermined results（结论先行）和回避问题。主动讲清局限、
#   给出后续方向，恰恰显示你想清楚了边界，是成熟度的体现。下面四条照着展开写。
#
# 【看什么】这段是文字总结 + 模型保存确认。重点读 Limitations 四条和 logbook 三问——
#   它们是把这节课内化成"你自己能在答辩里讲出来的话"的关键。
full_model.save("soilguard_ppo_standard")     # 标准土箱训的（峰值模型，演示用）
dr_model.save("soilguard_ppo_domainrand")     # 域随机化训的（部署候选，鲁棒）
print("实验 5-2：模型已保存——soilguard_ppo_standard.zip（标准）/ soilguard_ppo_domainrand.zip（域随机化）")
print("  部署候选选『域随机化版』：真机探头有噪声，抗噪的宽泛化版比只在干净读数下好看的峰值版更靠谱。")

print(r"""
================ Limitations（写论文时照这个思路展开）================
1. 简化物理：本数字孪生用规则化的电渗/pH 近似，未求解完整 Navier-Stokes / Nernst-Planck
   PDE。结论的物理保真度受此限制，真机校准前不宜外推。
2. 低湿失效仅部分建模：可动性随含水率下降而衰减（见 0-3 的 mobility），把课题硬伤"低湿区电渗
   失效"建进了物理，但真实非饱和多孔介质的电渗更复杂，极端干燥工况未充分验证。
3. 盐分积累【未建模】：长期通电会使离子在阴极富集、导致土壤盐碱化——本模型未包含，
   是已知重要局限，列为后续工作（也是答辩必问，要主动讲）。
4. sim-to-real gap：本节用域随机化让策略学会扛"探头观测噪声"（实验4 已在仿真内复现"训练见过
   噪声→部署到脏探头也能用"），但①它只在训练覆盖的噪声范围内有效，噪声超出就照样崩（见实验4-2
   的 noise=0.2）；②真机还有没纳入随机化的因素（执行器延迟、温度漂移、未建模电化学反应、盐分）。
   真正下结论前，需用真实土箱数据校准数字孪生、并做真机微调。域随机化是必要不充分条件。
诚实写清这些边界 + 给出后续方向 = 成熟度的体现，是加分项，不是减分项。
====================================================================
""")

# 【logbook 三问——做完实验、用自己的话写，这是把课内化的关键】
#   1. proxy 版把"搬水活动量"刷到很高、水却没搬匀，为什么仍算"失败"？错在智能体还是错在你？
#      （答案要落到"奖励挂在了和真实目标不等价的 proxy 上"，并说清这对你的真实课题意味着什么风险）
#   2. full_model 在干净读数下表现很好，为什么不能直接说它"能用于真机"？
#      要补做哪个实验、用什么判据，才能下"能扛真机探头噪声部署"的结论？（这是本节最核心一问）
#   3. 把这套搬到你的真机：训练时还该把哪些东西纳入域随机化，才扛得住真实部署？至少列三个。
#      （参考：探头基线漂移/噪声幅度、执行器延迟、温度对电化学的影响、盐分积累……）
print("logbook 三问（见 5-2 代码注释，用自己的话写）：")
print("  ① proxy 版活动量刷爆、水却没搬匀，为何算失败？错在智能体还是错在你？")
print("  ② full_model 干净读数下好，为何不能说它'能用于真机'？要补哪个实验+判据才能下结论？")
print("  ③ 搬到真机，训练时还该把哪些东西纳入域随机化才扛得住（探头噪声之外还有？）？")
print("\n【本讲小结】环境和奖励是你的活、算法是现成工具；奖励挂错 proxy 会 reward hacking、")
print("            训练好≠能用要靠'给探头加噪声测判据+域随机化'救、RL 要打过规则基线、局限要诚实讲。")
print("            这就是把一个 RL 课题从'仿真 demo'做成'经得起答辩的研究'的全套动作。")
