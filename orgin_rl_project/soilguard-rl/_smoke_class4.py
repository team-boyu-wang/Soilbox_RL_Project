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
    奖励函数就是你给智能体的"考卷"。你写错了考卷，它就考出你没料到的答案。
    我们先按最朴素的想法设奖励——"只要把水搬均匀就给分"，训出来一看：
    它为了多搬水、搬快水，疯狂猛通电，把土壤 pH 顶到严重越界（电解把土烧坏了）。
    它没有错——它完美地最大化了你给的分；错的是你的奖励漏了"别把土搞坏"这一项。
    这就是 reward hacking（奖励钻空子）。把 pH 越界的惩罚加回去，它立刻学会收着点用电。

  【暗线二 · 训练好≠能用 → sim-to-real（实验 2~4）】
    把奖励设对、在"标准土箱"上训出的策略，换一个没训过的土箱（土质不同导致电渗
    效率变了、装土时初始干湿不同）就抓瞎——搬不动水了。它没学会"调水"这件事的
    通用规律，只是把动作过拟合到了【训练时那一个特定土箱的参数】上。
    这正是 sim-to-real gap：你在仿真里把土箱策略训到很好，搬到真实土箱（土质、
    探头、装填每次都不同）就不一定能用。诊断清楚后，用【域随机化】救：训练时每局
    随机换一种土箱，逼它学一个"对各种土箱都管用"的策略，再测——换土箱也稳了。

  这两条暗线，第 1 讲就都埋了：①"真实土箱试不起几万次"→ 数字孪生 + sim-to-real；
  ②"你给什么它追什么"→ 奖励设计 + reward hacking + 必须对比规则基线。本节全部收口。

== 这是"实验设计"，不是"跑个 demo"==
  实验 1（奖励）：先按朴素想法只奖搬水 → 亲手训出 reward hacking（pH 越界）→
    诊断"是奖励漏了惩罚项，不是算法坏了" → 把三项奖励配齐救回 → 判据函数量化前后差距。
  实验 2~4（泛化）：标准土箱训好 → 亲手把它在"漂移土箱"上跑垮 → 诊断是过拟合到单一
    土箱 → 域随机化救回 → 鲁棒性曲线对照。
  这才是论文/答辩里的"消融实验 / 泛化性分析 / sim-to-real"。

  "控制变量"贯穿全文：实验 2 每次只改一个土箱参数（先只改电渗系数 k_eo、再只改初始
  干湿 m_dry），其它锁死，才能精确说清"是哪个差异害它垮的"，而不是笼统的"换土箱就崩"。

== 判据写成代码（科研习惯，不靠肉眼看一局）==
  全文用一把统一的尺子 evaluate()：固定 30 局、固定评估 seed，量出五个指标
  （平均回报 / 搬匀程度 moved_frac / 耗电 energy / pH 越界 ph_viol / 成功率）。
  再写成若干 judge_*() 判据函数，把"reward hacking 复现了吗 / 泛化崩了吗 /
  域随机化救回了吗 / RL 真比规则强吗"这些结论，量化成可打印的 True/False + 数字，
  而不是散落各处、靠肉眼读 print 拍脑袋。判据一致，结论才可比、才可写进论文。

== 真实情况（服务器实测，下面 print/注释里标"服务器实测"的数都是真值，不是编的）==
  【★ 待 V100 真跑后回填——当前为占位，请勿引用其中具体数字 ★】
  环境：SoilBoxEnv（1D 简化电渗土箱），device=cpu、seed=42。
  · 实验 1 朴素奖励（只奖搬水）→ pH 越界 = TODO；三项配齐 → pH 越界 = TODO（救回）。
  · 实验 2 标准模型换漂移土箱：k_eo 扫描 = TODO；m_dry 扫描 = TODO；对照维度 = TODO。
  · 实验 4 域随机化模型换漂移土箱 = TODO（救回，对比标准模型）。
  · 收口 PPO vs 规则基线：energy/ph_viol = TODO（PPO 更低）。

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
  0-3  ★SoilBoxEnv：把土箱写成 Gym 环境（reset/step/reward）+ 域随机化开关
  0-4  自检：随机动作跑一局、画湿度演化（确认环境没写错）
  0-5  ★判据写成代码：make_soilbox 工厂 + evaluate + 规则基线 + train_ppo + judge_*
【实验 1 · 奖励怎么设：你给什么它追什么（reward hacking）】
  1-1  朴素奖励：只奖搬水（去掉耗电/pH 惩罚）→ 训一版
  1-2  看它学成什么 + 判据：pH 疯狂越界（亲手跑出 reward hacking）
  1-3  诊断 + 救：三项奖励配齐 → 重训 → pH 回安全带 → 判据量化前后差距
【实验 2 · 训练好≠能用：换个土箱就抓瞎（sim-to-real 仿真内缩影）】
  2-1  只改电渗系数 k_eo（土质差异）→ 标准模型暴跌
  2-2  只改初始干湿 m_dry（装土差异）→ 标准模型暴跌
  2-3  对照：改一个"扛得住"的参数 → 分清哪些漂移致命（研究判断）
【实验 3 · 诊断：是过拟合到单一土箱，不是训练不够】
【实验 4 · 域随机化救：训练时每局随机换土箱】
  4-1  域随机化训练（每局随机 k_eo/m_dry/m_threshold）
  4-2  域随机化模型 vs 标准模型在各土箱下对比 → 救回 + 鲁棒性曲线
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
    # --- 奖励权重（三个目标的天平，实验 1 的主角）---
    "w_progress": 1.0,     # 把水搬均匀的奖励权重
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
#   外加一个域随机化开关（randomize）：训练时每局随机换一种土箱——实验 4 的救命工具。
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
# 【域随机化开关 randomize 是什么（实验 4 才会打开，先理解原理）】
#   randomize=False（默认）时，土箱参数固定 = CONFIG，这是"标准土箱"。
#   randomize=True 时，每局 reset 都从 rand_ranges 里随机抽一组 k_eo/m_dry/m_threshold——
#   智能体每局面对不一样的土箱，逼它学"对各种土箱都管用"的策略，而不是死记一个土箱。
#   这跟 Class 3 给 CartPole 随机杆长是同一招，只是这次随机的是土箱物理参数。
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

    def __init__(self, config=CONFIG, randomize=False, rand_ranges=None):
        super().__init__()
        self.base_cfg = dict(config)          # 记住"基准配置"，randomize 时在它上面随机
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
        # 域随机化设置：训练时每局随机抽这些参数（实验 4 打开）
        self.randomize = randomize
        self.rand_ranges = rand_ranges or {
            "k_eo": (0.06, 0.24),          # 土质：电渗效率高低
            "m_dry": (0.04, 0.16),         # 装土：初始干区干到什么程度
            "m_threshold": (0.08, 0.18),   # 土质：低湿失效拐点高低
        }
        self._rng = np.random.RandomState(SEED)
        self._init_spread = None

    # ---- 把内部物理量打包成给智能体看的观测向量 ----
    def _get_obs(self):
        ph_norm = (self.pH - 7.0) / 7.0
        return np.concatenate([self.m, ph_norm]).astype(np.float32)

    # ---- 不均匀度：含水率离平均值的总偏离，越小越均匀 ----
    def _spread(self):
        return float(np.sum(np.abs(self.m - self.m.mean())))

    # ---- reset：每局开始，把土箱拨回「左湿右干」初始状态 ----
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 域随机化：训练时每局随机抽一组土箱参数（覆盖到 self.cfg）
        if self.randomize:
            self.cfg = dict(self.base_cfg)
            for key, (lo, hi) in self.rand_ranges.items():
                self.cfg[key] = float(self._rng.uniform(lo, hi))
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
            dm[j] -= flux
            dm[j + 1] += flux
        self.m = np.clip(self.m + dm, 0.0, 0.5)

        # (2) pH：通电使靠正极侧酸化、靠负极侧碱化；极性反转可把 pH 拉回
        dp = np.zeros(self.n)
        for j in range(self.n - 1):
            dp[j] -= cfg["k_ph"] * a[j]
            dp[j + 1] += cfg["k_ph"] * a[j]
        self.pH = np.clip(self.pH + dp, 0.0, 14.0)

        # (3) 奖励：进展 − 耗电 − pH 越界
        reward, terms = self._reward(a)

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

    # ---- 奖励函数：把任务三要求翻译成三笔账（实验 1 会拆开它做文章）----
    def _reward(self, a):
        cfg = self.cfg
        cur_spread = self._spread()
        # 进展：这一步让不均匀度减小了多少（越搬越均匀 → 正分）
        progress = self.prev_spread - cur_spread
        # 耗电：电压平方 ∝ 功率
        energy = float(np.mean(a ** 2))
        # pH 越界：超出容忍带的部分才罚
        violation = float(np.mean(np.maximum(0.0, np.abs(self.pH - 7.0) - cfg["ph_tol"])))
        r = (cfg["w_progress"] * progress
             - cfg["w_energy"] * energy
             - cfg["w_ph"] * violation)
        return r, {"progress": progress, "energy": energy, "ph_violation": violation}


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

def evaluate(policy_fn, cfg=CONFIG, n_episodes=30, seed0=1000):
    """判据：把策略丢进 cfg 指定的土箱跑 n 局，返回五个平均指标。policy_fn(obs)->action。"""
    e = SoilBoxEnv(cfg)
    tot_r, tot_energy, tot_phviol, succ, moved = [], [], [], [], []
    for ep in range(n_episodes):
        obs, _ = e.reset(seed=seed0 + ep)      # 固定评估 seed：所有模型在相同初始局面上比
        init_spread = e._init_spread
        ep_r = ep_e = ep_p = 0.0
        info = {"success": False, "spread": init_spread}
        for _ in range(cfg["max_steps"]):
            a = policy_fn(obs)
            obs, r, term, trunc, info = e.step(a)
            ep_r += r
            ep_e += info["terms"]["energy"]
            ep_p += info["terms"]["ph_violation"]
            if term or trunc:
                break
        tot_r.append(ep_r); tot_energy.append(ep_e); tot_phviol.append(ep_p)
        succ.append(1.0 if info["success"] else 0.0)
        moved.append((init_spread - info["spread"]) / init_spread)  # 搬匀了百分之多少
    return {
        "reward": float(np.mean(tot_r)), "energy": float(np.mean(tot_energy)),
        "ph_viol": float(np.mean(tot_phviol)), "success": float(np.mean(succ)),
        "moved_frac": float(np.mean(moved)),
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
def judge_reward_hacking(res_naive, res_full, ratio=2.0):
    """实验 1 判据：朴素奖励版的 pH 越界，是否显著高于三项配齐版（≥ratio 倍）= reward hacking 复现。"""
    ok = res_naive["ph_viol"] > res_full["ph_viol"] * ratio
    return ok, (f"朴素奖励 pH越界={res_naive['ph_viol']:.3f}  vs  三项配齐={res_full['ph_viol']:.3f}"
                f"（{'复现' if ok else '未复现'}：{'前者≥'+str(ratio)+'倍后者' if ok else '差距不够'}）")

def judge_generalization_fail(res_in, res_out, metric="moved_frac", drop=0.30):
    """实验 2 判据：标准模型从'标准土箱'换到'漂移土箱'，主指标是否掉超过 drop（默认 30%）= 泛化失败。"""
    base = abs(res_in[metric]) + 1e-9
    rel = (res_in[metric] - res_out[metric]) / base
    ok = rel > drop
    return ok, (f"{metric}: 标准土箱={res_in[metric]:.3f} → 漂移土箱={res_out[metric]:.3f}"
                f"（掉 {rel*100:.0f}%，{'泛化失败' if ok else '扛得住'}）")

def judge_dr_rescue(res_dr_out, res_std_out, metric="moved_frac", margin=1.2):
    """实验 4 判据：域随机化模型在漂移土箱上的主指标，是否明显优于标准模型（≥margin 倍）= 救回。"""
    ok = res_dr_out[metric] > res_std_out[metric] * margin
    return ok, (f"{metric} 在漂移土箱：标准模型={res_std_out[metric]:.3f}  vs  "
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

# %% 1-1　朴素奖励：只奖搬水（去掉耗电/pH 惩罚）→ 训一版
# 【这段做什么】先按一个初学者最自然的想法来设奖励："我只要把水搬均匀就好，给搬匀加分"——
#   也就是把耗电惩罚和 pH 惩罚的权重都设成 0（cfg_naive），只留搬匀这一项，训一个 PPO。
#   这一版我们故意叫它 naive_model（朴素版）。
#
# 【为什么要先做"错误"的版本（这是实验设计的关键）】
#   如果一上来就把三项奖励配齐、训出一个好模型，你只会看到"它学会了"，
#   永远不知道"奖励设歪会怎样"。而"奖励设歪会怎样"恰恰是这门课最值钱的一课，
#   也是答辩高频考点。所以我们先亲手把坑踩出来：只奖搬水，看它学成什么样。
#   这跟 Class 2 先让纯贪心撞墙、Class 3 先让标准模型在长杆上垮，是同一个套路：
#   先制造失败，再诊断、再救——失败本身就是教学内容。
#
# 【朴素奖励到底漏了什么】
#   任务其实有三个要求：① 把水搬匀（要做到的事）；② 别太费电（成本约束）；
#   ③ 别把 pH 搞越界毁了土（安全约束）。朴素版只把 ① 写进了奖励，
#   把 ②③ 这两个"约束"忘了。智能体看不到约束，就只会一根筋地最大化 ①——
#   而搬水要靠通电，通电就会让 pH 漂移，于是它会为了多搬快搬而疯狂猛通电。
#
# 【看什么】这段只训练，输出是训练进度。真正的"后果"在 1-2 揭晓。
#   先记住：这一版的奖励里【没有任何东西拦着它把 pH 搞坏】。
cfg_naive = dict(CONFIG); cfg_naive["w_energy"] = 0.0; cfg_naive["w_ph"] = 0.0  # 只奖搬水
TRAIN_STEPS = 1500
print("实验 1-1：用『只奖搬水』的朴素奖励训练（w_energy=0, w_ph=0）")
naive_model, naive_log = train_ppo(SoilBoxEnv(cfg_naive), TRAIN_STEPS, "朴素奖励·只奖搬水")
print("  训练完成。它在『只有搬水加分、没有任何惩罚』的考卷下，会学成什么样？看 1-2。")

# %% 1-2　看它学成什么 + 判据：pH 疯狂越界（亲手跑出 reward hacking）
# 【这段做什么】跑一局朴素版策略，把含水率和 pH 的演化画出来，并用判据量化它的 pH 越界。
#   你会看到：它确实把水搬得挺匀（它擅长这个，因为只有这个给分），
#   但代价是 pH 被它通电通到严重越界——土壤在仿真里被"电解烧坏"了。
#
# 【这就是 reward hacking（奖励钻空子）——讲透它为什么"没错又很错"】
#   智能体没有任何 bug，它完美地最大化了你给它的分数。它发现"猛通电能更快搬匀、
#   多拿搬匀分"，而"pH 越界"在朴素奖励里【不扣分】，那它当然就肆无忌惮地猛通电。
#   错的不是它，是你的奖励函数漏了约束。这正是第 1 讲埋的那句"你给什么它追什么"：
#   你给的考卷只考"搬匀"，它就只优化"搬匀"，哪怕把土烧了也在所不惜。
#   现实里 reward hacking 的著名例子：让 AI 玩赛艇游戏只奖励"得分"，它发现原地绕圈
#   反复吃分球比好好比赛得分更高，于是干脆不跑完赛道、在原地刷分——和这里一模一样。
#
# 【为什么这对你的课题是致命的】
#   你的土箱真实目标是"修复土壤"，绝不是"不惜代价把水搬匀"。如果奖励设成朴素版，
#   你训出来的策略搬到真机上，会把土壤 pH 搞到植物没法活——那叫"修复成功"还是"破坏成功"？
#   答辩时评委一定会问"你怎么保证智能体不会为了完成任务而制造新危害"，
#   这段实验就是你的答案：把危害写进奖励的惩罚项，让它在优化时就不敢碰。
#
# 【看什么】
#   · 左图含水率：朴素版能把 6 条线收拢到一起（搬匀做得不错）。
#   · 中图 pH：多条线会冲出绿色安全带（pH 6~8）——这就是它"通电通过头"的铁证。
#   · 右图平均电压：朴素版会用很大的电压（猛干），不像后面配齐版那样收着用。
#   · 判据 judge_reward_hacking 会打印 True/False：朴素版 pH 越界是否≥配齐版的 2 倍。
#     （配齐版的数 1-3 才训出来，这里先把朴素版的数算好、图画好。）
naive_policy = ppo_policy_of(naive_model)
e = SoilBoxEnv(cfg_naive)
obs, _ = e.reset(seed=SEED)
m_hist, ph_hist, v_hist = [e.m.copy()], [e.pH.copy()], []
for _ in range(CONFIG["max_steps"]):
    a = naive_policy(obs)
    obs, r, term, trunc, info = e.step(a)
    m_hist.append(e.m.copy()); ph_hist.append(e.pH.copy()); v_hist.append(np.mean(np.abs(a)))
    if term or trunc:
        break
m_hist, ph_hist = np.array(m_hist), np.array(ph_hist)

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for i in range(e.n): ax[0].plot(m_hist[:, i], label=f"cell {i}")
ax[0].set_title("Naive reward: humidity (it DOES level water)")
ax[0].set_xlabel("step"); ax[0].set_ylabel("m"); ax[0].legend(fontsize=7)
for i in range(e.n): ax[1].plot(ph_hist[:, i])
ax[1].axhspan(6, 8, color="green", alpha=0.1)
ax[1].set_title("Naive reward: pH BREAKS the safe band")
ax[1].set_xlabel("step"); ax[1].set_ylabel("pH")
ax[2].plot(v_hist, color="orange")
ax[2].set_title("Naive reward: avg |voltage| (it floors it)")
ax[2].set_xlabel("step"); ax[2].set_ylabel("|v|")
plt.tight_layout(); plt.show()

res_naive = evaluate(naive_policy, cfg=cfg_naive)
print("实验 1-2：朴素奖励版的判据成绩")
print(f"  {({k: round(v,3) for k,v in res_naive.items()})}")
print(f"  → 搬匀程度 moved_frac={res_naive['moved_frac']:.3f}（它擅长搬水），"
      f"但 pH 越界 ph_viol={res_naive['ph_viol']:.3f}（把土烧了）。")
print("  （服务器实测：待回填）这就是 reward hacking：它没错，是你的奖励漏了 pH 惩罚。")

# %% 1-3　诊断 + 救：三项奖励配齐 → 重训 → pH 回安全带 → 判据量化
# 【这段做什么】对症下药——把奖励三项配齐（用标准 CONFIG：w_progress + w_energy + w_ph 都在），
#   重新训一个 full_model（"奖励配齐的标准模型"），再跑一局画图、用判据和朴素版对比。
#   注意：这个 full_model 不只是实验 1 的"正确答案"，它还是实验 2~4 的主角——
#   后面我们就是要看"这个在标准土箱上奖励设对、训好的模型，换个土箱还能不能用"。
#
# 【诊断：病根是奖励，不是算法】
#   有人看到朴素版把 pH 搞坏，第一反应是"PPO 不行 / 再多训训 / 换个算法"。错。
#   朴素版的 PPO 工作得非常好——它精确地最大化了你给的分。问题 100% 出在奖励函数上：
#   你没把"别把 pH 搞坏"写进分数。所以救法不是动算法，是动奖励：把 pH 越界的惩罚加回去
#   （w_ph>0），再把耗电惩罚也加上（w_energy>0）。这就是"诊断要诊断到病根"——
#   病根在奖励设计，就改奖励，别瞎换算法。
#
# 【三项奖励配齐后，智能体的权衡变了】
#   现在它的考卷变成"搬匀加分、但费电扣分、pH 越界重扣分"。它必须在三个打架的目标间
#   找平衡：搬水还得搬，但不能为了搬水把 pH 搞越界（罚得比搬水的收益还多）。
#   于是它学会"该用力时用力、pH 快越界就收手/反转极性"——这套权衡没人教，是它在
#   新考卷下自己试出来的。这就是奖励设计的威力：改几个权重，就重塑了它的整个行为。
#
# 【看什么】
#   · 中图 pH：配齐版的 pH 线基本待在绿色安全带里（对比 1-2 朴素版冲出去）。
#   · 右图电压：配齐版会"先用力、接近均匀就收手省电"，不像朴素版一路猛干。
#   · 判据 judge_reward_hacking：打印 True = 朴素版 pH 越界确实≥配齐版 2 倍，reward hacking 复现成立。
#   · 这一对比图（朴素 vs 配齐）就是答辩时回答"你怎么防止智能体制造新危害"的最佳证据。
print("实验 1-3：把三项奖励配齐（w_progress + w_energy + w_ph）重新训练")
full_model, full_log = train_ppo(SoilBoxEnv(CONFIG), TRAIN_STEPS, "三项配齐·标准土箱")
full_policy = ppo_policy_of(full_model)

# 跑一局配齐版，画含水率/pH/电压
e = SoilBoxEnv(CONFIG)
obs, _ = e.reset(seed=SEED)
m_hist, ph_hist, v_hist = [e.m.copy()], [e.pH.copy()], []
for _ in range(CONFIG["max_steps"]):
    a = full_policy(obs)
    obs, r, term, trunc, info = e.step(a)
    m_hist.append(e.m.copy()); ph_hist.append(e.pH.copy()); v_hist.append(np.mean(np.abs(a)))
    if term or trunc:
        break
m_hist, ph_hist = np.array(m_hist), np.array(ph_hist)

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for i in range(e.n): ax[0].plot(m_hist[:, i], label=f"cell {i}")
ax[0].set_title("Full reward: humidity (still levels water)")
ax[0].set_xlabel("step"); ax[0].set_ylabel("m"); ax[0].legend(fontsize=7)
for i in range(e.n): ax[1].plot(ph_hist[:, i])
ax[1].axhspan(6, 8, color="green", alpha=0.1)
ax[1].set_title("Full reward: pH STAYS in safe band")
ax[1].set_xlabel("step"); ax[1].set_ylabel("pH")
ax[2].plot(v_hist, color="orange")
ax[2].set_title("Full reward: avg |voltage| (eases off)")
ax[2].set_xlabel("step"); ax[2].set_ylabel("|v|")
plt.tight_layout(); plt.show()

res_full = evaluate(full_policy, cfg=CONFIG)
print("实验 1-3：三项配齐版的判据成绩")
print(f"  {({k: round(v,3) for k,v in res_full.items()})}")
hacked, msg = judge_reward_hacking(res_naive, res_full)
print(f"  判据 judge_reward_hacking = {hacked}")
print(f"    {msg}")
print("  → 同样能把水搬匀，但配齐版的 pH 老老实实待在安全带里。")
print("    结论：奖励写错 → 学出 reward hacking；把约束写进奖励的惩罚项 → 行为立刻被纠正。")
print("    （这就是『你给什么它追什么』。下面带着这个『奖励设对、标准土箱训好』的 full_model，")
print("     去问下一个问题：它换个土箱还能用吗？——这就是实验 2。）")

# ============================================================================
# 实验 2 · 训练好≠能用：换个土箱就抓瞎（sim-to-real 仿真内缩影）
# ============================================================================

# %% 2-1　只改电渗系数 k_eo（土质差异）→ 标准模型暴跌
# 【这段做什么】把实验 1 训好的 full_model（在标准土箱 k_eo=0.15 上训的），丢进一系列
#   "电渗效率不同"的土箱里测——k_eo 从 0.06 扫到 0.24，逐一看它还能不能把水搬匀。
#   其它所有参数（初始干湿、低湿拐点、pH 物理）全锁死，绝对不动。
#
# 【k_eo 漂移对应真实世界的什么——这是 sim-to-real 的核心】
#   k_eo（电渗系数）= 同样的电压能搬动多少水，本质上是土质决定的：
#   黏土、砂土、有机质含量不同，孔隙结构不同，电渗效率天差地别。
#   你在仿真里假设 k_eo=0.15 训出策略，搬到真实土箱——真土的 k_eo 几乎不可能正好是 0.15。
#   如果策略只在 k_eo=0.15 下管用，那它在真机上就是废的。这就是最典型的 sim-to-real gap：
#   仿真用的物理系数和真实物理系数对不上。
#
# 【控制变量：为什么只扫 k_eo、其他全锁死】
#   只改一个参数，成绩的变化才只有一种解释（"是 k_eo 害的"）。如果同时改 k_eo 和 m_dry，
#   看到掉分你没法归因。这是单因素消融实验，论文里的标准做法。
#
# 【为什么 full_model 可能扛不住 k_eo 变化——直觉】
#   full_model 训练时一直面对 k_eo=0.15。它学到的是"在这个电渗效率下，用多大电压、
#   用多久能把水搬到位"。换成 k_eo=0.06（电渗效率只剩 1/3），同样的电压搬动的水少得多，
#   它按"0.15 的力道"出招会严重搬不动；换成 k_eo=0.24（效率高），又可能搬过头。
#   它没学会"看效果调力道"，只学会了"在 0.15 这个效率下的固定打法"。
#
# 【看什么】
#   · 输出每行是一个 k_eo 下的搬匀程度 moved_frac。k_eo=0.15（训练值）那行应该最高。
#   · 判据 judge_generalization_fail 会逐行标"泛化失败/扛得住"：偏离 0.15 越远，越可能失败。
#   · 留意是不是"离训练点越远掉得越狠"——这正是过拟合到单点的典型形状。
# 【预期（服务器实测）】待 V100 真跑回填。
print("实验 2-1：把标准模型丢进『电渗效率 k_eo 不同』的土箱（只改 k_eo）")
res_std_standard = evaluate(full_policy, cfg=CONFIG)   # 标准土箱上的基准成绩（对比锚点）
keo_list = [0.06, 0.10, 0.15, 0.20, 0.24]
std_by_keo = []
for ke in keo_list:
    res = evaluate(full_policy, cfg=dict(CONFIG, k_eo=ke))
    std_by_keo.append(res)
    fail, _ = judge_generalization_fail(res_std_standard, res)
    flag = "  ← 泛化失败" if (fail and abs(ke - CONFIG["k_eo"]) > 1e-9) else ""
    print(f"  k_eo={ke:<5} moved_frac={res['moved_frac']:6.3f}  energy={res['energy']:6.3f}"
          f"  ph_viol={res['ph_viol']:6.3f}{flag}")
print("  → 离训练值 k_eo=0.15 越远，搬匀程度越掉（服务器实测：待回填）。它只会『0.15 这个土的打法』。")

# %% 2-2　只改初始干湿 m_dry（装土差异）→ 标准模型暴跌
# 【这段做什么】k_eo 锁回默认 0.15，改成扫"初始干区有多干"——m_dry 从 0.04 扫到 0.16。
#   和上一组一样只动一个变量，这次只动 m_dry，其余全锁死。
#
# 【m_dry 漂移对应真实世界的什么】
#   m_dry = 装土时干区的初始含水率。你每次往土箱里装土、做实验，初始干湿不可能每次都
#   一模一样：今天这批土干一点、明天那批湿一点，浇没浇过水、放了多久都会变。
#   仿真里你固定 m_dry=0.10 训练，真实实验的初始干湿却是飘的。如果策略只在 m_dry=0.10
#   下管用，那每次装土稍有不同它就失灵——这又是一种 sim-to-real gap（初始条件对不上）。
#
# 【为什么 m_dry 变化可能害它垮，还牵连到低湿失效】
#   m_dry 越小（越干），干区越接近 m_threshold（低湿失效拐点）——电渗可动性趋近 0，
#   水根本搬不进去。full_model 训练时见的是 m_dry=0.10（还能搬动），没见过 m_dry=0.04
#   这种"干到电渗几乎失效"的局面，它不知道该怎么办（其实物理上也很难办）。
#   这一组同时考了两件事：策略对初始条件的鲁棒性 + 课题硬伤"低湿区电渗失效"的体现。
#
# 【看什么】
#   · 每行一个 m_dry 下的 moved_frac。m_dry=0.10（训练值）附近应最好，越干越掉。
#   · 结合 2-1：标准模型对"土质（k_eo）"和"装土干湿（m_dry）"两种最常见的真实差异都脆——
#     说明它对训练土箱的依赖很重，换个真土箱很可能不能用。
# 【预期（服务器实测）】待 V100 真跑回填。
print("实验 2-2：把标准模型丢进『初始干湿 m_dry 不同』的土箱（只改 m_dry）")
mdry_list = [0.04, 0.07, 0.10, 0.13, 0.16]
std_by_mdry = []
for md in mdry_list:
    res = evaluate(full_policy, cfg=dict(CONFIG, m_dry=md))
    std_by_mdry.append(res)
    fail, _ = judge_generalization_fail(res_std_standard, res)
    flag = "  ← 泛化失败" if (fail and abs(md - CONFIG["m_dry"]) > 1e-9) else ""
    print(f"  m_dry={md:<5} moved_frac={res['moved_frac']:6.3f}  energy={res['energy']:6.3f}"
          f"  ph_viol={res['ph_viol']:6.3f}{flag}")
print("  → 初始越干，越搬不动（服务器实测：待回填）。这同时暴露了课题硬伤『低湿区电渗失效』。")

# 画实验 2 的两条暴跌曲线（搬匀程度 vs 漂移参数）
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(keo_list, [r["moved_frac"] for r in std_by_keo], "o-", color="red")
axes[0].axvline(CONFIG["k_eo"], ls=":", color="gray"); axes[0].set_ylim(0, 1)
axes[0].set_xlabel("k_eo (soil electro-osmotic coeff)"); axes[0].set_ylabel("moved_frac")
axes[0].set_title("Standard model vs soil k_eo"); axes[0].grid(alpha=0.3)
axes[1].plot(mdry_list, [r["moved_frac"] for r in std_by_mdry], "o-", color="purple")
axes[1].axvline(CONFIG["m_dry"], ls=":", color="gray"); axes[1].set_ylim(0, 1)
axes[1].set_xlabel("m_dry (initial dryness)"); axes[1].set_ylabel("moved_frac")
axes[1].set_title("Standard model vs initial dryness"); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()

# %% 2-3　对照：改一个"扛得住"的参数 → 分清哪些漂移致命（研究判断）
# 【这段做什么】刚才看到两种"会害死模型"的漂移（k_eo、m_dry）。但如果就此说"换土箱就崩"，
#   太宽泛、没价值。这一组测一个 full_model 大概率扛得住的漂移——k_ph（pH 漂移系数）。
#   用它做对照：证明不是"一漂移就崩"，而是对某些漂移敏感、对某些鲁棒——
#   分清哪些致命、哪些无所谓，才是有研究价值的结论。
#
# 【为什么 full_model 对 k_ph 漂移大概率扛得住——直觉】
#   k_ph 只影响 pH 怎么漂，不改变"电压→搬多少水"这条搬水动力学。full_model 的搬水能力
#   （moved_frac）依赖的是 k_eo 和初始湿度，跟 k_ph 没关系。所以改 k_ph，它的搬匀程度
#   应该基本不动（扛得住）——虽然 pH 越界量 ph_viol 会随 k_ph 变大而变（那是物理变了，
#   不是策略失灵）。所以这一组我们盯 moved_frac 看"搬水能力扛不扛得住"。
#
# 【为什么这组对照极重要：它是研究能力的体现】
#   只做 2-1/2-2，你只能说"模型对某些漂移垮了"。加上 2-3 对照，你能说的是一句精确结论：
#   "策略对【改变搬水动力学的漂移】（土质 k_eo、初始干湿 m_dry）敏感，
#    对【不改搬水动力学的漂移】（pH 系数 k_ph）的搬水能力则鲁棒。"
#   这告诉你部署时该重点担心什么（土质、装土干湿），不必过度担心什么——这就是研究判断。
# 【预期（服务器实测）】待 V100 真跑回填。
print("实验 2-3：对照组——改 pH 漂移系数 k_ph（看搬水能力扛不扛得住）")
for kp in [0.04, 0.08, 0.16]:
    res = evaluate(full_policy, cfg=dict(CONFIG, k_ph=kp))
    fail, _ = judge_generalization_fail(res_std_standard, res, metric="moved_frac")
    flag = "  ← 搬水也崩" if (fail and abs(kp - CONFIG["k_ph"]) > 1e-9) else "  （搬水扛得住）"
    print(f"  k_ph={kp:<5} moved_frac={res['moved_frac']:6.3f}  ph_viol={res['ph_viol']:6.3f}{flag}")
print("  →（服务器实测：待回填）搬水能力对 k_ph 鲁棒，只栽在改搬水动力学的 k_eo/m_dry 上。")
print("    研究判断：部署时重点防『土质 + 装土干湿』差异，这就是下一步域随机化要覆盖的方向。")

# ============================================================================
# 实验 3 · 诊断：是过拟合到单一土箱，不是训练不够
# ============================================================================
# %% 3　诊断结论
# 【这段做什么】把实验 2 三组证据归纳成一句精确诊断，讲透两个问题：
#   ① 为什么是"过拟合到单一土箱"？ ② 为什么不是"训练不够"？
#   这段只有 print，没有新实验——但它是逻辑链的转折点：诊断到病根，才能对症下药（实验 4）。
#
# 【实验 2 三组证据的完整逻辑链】
#   2-1：k_eo 偏离 0.15 → 搬匀程度暴跌      ← 改了搬水动力学，扛不住
#   2-2：m_dry 偏离 0.10 → 搬匀程度暴跌     ← 改了初始条件/触发低湿失效，扛不住
#   2-3：k_ph 大幅变化 → 搬水能力仍稳       ← 没改搬水动力学，扛住了
#   结论：full_model 不是对任何漂移都脆，而是专门对"训练时没见过的搬水条件变化"脆——
#   这就是"过拟合到单一土箱配置"的特征。
#
# 【为什么不是"训练不够"——把这条路堵死】
#   你可能想：是不是 6 万步不够，多训几万步就好了？不是。理由很直接：
#   full_model 在标准土箱（k_eo=0.15、m_dry=0.10）上已经训得很好了，继续在【同一个土箱】
#   上训，它也永远遇不到 k_eo=0.06 或 m_dry=0.04 的局面，当然学不到那些的应对。
#   缺的不是训练量，是【见过的土箱种类】。这和 Class 2「加训练局数救不了纯贪心」、
#   Class 3「多训也救不了过拟合」是同一种病：缺的是某种"经历"本身，不是"训练量"。
#
# 【这就是 sim-to-real gap——接土箱项目】
#   你在仿真（单一土箱）里把策略训到很好，搬到真实土箱（土质 k_eo 不同、装土 m_dry 不同、
#   还有仿真没建模的盐分/温度/探头噪声……）就垮——中间这道沟就是 sim-to-real gap。
#   这节你亲手在仿真内部就复现了这道沟：哪怕都在仿真里，只要测试土箱≠训练土箱，策略就崩。
#   真实世界的 gap 只会比这更大。
#
# 【下一步：对症下药】
#   病根是"只见过一种土箱"，那就让它训练时"见过很多种土箱"——域随机化，实验 4 去做。
print("=== 实验 3 诊断结论 ===")
print(f"· 现象：标准土箱搬匀 moved_frac={res_std_standard['moved_frac']:.3f}，"
      f"k_eo/m_dry 一偏离就暴跌（见 2-1/2-2），但改 k_ph 搬水仍稳（2-3）。")
print("· 不是训练不够：标准土箱已训好，在同一个土箱上再多训也遇不到别的土质/干湿。")
print("· 病根：过拟合到单一土箱配置——只在 k_eo=0.15、m_dry=0.10 上训，换搬水条件就失效。")
print("· 这就是 sim-to-real gap：仿真单一土箱训得好 ≠ 真实土箱能用。要救，得让它训练时见多种土箱。")

# ============================================================================
# 实验 4 · 域随机化救：训练时每局随机换土箱
# ============================================================================

# %% 4-1　域随机化训练（每局随机 k_eo/m_dry/m_threshold）
# 【这段做什么】对症下药——重新训一个 PPO（dr_model），但这次训练时每一局开局都随机换一种
#   土箱：k_eo 在 [0.06,0.24]、m_dry 在 [0.04,0.16]、m_threshold 在 [0.08,0.18] 里各随机抽一个
#   （就是 0-3 里 SoilBoxEnv(randomize=True) 那个开关）。奖励仍用配齐的三项（和 full_model 一样，
#   保证唯一的差别只是"训练时见的土箱多不多"）。
#
# 【域随机化的核心思想——为什么这样能救泛化】
#   病根是"只见过一种土箱（k_eo=0.15、m_dry=0.10）"。那就让它训练时见过整个范围的土箱。
#   当每局土质和干湿都不一样，"死记 0.15 这个土的固定打法"就没用了——它必须学一个
#   "看当前湿度/pH 反馈、对各种土都管用"的调水策略，也就是更通用的规律。
#   训练时见过 k_eo 0.06~0.24 的各种土，测试时遇到哪个都不陌生。
#   这跟 Class 3 给 CartPole 随机杆长是同一招，只是这次随机的是真实部署最会变的土箱参数。
#
# 【为什么域随机化是 sim-to-real 最主流的救法（接你的课题）】
#   真实部署时你没法精确知道土箱的 k_eo、m_dry 是多少（每批土不同、装填不同、还会随时间变）。
#   既然无法精确知道，就用"覆盖一个范围"代替"精确匹配"：把可能出现的土质/干湿范围都在训练里
#   随机到，部署时不管来的是哪种土，模型都见过接近的，就能扛住。这是工程上最实用的泛化手段。
#
# 【为什么训得比标准版久（步数加倍）】
#   域随机化任务更难：每局土箱都不一样，等于要"同时学会应付各种土"，是个更复杂的函数。
#   更难的任务需要更多经历才收敛，所以步数从 6 万加到 12 万。
#
# 【看什么】这段训练时间比前两次长（约翻倍）。训完看 dr_on_standard：它在标准土箱上的成绩。
#   关键期待：见多识广后，它在标准土箱上的搬匀程度不应明显低于 full_model——
#   即"学会扛各种土"不以"牺牲标准土箱表现"为代价。
# 【预期（服务器实测）】待 V100 真跑回填。
DR_TRAIN_STEPS = 3000
print("实验 4-1：域随机化训练（每局随机换土箱 k_eo/m_dry/m_threshold）")
dr_model, dr_log = train_ppo(SoilBoxEnv(CONFIG, randomize=True), DR_TRAIN_STEPS, "域随机化·每局换土箱")
dr_policy = ppo_policy_of(dr_model)
res_dr_standard = evaluate(dr_policy, cfg=CONFIG)
print(f"  域随机化模型在标准土箱：moved_frac={res_dr_standard['moved_frac']:.3f}"
      f"（full_model 是 {res_std_standard['moved_frac']:.3f}）。")
print("  （服务器实测：待回填）见多识广，标准土箱表现不应明显变差。下面看它换土箱救没救回。")

# %% 4-2　域随机化模型 vs 标准模型在各土箱下对比 → 救回 + 鲁棒性曲线
# 【这段做什么】用实验 2-1/2-2 完全相同的判据、完全相同的 k_eo/m_dry 扫描点，测域随机化模型，
#   和标准模型并排对比，再把两条鲁棒性曲线画在一起。
#
# 【控制变量：为什么这个对比是干净的】
#   full_model 和 dr_model 用同样配齐的奖励、同样的评估判据、同样的测试土箱、同样的评估 seed——
#   唯一的差别就是"训练时见的是一种土箱还是很多种土箱"。所以任何成绩差异只能归因于
#   "有没有做域随机化"，干净。
#
# 【看什么】
#   · 并排对比表：偏离训练值的那些 k_eo/m_dry 点上，标准模型（full）低、域随机化模型（dr）高。
#   · 判据 judge_dr_rescue 在最极端的漂移点（k_eo=0.06、m_dry=0.04）打印 True = 救回成立。
#   · 鲁棒性曲线：红线（标准）在偏离训练点后跳水，绿线（域随机化）在整段范围都比较平——
#     这条"平不平"的对比，就是"能不能换土箱部署"的依据。和 Class 3 的杆长鲁棒性曲线一个道理。
#
# 【到这里，泛化这条线的逻辑链合拢了】
#   标准土箱训好（实验1的 full_model）→ 换土箱就崩（实验2）→ 诊断是过拟合到单一土箱（实验3）
#   → 域随机化救回（实验4）。每步都有判据证据，不是凭直觉跳跃。
# 【预期（服务器实测）】待 V100 真跑回填。
print("实验 4-2：域随机化模型 vs 标准模型（各土箱并排对比）")
dr_by_keo = [evaluate(dr_policy, cfg=dict(CONFIG, k_eo=ke)) for ke in keo_list]
dr_by_mdry = [evaluate(dr_policy, cfg=dict(CONFIG, m_dry=md)) for md in mdry_list]

print(f"  【按 k_eo】{'k_eo':<8}{'标准模型':<12}{'域随机化':<12}")
for ke, a, b in zip(keo_list, std_by_keo, dr_by_keo):
    print(f"            {ke:<8}{a['moved_frac']:<12.3f}{b['moved_frac']:<12.3f}")
print(f"  【按 m_dry】{'m_dry':<7}{'标准模型':<12}{'域随机化':<12}")
for md, a, b in zip(mdry_list, std_by_mdry, dr_by_mdry):
    print(f"            {md:<8}{a['moved_frac']:<12.3f}{b['moved_frac']:<12.3f}")

rescued_keo, msg_keo = judge_dr_rescue(dr_by_keo[0], std_by_keo[0])     # 最干土质 k_eo=0.06
rescued_mdry, msg_mdry = judge_dr_rescue(dr_by_mdry[0], std_by_mdry[0]) # 最干装填 m_dry=0.04
print(f"  判据 judge_dr_rescue @k_eo=0.06 = {rescued_keo}： {msg_keo}")
print(f"  判据 judge_dr_rescue @m_dry=0.04 = {rescued_mdry}： {msg_mdry}")

# 鲁棒性曲线：标准 vs 域随机化，两个漂移维度各一条
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].plot(keo_list, [r["moved_frac"] for r in std_by_keo], "o-", color="red", label="standard")
axes[0].plot(keo_list, [r["moved_frac"] for r in dr_by_keo], "s-", color="green", label="domain rand.")
axes[0].axvline(CONFIG["k_eo"], ls=":", color="gray"); axes[0].set_ylim(0, 1)
axes[0].set_xlabel("k_eo (test soil)"); axes[0].set_ylabel("moved_frac")
axes[0].set_title("Robustness vs soil k_eo"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(mdry_list, [r["moved_frac"] for r in std_by_mdry], "o-", color="red", label="standard")
axes[1].plot(mdry_list, [r["moved_frac"] for r in dr_by_mdry], "s-", color="green", label="domain rand.")
axes[1].axvline(CONFIG["m_dry"], ls=":", color="gray"); axes[1].set_ylim(0, 1)
axes[1].set_xlabel("m_dry (test dryness)"); axes[1].set_ylabel("moved_frac")
axes[1].set_title("Robustness vs initial dryness"); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()
print("  → 红线（标准）偏离训练点就跳水，绿线（域随机化）整段都比较平（服务器实测：待回填）。")
print("    这就是'救回'：换土箱也能用，才是真能部署的策略。宽泛化 > 单点峰值。")

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
#   规则基线只会"猛干"：湿度差大就满电压推，顾了搬水顾不上省电和 pH。
#   PPO 学到的是"在搬匀、省电、护 pH 三个打架的目标间做权衡"——
#   该用力时用力、接近均匀就收手省电、pH 快越界就收着点。所以典型结果是：
#   两者都能把水搬得差不多匀，但 PPO 的耗电和 pH 越界明显更低。这就是 RL 的价值。
#
# 【看什么】
#   · 柱状图三组：搬匀程度（越高越好）、耗电（越低越好）、pH 越界（越低越好）。
#   · 期待：moved 两者接近，energy 和 ph_viol 上 PPO（绿）明显低于规则（灰）。
#   · 判据 judge_beats_baseline 打印 True = "搬匀不输 + 耗电更低 + pH 越界更低"全部成立。
#     这张图 + 这个 True，就是回答"非得用 RL 吗"的标准答案。
# 【预期（服务器实测）】待 V100 真跑回填。
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
print("  → 两者都能搬匀，但 PPO 在耗电/pH 越界上更低（服务器实测：待回填）。")
print("    答辩被问『非得用强化学习吗』，就甩这张对比图：RL 赢在多目标的非平凡权衡。")

# %% 5-2　保存模型 + Limitations（诚实交代局限）+ logbook
# 【这段做什么】保存两个模型，打印 Limitations（写论文照这个思路展开），留 logbook 三问。
#
# 【该部署哪个模型——一句研究判断】
#   部署到真机，应该选 dr_model（域随机化版），不是 full_model。理由：真实土箱的 k_eo/m_dry
#   不可能正好是训练值，full_model 在标准土箱上"峰值好看"但换土就崩，dr_model 在整段范围都
#   稳。这正是 Class 3 那句"宽泛化 > 单点峰值"在你课题上的落地。
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
print("  部署候选选『域随机化版』：真土箱参数和训练值对不上，宽泛化比单点峰值更靠谱。")

print(r"""
================ Limitations（写论文时照这个思路展开）================
1. 简化物理：本数字孪生用规则化的电渗/pH 近似，未求解完整 Navier-Stokes / Nernst-Planck
   PDE。结论的物理保真度受此限制，真机校准前不宜外推。
2. 低湿失效仅部分建模：可动性随含水率下降而衰减（见 0-3 的 mobility），实验 2-2 也复现了
   "初始太干就搬不动"，但真实非饱和多孔介质的电渗更复杂，极端干燥工况未充分验证。
3. 盐分积累【未建模】：长期通电会使离子在阴极富集、导致土壤盐碱化——本模型未包含，
   是已知重要局限，列为后续工作（也是答辩必问，要主动讲）。
4. sim-to-real gap：本节用域随机化在【仿真内部】缩小了 gap（换土箱也能用），但真机还有
   仿真没建模的因素（探头噪声、执行器延迟、温度、未建模电化学反应）。真正下结论前，
   需用真实土箱数据校准数字孪生、并做真机微调。域随机化是必要不充分条件。
诚实写清这些边界 + 给出后续方向 = 成熟度的体现，是加分项，不是减分项。
====================================================================
""")

# 【logbook 三问——做完实验、用自己的话写，这是把课内化的关键】
#   1. 朴素奖励版把水搬匀了，为什么仍然说它"失败"？reward hacking 错在智能体还是错在你？
#      （答案要落到"约束没写进奖励"，并说清这对你的真实土箱意味着什么风险）
#   2. full_model 在标准土箱表现很好，为什么不能直接说它"能用于真实土箱"？
#      要补做哪个实验、用什么判据，才能下"能换土箱部署"的结论？（这是本节最核心一问）
#   3. 把这套搬到你的真实土箱：训练时还该把哪些东西纳入域随机化，才扛得住真机？至少列三个。
#      （参考：探头基线漂移/噪声、土质 k_eo、装填初始干湿、温度对电化学的影响、盐分……）
print("logbook 三问（见 5-2 代码注释，用自己的话写）：")
print("  ① 朴素版搬匀了为何仍算失败？reward hacking 错在智能体还是错在你？")
print("  ② full_model 标准土箱好，为何不能说它'能用于真机'？要补哪个实验+判据才能下结论？")
print("  ③ 搬到真实土箱，训练时还该把哪三样纳入域随机化才扛得住真机？")
print("\n【本讲小结】环境和奖励是你的活、算法是现成工具；奖励设错会 reward hacking、")
print("            训练好≠能用要靠'换土箱测判据+域随机化'救、RL 要打过规则基线、局限要诚实讲。")
print("            这就是把一个 RL 课题从'仿真 demo'做成'经得起答辩的研究'的全套动作。")
