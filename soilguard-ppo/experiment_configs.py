# -*- coding: utf-8 -*-
"""Named experiment configurations for SoilGuard PPO training.

每个实验条目把全部参数写全，直接在字典里改即可。
评估 = 固定百题考卷（evaluation.py，含躺平/规则基线陪考），无需配置。
注意：旧协议的 episodes 已不存在——考卷固定 100 个场景，没有"评估局数"这个参数。

实验演进历史（v1–v5 属旧环境时代，配置条目已删除，运行记录仍在 artifacts/）：
# v1是当前基线实验的基线
# v2在v1的基础上增加了w_ph的惩罚权重
# v3在v2的基础上改变了土壤初始含水量分布（从中心多周围少到左边多右边少）
# v4在v3的基础上增强均匀化的驱动力，增大电压的动作幅度
# v5在v4的基础上最大步数从60改为120
# v6在v5的基础上重写环境以及动作环境交互
# v7在v6的基础上增大w_ph和w_local的惩罚系数
# v8在v7的基础上回调了w_ph以及w_local的惩罚系数
# v9在v8的基础上增大了水分均匀w_uniformity_goal的奖励系数
# v10在v9的基础上把每局步数从120延长到180
# v11在v10的基础上开启多核训练，评估方式彻底改写
# v12在v11的基础上开启初始环境随机化，总步数从500000增加到1000000步
# v13在v12的基础上修改delta cv的奖励方式（由于水分自然扩散导致水分变均匀的部分不作奖励），
#     并且训练不再"达标即终止"（每局跑满180步，目标变为最小化CV曲线下面积），冲线奖金置0，
#     总步数从1000000增加到1200000步
# v14在v13的基础上部署到了远程终端,进程数8->16
# v15在v14的基础上新增保持奖金hold_bonus：CV达标的每一步发正奖励（不是一次性冲线奖金）。
# v16在v15的基础上总步数增加到150w步
# v17在v16的基础上加行为克隆热启动：PPO开训前先监督模仿比例控制器。
#     奖励/环境/考卷全部冻结（hold_bonus=3.0 不变），唯一变量=策略初始化方式。
#     动机：v15证明奖励排序已修对但PPO仍躺平（方向梯度对冲+工资从未采样到+critic被阈值致盲EV=0）。
#     验收：纯BC版先考百题卷做锚点(ppo_bc_anchor)，微调版不得低于锚点；盯EV回升、action_abs不再萎缩。
# v18在v17的基础上新增到200w步
# v19在v18的基础上把log_std_init的值从2.0->0.0
# v20在v19的基础上把pH惩罚从"罚存量"改为"罚净酸增量"(ph_penalty_mode=acid_increment)：
#     旧账本每步重罚不可逆酸债(~-300/局)，持续抽血侵蚀搬运策略(v19锚点84.7被磨到72.9)。
#     新账本只在"最坏格酸侧越界较上一步变大"时按增量收一次费，保持/稀释/中和均不再计费。
#     增量telescoping使总量≈峰值酸(~1.5)，故w_ph由reward_audit.py重标1.2->30(pH成本~-48/局,
#     排序仍proportional257>rule137>idle-103)。考卷ph_exposure仍用存量口径,不受影响。
# v21在v20的基础上做了一轮"动作接口+奖励口径"对齐修订（BC热启动照旧保留——它对
#     "躺平不放电"探索死结的解法优于从零PPO），共四处改动：
#     ①动作空间归一化：SB3动作从物理电压改为[-1,1]，环境内部按 command×max_electrode_voltage_v
#       映射到±20V物理电极电压（rule/proportional基线策略同步改为输出归一化动作）；
#     ②CV罚改"超标gap"口径：只罚 max(0, moisture_cv - target_moisture_cv)，
#       已达标的均匀状态不再被CV底噪每步持续扣分；
#     ③能耗罚改"电极对电压差"：电渗只取决于对间电压差(left-right / back-front)，
#       按 pair_voltage_energy 计费，整体同抬共模电压不再被罚；
#     ④pH账本改混合计费：净酸增量 + ph_stock_penalty_fraction(0.01)×存量暴露
#       （w_ph=30下有效存量权重仅0.3）——保留v20"只为产酸行为付一次费"的主杠杆，
#       同时与考卷仍按存量口径计的ph_exposure保持弱对齐。
# v22在v21的基础上把log_std_init从0.0改为-2.7（唯一变量，步数保持200w与v19-v21同基准）：
#     动作空间归一化后0.0的含义变了——σ=e^0=1.0是半量程的100%（物理≈20V，比毒杀v17/v18
#     的σ≈7.4V更猛），BC克隆的均值(~0.1)被采样噪声淹没（首rollout action_abs 0.64/饱和35%，
#     熵17.0=12维σ=1高斯），策略靠推饱和均值适应噪声，确定性考试时满电压狂轰：
#     能耗58.9(bad线60)、酸暴露1.47、succ 0.64、考分56.7<躺平61.6（训练ep_rew却涨到+135，
#     EV 0.98——训练与考卷彻底错位）。-2.7即σ≈0.067，恢复v19验证过的噪声比例(1V/15V≈6.7%)。
#     实现：log_std_init接入实验配置并由run_train_standard透传给train_ppo（参数原已存在）。
# v23在v22的基础上log_std_init": -2.7（未上传到远程）
# v24上传到远程之后重新跑：
#     修复确认——熵起点-15.37、首rollout action_abs 0.096=BC均值、143k步ep_rew峰值+142。
#     但v19式侵蚀以最纯形态重现：ep_rew +142->-80、训练cv 0.167->0.38、产酸降到0.15，
#     策略用均匀性换pH减负成"安全但无能"轻推手（考卷succ 0.32/分52.1/phExp 0.289/能耗2.77）。
#     新病灶：σ自发回涨0.067->0.149（熵-15.4->-5.8，ent_coef=0下仍涨）。
#     核心病理（四制度全复现：v19存量/v20增量/v21大噪声/v24小噪声）：锚点处hold工资=批内
#     近常数（全达标）->对策略梯度不可见，批内方差全在成本侧->一致梯度方向="省成本少推"，
#     滑出工资区后想爬回=v15爬不动的谷地，单向棘轮。BC锚点(85)至今是项目最强策略。
# v25在v24的基础上加两件事（针对上述病理的组合拳）：
#     ①KL-to-BC系绳（主训练变量）：BC结束后冻结策略副本π_BC，PPO损失加
#       kl_bc_coef×KL(π‖π_BC)。高斯KL含方差项，同时钉住均值漂移+压制σ回涨（一石二鸟）。
#       kl_bc_coef=0.1的量级依据：σ_BC=0.067下KL对均值漂移极陡（Δμ=0.05≈1V -> ~3.3 nats），
#       0.1×3.3≈0.33与policy loss同量级。校准看dashboard新增的kl_to_bc曲线：
#       健康=KL稳定在0.5~5且ep_rew守住早期峰值；KL>20仍下滑->λ×5再跑；KL≈0且分守住->可降λ。
#     ②最优检查点存档（观测性收割，不改训练动力学）：每10w步用20题子卷（每家族s0变体、
#       官方种子）考当前策略，存最优快照soilguard_ppo_best.zip；训练结束后best快照重考
#       全卷记入ppo_best_battery。动机=v24的+142峰值策略未存档被浪费。BC锚点同时存zip。
#     验收：①final或best不低于锚点84.98附近；②kl_to_bc不爆炸且ep_rew不再单调下滑；
#     ③σ不回涨（熵曲线不再爬升）。若KL只能"守住"锚点而无超越，说明PPO微调无增量价值，
#     下一步转向改良BC教师（控制器调度/DAgger）而非继续调PPO。
#     【v25结果】侵蚀首次止住，四条验收全过：ep_rew全程平稳225~255（对照v24滑坡+142->-80，
#     反证锚点策略真实训练回报≈240）、kl_to_bc稳0.3~1.7、熵稳-15.1（σ无回涨）、EV0.98。
#     考卷final 84.78≈锚点、best@1.3M全卷85.13首次微超锚点（+0.15，在标准误0.54内不显著）、
#     succ 1.00、phExp 0.954（<1.06bonus线✓且优于锚点0.966）、能耗1.96。
#     21个检查点全在84.88~85.27窄带=守成不超越。病理机制假说获干净验证：系绳一上棘轮消失。
#     剩余损失结构：pH分项≈14.3分（phExp贴BAD线1.0，ph_score仅0.05）=最后的金矿；
#     dry≈4分；uniformity/energy已满分。教师本身phExp 0.966（达标后仍小电压产酸）。
# v26在v25的基础上把kl_bc_coef从0.1放松到0.02（唯一变量）：
#     动机：v25的KL全程0.3~1.7贴健康区(0.5~5)下沿，系绳可能紧到连有益的偏离也被压制，
#     PPO只能守成。放松5倍试探"守住的同时能否超越"。
#     风险已被检查点机制cap住：即使松绳后侵蚀部分回归，best快照仍收割途中峰值。
#     验收：①主看best是否显著超锚点（>85.5起算，超出0.54标准误一倍）；②kl_to_bc允许
#     升高但应稳定（不单调爆炸）；③若ep_rew重现单调下滑=侵蚀回归，则0.02过松，
#     结论区间收窄到(0.02,0.1)；④若与v25无差异（best仍≈85）则PPO微调确无增量价值，
#     转向改良BC教师攻pH分项（方向B）。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


COMMON_CONFIG: dict[str, Any] = {}


EXPERIMENT_OVERRIDES: dict[str, dict[str, Any]] = {
    "v26_reward_full": {
        "steps": 2000000,               # total PPO environment steps (v19-v25 基准)
        "n_envs": 16,                   # parallel rollout environments
        "seed": 42,                     # random seed
        "verbose": 0,                   # SB3 log verbosity
        "log_std_init": -2.7,           # v24 修复保持：sigma ~= 6.7% 半量程 ~= 1.3 V
        # v26 唯一变量：系绳放松 0.1 -> 0.02（v25 的 KL 贴健康区下沿，守成不超越；
        # 松绳 5 倍试探超越空间，检查点存档 cap 住侵蚀回归的下行风险）。
        "kl_bc_coef": 0.02,
        "checkpoint_every": 100000,     # 每 10w 步考 20 题子卷存最优快照
        "randomize_physics": False,     # keep physics fixed; only initial moisture field is randomized
        "randomize_obs_noise": False,   # no observation-noise domain randomization
        "randomize_initial_field": True,# train on randomized initial moisture patterns
        "eval_noise_levels": "",        # use the fixed 100-scenario evaluation set only
        "clearml": True,
        "clearml_online": True,
        "clearml_no_artifacts": False,
        "clearml_project": "soilbox_ppo_train",
        "clearml_task_name": "v26_reward_full",
        "bc_pretrain": True,
        "bc_episodes": 300,
        "bc_epochs": 20,
        "reward_overrides": {
            "max_electrode_voltage_v": 20.0,
            "ph_penalty_mode": "acid_increment",
            "w_ph": 30.0,
            # With w_ph=30, this adds a small effective stock penalty of 0.3.
            "ph_stock_penalty_fraction": 0.01,
        },
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
