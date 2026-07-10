# -*- coding: utf-8 -*-
"""Reward audit: score reference policies with the TRAINING reward.

PPO optimizes whatever the reward function actually pays, not what we meant.
v13 proved it the hard way: under those weights the do-nothing policy scored
~ -160 per episode while the rule baseline (0.79 battery success) scored
~ -310, so PPO faithfully collapsed into idling. The failure was in the exam
paper, not in the student.

This script is the pre-training judge for that failure mode: it rolls three
reference policies through the SAME reward the trainer will use, on a batch of
random initial fields drawn from the SAME training dice
(``randomize_initial``), and checks the ordering that must hold before any
training run is worth launching:

    competent active policy  >  idle   (by a clear margin)

Run it whenever reward weights change; if the judge fails, fix the weights and
re-audit — do NOT start training.

Usage::

    python reward_audit.py                       # audit DEFAULT_CONFIG (v13 weights)
    python reward_audit.py --hold-bonus 2.0      # audit a candidate override
    python reward_audit.py --experiment v14_reward_hold   # audit an experiment entry
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Callable

import numpy as np

from evaluation import idle_policy
from experiment_configs import EXPERIMENTS
from soilbox_env import (
    DEFAULT_CONFIG,
    SoilBoxConfig,
    SoilBoxEnv,
    max_electrode_voltage,
    rule_policy,
)

N_FIELDS = 50
FIELD_SEED0 = 7000
# The active policy must beat idle by at least this much episode return.
# Units since v5.0: EXAM POINTS on the 0-100 scale (the return equals the
# exam score by construction), so +10 means "clearly better by 10 exam
# points". (The old +50 was calibrated for the legacy weighted-penalty
# reward whose returns spanned roughly -400..+250.)
MARGIN = 10.0

# v5.0 reward structure: per-step quality instalments (u/ph/e) + optional hooks.
REWARD_COMPONENTS = (
    "reward_u",
    "reward_ph",
    "reward_e",
    "shaping_excess",
    "dry_penalty",
)


def proportional_policy(obs: np.ndarray, cfg: SoilBoxConfig, gain: float = 80.0) -> np.ndarray:
    """Gentle closed-loop controller: per-row/column moisture-difference pushes.

    2-D generalization of the x-axis controller pinned in
    ``test_electroosmosis_plus_diffusion_reaches_target``; throttles to half
    voltage when any cell's pH leaves the tolerance band (same protection as
    ``rule_policy``, but proportional instead of bang-bang, so it eases off as
    the field approaches uniformity).
    """

    n = cfg.n_cells
    rows, cols = cfg.grid_rows, cfg.grid_cols
    max_v = max_electrode_voltage(cfg)
    moisture = obs[:n]
    ph = obs[n:] * 7.0 + 7.0
    grid = moisture.reshape(rows, cols)
    vx_volts = np.clip(gain * (grid[:, 0] - grid[:, -1]), -max_v, max_v)   # per row
    vy_volts = np.clip(gain * (grid[0, :] - grid[-1, :]), -max_v, max_v)   # per column
    vx = vx_volts / max(max_v, 1e-9)
    vy = vy_volts / max(max_v, 1e-9)
    if float(np.max(np.abs(ph - 7.0))) > cfg.ph_tol:
        vx *= 0.5
        vy *= 0.5
    left, right = vx, -vx
    back, front = vy, -vy
    return np.concatenate([left, right, front, back]).astype(np.float32)


def sample_training_fields(
    cfg: SoilBoxConfig,
    n_fields: int,
    seed0: int,
    *,
    curriculum_difficulty: float | None = None,
    curriculum_min_contrast: float = 0.6,
) -> list[np.ndarray]:
    """Draw initial fields from the exact dice the trainer uses."""

    env = SoilBoxEnv(
        cfg,
        randomize_initial=True,
        seed=seed0,
        curriculum_difficulty=curriculum_difficulty,
        curriculum_min_contrast=curriculum_min_contrast,
    )
    fields = []
    for k in range(n_fields):
        env.reset(seed=seed0 + k)
        fields.append(env.m.copy().reshape(cfg.grid_rows, cfg.grid_cols))
    return fields


def rollout(policy_fn: Callable[[np.ndarray], np.ndarray], cfg: SoilBoxConfig,
            field: np.ndarray, seed: int) -> dict[str, float]:
    """Full-horizon rollout; returns training return + weighted components."""

    env = SoilBoxEnv(cfg, seed=seed)
    obs, _ = env.reset(seed=seed, options={"initial_moisture": field})
    total = 0.0
    comp = dict.fromkeys(REWARD_COMPONENTS, 0.0)
    hold_steps = 0
    reached = False
    ph_exposure_sum = 0.0
    for _ in range(cfg.max_steps):
        obs, reward, terminated, truncated, info = env.step(policy_fn(obs))
        total += reward
        terms = info["terms"]
        # v5.0: the reward IS the per-step exam instalment; read the actual
        # paid components straight from terms.
        comp["reward_u"] += terms["reward_u"]
        comp["reward_ph"] += terms["reward_ph"]
        comp["reward_e"] += terms["reward_e"]
        comp["shaping_excess"] += cfg.w_uniformity_goal * terms["uniformity_excess"]
        comp["dry_penalty"] -= cfg.w_local_dry * terms["local_dry_violation"]
        ph_exposure_sum += float(terms["ph_violation"])
        if info["success"]:
            hold_steps += 1
            reached = True
        if terminated or truncated:
            break
    return {
        "return": total,
        **comp,
        "reached": float(reached),
        "hold_steps": float(hold_steps),
        "ph_exposure": ph_exposure_sum / cfg.max_steps,
        "final_cv": float(info["moisture_cv"]),
    }


def audit(
    cfg: SoilBoxConfig,
    n_fields: int = N_FIELDS,
    *,
    curriculum_difficulty: float | None = None,
    curriculum_min_contrast: float = 0.6,
) -> dict[str, dict[str, float]]:
    """Mean per-policy stats over the shared field batch."""

    policies: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "idle": idle_policy,
        "rule": lambda obs: rule_policy(obs, cfg.n_cells, max_electrode_voltage(cfg)),
        "proportional": lambda obs: proportional_policy(obs, cfg),
    }
    fields = sample_training_fields(
        cfg,
        n_fields,
        FIELD_SEED0,
        curriculum_difficulty=curriculum_difficulty,
        curriculum_min_contrast=curriculum_min_contrast,
    )
    results: dict[str, dict[str, float]] = {}
    for name, policy in policies.items():
        rows = [rollout(policy, cfg, field, seed=FIELD_SEED0 + i) for i, field in enumerate(fields)]
        mean = {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
        mean["return_std"] = float(np.std([r["return"] for r in rows]))
        results[name] = mean
    return results


CURRICULUM_AUDIT_DIFFICULTIES = (0.0, 0.25, 0.5, 0.75, 1.0)


def curriculum_audit(
    cfg: SoilBoxConfig,
    n_fields: int = N_FIELDS,
    *,
    min_contrast: float = 0.6,
    difficulties: tuple[float, ...] = CURRICULUM_AUDIT_DIFFICULTIES,
    margin: float = MARGIN,
    idle_reach_slack: float = 0.10,
) -> bool:
    """The curriculum gate (v27): the reward ordering must hold at EVERY level.

    The trap this guards against: easy fields sit closer to the target, and if
    diffusion alone (idle) can finish the job there disproportionately often,
    the hold wage pays the do-nothing policy and the curriculum's first lesson
    becomes idling — the v12 collapse, served on purpose.

    Two checks per level:
    1. Reward ordering (the battle-tested gate): proportional out-earns idle
       by the margin, and idle is not the top scorer.
    2. COMPARATIVE idle reach: measured d=1.0 (the historical full dice, which
       never collapsed under a correct ordering) has ~0.36 idle-reachable
       episodes — an inherent property of the pattern pool, not curable by
       contrast (single wet cells diffuse away regardless). Easy levels must
       not be materially more idle-solvable than that baseline
       (reach(d) <= reach(1.0) + idle_reach_slack).
    """

    baseline = audit(
        cfg, n_fields, curriculum_difficulty=1.0, curriculum_min_contrast=min_contrast
    )
    baseline_idle_reach = baseline["idle"]["reached"]
    print(f"baseline (d=1.00 = historical dice): idle reach rate = {baseline_idle_reach:.2f}")
    all_ok = True
    for d in difficulties:
        results = (
            baseline
            if d == 1.0
            else audit(cfg, n_fields, curriculum_difficulty=d, curriculum_min_contrast=min_contrast)
        )
        idle_reach = results["idle"]["reached"]
        ok, msg = judge(results, margin=margin)
        idle_ok = idle_reach <= baseline_idle_reach + idle_reach_slack
        level_ok = ok and idle_ok
        all_ok = all_ok and level_ok
        print(f"--- difficulty d={d:.2f} (min_contrast={min_contrast:g}) ---")
        print(format_audit_table(results))
        print(f"  ordering: {msg}")
        print(
            f"  idle reach rate = {idle_reach:.2f} "
            f"(need <= baseline {baseline_idle_reach:.2f} + {idle_reach_slack:g}) -> {idle_ok}"
        )
        print(f"  level PASS = {level_ok}")
    return all_ok


def format_audit_table(results: dict[str, dict[str, float]]) -> str:
    header = (
        f"{'policy':<14}{'return':>10}{'std':>8}{'r_u':>8}{'r_ph':>8}{'r_e':>8}"
        f"{'excess':>8}{'dry_pen':>9}{'reach':>7}{'hold_n':>8}{'phExp':>8}{'final_cv':>10}"
    )
    lines = [header]
    for name, m in results.items():
        lines.append(
            f"{name:<14}{m['return']:>10.1f}{m['return_std']:>8.1f}{m['reward_u']:>8.1f}"
            f"{m['reward_ph']:>8.1f}{m['reward_e']:>8.1f}{m['shaping_excess']:>8.2f}"
            f"{m['dry_penalty']:>9.1f}{m['reached']:>7.2f}{m['hold_steps']:>8.1f}"
            f"{m['ph_exposure']:>8.3f}{m['final_cv']:>10.3f}"
        )
    return "\n".join(lines)


def judge(results: dict[str, dict[str, float]], margin: float = MARGIN) -> tuple[bool, str]:
    """The gate: an active competent policy must clearly out-earn idling."""

    idle = results["idle"]["return"]
    prop = results["proportional"]["return"]
    rule = results["rule"]["return"]
    ok_margin = prop - idle >= margin
    ok_not_top = idle < max(prop, rule)
    ok = ok_margin and ok_not_top
    msg = (
        f"proportional - idle = {prop - idle:+.1f} (need >= +{margin:g}) -> {ok_margin} | "
        f"idle is top scorer -> {not ok_not_top}"
    )
    return ok, msg


def build_config(args: argparse.Namespace) -> SoilBoxConfig:
    overrides: dict[str, Any] = {}
    if args.experiment:
        overrides.update(EXPERIMENTS[args.experiment].get("reward_overrides", {}))
    if args.hold_bonus is not None:
        overrides["hold_bonus"] = args.hold_bonus
    if args.w_uniformity_goal is not None:
        overrides["w_uniformity_goal"] = args.w_uniformity_goal
    if args.w_ph is not None:
        overrides["w_ph"] = args.w_ph
    if args.ph_penalty_mode is not None:
        overrides["ph_penalty_mode"] = args.ph_penalty_mode
    return DEFAULT_CONFIG.with_overrides(**overrides)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default=None, choices=sorted(EXPERIMENTS),
                        help="Audit the reward_overrides of a named experiment.")
    parser.add_argument("--hold-bonus", type=float, default=None)
    parser.add_argument("--w-uniformity-goal", type=float, default=None)
    parser.add_argument("--w-ph", type=float, default=None)
    parser.add_argument("--ph-penalty-mode", default=None, choices=["stock", "acid_increment"])
    parser.add_argument("--fields", type=int, default=N_FIELDS)
    parser.add_argument("--curriculum-audit", action="store_true",
                        help="v27 gate: rerun the three-policy audit at each curriculum "
                             "difficulty level; every level must pass.")
    parser.add_argument("--curriculum-min-contrast", type=float, default=0.6)
    args = parser.parse_args()

    cfg = build_config(args)
    if args.curriculum_audit:
        print(f"Curriculum audit on {args.fields} fields per difficulty (seed0={FIELD_SEED0})")
        print(f"weights: hold_bonus={cfg.hold_bonus:g} w_ph={cfg.w_ph:g} "
              f"ph_penalty_mode={cfg.ph_penalty_mode}")
        ok = curriculum_audit(cfg, n_fields=args.fields, min_contrast=args.curriculum_min_contrast)
        print(f"curriculum_audit_pass = {ok}")
        if not ok:
            print("CURRICULUM AUDIT FAILED: some difficulty level lets idling compete "
                  "(likely the easy end). Raise --curriculum-min-contrast and re-audit. "
                  "Do not launch training.")
        sys.exit(0 if ok else 1)
    print(f"Reward audit on {args.fields} random training fields (seed0={FIELD_SEED0})")
    print(f"weights: hold_bonus={cfg.hold_bonus:g} w_uniformity_goal={cfg.w_uniformity_goal:g} "
          f"w_uniformity={cfg.w_uniformity:g} w_ph={cfg.w_ph:g} w_local_dry={cfg.w_local_dry:g} "
          f"ph_penalty_mode={cfg.ph_penalty_mode}")
    results = audit(cfg, n_fields=args.fields)
    print(format_audit_table(results))
    ok, msg = judge(results)
    print(f"judge_reward_prefers_action = {ok}")
    print(f"  {msg}")
    if not ok:
        print("AUDIT FAILED: the training reward does not clearly prefer acting "
              "over idling. Since v5.0 reward == exam, fix the LEVEL CURVES "
              "(uniformity_level_points first) or the component weights, then "
              "re-audit. Do not launch training.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
