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
# The active policy must beat idle by at least this much episode return,
# otherwise the gradient toward "act" is too weak to trust.
MARGIN = 50.0

REWARD_COMPONENTS = (
    "goal_excess",
    "hold_bonus",
    "cv_penalty",
    "ph_penalty",
    "dry_penalty",
    "energy_penalty",
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


def sample_training_fields(cfg: SoilBoxConfig, n_fields: int, seed0: int) -> list[np.ndarray]:
    """Draw initial fields from the exact dice the trainer uses."""

    env = SoilBoxEnv(cfg, randomize_initial=True, seed=seed0)
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
        comp["goal_excess"] += cfg.w_uniformity_goal * terms["uniformity_excess"]
        comp["hold_bonus"] += terms.get("hold_bonus", 0.0)
        comp["cv_penalty"] -= cfg.w_uniformity * terms.get("moisture_cv_gap", terms["moisture_cv"])
        # The pH component of the RETURN uses whatever the env actually charged
        # (stock or acid-increment); the stock exposure is tracked separately as
        # the exam-relevant metric.
        comp["ph_penalty"] -= cfg.w_ph * terms.get("ph_penalty_charge", terms["ph_violation"])
        comp["dry_penalty"] -= cfg.w_local_dry * terms["local_dry_violation"]
        comp["energy_penalty"] -= cfg.w_energy * terms["energy"]
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


def audit(cfg: SoilBoxConfig, n_fields: int = N_FIELDS) -> dict[str, dict[str, float]]:
    """Mean per-policy stats over the shared field batch."""

    policies: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "idle": idle_policy,
        "rule": lambda obs: rule_policy(obs, cfg.n_cells, max_electrode_voltage(cfg)),
        "proportional": lambda obs: proportional_policy(obs, cfg),
    }
    fields = sample_training_fields(cfg, n_fields, FIELD_SEED0)
    results: dict[str, dict[str, float]] = {}
    for name, policy in policies.items():
        rows = [rollout(policy, cfg, field, seed=FIELD_SEED0 + i) for i, field in enumerate(fields)]
        mean = {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
        mean["return_std"] = float(np.std([r["return"] for r in rows]))
        results[name] = mean
    return results


def format_audit_table(results: dict[str, dict[str, float]]) -> str:
    header = (
        f"{'policy':<14}{'return':>10}{'std':>8}{'excess':>8}{'hold':>8}{'cv_pen':>9}"
        f"{'ph_pen':>9}{'dry_pen':>9}{'reach':>7}{'hold_n':>8}{'phExp':>8}{'final_cv':>10}"
    )
    lines = [header]
    for name, m in results.items():
        lines.append(
            f"{name:<14}{m['return']:>10.1f}{m['return_std']:>8.1f}{m['goal_excess']:>8.2f}"
            f"{m['hold_bonus']:>8.1f}{m['cv_penalty']:>9.1f}{m['ph_penalty']:>9.1f}"
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
    args = parser.parse_args()

    cfg = build_config(args)
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
        print("AUDIT FAILED: the training reward still prefers idling. "
              "Fix the weights (hold_bonus first, then w_uniformity_goal) and re-audit. "
              "Do not launch training.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
