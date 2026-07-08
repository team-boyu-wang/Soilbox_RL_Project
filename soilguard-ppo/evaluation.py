# -*- coding: utf-8 -*-
"""Scenario-battery evaluation (protocol v2) for the SoilGuard soil box.

Replaces the old single-scenario, terminate-on-success report protocol:

- **Fixed scenario battery**: a versioned, numbered list of initial moisture
  fields. Rotated variants expose memorisation (a policy that truly
  uniformises must handle left-wet and right-wet alike); amplitude variants
  and fixed-seed random fields probe generalisation. The battery is frozen so
  every model version takes the same exam.
- **Full-horizon rollouts**: evaluation ignores the env's terminate-on-success
  flag and always runs ``max_steps``, so we can measure whether uniformity is
  held after the target is reached and what the side effects of the whole
  trajectory are.
- **Graded scoring**: reaching the target is a weighted component of the
  scenario score, not a multiplicative gate — a near-miss (CV 0.19) now scores
  higher than a far miss (CV 0.29) instead of both collapsing to 0.
- **Baseline anchoring**: callers are expected to also run ``idle_policy``
  (diffusion only, the floor) and ``rule_policy`` on the same battery.

Aggregates are reported as mean +/- std over scenarios plus the worst
scenario, which is the honest engineering number ("how bad does it get").
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from soilbox_env import (
    DEFAULT_CONFIG,
    FINAL_SCORE_WEIGHTS,
    FINAL_SCORE_NORMALIZATION,
    SoilBoxConfig,
    SoilBoxEnv,
    _clip01,
    _coerce_config,
    _low_is_good_score,
)

BATTERY_VERSION = "v3.0"

# Weight of "reached the CV target at least once" in the per-scenario score;
# the remaining weight goes to the quality score. Engineering choice: reaching
# the goal stays the single most valuable behaviour but no longer zeroes
# everything else out.
REACHED_WEIGHT = 0.40

# Exposure normalisation lines (engineering choices, to be recalibrated
# against real-box data): exposure = per-step mean violation over the episode.
# Peaks are reported alongside but not scored — the old peak-based zero-lines
# proved structurally unreachable for this task (see v6-v9 analysis).
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
        else:  # "top"
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


# 20 distribution families. Each family is a deterministic base pattern; the
# battery instantiates 5 perturbed variants per family (small +/- noise on the
# baseline, fixed seeds), giving a frozen 100-scenario exam.
FAMILIES: list[tuple[str, Callable[[SoilBoxConfig], np.ndarray]]] = [
    ("left_col", _column_wet(0, 0.35)),          # canonical training scenario
    ("right_col", _column_wet(2, 0.35)),         # mirror: memorisation check
    ("back_row", _row_wet(0, 0.35)),             # 90-degree rotations
    ("front_row", _row_wet(2, 0.35)),
    ("middle_col", _column_wet(1, 0.35)),        # interior wet: fill-outward
    ("middle_row", _row_wet(1, 0.35)),
    ("left_col_low", _column_wet(0, 0.30)),      # amplitude variants
    ("left_col_high", _column_wet(0, 0.40)),
    ("corner_tl", _cells([(0, 0)], 0.40)),       # single wet cell topologies
    ("corner_br", _cells([(2, 2)], 0.40)),
    ("center_cell", _cells([(1, 1)], 0.40)),
    ("diag_main", _cells([(0, 0), (1, 1), (2, 2)], 0.35)),
    ("diag_anti", _cells([(0, 2), (1, 1), (2, 0)], 0.35)),
    ("opposite_corners", _cells([(0, 0), (2, 2)], 0.35)),
    ("adjacent_corners", _cells([(0, 0), (0, 2)], 0.35)),
    ("cross_wet", _cells([(1, 1), (0, 1), (1, 0), (1, 2), (2, 1)], 0.30)),
    ("left_half", _half_wet("left", 0.28)),
    ("top_half", _half_wet("top", 0.28)),
    ("border_ring", _border_ring(0.30)),         # wet ring, dry centre
    ("gradient_x", _gradient("x")),              # smooth ramp
]

VARIANTS_PER_FAMILY = 5
# Perturbation amplitude for the seeded variants: small +/- noise on the base
# pattern so each family probes a neighbourhood, not a single point.
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


# The frozen battery (20 families x 5 seeded variants = 100 scenarios).
# Do not reorder or edit entries in place — add families / bump
# BATTERY_VERSION so scores stay comparable across model versions.
SCENARIOS: list[tuple[str, Callable[[SoilBoxConfig], np.ndarray]]] = _build_battery()


def idle_policy(obs: np.ndarray) -> np.ndarray:
    """Do-nothing baseline: diffusion alone sets the floor of the battery."""

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
    ph_peak = 0.0
    dry_sum = 0.0
    dry_peak = 0.0
    energy = 0.0
    energy_to_target = 0.0
    final_cv = initial_cv

    for t in range(1, cfg.max_steps + 1):
        action = policy_fn(obs)
        # Full-horizon protocol: the env may flag terminated on success, but
        # evaluation keeps stepping to measure stability and total exposure.
        obs, _, _, _, info = env.step(action)
        terms = info["terms"]
        final_cv = float(info["moisture_cv"])
        min_cv = min(min_cv, final_cv)
        ph_sum += float(terms["ph_violation"])
        ph_peak = max(ph_peak, float(terms["ph_violation"]))
        dry_sum += float(terms["local_dry_violation"])
        dry_peak = max(dry_peak, float(terms["local_dry_violation"]))
        energy += float(terms["energy"])
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
        "ph_peak": ph_peak,
        "dry_exposure": dry_sum / steps,
        "dry_peak": dry_peak,
        "energy": energy,
        "energy_to_target": energy_to_target if reached else energy,
        "final_mean_pH": float(np.mean(env.pH)),
        "final_mean_moisture": float(np.mean(env.m)),
    }


def score_scenario(metrics: Mapping[str, float]) -> dict[str, float]:
    """Component scores in [0, 1] and the graded per-scenario score in [0, 100]."""

    norm = FINAL_SCORE_NORMALIZATION
    uniformity_score = _low_is_good_score(
        metrics["final_cv"], norm["moisture_cv_good"], norm["moisture_cv_bad"]
    )
    ph_score = _clip01(1.0 - metrics["ph_exposure"] / PH_EXPOSURE_BAD)
    dry_score = _clip01(1.0 - metrics["dry_exposure"] / DRY_EXPOSURE_BAD)
    energy_score = _low_is_good_score(metrics["energy"], norm["energy_good"], norm["energy_bad"])
    quality = (
        FINAL_SCORE_WEIGHTS["uniformity_score"] * uniformity_score
        + FINAL_SCORE_WEIGHTS["ph_score"] * ph_score
        + FINAL_SCORE_WEIGHTS["local_dry_score"] * dry_score
        + FINAL_SCORE_WEIGHTS["energy_score"] * energy_score
    )
    scenario_score = 100.0 * (REACHED_WEIGHT * metrics["reached"] + (1.0 - REACHED_WEIGHT) * quality)
    return {
        "uniformity_score": float(uniformity_score),
        "ph_score": float(ph_score),
        "dry_score": float(dry_score),
        "energy_score": float(energy_score),
        "quality_score": float(quality),
        "scenario_score": float(scenario_score),
    }


def evaluate_v2(
    policy_fn: Callable[[np.ndarray], np.ndarray],
    *,
    config: SoilBoxConfig | Mapping[str, Any] | None = None,
    obs_noise: float = 0.0,
    seed0: int = 1000,
    scenario_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Run the frozen scenario battery and return per-scenario + aggregate results.

    ``scenario_indices`` selects a battery SUBSET by original scenario index
    (per-scenario seeds stay ``seed0 + original_index``, so subset metrics are
    directly comparable with the same scenarios inside a full run). Default
    None runs the full frozen exam — the official protocol is untouched.
    Subset runs are for cheap in-training checkpoint ranking only; official
    scores must come from the full battery.
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
    """Console table aggregated per family (full per-scenario data stays in
    the returned dict / diagnostics json)."""

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
