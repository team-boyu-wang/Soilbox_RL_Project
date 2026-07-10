# -*- coding: utf-8 -*-
"""Reusable SoilGuard electro-osmosis soil-box environment.

The original class-4 script is intentionally written like a lesson notebook.
This module keeps the same environment idea in a small importable package:
reset/step/reward, optional observation noise, optional domain randomization,
evaluation helpers, and a simple rule baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - exercised only on missing deps
    raise ImportError(
        "soilbox_env requires gymnasium. Install dependencies with "
        "`pip install -r requirements.txt` from the soilguard-ppo folder."
    ) from exc


SEED = 42


@dataclass(frozen=True)
class SoilBoxConfig:
    """Physical and reward parameters for the 3x3 uniformity soil box."""

    grid_rows: int = 3
    grid_cols: int = 3
    n_cells: int = 9
    max_steps: int = 180
    box_length_cm: float = 30.0
    box_width_cm: float = 30.0
    box_height_cm: float = 30.0
    cell_length_cm: float = 10.0
    cell_width_cm: float = 10.0
    cell_height_cm: float = 30.0
    electrode_layout: str = "four_side_3_by_3_uniformity"
    left_electrode_count: int = 3
    right_electrode_count: int = 3
    front_electrode_count: int = 3
    back_electrode_count: int = 3
    top_electrode_count: int = 0
    bottom_electrode_count: int = 0
    total_electrode_count: int = 12
    independent_power_sources: bool = True
    electrode_material: str = "graphite_or_carbon"
    voltage_gradient_v_per_cm: float = 1.0
    max_voltage_gradient_v_per_cm: float = 1.0
    max_electrode_voltage_v: float = 20.0
    voltage_gradient_levels: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5)
    effective_voltage_ratio: float = 0.656
    contact_resistance_growth: float = 0.005
    corrosion_rate: float = 0.01
    gas_blocking_rate: float = 0.0
    anode_water_injection: bool = False
    anode_water_injection_amount: float = 0.015
    anode_water_injection_interval_steps: int = 12
    v_max: float = 1.0
    electroosmotic_permeability_cm2_per_s_v: float = 3.775e-5
    time_step_s: float = 3600.0
    # Matric-suction moisture diffusion between adjacent cells, per step:
    # ``diffused = moisture_diffusion_per_step x (theta_wet - theta_dry)`` on each
    # internal edge. Unsaturated soil redistributes water down the moisture
    # gradient on its own (suction is higher in dry soil); electro-osmosis does
    # the coarse transport and this term does the slow smoothing.
    # Physically grounded, not tuned for RL: the dimensionless per-step factor is
    # ``k = D_w x dt / dx^2 = D_w x 3600 s / (10 cm)^2 = 36 x D_w``, and soil-water
    # diffusivity D_w for soft clay at moderate-to-high moisture is ~1e-5..1e-4
    # cm^2/s (Richards-equation literature), giving k ~ 0.0004..0.0036. The default
    # 0.003 corresponds to D_w ~ 8.3e-5 cm^2/s, a typical wet soft clay. Constant-D
    # is itself a simplification: real D_w(theta) varies by orders of magnitude.
    # Sanity check (not the justification): at this value diffusion alone cannot
    # reach the uniformity target within an episode (idle best CV ~ 0.40 >> 0.15)
    # while electro-osmosis + diffusion can.
    moisture_diffusion_per_step: float = 0.003
    # Half-saturation moisture for the pore-water conductivity factor
    # ``m / (m + m_threshold)``. Reused by BOTH water mobility and pH current so
    # the moisture dependence is defined once (see ``_moisture_conductivity``).
    m_threshold: float = 0.12
    # --- pH electrolysis + acid/base transport kinetics (see step()) ---
    # Acid (H+) is generated at anodic electrode-contact cells, base (OH-) at
    # cathodic ones; both then migrate and neutralize. The acid front advances
    # faster and farther than the base front (H+ has higher mobility and the
    # electro-osmotic flow drags the base front back), so the treated soil ends up
    # net-acidic (Asadi 2013). This is lumped into two asymmetries: acid is both
    # generated more (k_acid_gen > k_base_gen) and migrated faster (k_acid_mig >
    # k_base_mig). Charge is deliberately not strictly balanced here.
    # Units: per SECOND of simulated time, per unit of normalized implicit
    # current — usage sites multiply by ``time_step_s``, so pH kinetics scale
    # consistently with water transport if the step length ever changes. At the
    # default 3600 s step these equal the historical per-step values
    # (0.10 / 0.06 / 0.06 / 0.03). They remain simulation assumptions, not
    # literature rate constants.
    k_acid_gen_per_s: float = 0.10 / 3600.0
    k_base_gen_per_s: float = 0.06 / 3600.0
    k_acid_mig_per_s: float = 0.06 / 3600.0
    k_base_mig_per_s: float = 0.03 / 3600.0
    # Net acid/base amount that maps to pH ~4.5 / ~9.5 (saturating pH curve).
    ph_half_amount: float = 0.5
    ph_tol: float = 1.0
    # How the pH penalty is charged (see _reward):
    #   "stock" (v13-v19): every step pays w_ph * max-over-cells |pH-7| beyond
    #       tolerance. The acid debt is irreversible within an episode, so a
    #       one-off electrolysis event is re-billed on every remaining step
    #       (~150x) — this persistent bleed is what erodes the transport policy
    #       (v17/v19: PPO weakens coordinated pushes to cut the pH stream).
    #   "acid_increment" (v20): pay only for the POSITIVE step-over-step
    #       increase in the worst cell's ACID-side violation, i.e. bill the act
    #       of generating fresh net acid exactly once, on the step it happens.
    #       Acid diluted/neutralised later gives 0 (max(0,.) blocks a
    #       "make-then-remove" refund loop); holding uniformity with no new acid
    #       is free. Base (pH>7) is not charged here — net acidification is the
    #       physical main axis (Asadi 2013); a base-side term is left for later.
    #       Because this accumulates ~1/horizon of the stock total, w_ph is
    #       recalibrated per reward_audit.py before training.
    ph_penalty_mode: str = "stock"
    # In acid_increment mode, add this small fraction of the stock pH exposure
    # to keep training aligned with the evaluation ph_exposure metric.
    ph_stock_penalty_fraction: float = 0.05
    # ===== v5.0 per-step quality scoring (SHARED by reward and evaluation) =====
    # Every step is graded like a mini-exam: q = 0.60*U + 0.20*P + 0.20*E, each
    # component in [0, 1]. The evaluation score is 100 * mean(q) over the
    # episode; the training reward pays (100/max_steps)*q per step — so the
    # undiscounted episode return EQUALS the exam score by construction, and
    # the training-vs-exam alignment gap is eliminated structurally.
    # Level curves are PIECEWISE-LINEAR breakpoints ((x, score), np.interp,
    # clamped at both ends): continuous everywhere with region-wise slopes —
    # graded "levels" without the zero-gradient plateaus / jumps of a staircase
    # (staircase thresholds are what blinded the critic in v12/v15).
    #
    # U(cv): full credit at cv<=0.10; the teacher's working band 0.08-0.15
    # scores 0.8-1.0; idle's standing 0.3-0.5 scores ~0-0.2 (kills both the
    # lazy and the delay-sprint degenerate optima via the time integral).
    uniformity_level_points: tuple[tuple[float, float], ...] = (
        (0.10, 1.0), (0.15, 0.8), (0.25, 0.4), (0.40, 0.0))
    # P(viol): viol = ph_peak_weight * worst-cell violation + (1-w) * mean
    # violation (stock, symmetric, beyond ph_tol). The physically unavoidable
    # teacher acid (~0.55 mixed) lands mid-curve -> gradients exist BOTH ways
    # (the old hard 1.0 zero-line sat exactly on the physical floor and made
    # pH an unwinnable axis).
    ph_level_points: tuple[tuple[float, float], ...] = (
        (0.0, 1.0), (0.5, 0.8), (1.0, 0.45), (2.0, 0.0))
    # Peak-vs-mean mix for the pH stock violation: peak guards the worst cell
    # (local damage is damage), mean gives dense gradients (dilution/
    # neutralisation anywhere counts). User decision 2026-07-09: 0.7 peak.
    ph_peak_weight: float = 0.7
    # E(power): E = clip01(1 - power/energy_power_ref), where power is the
    # PHYSICAL per-step power proxy: mean over the 6 electrode pairs of
    # conductivity(row/col mean moisture) * (pair voltage ratio)^2 — i.e.
    # P = U*I with I ~ conductivity*U (wet soil conducts more, costs more).
    # Reference calibrated so the teacher's mean power scores ~0.9
    # (measured teacher mean power_phys = 0.00716 on the 20-scenario subset).
    energy_power_ref: float = 0.0716
    w_flux: float = 0.0
    # Optional shaping hook (v13 lesson: use the beyond-diffusion EXCESS, never
    # raw improvement). Default 0 = pure "reward == exam".
    w_uniformity_goal: float = 0.0
    # ---- legacy weights below: NO LONGER used by the v5.0 reward. Kept as
    # config fields so historical experiment reward_overrides still resolve.
    w_uniformity: float = 8.0
    w_energy: float = 0.005
    w_ph: float = 1.2
    w_anode_dry: float = 0.0
    # Out of the user's three axes since v4.0; reporting-only default 0.
    w_local_dry: float = 0.0
    w_voltage_jump: float = 0.0
    # Reporting-only: used for the reported success / reached / hold_frac
    # metrics, not by any reward or score term.
    target_moisture_cv: float = 0.15
    # Retired to 0 (v4.0): no target-line income of any kind.
    success_bonus: float = 0.0
    hold_bonus: float = 0.0
    # Episodes always run max_steps; the objective is the integral of per-step
    # quality over the full horizon, identical to the evaluation.
    terminate_on_success: bool = False
    anode_limiting_moisture: float = 0.22
    local_dry_limiting_moisture: float = 0.22
    m_wet: float = 0.35
    m_dry: float = 0.10

    def with_overrides(self, **overrides: Any) -> "SoilBoxConfig":
        return replace(self, **overrides)


DEFAULT_CONFIG = SoilBoxConfig()


def max_electrode_voltage(config: SoilBoxConfig | Mapping[str, Any] | None = None) -> float:
    """Maximum absolute voltage for each physical electrode in volts."""

    cfg = _coerce_config(config)
    return float(cfg.max_electrode_voltage_v)

# Battery v4.0 exam weights: uniformity dominant, pH and energy equal.
# (v3.0 was 0.40/0.25/0.20/0.15 over uniformity/pH/local_dry/energy plus a
# separate 0.40 reached-gate — all superseded.)
FINAL_SCORE_WEIGHTS: dict[str, float] = {
    "uniformity_score": 0.60,
    "ph_score": 0.20,
    "energy_score": 0.20,
}

FINAL_SCORE_NORMALIZATION: dict[str, float] = {
    "energy_good": 20.0,
    "energy_bad": 60.0,
    "ph_viol_bad": 0.5,
    "local_dry_viol_bad": 0.05,
    "moisture_cv_good": 0.10,
    "moisture_cv_bad": 0.45,
}

DEFAULT_DOMAIN_RANGES: dict[str, tuple[float, float]] = {
    "electroosmotic_permeability_cm2_per_s_v": (1.51e-5, 6.04e-5),
    "m_dry": (0.04, 0.16),
    "m_threshold": (0.08, 0.18),
    "effective_voltage_ratio": (0.566, 0.656),
    # Physical clay range (k = 36 x D_w, D_w ~ 5.6e-5..1e-4 cm^2/s) intersected
    # with the range where the task stays well-posed (idle fails, EO wins).
    "moisture_diffusion_per_step": (0.002, 0.0036),
    "obs_noise": (0.0, 0.2),
}


MATERIAL_EFFECTS: dict[str, dict[str, float]] = {
    "composite_carbon_fiber": {
        "ratio_multiplier": 1.12,
        "corrosion_rate": 0.0002,
        "contact_loss": 0.08,
    },
    "graphite_or_carbon": {
        "ratio_multiplier": 1.05,
        "corrosion_rate": 0.0005,
        "contact_loss": 0.10,
    },
    "steel": {
        "ratio_multiplier": 0.82,
        "corrosion_rate": 0.0040,
        "contact_loss": 0.20,
    },
    "copper": {
        "ratio_multiplier": 0.76,
        "corrosion_rate": 0.0055,
        "contact_loss": 0.24,
    },
    "aluminum": {
        "ratio_multiplier": 0.62,
        "corrosion_rate": 0.0090,
        "contact_loss": 0.32,
    },
}


PolicyFn = Callable[[np.ndarray], np.ndarray]


def _coerce_config(config: SoilBoxConfig | Mapping[str, Any] | None) -> SoilBoxConfig:
    if config is None:
        return DEFAULT_CONFIG
    if isinstance(config, SoilBoxConfig):
        return config
    return SoilBoxConfig(**dict(config))


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _low_is_good_score(value: float, good: float, bad: float) -> float:
    if bad <= good:
        raise ValueError(f"Bad threshold must be greater than good threshold: {good=} {bad=}")
    return _clip01((bad - value) / (bad - good))


def _level_score(value: float, points: tuple[tuple[float, float], ...]) -> float:
    """Piecewise-linear level curve: interpolate score over (x, score) breakpoints.

    Continuous everywhere (region-wise slopes, no staircase jumps); clamped to
    the first/last breakpoint score outside the range. Breakpoint x values
    must be increasing.
    """

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return float(np.interp(value, xs, ys))


class SoilBoxEnv(gym.Env):
    """Closed 3x3 electro-osmosis soil box for moisture *redistribution*.

    This is a sealed box: total water is conserved and only redistributed
    between the nine cells (no cathode drainage). The goal is to make the
    moisture field uniform. Two transport mechanisms act each step:
    electro-osmosis (voltage-driven, the agent's lever) and a slow
    matric-suction diffusion (moisture-gradient-driven, always on) that lets
    interior cells fill up — without it, a wall-electrode layout cannot make
    any interior cell a source/sink and uniformity is unreachable.

    Observation:
        ``[moisture_00..moisture_22, normalized_pH_00..normalized_pH_22]``.
        pH is normalized as ``(pH - 7) / 7``.

    Action:
        twelve normalized side-electrode commands in [-1, 1]:
        ``[L0,L1,L2,R0,R1,R2,F0,F1,F2,B0,B1,B2]``. The left/right electrodes
        drive the horizontal (x) field row by row; the front/back electrodes
        drive the vertical (y) field column by column. The environment converts
        commands to physical electrode voltages with
        ``voltage = command * max_electrode_voltage_v``. The two axes are
        decoupled (see ``step``), so an idle axis never weakens the active one.

    Reward:
        moisture-CV improvement and low final CV, minus energy, pH, local
        drying, and optional voltage-jump penalties.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: SoilBoxConfig | Mapping[str, Any] | None = None,
        *,
        obs_noise: float = 0.0,
        randomize: bool = False,
        randomize_noise: bool = False,
        randomize_initial: bool = False,
        domain_ranges: Mapping[str, tuple[float, float]] | None = None,
        seed: int = SEED,
        curriculum_difficulty: float | None = None,
        curriculum_min_contrast: float = 0.6,
    ) -> None:
        super().__init__()
        self.base_config = _coerce_config(config)
        self.cfg = self.base_config
        if self.cfg.n_cells != self.cfg.grid_rows * self.cfg.grid_cols:
            raise ValueError("n_cells must equal grid_rows * grid_cols")
        self.n = self.cfg.n_cells
        self.max_electrode_voltage_v = max_electrode_voltage(self.cfg)
        self.obs_noise = float(obs_noise)
        self.randomize = bool(randomize)
        self.randomize_noise = bool(randomize_noise)
        self.randomize_initial = bool(randomize_initial)
        # Curriculum (v27): None = off (full training dice, historical
        # behaviour). A float in [0, 1] reshapes the initial-field dice from
        # easy to hard — see _sample_initial_field. Takes effect on reset.
        self.curriculum_difficulty = (
            None if curriculum_difficulty is None else float(np.clip(curriculum_difficulty, 0.0, 1.0))
        )
        self.curriculum_min_contrast = float(curriculum_min_contrast)
        self.domain_ranges = dict(DEFAULT_DOMAIN_RANGES)
        if domain_ranges:
            self.domain_ranges.update(domain_ranges)

        self._base_seed = seed
        self._rng = np.random.default_rng(seed)
        self._init_spread = 0.0
        self._init_total_water = 0.0
        self.prev_total_water = 0.0
        self.prev_spread = 0.0
        self.prev_moisture_cv = 0.0
        self.prev_action = np.zeros(self.cfg.total_electrode_count, dtype=np.float64)
        self.prev_normalized_action = np.zeros(self.cfg.total_electrode_count, dtype=np.float64)
        self.steps = 0
        self._reached_target = False
        self.m = np.zeros(self.n, dtype=np.float64)
        # Acid (H+) and base (OH-) amounts per cell; pH is derived from the net.
        self.acid = np.zeros(self.n, dtype=np.float64)
        self.base = np.zeros(self.n, dtype=np.float64)
        self.pH = np.full(self.n, 7.0, dtype=np.float64)
        # Previous step's worst-cell acid-side pH violation, for the
        # "acid_increment" penalty mode (charges only positive step increases).
        self._prev_ph_acid_viol = 0.0

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.cfg.total_electrode_count,),
            dtype=np.float32,
        )
        obs_low = np.array([0.0] * self.n + [-1.0] * self.n, dtype=np.float32)
        obs_high = np.array([0.5] * self.n + [1.0] * self.n, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

    def _sample_domain(self) -> None:
        updates: dict[str, float] = {}
        for key in (
            "electroosmotic_permeability_cm2_per_s_v",
            "m_dry",
            "m_threshold",
            "effective_voltage_ratio",
            "moisture_diffusion_per_step",
        ):
            if key in self.domain_ranges:
                lo, hi = self.domain_ranges[key]
                updates[key] = float(self._rng.uniform(lo, hi))
        self.cfg = self.base_config.with_overrides(**updates)

    def _sample_noise(self) -> None:
        lo, hi = self.domain_ranges.get("obs_noise", (0.0, 0.2))
        self.obs_noise = float(self._rng.uniform(lo, hi))

    def set_curriculum_difficulty(self, difficulty: float) -> float:
        """Set curriculum difficulty in [0, 1]; applies from the next reset.

        Called by the curriculum callback via VecEnv.env_method so all
        parallel workers advance together. Returns the clamped value.
        """

        self.curriculum_difficulty = float(np.clip(difficulty, 0.0, 1.0))
        return self.curriculum_difficulty

    def _sample_initial_field(self) -> np.ndarray:
        """Draw a fresh initial moisture field (training-time randomization).

        Default dice (curriculum_difficulty=None), deliberately WIDER than and
        independent from the frozen evaluation battery: 40% structured wet
        patterns (column/row/half/diagonal/cells), 30% linear ramps in a
        random direction, 30% iid random fields. A CV floor rejects
        near-uniform draws so every episode is a real task.

        Curriculum mode (v27): ``curriculum_difficulty`` d in [0, 1] reshapes
        the dice along two axes, easy -> hard:
          - pattern pool: d < 1/3 only 1-D column/row wet fields; d < 2/3
            adds the remaining structured patterns and ramps; d >= 2/3 is the
            full default dice including iid fields.
          - contrast: the accepted grid is scaled toward its mean by
            ``c = min_contrast + (1 - min_contrast) * d`` (total water
            preserved, initial CV scales with c), so easy fields start closer
            to the target and the hold wage is sampled early.
        The CV floor is enforced on the SCALED grid — no episode may start
        near/below the success target, otherwise easy stages would teach
        idling (the diffusion-solves-it trap: the wage does not ask HOW the
        field became uniform). min_contrast must be calibrated so the reward
        audit's ordering (proportional > idle) holds at EVERY difficulty.
        """

        rows, cols = self.cfg.grid_rows, self.cfg.grid_cols
        rng = self._rng
        d = self.curriculum_difficulty
        for _ in range(20):
            wet = float(rng.uniform(0.25, 0.45))
            dry = float(rng.uniform(0.06, 0.14))
            mode = float(rng.random())
            if d is not None:
                # Curriculum pattern-pool gating: remap the mode roll so easy
                # stages only see the simple end of the distribution.
                if d < 1.0 / 3.0:
                    mode = 0.0      # structured only (pattern further gated below)
                elif d < 2.0 / 3.0:
                    mode = mode * 0.7   # structured + ramps, no iid fields
            if mode < 0.4:
                grid = np.full((rows, cols), dry)
                if d is not None and d < 1.0 / 3.0:
                    pattern = int(rng.integers(0, 2))   # columns/rows only (1-D tasks)
                else:
                    pattern = int(rng.integers(0, 5))
                if pattern == 0:
                    grid[:, int(rng.integers(0, cols))] = wet
                elif pattern == 1:
                    grid[int(rng.integers(0, rows)), :] = wet
                elif pattern == 2:
                    if rng.random() < 0.5:
                        grid[:, : cols // 2 + 1] = wet
                    else:
                        grid[: rows // 2 + 1, :] = wet
                elif pattern == 3:
                    idx = np.arange(min(rows, cols))
                    if rng.random() < 0.5:
                        grid[idx, idx] = wet
                    else:
                        grid[idx, cols - 1 - idx] = wet
                else:
                    k = int(rng.integers(1, 4))
                    cells = rng.choice(rows * cols, size=k, replace=False)
                    grid.reshape(-1)[cells] = wet
            elif mode < 0.7:
                xs, ys = np.meshgrid(np.linspace(0, 1, cols), np.linspace(0, 1, rows))
                theta = float(rng.uniform(0.0, 2.0 * np.pi))
                ramp = np.cos(theta) * xs + np.sin(theta) * ys
                ramp = (ramp - ramp.min()) / max(float(ramp.max() - ramp.min()), 1e-9)
                grid = dry + (wet - dry) * ramp
            else:
                grid = rng.uniform(0.08, 0.42, size=(rows, cols))
            if d is not None:
                contrast = self.curriculum_min_contrast + (1.0 - self.curriculum_min_contrast) * d
                grid = float(np.mean(grid)) + contrast * (grid - float(np.mean(grid)))
            grid = np.clip(grid, 0.02, 0.5)
            if float(np.std(grid) / (np.mean(grid) + 1e-9)) >= 0.25:
                return grid
        # Extremely unlucky streak of near-uniform draws: fall back to the
        # canonical left-wet field, which is always well-posed.
        grid = np.full((rows, cols), self.cfg.m_dry)
        grid[:, 0] = self.cfg.m_wet
        return grid

    def _get_obs(self) -> np.ndarray:
        obs = np.concatenate([self.m, (self.pH - 7.0) / 7.0])
        if self.obs_noise > 0.0:
            obs = obs + self._rng.normal(0.0, self.obs_noise, size=obs.shape)
        return np.clip(obs, self.observation_space.low, self.observation_space.high).astype(np.float32)

    def spread(self) -> float:
        return float(np.sum(np.abs(self.m - self.m.mean())))

    def moisture_cv(self) -> float:
        mean = float(np.mean(self.m))
        return float(np.std(self.m) / (mean + 1e-9))

    def removable_water(self) -> float:
        return float(max(self._init_total_water - self.n * self.cfg.m_dry, 1e-9))

    def nominal_applied_voltage(self) -> float:
        """Voltage across the full box at the configured nominal gradient."""

        return float(self.cfg.voltage_gradient_v_per_cm * self.cfg.box_length_cm)

    def x_boundary_area_cm2(self) -> float:
        """Cell-face area for x-direction flow."""

        return float(self.cfg.cell_width_cm * self.cfg.cell_height_cm)

    def y_boundary_area_cm2(self) -> float:
        """Cell-face area for y-direction flow."""

        return float(self.cfg.cell_length_cm * self.cfg.cell_height_cm)

    def effective_cross_section_area_cm2(self) -> float:
        """Representative cell-face area kept for diagnostics compatibility."""

        return float(0.5 * (self.x_boundary_area_cm2() + self.y_boundary_area_cm2()))

    def cell_volume_ml(self) -> float:
        """Approximate cell volume; 1 cm3 of water is treated as 1 mL."""

        return float(self.cfg.cell_length_cm * self.cfg.cell_width_cm * self.cfg.cell_height_cm)

    def _grid(self, values: np.ndarray) -> np.ndarray:
        return values.reshape(self.cfg.grid_rows, self.cfg.grid_cols)

    def _material_effects(self) -> dict[str, float]:
        return MATERIAL_EFFECTS.get(self.cfg.electrode_material, MATERIAL_EFFECTS["graphite_or_carbon"])

    def _effective_voltage_ratio(self, action: np.ndarray) -> float:
        """Estimate the fraction of applied voltage that remains active in soil.

        The model is intentionally compact: dry electrode contact, corrosion, gas
        blocking, and material choice reduce the useful field before it drives
        electro-osmotic flow.
        """

        cfg = self.cfg
        material = self._material_effects()
        moisture_span = max(cfg.m_wet - cfg.m_dry, 1e-6)
        wettest_contact = np.clip((float(np.max(self.m)) - cfg.m_dry) / moisture_span, 0.0, 1.0)
        contact_factor = 1.0 - cfg.contact_resistance_growth * self.steps
        contact_factor *= 0.75 + 0.25 * wettest_contact

        corrosion_rate = max(cfg.corrosion_rate, material["corrosion_rate"])
        corrosion_factor = 1.0 - corrosion_rate * self.steps
        normalized_load = float(np.mean(np.abs(action)) / max(self.max_electrode_voltage_v, 1e-9))
        gas_factor = 1.0 - cfg.gas_blocking_rate * normalized_load * self.steps
        ratio = cfg.effective_voltage_ratio * material["ratio_multiplier"] * contact_factor
        ratio *= corrosion_factor * gas_factor
        return float(np.clip(ratio, 0.05, 0.95))

    def _split_electrodes(self, electrode_action: np.ndarray) -> tuple[np.ndarray, ...]:
        rows = self.cfg.grid_rows
        cols = self.cfg.grid_cols
        left = electrode_action[0:rows]
        right = electrode_action[rows : 2 * rows]
        front = electrode_action[2 * rows : 2 * rows + cols]
        back = electrode_action[2 * rows + cols : 2 * rows + 2 * cols]
        return left, right, front, back

    def _axis_gradients(self, electrode_action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Nominal (pre-ratio) voltage gradients per row (x) and per column (y).

        Each row is a 1-D resistor between its left and right electrodes; each
        column between its back and front electrodes. In a uniform linear field
        the gradient is constant along the span, so a single value per row/column
        applies to every internal edge on that line::

            grad_x[i] = (left[i]  - right[i]) / box_length_cm   # V/cm, row i
            grad_y[j] = (back[j]  - front[j]) / box_width_cm     # V/cm, column j

        Dividing by the full box span (not the cell length) keeps the geometry
        consistent: with left=+15 V, right=-15 V the row gradient is exactly the
        nominal 1 V/cm. The x and y axes are kept separate on purpose so an idle
        axis contributes 0 instead of halving the active axis.
        """

        left, right, front, back = self._split_electrodes(electrode_action)
        grad_x = (left - right) / max(self.cfg.box_length_cm, 1e-9)
        grad_y = (back - front) / max(self.cfg.box_width_cm, 1e-9)
        return grad_x.astype(np.float64), grad_y.astype(np.float64)

    def _diffusion_deltas(self, moisture_grid: np.ndarray) -> np.ndarray:
        """Per-step matric-suction diffusion deltas for the given field.

        Called twice per step: once as part of the real update, and once alone
        on the pre-step field to build the diffusion-only counterfactual that
        the uniformity reward is measured against (see step / _reward).
        """

        cfg = self.cfg
        deltas = np.zeros_like(moisture_grid)
        k_diff = cfg.moisture_diffusion_per_step
        if k_diff <= 0.0:
            return deltas
        for i in range(cfg.grid_rows):
            for j in range(cfg.grid_cols - 1):
                seep = k_diff * (moisture_grid[i, j] - moisture_grid[i, j + 1])
                deltas[i, j] -= seep
                deltas[i, j + 1] += seep
        for i in range(cfg.grid_rows - 1):
            for j in range(cfg.grid_cols):
                seep = k_diff * (moisture_grid[i, j] - moisture_grid[i + 1, j])
                deltas[i, j] -= seep
                deltas[i + 1, j] += seep
        return deltas

    def _moisture_conductivity(self, m_avg: float) -> float:
        """Pore-water continuity factor ``m / (m + m_threshold)`` in [0, 1).

        Phenomenological, not a literature coefficient: the drier the soil the
        less water/charge it can carry. Shared by water mobility and pH current
        so the moisture dependence lives in one place.
        """

        return float(m_avg / (m_avg + self.cfg.m_threshold))

    def _electrode_contact_dry_violation(self, electrode_action: np.ndarray) -> float:
        """Dryness penalty only for active high-potential electrode contact cells.

        The 0.22 moisture threshold is a contact/electrode-region limit from the
        electro-osmosis literature. Polarity is decided per electrode pair
        (left-vs-right per row, back-vs-front per column), consistent with the
        decoupled field model and with acid/base generation: only the anodic end
        of each driven pair injects current into its contact cell, so only that
        cell's dryness is penalized, weighted by the pair voltage.
        """

        cfg = self.cfg
        moisture_grid = self._grid(self.m)
        left, right, front, back = self._split_electrodes(electrode_action)
        deficits: list[float] = []
        weights: list[float] = []

        def check_pair(v_a: float, cell_a: tuple[int, int], v_b: float, cell_b: tuple[int, int]) -> None:
            drive = float(v_a) - float(v_b)
            if abs(drive) <= 1e-9:
                return
            anode_cell = cell_a if drive > 0.0 else cell_b
            deficits.append(max(0.0, cfg.local_dry_limiting_moisture - moisture_grid[anode_cell]))
            weights.append(abs(drive))

        for i in range(cfg.grid_rows):
            check_pair(left[i], (i, 0), right[i], (i, cfg.grid_cols - 1))
        for j in range(cfg.grid_cols):
            check_pair(back[j], (0, j), front[j], (cfg.grid_rows - 1, j))

        if not weights:
            return 0.0
        return float(np.average(deficits, weights=weights))

    def _transport_from_gradient(
        self,
        effective_gradient: float,
        boundary_area_cm2: float,
    ) -> tuple[float, float, float]:
        """Return signed internal transport for one 2D cell boundary."""

        cfg = self.cfg
        flow_ml_per_s = (
            cfg.electroosmotic_permeability_cm2_per_s_v
            * boundary_area_cm2
            * effective_gradient
        )
        transport_ml = flow_ml_per_s * cfg.time_step_s
        transport_fraction = transport_ml / max(self.cell_volume_ml(), 1e-9)
        return float(flow_ml_per_s), float(transport_ml), float(transport_fraction)

    def _inject_anode_water(self) -> float:
        cfg = self.cfg
        if not cfg.anode_water_injection:
            return 0.0
        interval = max(int(cfg.anode_water_injection_interval_steps), 1)
        if self.steps % interval != 0:
            return 0.0
        before = float(self.m[0])
        self.m[0] = min(0.5, self.m[0] + cfg.anode_water_injection_amount)
        return float(self.m[0] - before)

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.cfg = self.base_config
        if self.randomize:
            self._sample_domain()
        if self.randomize_noise:
            self._sample_noise()

        # Initial-field priority: explicit injection (evaluation battery) beats
        # the training dice, which beat the canonical left-wet default.
        if options and "initial_moisture" in options:
            m_grid = np.asarray(options["initial_moisture"], dtype=np.float64).reshape(
                self.cfg.grid_rows, self.cfg.grid_cols
            )
            if np.any(m_grid < 0.0) or np.any(m_grid > 0.5):
                raise ValueError("initial_moisture values must lie in [0, 0.5]")
        elif self.randomize_initial:
            m_grid = self._sample_initial_field()
        else:
            m_grid = np.full((self.cfg.grid_rows, self.cfg.grid_cols), self.cfg.m_dry, dtype=np.float64)
            m_grid[:, 0] = self.cfg.m_wet
        self.m = m_grid.reshape(-1)
        self.acid = np.zeros(self.n, dtype=np.float64)
        self.base = np.zeros(self.n, dtype=np.float64)
        self.pH = np.full(self.n, 7.0, dtype=np.float64)
        self._prev_ph_acid_viol = 0.0
        self.steps = 0
        self._reached_target = False
        self._init_spread = self.spread()
        self._init_total_water = float(np.sum(self.m))
        self.prev_total_water = self._init_total_water
        self.prev_spread = self._init_spread
        self.prev_moisture_cv = self.moisture_cv()
        self.prev_action = np.zeros(self.cfg.total_electrode_count, dtype=np.float64)
        self.prev_normalized_action = np.zeros(self.cfg.total_electrode_count, dtype=np.float64)
        return self._get_obs(), {}

    def step(self, action):
        cfg = self.cfg
        normalized_action = np.clip(
            np.asarray(action, dtype=np.float64),
            -1.0,
            1.0,
        )
        if normalized_action.shape != (cfg.total_electrode_count,):
            raise ValueError(
                f"Expected action shape ({cfg.total_electrode_count},), got {normalized_action.shape}"
            )
        electrode_action = normalized_action * self.max_electrode_voltage_v
        injected_water = self._inject_anode_water()
        effective_voltage_ratio = self._effective_voltage_ratio(electrode_action)
        # Decoupled fields: one gradient per row (x) and per column (y).
        raw_grad_x, raw_grad_y = self._axis_gradients(electrode_action)
        max_grad = cfg.max_voltage_gradient_v_per_cm
        moisture_grid = self._grid(self.m)
        acid_grid = self._grid(self.acid).copy()
        base_grid = self._grid(self.base).copy()
        dm_grid = np.zeros_like(moisture_grid)
        d_acid = np.zeros_like(moisture_grid)
        d_base = np.zeros_like(moisture_grid)
        flux_activity = 0.0
        x_flux = np.zeros((cfg.grid_rows, cfg.grid_cols - 1), dtype=np.float64)
        y_flux = np.zeros((cfg.grid_rows - 1, cfg.grid_cols), dtype=np.float64)
        x_grad = np.zeros_like(x_flux)
        y_grad = np.zeros_like(y_flux)
        x_transport_ml = np.zeros_like(x_flux)
        y_transport_ml = np.zeros_like(y_flux)

        # --- x-direction edges: water transport + acid/base migration ---
        for i in range(cfg.grid_rows):
            gradient = float(np.clip(effective_voltage_ratio * raw_grad_x[i], -max_grad, max_grad))
            for j in range(cfg.grid_cols - 1):
                m_avg = 0.5 * (moisture_grid[i, j] + moisture_grid[i, j + 1])
                conductivity = self._moisture_conductivity(m_avg)
                _, transport_ml, transport_fraction = self._transport_from_gradient(
                    gradient, self.x_boundary_area_cm2()
                )
                flux = transport_fraction * conductivity
                if flux > 0:
                    flux = min(flux, 0.8 * moisture_grid[i, j])
                else:
                    flux = max(flux, -0.8 * moisture_grid[i, j + 1])
                dm_grid[i, j] -= flux
                dm_grid[i, j + 1] += flux
                x_flux[i, j] = flux
                x_grad[i, j] = gradient
                x_transport_ml[i, j] = transport_ml * conductivity
                flux_activity += abs(flux)
                self._migrate_acid_base(
                    d_acid, d_base, acid_grid, base_grid,
                    (i, j), (i, j + 1), gradient, conductivity,
                )

        # --- y-direction edges: water transport + acid/base migration ---
        for j in range(cfg.grid_cols):
            gradient = float(np.clip(effective_voltage_ratio * raw_grad_y[j], -max_grad, max_grad))
            for i in range(cfg.grid_rows - 1):
                m_avg = 0.5 * (moisture_grid[i, j] + moisture_grid[i + 1, j])
                conductivity = self._moisture_conductivity(m_avg)
                _, transport_ml, transport_fraction = self._transport_from_gradient(
                    gradient, self.y_boundary_area_cm2()
                )
                flux = transport_fraction * conductivity
                if flux > 0:
                    flux = min(flux, 0.8 * moisture_grid[i, j])
                else:
                    flux = max(flux, -0.8 * moisture_grid[i + 1, j])
                dm_grid[i, j] -= flux
                dm_grid[i + 1, j] += flux
                y_flux[i, j] = flux
                y_grad[i, j] = gradient
                y_transport_ml[i, j] = transport_ml * conductivity
                flux_activity += abs(flux)
                self._migrate_acid_base(
                    d_acid, d_base, acid_grid, base_grid,
                    (i, j), (i + 1, j), gradient, conductivity,
                )

        # --- matric-suction diffusion: water seeps from wet to dry cells on its own ---
        # This is the field-free half of unsaturated flow: suction is higher in dry
        # soil, so moisture relaxes down its own gradient regardless of the applied
        # voltage. It is what lets interior cells fill up even though no electrode
        # can make them a potential extremum (maximum principle).
        diffusion_deltas = self._diffusion_deltas(moisture_grid)
        dm_grid += diffusion_deltas

        # Diffusion-only counterfactual: the CV this step would have reached with
        # zero applied field. The uniformity reward pays only for improvement
        # BEYOND this baseline, so idling earns nothing from diffusion's free
        # smoothing (the v12 lazy-collapse fix).
        m_diffusion_only = np.clip((moisture_grid + diffusion_deltas).reshape(-1), 0.0, 0.5)
        cv_diffusion_only = float(
            np.std(m_diffusion_only) / (float(np.mean(m_diffusion_only)) + 1e-9)
        )

        # --- electrolysis: generate acid at anodic / base at cathodic contact cells ---
        self._generate_acid_base(d_acid, d_base, electrode_action, effective_voltage_ratio)

        self.m = np.clip((moisture_grid + dm_grid).reshape(-1), 0.0, 0.5)
        # Apply migration + generation, then neutralize and derive pH.
        acid = np.maximum(0.0, (acid_grid + d_acid).reshape(-1))
        base = np.maximum(0.0, (base_grid + d_base).reshape(-1))
        neutralized = np.minimum(acid, base)
        self.acid = acid - neutralized
        self.base = base - neutralized
        self.pH = self._ph_from_net(self.acid - self.base)
        voltage_jump = float(
            np.mean(np.abs(normalized_action - self.prev_normalized_action)) / 2.0
        )
        self.prev_action = electrode_action.copy()
        self.prev_normalized_action = normalized_action.copy()

        moisture_cv_before = self.prev_moisture_cv
        moisture_cv_after = self.moisture_cv()
        uniformity_improvement = moisture_cv_before - moisture_cv_after
        # Improvement beyond what diffusion alone would have delivered this step.
        # Zero action => exactly 0 (both states share the same diffusion deltas).
        uniformity_excess = cv_diffusion_only - moisture_cv_after

        reward, terms = self._reward(
            electrode_action,
            x_grad,
            y_grad,
            flux_activity,
            effective_voltage_ratio,
            uniformity_improvement,
            uniformity_excess,
            voltage_jump,
        )
        self.steps += 1
        cur_spread = self.spread()
        success = moisture_cv_after <= cfg.target_moisture_cv
        if success and not self._reached_target:
            # One-shot: paid at the FIRST crossing only. Without this guard the
            # no-termination mode would re-pay the bonus on every below-target
            # step (up to ~180x per episode).
            self._reached_target = True
            reward += cfg.success_bonus
        # Per-step hold wage: deliberately NOT one-shot (see config docstring).
        hold_paid = cfg.hold_bonus if success else 0.0
        reward += hold_paid
        terms["hold_bonus"] = float(hold_paid)
        terminated = bool(success and cfg.terminate_on_success)
        truncated = bool(self.steps >= cfg.max_steps)
        self.prev_spread = cur_spread
        self.prev_moisture_cv = moisture_cv_after

        info = {
            "terms": terms,
            "spread": cur_spread,
            "success": success,
            "mean_moisture": float(np.mean(self.m)),
            "moisture_cv": moisture_cv_after,
            "mean_pH": float(np.mean(self.pH)),
            "config": self.cfg,
            "obs_noise": self.obs_noise,
            "nominal_applied_voltage_v": self.nominal_applied_voltage(),
            "max_electrode_voltage_v": self.max_electrode_voltage_v,
            "voltage_drop_v": float(np.max(electrode_action) - np.min(electrode_action)),
            "injected_water": injected_water,
            "electrode_action": electrode_action.astype(float).tolist(),
            "normalized_action": normalized_action.astype(float).tolist(),
            "boundary_action": np.concatenate([x_grad.reshape(-1), y_grad.reshape(-1)]).astype(float).tolist(),
            "effective_cross_section_area_cm2": self.effective_cross_section_area_cm2(),
            "x_boundary_area_cm2": self.x_boundary_area_cm2(),
            "y_boundary_area_cm2": self.y_boundary_area_cm2(),
            "cell_volume_ml": self.cell_volume_ml(),
            "internal_transport_ml": np.concatenate(
                [x_transport_ml.reshape(-1), y_transport_ml.reshape(-1)]
            ).astype(float).tolist(),
            "internal_transport_fraction": np.concatenate(
                [x_flux.reshape(-1), y_flux.reshape(-1)]
            ).astype(float).tolist(),
            "x_flux": x_flux.reshape(-1).astype(float).tolist(),
            "y_flux": y_flux.reshape(-1).astype(float).tolist(),
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def _edge_current(self, gradient: float, conductivity: float) -> float:
        """Normalized implicit current across one internal edge, in [0, 1).

        ``current = |potential drop| x conductivity``. The potential drop is the
        effective field times the cell length, normalized by the maximum drop so
        the current is O(1) and dimensionless; the physical duration of a step
        enters explicitly as ``time_step_s`` at the usage sites (generation and
        migration), matching how water transport handles time.
        """

        cfg = self.cfg
        max_drop = max(cfg.max_voltage_gradient_v_per_cm * cfg.cell_length_cm, 1e-9)
        potential_drop = abs(gradient) * cfg.cell_length_cm / max_drop
        return float(potential_drop * conductivity)

    def _migrate_acid_base(
        self,
        d_acid: np.ndarray,
        d_base: np.ndarray,
        acid_grid: np.ndarray,
        base_grid: np.ndarray,
        cell_a: tuple[int, int],
        cell_b: tuple[int, int],
        gradient: float,
        conductivity: float,
    ) -> None:
        """Move acid toward the cathode and base toward the anode across one edge.

        ``gradient > 0`` means ``cell_a`` is the higher-potential (anodic) side, so
        acid (H+) drifts a->b and base (OH-) drifts b->a; ``gradient < 0`` reverses
        it. Acid migrates with a larger coefficient than base, reproducing the
        faster acid front and the resulting net acidification (Asadi 2013).
        """

        cfg = self.cfg
        current = self._edge_current(gradient, conductivity)
        if current <= 0.0:
            return
        if gradient >= 0.0:
            anode, cathode = cell_a, cell_b
        else:
            anode, cathode = cell_b, cell_a
        dt = cfg.time_step_s
        acid_moved = min(cfg.k_acid_mig_per_s * dt * current * acid_grid[anode], acid_grid[anode])
        base_moved = min(cfg.k_base_mig_per_s * dt * current * base_grid[cathode], base_grid[cathode])
        d_acid[anode] -= acid_moved
        d_acid[cathode] += acid_moved
        d_base[cathode] -= base_moved
        d_base[anode] += base_moved

    def _generate_acid_base(
        self,
        d_acid: np.ndarray,
        d_base: np.ndarray,
        electrode_action: np.ndarray,
        effective_voltage_ratio: float,
    ) -> None:
        """Electrolysis source: acid at anodic contact cells, base at cathodic ones.

        Polarity is decided PER ELECTRODE PAIR, consistent with the decoupled
        field model that drives water: each row is the (left, right) pair, each
        column the (back, front) pair. The higher-potential end of a pair is the
        anode (produces H+ at its contact cell), the other end the cathode
        (produces OH-); both scale with the same pair current, taken as the
        normalized pair voltage times the mean conductivity of the two contact
        cells. A global-mean reference was used before, which could disagree
        with the water-flow direction for asymmetric action patterns.
        """

        cfg = self.cfg
        left, right, front, back = self._split_electrodes(electrode_action)
        moisture_grid = self._grid(self.m)
        max_pair_v = max(2.0 * self.max_electrode_voltage_v, 1e-9)
        dt = cfg.time_step_s

        def emit_pair(v_a: float, cell_a: tuple[int, int], v_b: float, cell_b: tuple[int, int]) -> None:
            drive = float(v_a) - float(v_b)
            if abs(drive) <= 1e-9:
                return
            m_avg = 0.5 * (moisture_grid[cell_a] + moisture_grid[cell_b])
            current = abs(drive) / max_pair_v * self._moisture_conductivity(m_avg) * effective_voltage_ratio
            anode_cell, cathode_cell = (cell_a, cell_b) if drive > 0.0 else (cell_b, cell_a)
            d_acid[anode_cell] += cfg.k_acid_gen_per_s * dt * current
            d_base[cathode_cell] += cfg.k_base_gen_per_s * dt * current

        for i in range(cfg.grid_rows):
            emit_pair(left[i], (i, 0), right[i], (i, cfg.grid_cols - 1))
        for j in range(cfg.grid_cols):
            emit_pair(back[j], (0, j), front[j], (cfg.grid_rows - 1, j))

    def _ph_from_net(self, net: np.ndarray) -> np.ndarray:
        """Map net acid/base amount per cell to a pH in [2, 12] (7 at net 0).

        Positive net (acid surplus) lowers pH; the saturating form keeps pH bounded
        no matter how much acid/base accumulates.
        """

        half = max(self.cfg.ph_half_amount, 1e-9)
        pH = 7.0 - 5.0 * net / (np.abs(net) + half)
        return np.clip(pH, 0.0, 14.0)

    def _reward(
        self,
        electrode_action: np.ndarray,
        x_grad: np.ndarray,
        y_grad: np.ndarray,
        flux_activity: float,
        effective_voltage_ratio: float,
        uniformity_improvement: float,
        uniformity_excess: float,
        voltage_jump: float,
    ) -> tuple[float, dict[str, float]]:
        """Per-step reward.

        The uniformity-goal term pays for ``uniformity_excess`` — CV improvement
        beyond the diffusion-only counterfactual — NOT for raw improvement.
        Diffusion smooths the field for free every step; paying for raw
        improvement let a do-nothing policy collect that free income (v12
        collapsed into exactly that). Raw improvement is still reported in
        ``terms`` for monitoring.
        """
        cfg = self.cfg
        normalized_electrode_action = electrode_action / max(self.max_electrode_voltage_v, 1e-9)
        absolute_voltage_energy = float(np.mean(normalized_electrode_action**2))
        left, right, front, back = self._split_electrodes(electrode_action)
        max_pair_voltage = max(2.0 * self.max_electrode_voltage_v, 1e-9)
        pair_voltage_ratios = np.concatenate([(left - right), (back - front)]) / max_pair_voltage
        pair_voltage_energy = float(np.mean(pair_voltage_ratios**2))
        energy = pair_voltage_energy
        boundary_values = np.concatenate([x_grad.reshape(-1), y_grad.reshape(-1)])
        field_energy = float(np.mean(boundary_values**2))
        effective_energy = field_energy
        voltage_loss = float(max(0.0, 1.0 - effective_voltage_ratio))

        # --- physical per-step power (v5.0 energy axis): P = U*I per pair,
        # I ~ conductivity(row/col mean moisture) * U. A row is one series
        # circuit, so the row MEAN moisture sets its conductivity; wet soil
        # conducts more and genuinely costs more at the same voltage.
        moisture_grid = self._grid(self.m)
        pair_powers: list[float] = []
        for i in range(cfg.grid_rows):
            ratio = float(left[i] - right[i]) / max_pair_voltage
            conductivity = self._moisture_conductivity(float(np.mean(moisture_grid[i, :])))
            pair_powers.append(conductivity * ratio * ratio)
        for j in range(cfg.grid_cols):
            ratio = float(back[j] - front[j]) / max_pair_voltage
            conductivity = self._moisture_conductivity(float(np.mean(moisture_grid[:, j])))
            pair_powers.append(conductivity * ratio * ratio)
        power_phys = float(np.mean(pair_powers))

        # --- pH stock violations (symmetric, beyond tolerance) ---
        cell_violations = np.maximum(0.0, np.abs(self.pH - 7.0) - cfg.ph_tol)
        ph_violation = float(np.max(cell_violations))          # worst cell (historic metric)
        ph_violation_mean = float(np.mean(cell_violations))    # whole-box mean
        ph_violation_mixed = float(
            cfg.ph_peak_weight * ph_violation
            + (1.0 - cfg.ph_peak_weight) * ph_violation_mean
        )
        # Acid-increment bookkeeping retained for monitoring only (v5.0 reward
        # charges the stock via the level curve; increment no longer priced).
        ph_acid_viol = float(np.max(np.maximum(0.0, (7.0 - self.pH) - cfg.ph_tol)))
        ph_acid_increment = max(0.0, ph_acid_viol - self._prev_ph_acid_viol)
        self._prev_ph_acid_viol = ph_acid_viol

        moisture_cv = self.moisture_cv()
        moisture_cv_gap = float(max(0.0, moisture_cv - cfg.target_moisture_cv))
        local_dry_violation = self._electrode_contact_dry_violation(electrode_action)
        anode_dry_violation = local_dry_violation

        # --- v5.0 per-step quality: the mini-exam this step, in [0, 1] ---
        u_score = _level_score(moisture_cv, cfg.uniformity_level_points)
        ph_score = _level_score(ph_violation_mixed, cfg.ph_level_points)
        energy_score = _clip01(1.0 - power_phys / max(cfg.energy_power_ref, 1e-9))
        step_quality = (
            FINAL_SCORE_WEIGHTS["uniformity_score"] * u_score
            + FINAL_SCORE_WEIGHTS["ph_score"] * ph_score
            + FINAL_SCORE_WEIGHTS["energy_score"] * energy_score
        )
        # Reward pays the mini-exam in instalments: undiscounted episode return
        # (with all hooks at 0) telescopes to EXACTLY the evaluation score.
        quality_scale = 100.0 / max(cfg.max_steps, 1)
        reward_u = quality_scale * FINAL_SCORE_WEIGHTS["uniformity_score"] * u_score
        reward_ph = quality_scale * FINAL_SCORE_WEIGHTS["ph_score"] * ph_score
        reward_e = quality_scale * FINAL_SCORE_WEIGHTS["energy_score"] * energy_score
        reward = (
            reward_u
            + reward_ph
            + reward_e
            + cfg.w_flux * flux_activity
            + cfg.w_uniformity_goal * uniformity_excess  # optional shaping, 0 by default
            - cfg.w_local_dry * local_dry_violation      # optional safety hook, 0 by default
            - cfg.w_voltage_jump * voltage_jump
        )
        terms = {
            "uniformity_improvement": float(uniformity_improvement),
            "uniformity_excess": float(uniformity_excess),
            "moisture_cv": moisture_cv,
            "moisture_cv_gap": moisture_cv_gap,
            "anode_dry_violation": anode_dry_violation,
            "local_dry_violation": local_dry_violation,
            "voltage_jump": float(voltage_jump),
            "energy": energy,
            "absolute_voltage_energy": absolute_voltage_energy,
            "pair_voltage_energy": pair_voltage_energy,
            "field_energy": field_energy,
            "effective_energy": effective_energy,
            "voltage_loss": voltage_loss,
            "voltage_drop_v": float(np.max(electrode_action) - np.min(electrode_action)),
            "effective_voltage_ratio": float(effective_voltage_ratio),
            "ph_violation": ph_violation,
            "ph_violation_mean": ph_violation_mean,
            "ph_violation_mixed": ph_violation_mixed,
            "ph_acid_violation": ph_acid_viol,
            "ph_acid_increment": float(ph_acid_increment),
            "power_phys": power_phys,
            "u_score": float(u_score),
            "ph_score": float(ph_score),
            "energy_score": float(energy_score),
            "step_quality": float(step_quality),
            "reward_u": float(reward_u),
            "reward_ph": float(reward_ph),
            "reward_e": float(reward_e),
            "flux": float(flux_activity),
        }
        return float(reward), terms


def make_soilbox(**overrides: Any) -> SoilBoxEnv:
    """Create a standard soil box with selected config overrides."""

    return SoilBoxEnv(DEFAULT_CONFIG.with_overrides(**overrides))


def rule_policy(
    obs: np.ndarray,
    n: int = DEFAULT_CONFIG.n_cells,
    max_electrode_voltage_v: float | None = None,
) -> np.ndarray:
    """Simple normalized-action baseline: push moisture from wetter sides to drier sides."""

    if max_electrode_voltage_v is None:
        max_electrode_voltage_v = max_electrode_voltage(DEFAULT_CONFIG)
    moisture = obs[:n]
    ph = obs[n:] * 7.0 + 7.0
    rows = DEFAULT_CONFIG.grid_rows
    cols = DEFAULT_CONFIG.grid_cols
    grid = moisture.reshape(rows, cols)
    left_moisture = float(np.mean(grid[:, 0]))
    right_moisture = float(np.mean(grid[:, -1]))
    front_moisture = float(np.mean(grid[-1, :]))
    back_moisture = float(np.mean(grid[0, :]))
    x_direction = np.sign(left_moisture - right_moisture)
    y_direction = np.sign(front_moisture - back_moisture)
    voltage = 0.5
    if float(np.max(np.abs(ph - 7.0))) > 1.0:
        voltage *= 0.5
    left = np.full(rows, voltage * x_direction, dtype=np.float32)
    right = -left
    front = np.full(cols, voltage * y_direction, dtype=np.float32)
    back = -front
    return np.concatenate([left, right, front, back]).astype(np.float32)

