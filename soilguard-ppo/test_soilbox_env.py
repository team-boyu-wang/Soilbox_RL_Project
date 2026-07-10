# -*- coding: utf-8 -*-
"""Lightweight checks for the reusable SoilBoxEnv module."""

from __future__ import annotations

import numpy as np

from soilbox_env import DEFAULT_CONFIG, SoilBoxEnv, max_electrode_voltage, rule_policy


def _normalized_volts(env: SoilBoxEnv, values) -> np.ndarray:
    """Convert physical electrode volts to the normalized action API."""

    return (np.asarray(values, dtype=np.float32) / env.max_electrode_voltage_v).astype(np.float32)


def test_reset_and_step_contract() -> None:
    env = SoilBoxEnv(seed=123)
    obs, info = env.reset(seed=123)
    assert info == {}
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)
    assert env.action_space.shape == (12,)

    action = np.zeros(env.action_space.shape, dtype=np.float32)
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert next_obs.shape == env.observation_space.shape
    assert env.observation_space.contains(next_obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert {
        "terms",
        "spread",
        "success",
        "config",
        "obs_noise",
        "nominal_applied_voltage_v",
        "max_electrode_voltage_v",
        "voltage_drop_v",
        "injected_water",
        "mean_moisture",
        "moisture_cv",
        "mean_pH",
        "electrode_action",
        "boundary_action",
        "effective_cross_section_area_cm2",
        "x_boundary_area_cm2",
        "y_boundary_area_cm2",
        "cell_volume_ml",
        "internal_transport_ml",
        "internal_transport_fraction",
    } <= set(info)
    assert {
        "moisture_cv",
        "moisture_cv_gap",
        "anode_dry_violation",
        "local_dry_violation",
        "uniformity_improvement",
        "voltage_jump",
        "energy",
        "absolute_voltage_energy",
        "pair_voltage_energy",
        "field_energy",
        "effective_energy",
        "voltage_loss",
        "effective_voltage_ratio",
        "ph_violation",
        "ph_violation_mean",
        "ph_violation_mixed",
        "power_phys",
        "u_score",
        "ph_score",
        "energy_score",
        "step_quality",
        "reward_u",
        "reward_ph",
        "reward_e",
        "flux",
    } <= set(info["terms"])
    assert 0.0 < info["terms"]["effective_voltage_ratio"] <= 1.0
    assert info["nominal_applied_voltage_v"] == 30.0
    assert info["max_electrode_voltage_v"] == 20.0
    assert info["voltage_drop_v"] == 0.0
    assert len(info["electrode_action"]) == DEFAULT_CONFIG.total_electrode_count
    assert len(info["boundary_action"]) == 12
    # Zero action: no field, no electrolysis -> pH stays neutral.
    assert info["mean_pH"] == 7.0


def test_rule_policy_action_shape() -> None:
    env = SoilBoxEnv()
    obs, _ = env.reset(seed=3)
    action = rule_policy(obs, DEFAULT_CONFIG.n_cells, max_electrode_voltage(DEFAULT_CONFIG))
    assert action.shape == env.action_space.shape


def test_contact_dry_penalty_only_uses_active_high_potential_contacts() -> None:
    env = SoilBoxEnv(seed=11)
    env.reset(seed=11)
    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
    _, _, _, _, info = env.step(zero_action)
    assert info["terms"]["local_dry_violation"] == 0.0

    env.reset(seed=11)
    right_high_action = np.zeros(env.action_space.shape, dtype=np.float32)
    right_high_action[3:6] = env.action_space.high[0]
    _, _, _, _, info = env.step(right_high_action)
    assert info["terms"]["local_dry_violation"] > 0.0


def test_decoupled_field_is_not_halved() -> None:
    """A pure single-axis field keeps its full nominal gradient (fixes A1)."""

    env = SoilBoxEnv(seed=1)
    env.reset(seed=1)
    action = np.zeros(env.action_space.shape, dtype=np.float64)
    action[0:3] = 0.5 * env.max_electrode_voltage_v
    action[3:6] = -0.5 * env.max_electrode_voltage_v
    grad_x, grad_y = env._axis_gradients(action)
    # (15 - (-15)) / 30 cm = 1.0 V/cm 鈥?full nominal, NOT the old halved ~0.5.
    assert np.allclose(grad_x, 2.0 / 3.0)
    # The idle y-axis contributes exactly zero instead of dragging x down.
    assert np.allclose(grad_y, 0.0)


def test_net_acidification_over_episode() -> None:
    """A sustained field drives the bulk soil net-acidic (mean pH < 7)."""

    env = SoilBoxEnv(seed=2)
    obs, _ = env.reset(seed=2)
    action_volts = np.concatenate(
        [
            np.full(3, env.max_electrode_voltage_v),
            np.full(3, -env.max_electrode_voltage_v),
            np.zeros(3),
            np.zeros(3),
        ]
    ).astype(np.float32)
    action = _normalized_volts(env, action_volts)
    mean_ph = 7.0
    for _ in range(30):
        obs, _, terminated, truncated, info = env.step(action)
        mean_ph = info["mean_pH"]
        if terminated or truncated:
            break
    assert mean_ph < 7.0


def test_diffusion_alone_cannot_win() -> None:
    """Idle policy: diffusion smooths a little but must NOT reach the target.

    Pins the calibration window 鈥?if this fails the task has become trivial
    (agent can do nothing and still succeed).
    """

    env = SoilBoxEnv(seed=4)
    env.reset(seed=4)
    zero = np.zeros(env.action_space.shape, dtype=np.float32)
    cv0 = env.moisture_cv()
    for _ in range(env.cfg.max_steps):
        _, _, terminated, truncated, info = env.step(zero)
        if terminated or truncated:
            break
    assert info["moisture_cv"] < cv0  # diffusion does equalize a bit...
    assert not info["success"]  # ...but never wins on its own


def test_electroosmosis_plus_diffusion_reaches_target() -> None:
    """A simple proportional x-controller reaches CV <= target with diffusion.

    This is the reachability guarantee: without diffusion the same controller
    plateaus at CV ~ 0.24 because the middle column cannot fill.
    """

    env = SoilBoxEnv(seed=5)
    obs, _ = env.reset(seed=5)
    success = False
    for _ in range(env.cfg.max_steps):
        grid = env._grid(env.m)
        v = float(np.clip((grid[:, 0].mean() - grid[:, 2].mean()) * 80.0, -15.0, 15.0))
        action = _normalized_volts(env, [v, v, v, -v, -v, -v, 0, 0, 0, 0, 0, 0])
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            success = bool(info["success"])
            break
    assert success


def test_reset_accepts_initial_moisture_option() -> None:
    env = SoilBoxEnv(seed=6)
    field = np.full((3, 3), 0.2)
    field[1, 1] = 0.4
    obs, _ = env.reset(seed=6, options={"initial_moisture": field})
    assert np.allclose(env.m.reshape(3, 3), field)
    assert np.allclose(obs[:9], field.reshape(-1), atol=1e-6)


def test_no_terminate_mode_full_horizon_and_bonus_once() -> None:
    """terminate_on_success=False: episodes always run max_steps, and the
    success bonus is paid exactly once at the first crossing."""

    env = SoilBoxEnv(
        config={"terminate_on_success": False, "success_bonus": 100.0}, seed=5
    )
    env.reset(seed=5)
    steps = 0
    bonus_steps = 0
    terminated_any = False
    for _ in range(env.cfg.max_steps):
        grid = env._grid(env.m)
        v = float(np.clip((grid[:, 0].mean() - grid[:, 2].mean()) * 80.0, -15.0, 15.0))
        action = _normalized_volts(env, [v, v, v, -v, -v, -v, 0, 0, 0, 0, 0, 0])
        _, reward, terminated, truncated, info = env.step(action)
        steps += 1
        terminated_any = terminated_any or terminated
        if reward > 50.0:  # only the 100-point bonus can push a step this high
            bonus_steps += 1
        if truncated:
            break
    assert not terminated_any  # crossing no longer ends the episode
    assert steps == env.cfg.max_steps  # ran the full horizon
    assert bonus_steps == 1  # bonus paid exactly once


def test_hold_bonus_paid_per_step_below_target() -> None:
    """hold_bonus is a per-step wage: paid on EVERY below-target step, exactly
    the configured amount, and only there (v13 lazy-collapse income fix)."""

    def run(bonus: float) -> tuple[np.ndarray, np.ndarray]:
        env = SoilBoxEnv(config={"hold_bonus": bonus}, seed=5)
        env.reset(seed=5)
        rewards, successes = [], []
        for _ in range(env.cfg.max_steps):
            grid = env._grid(env.m)
            v = float(np.clip((grid[:, 0].mean() - grid[:, 2].mean()) * 80.0, -15.0, 15.0))
            action = _normalized_volts(env, [v, v, v, -v, -v, -v, 0, 0, 0, 0, 0, 0])
            _, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            successes.append(bool(info["success"]))
            assert info["terms"]["hold_bonus"] == (bonus if info["success"] else 0.0)
            if terminated or truncated:
                break
        return np.array(rewards), np.array(successes)

    # Same seed + same deterministic controller => identical trajectories, so
    # the reward difference isolates the bonus exactly.
    rewards_zero, success_zero = run(0.0)
    rewards_paid, success_paid = run(5.0)
    assert np.array_equal(success_zero, success_paid)  # dynamics unchanged
    assert int(success_paid.sum()) > 1  # reached AND held for multiple steps
    diff = rewards_paid - rewards_zero
    assert np.allclose(diff[success_paid], 5.0)   # paid on every below-target step
    assert np.allclose(diff[~success_paid], 0.0)  # and nowhere else

    # Idle cannot collect: diffusion alone never reaches the target
    # (guaranteed by test_diffusion_alone_cannot_win).
    env = SoilBoxEnv(config={"hold_bonus": 5.0}, seed=4)
    env.reset(seed=4)
    zero = np.zeros(env.action_space.shape, dtype=np.float32)
    for _ in range(env.cfg.max_steps):
        _, _, terminated, truncated, info = env.step(zero)
        assert info["terms"]["hold_bonus"] == 0.0
        if terminated or truncated:
            break


def test_v5_reward_is_exam_instalment() -> None:
    """v5.0 pin: with all hooks at their 0 defaults, the per-step reward is
    EXACTLY (100/max_steps) * step_quality — the construction that makes the
    undiscounted episode return equal the exam score. Changing any default
    hook breaks this identity loudly."""

    assert DEFAULT_CONFIG.hold_bonus == 0.0
    assert DEFAULT_CONFIG.success_bonus == 0.0
    assert DEFAULT_CONFIG.w_uniformity_goal == 0.0
    assert DEFAULT_CONFIG.w_local_dry == 0.0
    env = SoilBoxEnv(seed=8)
    env.reset(seed=8)
    push = _normalized_volts(env, [15, 15, 15, -15, -15, -15, 0, 0, 0, 0, 0, 0])
    scale = 100.0 / env.cfg.max_steps
    for _ in range(10):
        _, reward, _, _, info = env.step(push)
        terms = info["terms"]
        assert abs(reward - scale * terms["step_quality"]) < 1e-9
        assert abs(
            terms["reward_u"] + terms["reward_ph"] + terms["reward_e"]
            - scale * terms["step_quality"]
        ) < 1e-9


def test_bc_pretrain_learns_direction_and_is_off_by_default() -> None:
    """BC warm start (v16): a short supervised pass must reduce the imitation
    loss and leave the policy pushing water in the RIGHT direction on the
    canonical left-wet field 鈥?the exact capability PPO-from-scratch never
    develops under random initial fields."""

    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    from train_core import base_defaults, pretrain_policy_bc

    # Off by default: enabling it must be an explicit experiment decision.
    assert base_defaults()["bc_pretrain"] is False

    make_env = lambda s: SoilBoxEnv(seed=s, randomize_initial=True)
    model = PPO(
        "MlpPolicy",
        Monitor(make_env(0)),
        seed=0,
        device="cpu",
        policy_kwargs=dict(log_std_init=2.0),
    )
    stats = pretrain_policy_bc(model, make_env, episodes=20, epochs=20, seed=0)
    assert stats["bc_loss_last_epoch"] < stats["bc_loss_first_epoch"]

    # Canonical left-wet field: the cloned policy must push rightward
    # (left electrodes positive, right negative) with a non-trivial normalized command.
    probe = SoilBoxEnv(seed=1)
    obs, _ = probe.reset(seed=1)
    action, _ = model.predict(obs, deterministic=True)
    assert float(np.mean(action[0:3])) > 0.05   # left electrodes push
    assert float(np.mean(action[3:6])) < -0.05  # right electrodes pull


def test_ph_mixed_violation_is_peak_mean_blend() -> None:
    """v5.0 pH input: mixed = 0.7 * worst-cell + 0.3 * whole-box mean stock
    violation (user decision 2026-07-09). Peak guards the hotspot, mean gives
    dense gradients; the blend must sit between the two."""

    assert DEFAULT_CONFIG.ph_peak_weight == 0.7
    env = SoilBoxEnv(seed=2)
    env.reset(seed=2)
    push = _normalized_volts(env, [15, 15, 15, -15, -15, -15, 0, 0, 0, 0, 0, 0])
    saw_violation = False
    for _ in range(40):
        _, _, _, _, info = env.step(push)
        terms = info["terms"]
        expected = 0.7 * terms["ph_violation"] + 0.3 * terms["ph_violation_mean"]
        assert abs(terms["ph_violation_mixed"] - expected) < 1e-12
        assert (
            terms["ph_violation_mean"] - 1e-12
            <= terms["ph_violation_mixed"]
            <= terms["ph_violation"] + 1e-12
        )
        if terms["ph_violation"] > 0.5:
            saw_violation = True
    assert saw_violation  # sustained push must acidify the anode cells


def test_level_curves_piecewise_linear_and_bounded() -> None:
    """The scoring levels are continuous piecewise-linear curves (no staircase
    plateaus/jumps — the v12/v15 critic-blindness lesson), clamped to [0, 1]."""

    from soilbox_env import _level_score

    u = DEFAULT_CONFIG.uniformity_level_points
    assert _level_score(0.05, u) == 1.0          # below first breakpoint: clamp
    assert _level_score(0.15, u) == 0.8          # exact breakpoint
    assert abs(_level_score(0.20, u) - 0.6) < 1e-12   # midpoint interpolation
    assert _level_score(0.60, u) == 0.0          # beyond last breakpoint: clamp
    p = DEFAULT_CONFIG.ph_level_points
    assert _level_score(0.0, p) == 1.0
    assert abs(_level_score(1.0, p) - 0.45) < 1e-12
    assert _level_score(3.0, p) == 0.0
    # Monotone non-increasing over a fine sweep (more violation never scores higher).
    for pts in (u, p):
        xs = np.linspace(0.0, 3.0, 301)
        ys = [_level_score(float(x), pts) for x in xs]
        assert all(ys[i] >= ys[i + 1] - 1e-12 for i in range(len(ys) - 1))
        assert all(0.0 <= y <= 1.0 for y in ys)


def test_energy_uses_pair_voltage_difference_not_absolute_offset() -> None:
    """Energy follows electrode-pair voltage differences, not absolute voltage."""

    env = SoilBoxEnv(seed=12)
    env.reset(seed=12)
    offset_action = _normalized_volts(env, [5, 5, 5, 15, 15, 15, 0, 0, 0, 0, 0, 0])
    _, _, _, _, info_offset = env.step(offset_action)

    env.reset(seed=12)
    centered_action = _normalized_volts(env, [-5, -5, -5, 5, 5, 5, 0, 0, 0, 0, 0, 0])
    _, _, _, _, info_centered = env.step(centered_action)

    assert np.isclose(info_offset["terms"]["energy"], info_centered["terms"]["energy"])
    assert np.isclose(
        info_offset["terms"]["pair_voltage_energy"],
        info_centered["terms"]["pair_voltage_energy"],
    )
    assert not np.isclose(
        info_offset["terms"]["absolute_voltage_energy"],
        info_centered["terms"]["absolute_voltage_energy"],
    )


def test_uniformity_reward_pays_no_diffusion_freeride() -> None:
    """Idle earns exactly zero from the uniformity-goal term (v12 lazy fix).

    Diffusion still improves CV every step (raw improvement > 0), but the paid
    basis is the excess over the diffusion-only counterfactual, which is zero
    when no field is applied.
    """

    env = SoilBoxEnv(seed=21)
    env.reset(seed=21)
    zero = np.zeros(env.action_space.shape, dtype=np.float32)
    _, _, _, _, info = env.step(zero)
    assert info["terms"]["uniformity_improvement"] > 0.0  # diffusion did work...
    assert abs(info["terms"]["uniformity_excess"]) < 1e-12  # ...but pays nothing

    # A sensible push must be able to earn a positive excess.
    env.reset(seed=21)
    push = _normalized_volts(env, [15, 15, 15, -15, -15, -15, 0, 0, 0, 0, 0, 0])
    _, _, _, _, info = env.step(push)
    assert info["terms"]["uniformity_excess"] > 0.0


def test_randomized_initial_fields_vary_and_are_wellposed() -> None:
    env = SoilBoxEnv(seed=9, randomize_initial=True)
    fields = []
    for ep in range(8):
        env.reset(seed=90 + ep)
        grid = env.m.copy()
        assert np.all(grid >= 0.02 - 1e-9) and np.all(grid <= 0.5 + 1e-9)
        cv = float(np.std(grid) / (np.mean(grid) + 1e-9))
        assert cv >= 0.25  # no trivially-uniform episodes
        fields.append(tuple(np.round(grid, 6)))
    assert len(set(fields)) > 1  # the dice actually roll


def test_injection_beats_initial_field_dice() -> None:
    env = SoilBoxEnv(seed=9, randomize_initial=True)
    fixed = np.full((3, 3), 0.2)
    fixed[1, 1] = 0.4
    env.reset(seed=123, options={"initial_moisture": fixed})
    assert np.allclose(env.m.reshape(3, 3), fixed)  # exam questions override dice


def test_evaluate_v2_battery_is_wellposed_and_deterministic() -> None:
    from evaluation import SCENARIOS, evaluate_v2, idle_policy
    from soilbox_env import max_electrode_voltage, rule_policy

    # Every scenario must start meaningfully non-uniform.
    for name, build in SCENARIOS:
        grid = build(DEFAULT_CONFIG)
        cv0 = float(np.std(grid) / (np.mean(grid) + 1e-9))
        assert cv0 > 0.18, f"{name} starts too uniform (cv0={cv0:.3f})"

    assert len(SCENARIOS) == 100  # 20 families x 5 seeded variants

    idle_result = evaluate_v2(idle_policy)
    # Diffusion alone must not pass the exam (task stays non-trivial).
    assert idle_result["aggregate"]["success"] < 0.5

    rule_fn = lambda obs: rule_policy(obs, DEFAULT_CONFIG.n_cells, max_electrode_voltage(DEFAULT_CONFIG))
    first = evaluate_v2(rule_fn)
    second = evaluate_v2(rule_fn)
    assert first["aggregate"] == second["aggregate"]  # frozen battery => same exam
    assert len(first["per_scenario"]) == len(SCENARIOS)
    assert 0.0 <= first["aggregate"]["final_score"] <= 100.0


def test_domain_randomization_changes_config_and_noise() -> None:
    env = SoilBoxEnv(randomize=True, randomize_noise=True, seed=7)
    seen = []
    for seed in range(3):
        env.reset(seed=seed)
        seen.append(
            (
                env.cfg.electroosmotic_permeability_cm2_per_s_v,
                env.cfg.m_dry,
                env.cfg.m_threshold,
                env.obs_noise,
            )
        )
    assert len(set(seen)) > 1
    for electroosmotic_permeability, m_dry, m_threshold, obs_noise in seen:
        assert 1.51e-5 <= electroosmotic_permeability <= 6.04e-5
        assert 0.04 <= m_dry <= 0.16
        assert 0.08 <= m_threshold <= 0.18
        assert 0.0 <= obs_noise <= 0.2


def test_literature_backed_config_knobs() -> None:
    env = SoilBoxEnv(
        config={
            "electrode_material": "aluminum",
            "anode_water_injection": True,
            "anode_water_injection_amount": 0.02,
        },
        seed=5,
    )
    obs, _ = env.reset(seed=5)
    action = np.concatenate(
        [
            np.full(3, env.action_space.high[0]),
            np.full(3, env.action_space.low[0]),
            np.full(3, env.action_space.high[0]),
            np.full(3, env.action_space.low[0]),
        ]
    ).astype(np.float32)
    _, _, _, _, info = env.step(action)
    assert env.cfg.box_length_cm == 30.0
    assert env.cfg.box_width_cm == 30.0
    assert env.cfg.box_height_cm == 30.0
    assert env.cfg.electrode_layout == "four_side_3_by_3_uniformity"
    assert env.cfg.total_electrode_count == 12
    assert env.cfg.voltage_gradient_levels == (0.5, 0.75, 1.0, 1.25, 1.5)
    assert info["injected_water"] > 0.0
    assert info["terms"]["voltage_drop_v"] == 40.0
    assert info["terms"]["effective_voltage_ratio"] < env.cfg.effective_voltage_ratio


if __name__ == "__main__":
    test_reset_and_step_contract()
    test_rule_policy_action_shape()
    test_contact_dry_penalty_only_uses_active_high_potential_contacts()
    test_decoupled_field_is_not_halved()
    test_net_acidification_over_episode()
    test_diffusion_alone_cannot_win()
    test_electroosmosis_plus_diffusion_reaches_target()
    test_reset_accepts_initial_moisture_option()
    test_no_terminate_mode_full_horizon_and_bonus_once()
    test_hold_bonus_paid_per_step_below_target()
    test_default_hold_bonus_matches_audited_value()
    test_bc_pretrain_learns_direction_and_is_off_by_default()
    test_ph_stock_mode_is_default_and_charges_stock_exposure()
    test_ph_acid_increment_bills_generation_once_not_standing_debt()
    test_energy_uses_pair_voltage_difference_not_absolute_offset()
    test_uniformity_reward_pays_no_diffusion_freeride()
    test_randomized_initial_fields_vary_and_are_wellposed()
    test_injection_beats_initial_field_dice()
    test_evaluate_v2_battery_is_wellposed_and_deterministic()
    test_domain_randomization_changes_config_and_noise()
    test_literature_backed_config_knobs()
    print("SoilBoxEnv smoke tests passed.")

