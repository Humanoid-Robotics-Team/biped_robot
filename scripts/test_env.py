"""
Milestone 3 verification: the Gymnasium environment (env/biped_env.py).

CHECKS
  1. Gymnasium API compliance (gymnasium.utils.env_checker.check_env).
  2. Determinism: same seed + same actions -> identical observations.
  3. Zero-action (posture-only, no learned balance) topples at ~1.8 s --
     the same number M1/M2 measured -- confirming the env wraps the model
     faithfully rather than accidentally changing the dynamics.
  4. Reward sanity at the nominal pose: all terms finite, upright/height/
     alive terms near their maximum (1.0).
  5. Random-action robustness: no NaN/crash over many episodes.

USAGE
  python scripts/test_env.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from env.biped_env import BipedEnv  # noqa: E402


class Checklist:
    def __init__(self):
        self.ok = True

    def __call__(self, cond, msg):
        cond = bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
        self.ok &= cond


def check_gym_api() -> bool:
    print("\n[1] GYMNASIUM API COMPLIANCE")
    from gymnasium.utils.env_checker import check_env
    env = BipedEnv(seed=0)
    c = Checklist()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            check_env(env.unwrapped, skip_render_check=True)
            c(True, "check_env passed (reset/step contracts, spaces, determinism)")
        except Exception as e:  # noqa: BLE001
            c(False, f"check_env FAILED: {e!r}")
    return c.ok


def check_determinism() -> bool:
    print("\n[2] DETERMINISM  (same seed -> identical trajectory)")
    rng = np.random.default_rng(42)
    actions = [rng.uniform(-1, 1, 12).astype(np.float32) for _ in range(20)]

    def rollout(seed):
        env = BipedEnv(seed=seed)
        obs, _ = env.reset(seed=seed)
        obs_hist = [obs.copy()]
        for a in actions:
            obs, r, term, trunc, _ = env.step(a)
            obs_hist.append(obs.copy())
            if term or trunc:
                break
        return obs_hist

    h1, h2 = rollout(7), rollout(7)
    c = Checklist()
    same_len = len(h1) == len(h2)
    c(same_len, f"same episode length for seed=7 twice ({len(h1)} vs {len(h2)})")
    if same_len:
        max_diff = max(float(np.abs(a - b).max()) for a, b in zip(h1, h2))
        c(max_diff < 1e-9, f"identical observations (max diff {max_diff:.2e})")
    return c.ok


def check_zero_action_topple() -> bool:
    print("\n[3] ZERO-ACTION SANITY  (should topple ~1.8s, matches M1/M2 measurement)")
    env = BipedEnv(seed=1)
    env.reset(seed=1)
    zero = np.zeros(12, dtype=np.float32)
    t_topple = None
    for k in range(300):
        _, _, term, trunc, info = env.step(zero)
        if term:
            t_topple = (k + 1) * env.dt_ctrl
            break
    print(f"  topple time: {t_topple}")
    c = Checklist()
    c(t_topple is not None, "episode terminated (did not run forever)")
    c(t_topple is not None and 1.0 < t_topple < 3.0, "topple time in the 1-3 s band (~1.8s expected)")
    return c.ok


def check_reward_sanity() -> bool:
    print("\n[4] REWARD SANITY AT NOMINAL POSE")
    env = BipedEnv(seed=2)
    env.reset(seed=2)
    _, reward, _, _, info = env.step(np.zeros(12, dtype=np.float32))
    terms = info["reward_terms"]
    for k, v in terms.items():
        print(f"    {k:16s} {v:+.4f}")
    c = Checklist()
    c(np.isfinite(reward), "total reward is finite")
    c(all(np.isfinite(v) for v in terms.values()), "all reward terms finite")
    c(terms["upright"] > 0.95, "upright term near max at nominal pose")
    c(terms["height"] > 0.9, "height term near max at nominal pose")
    return c.ok


def check_random_robustness(n_seeds=5, n_steps=2000) -> bool:
    print(f"\n[5] RANDOM-ACTION ROBUSTNESS  ({n_seeds} seeds x {n_steps} steps)")
    rng = np.random.default_rng(0)
    c = Checklist()
    for seed in range(n_seeds):
        env = BipedEnv(seed=seed)
        obs, _ = env.reset(seed=seed)
        bad = False
        for _ in range(n_steps):
            a = rng.uniform(-1, 1, size=12).astype(np.float32)
            obs, r, term, trunc, _ = env.step(a)
            if not (np.all(np.isfinite(obs)) and np.isfinite(r)):
                bad = True
                break
            if term or trunc:
                obs, _ = env.reset()
        c(not bad, f"seed={seed}: {n_steps} random steps, no NaN")
    return c.ok


def main():
    ok = check_gym_api()
    ok &= check_determinism()
    ok &= check_zero_action_topple()
    ok &= check_reward_sanity()
    ok &= check_random_robustness()
    print("\n" + ("MILESTONE 3: ALL CHECKS PASSED" if ok else "MILESTONE 3: SOME CHECKS FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
