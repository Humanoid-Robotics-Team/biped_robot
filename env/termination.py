"""
Episode termination logic (Part 11).

Thresholds are DELIBERATELY LOOSE for early training: a policy that has not
yet learned to balance needs room to wobble and still collect some reward
before the episode ends, or it never gets a gradient signal at all.
Tighten max_tilt_deg / min_base_height_m in config/training.yaml as the
curriculum advances (Part 12) -- no code change needed.

A velocity blow-up ("instability") is treated as a simulation problem, not
a fall, and is excluded from the fall_penalty in env/rewards.py -- it
should never happen with the M1-verified model, but termination.py must be
defensive regardless (Part 25: never let a policy exploit a physics glitch).
"""
from __future__ import annotations

import numpy as np


def check_termination(base_height: float, tilt_deg: float, qpos: np.ndarray, qvel: np.ndarray,
                       max_tilt_deg: float, min_base_height_m: float) -> tuple[bool, str]:
    if not (np.all(np.isfinite(qpos)) and np.all(np.isfinite(qvel))):
        return True, "nan"
    if np.abs(qvel).max() > 100.0:
        return True, "instability (velocity blow-up)"
    if base_height < min_base_height_m:
        return True, "fell (height)"
    if tilt_deg > max_tilt_deg:
        return True, "fell (tilt)"
    return False, ""
