"""
Reward function for BR-1 locomotion.

Reward terms are computed from PRIVILEGED, noiseless simulation state (base
linear velocity included) -- that is standard and correct: the reward
function is a training-time-only oracle, unlike the observation the policy
actually sees (env/observations.py), which must stay realistic.

R = w_vel * lin_vel_track + w_ang * ang_vel_track + w_up * upright
  + w_h * height + w_alive * alive
  - w_tau * torque - w_ar * action_rate - w_ja * joint_acc
  - w_slip * foot_slip - w_col * collision
  - fall_penalty * (1 if terminated by falling else 0)

REWARD-HACKING GUARDS (Part 10) -- what each penalty is there to stop:
  torque, action_rate, joint_acc  -> jittery, high-frequency, energy-wasting
                                      policies; also discourages "vibrating
                                      in place" to farm the alive bonus.
  foot_slip                        -> sliding feet instead of stepping
                                      (only meaningful once stepping starts,
                                      M5+; harmless at M4 with cmd=0).
  collision (non-foot ground hit)  -> "walking" by dragging a knee/torso,
                                      or falling forward while still racking
                                      up alive/velocity reward before the
                                      termination check fires.
  height term                      -> discourages crouching low to farm a
                                      cheap upright/stability score, or
                                      jumping (a height term alone doesn't
                                      stop jumping -- action_rate/torque and
                                      the foot-contact-based alive/height
                                      combination discourage the aerial
                                      phase a jump requires).
  tracking (not raw speed) reward  -> rewarding raw forward SPEED (instead
                                      of tracking a commanded speed) lets the
                                      policy farm reward by falling forward
                                      with increasing velocity; tracking a
                                      *bounded* command with a Gaussian
                                      kernel caps the achievable reward at
                                      "moving exactly at the requested
                                      speed", removing that incentive.
"""
from __future__ import annotations

import numpy as np


def compute_reward(state: dict, weights: dict, tracking_sigma: float,
                    fall_penalty: float) -> tuple[float, dict]:
    r = {}

    lin_err = float(np.sum((state["base_linvel"][:2] - state["cmd"][:2]) ** 2))
    r["lin_vel_track"] = float(np.exp(-lin_err / tracking_sigma))

    ang_err = float((state["base_angvel_z"] - state["cmd"][2]) ** 2)
    r["ang_vel_track"] = float(np.exp(-ang_err / tracking_sigma))

    r["upright"] = float(np.exp(-np.sum(np.square(state["projected_gravity"][:2])) / tracking_sigma))
    r["height"] = float(np.exp(-((state["base_height"] - state["height_target"]) ** 2) / tracking_sigma))
    r["alive"] = 1.0

    r["torque"] = -float(np.sum(np.square(state["torque"])))
    r["action_rate"] = -float(np.sum(np.square(state["action"] - state["prev_action"])))
    r["joint_acc"] = -float(np.sum(np.square((state["qvel"] - state["prev_qvel"]) / state["dt"])))

    slip = 0.0
    for in_contact, foot_xy_vel in zip(state["foot_contact"], state["foot_xy_vel"]):
        if in_contact:
            slip += float(np.sum(np.square(foot_xy_vel)))
    r["foot_slip"] = -slip
    r["collision"] = -1.0 if state["non_foot_contact"] else 0.0

    total = sum(weights.get(name, 0.0) * value for name, value in r.items())
    if state["terminated_by_fall"]:
        total -= fall_penalty

    r["total"] = float(total)
    return float(total), r
