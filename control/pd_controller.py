"""
Generic joint-space PD position controller.

  target joint position  --[PD]-->  torque

This is the block from the Part 5 control chain:

    RL Policy -> Target Joint Commands -> Low-Level Controller -> Actuator -> Joint

It has no knowledge of MuJoCo, the robot, or balance -- it is a pure
stateless math object so the *exact same class* can run in simulation
(control/actuator.py wraps it around mjData) and later inside the real
robot's high-level control loop or an embedded MCU port.
"""
from __future__ import annotations

import numpy as np


class PDController:
    """tau = kp * (q_target - q) + kd * (qd_target - qd), clamped to torque_limit."""

    def __init__(self, kp: np.ndarray, kd: np.ndarray, torque_limit: np.ndarray):
        self.kp = np.asarray(kp, dtype=float)
        self.kd = np.asarray(kd, dtype=float)
        self.torque_limit = np.asarray(torque_limit, dtype=float)
        n = self.kp.shape[0]
        assert self.kd.shape[0] == n and self.torque_limit.shape[0] == n, (
            "kp, kd, torque_limit must all have the same length"
        )

    def compute(self, q: np.ndarray, qd: np.ndarray, q_target: np.ndarray,
                qd_target: np.ndarray | None = None) -> np.ndarray:
        qd_target = np.zeros_like(qd) if qd_target is None else qd_target
        tau = self.kp * (q_target - q) + self.kd * (qd_target - qd)
        return np.clip(tau, -self.torque_limit, self.torque_limit)
