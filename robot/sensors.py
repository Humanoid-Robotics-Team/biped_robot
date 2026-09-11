"""
Sensor noise / latency model.

Wraps noiseless MuJoCo sensor readings with the imperfections a real BR-1
would actually see (config/robot.yaml -> sensors.noise_initial_assumption):
additive Gaussian noise, a fixed one-step latency, and a contact-sensor
bit-flip probability. All numbers are INITIAL ASSUMPTIONs -- Milestone 9
replaces them with characteristics measured off the real IMU/encoders.

Used by env/observations.py; NOT applied to anything the reward function
reads directly (the reward is allowed privileged, noiseless simulation
state -- see env/rewards.py's docstring). Only what the POLICY sees goes
through this.
"""
from __future__ import annotations

import numpy as np


class SensorNoise:
    def __init__(self, rng: np.random.Generator,
                 joint_pos_std: float = 0.003, joint_vel_std: float = 0.1,
                 gyro_std: float = 0.05, tilt_std: float = 0.02,
                 contact_flip_prob: float = 0.03, latency_steps: int = 1):
        self.rng = rng
        self.joint_pos_std = joint_pos_std
        self.joint_vel_std = joint_vel_std
        self.gyro_std = gyro_std
        self.tilt_std = tilt_std
        self.contact_flip_prob = contact_flip_prob
        self.latency_steps = latency_steps
        self._buffers: dict[str, list] = {}

    def _delay(self, key: str, value: np.ndarray) -> np.ndarray:
        if self.latency_steps <= 0:
            return value
        buf = self._buffers.setdefault(key, [])
        buf.append(value.copy())
        if len(buf) <= self.latency_steps:
            return value
        return buf.pop(0)

    def joint_pos(self, q: np.ndarray) -> np.ndarray:
        return self._delay("qpos", q + self.rng.normal(0, self.joint_pos_std, q.shape))

    def joint_vel(self, qd: np.ndarray) -> np.ndarray:
        return self._delay("qvel", qd + self.rng.normal(0, self.joint_vel_std, qd.shape))

    def gyro(self, w: np.ndarray) -> np.ndarray:
        return self._delay("gyro", w + self.rng.normal(0, self.gyro_std, w.shape))

    def projected_gravity(self, g: np.ndarray) -> np.ndarray:
        noisy = g + self.rng.normal(0, self.tilt_std, g.shape)
        n = np.linalg.norm(noisy)
        return self._delay("grav", noisy / n if n > 1e-6 else g)

    def contact(self, contact_bool: np.ndarray) -> np.ndarray:
        flips = self.rng.random(contact_bool.shape) < self.contact_flip_prob
        return np.where(flips, 1.0 - contact_bool, contact_bool)

    def reset(self) -> None:
        self._buffers.clear()
