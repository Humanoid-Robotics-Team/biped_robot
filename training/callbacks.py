"""
Custom SB3 callback for BR-1 training metrics.

SB3's own Monitor wrapper already logs ep_rew_mean / ep_len_mean to
TensorBoard. This adds the BR-1-specific numbers from Part 17/19 that
aren't generic RL metrics: fall rate (vs. timeout) and mean tilt, read from
the `info` dict env/biped_env.py attaches to every step.
"""
from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeStatsCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._fall_count = 0
        self._episode_count = 0
        self._tilt_buf: list[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            if "tilt_deg" in info:
                self._tilt_buf.append(info["tilt_deg"])
            if done:
                self._episode_count += 1
                if info.get("termination_reason", "").startswith("fell"):
                    self._fall_count += 1
        return True

    def _on_rollout_end(self) -> None:
        if self._episode_count > 0:
            self.logger.record("br1/fall_rate", self._fall_count / self._episode_count)
            self.logger.record("br1/episodes_this_rollout", self._episode_count)
        if self._tilt_buf:
            self.logger.record("br1/mean_tilt_deg", float(np.mean(self._tilt_buf)))
        self._fall_count = 0
        self._episode_count = 0
        self._tilt_buf = []
