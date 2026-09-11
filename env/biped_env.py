"""
BR-1 Gymnasium environment (Milestone 3).

    Observation -> RL Policy -> Action -> PD Controller -> MuJoCo Physics
         ^                                                        |
         '------------------------- Sensors <----------------------'

`step(action)` runs `control_decimation` physics steps (500 Hz sim /
decimation 10 = 50 Hz control, config/robot.yaml) per RL step. The action is
a 12-dim offset from the nominal standing pose (Part 8: residual position
action) converted to torque by the SAME PDController class used in
Milestone 2 -- but here the policy owns all 12 joint targets, including the
ankles. There is no hard-coded balance controller in this env: learning to
not fall is exactly the Milestone 4 task.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import yaml

from control.actuator import ActuatorInterface
from control.pd_controller import PDController
from control.balance_controller import POSTURE_GAINS
from robot.sensors import SensorNoise
from env.observations import build_observation, projected_gravity, OBS_DIM
from env.rewards import compute_reward
from env.termination import check_termination

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = REPO / "mujoco" / "scene.xml"
DEFAULT_TRAINING_CFG = REPO / "config" / "training.yaml"


class BipedEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, scene_path=DEFAULT_SCENE, training_cfg_path=DEFAULT_TRAINING_CFG,
                 use_sensor_noise: bool | None = None, seed: int | None = None):
        super().__init__()
        with open(training_cfg_path) as f:
            self.cfg = yaml.safe_load(f)

        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.act = ActuatorInterface(self.model)
        kp, kd = self.act.gains_by_suffix(POSTURE_GAINS)
        self.pd = PDController(kp, kd, self.act.torque_limit)

        self.kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        self.q_nominal = self.model.key_qpos[self.kid][self.act.qpos_adr].copy()
        self.height_target = float(self.model.key_qpos[self.kid][2])

        env_cfg = self.cfg["env"]
        self.decimation = int(env_cfg["control_decimation"])
        self.dt_ctrl = self.model.opt.timestep * self.decimation
        self.action_scale = float(env_cfg["action_scale_rad"])
        self.init_qpos_noise = float(env_cfg["init_qpos_noise_rad"])
        self.max_episode_steps = int(round(env_cfg["episode_length_s"] / self.dt_ctrl))
        self.max_tilt_deg = float(env_cfg["termination"]["max_tilt_deg"])
        self.min_base_height = float(env_cfg["termination"]["min_base_height_m"])
        self.cmd_ranges = env_cfg["command_ranges"]
        use_noise = env_cfg.get("use_sensor_noise", True) if use_sensor_noise is None else use_sensor_noise

        reward_cfg = self.cfg["reward"]
        self.reward_weights = reward_cfg["weights"]
        self.tracking_sigma = float(reward_cfg["tracking_sigma"])
        self.fall_penalty = float(reward_cfg["fall_penalty"])

        self.rng = np.random.default_rng(seed)
        self.noise = SensorNoise(self.rng) if use_noise else None

        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.act.n,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)

        self._prev_action = np.zeros(self.act.n)
        self._prev_qvel = np.zeros(self.act.n)
        self._cmd = np.zeros(3)
        self._clock_phase = 0.0
        self._step_count = 0

        self._foot_body = {
            "L": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "L_foot"),
            "R": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "R_foot"),
        }
        self._foot_geom = {
            "L": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "L_sole"),
            "R": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "R_sole"),
        }
        self._floor_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    # ------------------------------------------------------------------ #
    def _sample_command(self) -> np.ndarray:
        def u(pair):
            lo, hi = pair
            return self.rng.uniform(lo, hi) if hi > lo else lo
        return np.array([u(self.cmd_ranges["lin_vel_x"]),
                          u(self.cmd_ranges["lin_vel_y"]),
                          u(self.cmd_ranges["ang_vel_z"])])

    def _tilt_deg(self) -> float:
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, self.data.qpos[3:7])
        return float(np.degrees(np.arccos(np.clip(R.reshape(3, 3)[2, 2], -1.0, 1.0))))

    def _foot_contacts(self):
        """Returns (contact_bool_per_foot, foot_xy_vel_per_foot, non_foot_ground_hit)."""
        contact = {"L": False, "R": False}
        non_foot_hit = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if self._floor_geom not in (g1, g2):
                continue
            other = g2 if g1 == self._floor_geom else g1
            if other == self._foot_geom["L"]:
                contact["L"] = True
            elif other == self._foot_geom["R"]:
                contact["R"] = True
            else:
                non_foot_hit = True
        # cvel = [angular(3), linear(3)] about the body, world-aligned frame
        vel = {side: self.data.cvel[self._foot_body[side]][3:5].copy() for side in ("L", "R")}
        return contact, vel, non_foot_hit

    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            if self.noise is not None:
                self.noise.rng = self.rng  # keep the noise generator on the reseeded stream
        if self.noise is not None:
            self.noise.reset()

        mujoco.mj_resetDataKeyframe(self.model, self.data, self.kid)
        self.data.qpos[self.act.qpos_adr] += self.rng.uniform(
            -self.init_qpos_noise, self.init_qpos_noise, self.act.n)
        self.data.qpos[2] -= 0.020  # match the M1/M2 ground-clearance correction
        mujoco.mj_forward(self.model, self.data)

        self._prev_action = np.zeros(self.act.n)
        _, qd = self.act.joint_state(self.data)
        self._prev_qvel = qd.copy()
        self._cmd = self._sample_command()
        self._clock_phase = 0.0
        self._step_count = 0

        return self._get_obs(), {"cmd": self._cmd.copy()}

    def _get_obs(self) -> np.ndarray:
        return build_observation(self.data, self.act, self.q_nominal, self._cmd,
                                  self._prev_action, self._clock_phase, self.noise)

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        q_target = self.q_nominal + self.action_scale * action

        torque_sum = np.zeros(self.act.n)
        for _ in range(self.decimation):
            q, qd = self.act.joint_state(self.data)
            tau = self.pd.compute(q, qd, q_target)
            self.act.apply(self.data, tau)
            mujoco.mj_step(self.model, self.data)
            torque_sum += tau
        torque_avg = torque_sum / self.decimation

        mujoco.mj_forward(self.model, self.data)  # refresh contacts/CoM for the new qpos
        q, qd = self.act.joint_state(self.data)
        tilt = self._tilt_deg()
        base_height = float(self.data.qpos[2])
        contact, foot_vel, non_foot_hit = self._foot_contacts()

        terminated, reason = check_termination(
            base_height, tilt, self.data.qpos, self.data.qvel,
            self.max_tilt_deg, self.min_base_height)
        fall = terminated and reason not in ("instability (velocity blow-up)", "nan")

        state = {
            "base_linvel": self.data.qvel[0:3].copy(),
            "base_angvel_z": float(self.data.qvel[5]),
            "cmd": self._cmd,
            "projected_gravity": projected_gravity(self.data),
            "base_height": base_height,
            "height_target": self.height_target,
            "torque": torque_avg,
            "action": action,
            "prev_action": self._prev_action,
            "qvel": qd,
            "prev_qvel": self._prev_qvel,
            "dt": self.dt_ctrl,
            "foot_contact": [contact["L"], contact["R"]],
            "foot_xy_vel": [foot_vel["L"], foot_vel["R"]],
            "non_foot_contact": non_foot_hit,
            "terminated_by_fall": fall,
        }
        reward, reward_terms = compute_reward(
            state, self.reward_weights, self.tracking_sigma, self.fall_penalty)

        cmd_speed = float(np.linalg.norm(self._cmd[:2]))
        if cmd_speed > 1e-3:
            self._clock_phase += 2 * np.pi * cmd_speed * self.dt_ctrl
        self._prev_action = action.copy()
        self._prev_qvel = qd.copy()
        self._step_count += 1
        truncated = self._step_count >= self.max_episode_steps

        obs = self._get_obs()
        info = {
            "reward_terms": reward_terms,
            "termination_reason": reason,
            "tilt_deg": tilt,
            "base_height": base_height,
            "base_linvel": state["base_linvel"],  # privileged; logging/eval only
        }
        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self):
        pass
