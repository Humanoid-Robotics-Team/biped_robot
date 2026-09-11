"""
Actuator interface: bookkeeping between MuJoCo actuator order and joint
qpos/qvel addresses, and the single place where a torque command is clamped
before being written to mjData.ctrl.

Gen-1 model: ideal instantaneous torque source (config/robot.yaml
'actuation'). Milestone 9 (sim-to-real) will extend this class with a
measured torque-speed curve, gearbox backlash, and comms latency -- every
caller goes through `apply()`, so that upgrade does not touch env/ or
control/*_controller.py.
"""
from __future__ import annotations

import numpy as np
import mujoco


class ActuatorInterface:
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.n = model.nu
        self.names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                      for i in range(self.n)]
        self.joint_ids = np.array([int(model.actuator_trnid[i, 0]) for i in range(self.n)])
        self.qpos_adr = model.jnt_qposadr[self.joint_ids]
        self.qvel_adr = model.jnt_dofadr[self.joint_ids]
        self.torque_limit = model.actuator_forcerange[:, 1].copy()
        self.joint_range = model.jnt_range[self.joint_ids].copy()

    def joint_state(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        return data.qpos[self.qpos_adr].copy(), data.qvel[self.qvel_adr].copy()

    def apply(self, data: mujoco.MjData, torque: np.ndarray) -> None:
        data.ctrl[:] = np.clip(torque, -self.torque_limit, self.torque_limit)

    def index(self, actuator_name_suffix: str) -> np.ndarray:
        """Indices of actuators whose name ends with e.g. 'ankle_pitch'
        (matches both L_ankle_pitch and R_ankle_pitch)."""
        return np.array(
            [i for i, name in enumerate(self.names) if name.endswith(actuator_name_suffix)]
        )

    def gains_by_suffix(self, gain_table: dict[str, tuple[float, float]]):
        """Build (kp, kd) arrays in actuator order from a {joint_type: (kp, kd)} table,
        e.g. {"hip_yaw": (200, 6), ...} as in config/robot.yaml."""
        kp = np.zeros(self.n)
        kd = np.zeros(self.n)
        for i, name in enumerate(self.names):
            suffix = name.split("_", 1)[1]
            kp[i], kd[i] = gain_table[suffix]
        return kp, kd
