"""
Milestone 2 standing controller.

  6 joints/leg (hip yaw/roll/pitch, knee, ankle roll) hold a nominal posture
  via joint-space PD (control/pd_controller.py). The two ankle-pitch and two
  ankle-roll actuators are instead driven by a whole-body "ankle strategy"
  that regulates the horizontal CoM position over the support polygon --
  this is what actually keeps the robot from falling; posture PD alone
  cannot (see scripts/test_robot.py's time-to-topple measurement, ~1.8 s).

METHOD (classic double-support ankle strategy, e.g. Kajita et al.)
  Let  e = CoM_xy(t) - support_center_xy   (support_center = midpoint of the
  two sole centers, fixed while both feet stay planted -- correct for M2;
  Milestone 5+ recomputes it as the feet start moving).

      tau_ankle_pitch =  kp_x * e_x + kd_x * de_x/dt     (both legs)
      tau_ankle_roll  = -kp_y * e_y - kd_y * de_y/dt     (both legs)

  The two axes use opposite-looking signs because of the MuJoCo joint axis
  convention (ankle_pitch axis = +Y, ankle_roll axis = +X, see
  mujoco/robot.xml) -- NEITHER sign was assumed; both were found by seeding a
  small CoM velocity perturbation on each axis independently and keeping the
  sign for which the error decayed rather than grew (a naive same-sign guess
  for the roll axis is UNSTABLE and looks fine for ~10 s before diverging --
  see scripts/test_controller.py, which reproduces this check).

LIMITS  (measured in scripts/test_controller.py, Gen-1 geometry)
  - holds indefinitely with no disturbance (tested to 30 s, converges to
    machine-precision CoM error)
  - recovers a CoM velocity kick up to ~0.3 m/s fore-aft, ~0.2 m/s lateral
  - beyond that the required center-of-pressure leaves the sole
    ([-0.06, +0.14] m fore-aft, +-0.045 m lateral around the ankle) and no
    ankle torque can help -- this is a genuine physical limit of a fixed-
    footprint ankle-only strategy, not a bug. Larger disturbances need a hip
    or stepping strategy (Milestone 7/8) or a learned policy (Milestone 4+).
"""
from __future__ import annotations

import numpy as np
import mujoco

from control.actuator import ActuatorInterface
from control.pd_controller import PDController

# posture PD gains, keyed by joint-type suffix (applies to both legs) -- must
# match config/robot.yaml -> joints.*.kp/kd
POSTURE_GAINS = {
    "hip_yaw": (200.0, 6.0),
    "hip_roll": (200.0, 6.0),
    "hip_pitch": (200.0, 6.0),
    "knee": (200.0, 6.0),
    "ankle_pitch": (120.0, 3.0),  # unused for ankle_pitch (balance law overrides it)
    "ankle_roll": (120.0, 3.0),   # unused for ankle_roll  (balance law overrides it)
}

# ankle-strategy balance gains, empirically tuned in simulation (see module docstring)
KP_ANKLE_X, KD_ANKLE_X = 400.0, 40.0
KP_ANKLE_Y, KD_ANKLE_Y = 400.0, 40.0

# sole center offset forward of the ankle joint, meters -- must match mujoco/robot.xml
FOOT_X_OFFSET = 0.04


class StandingBalanceController:
    def __init__(self, model: mujoco.MjModel, q_nominal: np.ndarray):
        self.model = model
        self.act = ActuatorInterface(model)
        self.q_nominal = np.asarray(q_nominal, dtype=float).copy()

        kp, kd = self.act.gains_by_suffix(POSTURE_GAINS)
        self.posture = PDController(kp, kd, self.act.torque_limit)

        self.ap_idx = self.act.index("ankle_pitch")
        self.ar_idx = self.act.index("ankle_roll")

        self._support_center = None
        self._prev_err = None

    def reset(self, data: mujoco.MjData) -> None:
        """Call once after placing the robot (e.g. after mj_resetDataKeyframe)."""
        mujoco.mj_forward(self.model, data)
        lf = data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "L_foot")]
        rf = data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "R_foot")]
        self._support_center = np.array([(lf[0] + rf[0]) / 2.0 + FOOT_X_OFFSET, 0.0])
        self._prev_err = None

    def compute(self, data: mujoco.MjData, dt: float) -> np.ndarray:
        """Returns the 12-vector of actuator torques for this control step.
        Refreshes CoM/kinematics for the CURRENT data.qpos before computing
        (cheap at 12 DOF; keeps the balance term one step fresher)."""
        if self._support_center is None:
            self.reset(data)
        mujoco.mj_forward(self.model, data)

        q, qd = self.act.joint_state(data)
        tau = self.posture.compute(q, qd, self.q_nominal)

        com_xy = data.subtree_com[0][:2]
        err = com_xy - self._support_center
        derr = np.zeros(2) if self._prev_err is None else (err - self._prev_err) / dt
        self._prev_err = err

        tau[self.ap_idx] = KP_ANKLE_X * err[0] + KD_ANKLE_X * derr[0]
        tau[self.ar_idx] = -KP_ANKLE_Y * err[1] - KD_ANKLE_Y * derr[1]

        return np.clip(tau, -self.act.torque_limit, self.act.torque_limit)
