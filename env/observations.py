"""
Build the RL policy observation vector from raw simulation state.

    obs = [ projected_gravity(3),            <- IMU-derivable orientation, no
                                                 gimbal lock / no double cover
                                                 (see design notes Part 7)
            base_angular_velocity(3)*0.25,   <- IMU gyro
            velocity_command(3),             <- task input: vx, vy, wz
            joint_pos_relative(12),          <- encoders, minus nominal pose
            joint_vel(12)*0.05,              <- encoders (finite-diff on HW)
            previous_action(12),             <- internal state, always known
            gait_clock(2) ]                  <- sin/cos of a commanded-speed
                                                 -linked phase (Milestone 5+)
    = 3+3+3+12+12+12+2 = 47-dim, float32.
    (The Milestone-0 design notes quoted 49 for this vector -- that was an
    arithmetic slip in the write-up, caught here by the runtime assertion
    below; 47 is the actual, verified dimension.)

Everything here is a signal the physical robot could plausibly produce
onboard, or an internal/task quantity -- see config/robot.yaml
'sensors.actor_can_use'. Base linear velocity and absolute world position
are deliberately excluded: not directly measurable on hardware. They remain
available to the REWARD function (which is allowed privileged simulator
state, see env/rewards.py) and are exposed in `info` for logging/evaluation
only -- never fed to the policy.
"""
from __future__ import annotations

import numpy as np
import mujoco

OBS_DIM = 47
ANG_VEL_SCALE = 0.25
JOINT_VEL_SCALE = 0.05


def projected_gravity(data: "mujoco.MjData") -> np.ndarray:
    """Unit gravity vector expressed in the base (pelvis) frame -- what an
    IMU accelerometer reads at rest, normalized. (0, 0, -1) when upright."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, data.qpos[3:7])
    return R.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])


def build_observation(data, act, q_nominal: np.ndarray, velocity_command: np.ndarray,
                       prev_action: np.ndarray, clock_phase: float, noise=None) -> np.ndarray:
    q, qd = act.joint_state(data)
    grav = projected_gravity(data)
    ang_vel = data.qvel[3:6].copy()  # base angular velocity, body frame (MuJoCo free-joint qvel)

    if noise is not None:
        grav = noise.projected_gravity(grav)
        ang_vel = noise.gyro(ang_vel)
        q = noise.joint_pos(q)
        qd = noise.joint_vel(qd)

    clock = np.array([np.sin(clock_phase), np.cos(clock_phase)])

    obs = np.concatenate([
        grav,
        ang_vel * ANG_VEL_SCALE,
        velocity_command,
        q - q_nominal,
        qd * JOINT_VEL_SCALE,
        prev_action,
        clock,
    ]).astype(np.float32)
    assert obs.shape[0] == OBS_DIM, f"observation dim mismatch: {obs.shape[0]} != {OBS_DIM}"
    return obs
