"""Fast pytest checks for the Milestone 2 standing controller."""
from pathlib import Path

import mujoco
import numpy as np
import pytest

SCENE = Path(__file__).resolve().parents[1] / "mujoco" / "scene.xml"

from control.actuator import ActuatorInterface
from control.balance_controller import StandingBalanceController


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(SCENE))


def _reset_standing(model, data):
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(model, data, kid)
    data.qpos[2] -= 0.020
    mujoco.mj_forward(model, data)


def _tilt_deg(data) -> float:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, data.qpos[3:7])
    return float(np.degrees(np.arccos(np.clip(R.reshape(3, 3)[2, 2], -1.0, 1.0))))


def test_holds_for_3_seconds(model):
    data = mujoco.MjData(model)
    _reset_standing(model, data)
    act = ActuatorInterface(model)
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    q_nom = model.key_qpos[kid][act.qpos_adr]
    ctrl = StandingBalanceController(model, q_nom)
    ctrl.reset(data)
    dt = model.opt.timestep
    for _ in range(int(3.0 / dt)):
        act.apply(data, ctrl.compute(data, dt))
        mujoco.mj_step(model, data)
    assert data.qpos[2] > 0.75
    assert _tilt_deg(data) < 10.0
