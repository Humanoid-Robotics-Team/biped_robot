"""Fast pytest checks on the BR-1 MuJoCo model (no simulation)."""
from pathlib import Path

import mujoco
import pytest

SCENE = Path(__file__).resolve().parents[1] / "mujoco" / "scene.xml"


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(SCENE))


def test_compiles(model):
    assert model.nbody > 1


def test_dof_counts(model):
    assert model.nu == 12
    assert model.nv == 18
    assert model.nq == 19


def test_total_mass(model):
    assert abs(model.body_mass.sum() - 32.0) < 0.5


def test_keyframe(model):
    assert model.nkey >= 1
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    assert kid >= 0
    assert model.key_qpos[kid].shape[0] == model.nq


def test_actuator_torque_limits(model):
    assert (model.actuator_forcerange[:, 1] > 0).all()


def test_expected_joint_names(model):
    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)}
    for side in ("L", "R"):
        for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll"):
            assert f"{side}_{j}" in names
