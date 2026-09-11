"""
Milestone 1 verification for the BR-1 MuJoCo model.

SCOPE OF MILESTONE 1
  M1 validates the *model* and the *simulation*, NOT balance:
    - correct kinematic tree, DOF count, mass, inertia, actuator limits
    - numerically stable integration (no NaN / blow-up / tunnelling)
    - correct foot-ground contact: both soles touch, total vertical ground
      reaction ~= m*g, negligible penetration
  Active standing balance is Milestone 2 (a floating-base biped held only by
  joint-level PD is an uncontrolled inverted pendulum and WILL slowly topple
  - this script measures that time-to-topple for reference, it is not a
  failure of the model).

USAGE
  python scripts/test_robot.py
  python scripts/test_robot.py --seconds 3

EXIT CODE 0 = all M1 checks passed, 1 = a check failed, 2 = model file missing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import mujoco

REPO = Path(__file__).resolve().parents[1]
SCENE = REPO / "mujoco" / "scene.xml"

GRAVITY = 9.81
# joint "type" (prefix stripped) -> PD gains, matches config/robot.yaml
KP = {"hip_yaw": 200.0, "hip_roll": 200.0, "hip_pitch": 200.0, "knee": 200.0,
      "ankle_pitch": 120.0, "ankle_roll": 120.0}
KD = {"hip_yaw": 6.0, "hip_roll": 6.0, "hip_pitch": 6.0, "knee": 6.0,
      "ankle_pitch": 3.0, "ankle_roll": 3.0}


# --------------------------------------------------------------------------- #
def load():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    return model, mujoco.MjData(model)


def actuator_jtype(model, i: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i).split("_", 1)[1]


def actuator_qv_adr(model):
    q_adr, v_adr = [], []
    for i in range(model.nu):
        jid = int(model.actuator_trnid[i, 0])
        q_adr.append(int(model.jnt_qposadr[jid]))
        v_adr.append(int(model.jnt_dofadr[jid]))
    return np.array(q_adr), np.array(v_adr)


def gains(model):
    kp = np.array([KP[actuator_jtype(model, i)] for i in range(model.nu)])
    kd = np.array([KD[actuator_jtype(model, i)] for i in range(model.nu)])
    return kp, kd


def sole_min_corner_z(model, data, geom_name: str) -> float:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    R = data.geom_xmat[gid].reshape(3, 3)
    c = data.geom_xpos[gid]
    h = model.geom_size[gid]
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return float((c + (signs * h) @ R.T)[:, 2].min())


def base_tilt_deg(data) -> float:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, data.qpos[3:7])
    return float(np.degrees(np.arccos(np.clip(R.reshape(3, 3)[2, 2], -1.0, 1.0))))


class Checklist:
    def __init__(self):
        self.ok = True

    def __call__(self, cond, msg):
        cond = bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
        self.ok &= cond


# --------------------------------------------------------------------------- #
def report_structure(model) -> float:
    print("=" * 70)
    print("BR-1 MODEL STRUCTURE")
    print("=" * 70)
    print(f"  nq generalized coordinates : {model.nq}")
    print(f"  nv degrees of freedom      : {model.nv}")
    print(f"  nu actuators               : {model.nu}")
    print(f"  nbody (incl. world)        : {model.nbody}")
    print(f"  njnt                       : {model.njnt}")
    total = float(model.body_mass.sum())
    print(f"  total mass                 : {total:8.3f} kg   (weight {total * GRAVITY:6.1f} N)")
    print("-" * 70)
    for i in range(model.nbody):
        if model.body_mass[i] <= 0:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        ix, iy, iz = model.body_inertia[i]
        print(f"    {name:16s} {model.body_mass[i]:7.3f} kg   I=[{ix:.4f} {iy:.4f} {iz:.4f}]")
    print("-" * 70)
    for i in range(model.nu):
        aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        jid = int(model.actuator_trnid[i, 0])
        lo, hi = np.degrees(model.jnt_range[jid])
        print(f"    {aname:14s} range [{lo:7.1f}, {hi:7.1f}] deg   |tau| <= "
              f"{model.actuator_forcerange[i, 1]:6.1f} Nm")
    print("=" * 70)
    return total


def check_structure(model, total) -> bool:
    print("\n[1] STRUCTURE CHECKS")
    c = Checklist()
    c(model.nu == 12, f"12 actuators                    (got {model.nu})")
    c(model.nv == 18, f"18 DoF = 6 floating base + 12   (got {model.nv})")
    c(model.nq == 19, f"19 coords = 7 base + 12         (got {model.nq})")
    c(abs(total - 32.0) < 0.5, f"total mass 32.0 +/- 0.5 kg      (got {total:.3f})")
    c((model.actuator_forcerange[:, 1] > 0).all(), "all actuators have a torque limit")
    c(model.nkey >= 1, "'stand' keyframe present")
    return c.ok


def check_contact(model, data, total) -> bool:
    """Place the soles ~2 mm above the floor, hold joints with PD, let it
    settle for 0.4 s (well inside the stable window) and check the contact."""
    print("\n[2] CONTACT / GROUND-REACTION CHECK  (0.4 s PD-held settle)")
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(model, data, kid)
    data.qpos[2] -= 0.020
    mujoco.mj_forward(model, data)

    q_adr, v_adr = actuator_qv_adr(model)
    kp, kd = gains(model)
    tau_lim = model.actuator_forcerange[:, 1].copy()
    q_star = data.qpos[q_adr].copy()

    for _ in range(int(0.4 / model.opt.timestep)):
        tau = kp * (q_star - data.qpos[q_adr]) - kd * data.qvel[v_adr]
        data.ctrl[:] = np.clip(tau, -tau_lim, tau_lim)
        mujoco.mj_step(model, data)

    lt = float(data.sensor("L_touch").data[0])
    rt = float(data.sensor("R_touch").data[0])
    grf_ratio = (lt + rt) / (total * GRAVITY)
    pen = -min(sole_min_corner_z(model, data, "L_sole"),
               sole_min_corner_z(model, data, "R_sole"))
    print(f"  base height        : {data.qpos[2]:.3f} m   (tilt {base_tilt_deg(data):.1f} deg)")
    print(f"  vertical GRF       : L={lt:7.1f} N   R={rt:7.1f} N   total/mg = {grf_ratio:.2f}")
    print(f"  max foot penetration: {pen * 1000:.2f} mm")

    c = Checklist()
    c(np.all(np.isfinite(data.qpos)), "no NaN/Inf")
    c(lt > 40.0 and rt > 40.0, "both feet carry load (> 40 N each)")
    c(0.75 < grf_ratio < 1.25, "total vertical GRF ~= body weight (0.75-1.25 mg)")
    c(pen < 0.003, "foot penetration < 3 mm")
    return c.ok


def check_numeric(model, data, seconds) -> bool:
    """Small random joint torques -> integration must stay finite and bounded."""
    print(f"\n[3] NUMERIC ROBUSTNESS  ({seconds:.1f}s of small random torques)")
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(model, data, kid)
    tau_lim = model.actuator_forcerange[:, 1].copy()
    rng = np.random.default_rng(0)
    nan_hit = False
    for _ in range(int(seconds / model.opt.timestep)):
        data.ctrl[:] = 0.10 * tau_lim * rng.uniform(-1, 1, model.nu)
        mujoco.mj_step(model, data)
        if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))):
            nan_hit = True
            break
    vmax = float(np.abs(data.qvel).max())
    print(f"  max |qvel| over run : {vmax:.2f}")
    c = Checklist()
    c(not nan_hit, "no NaN/Inf during 500 Hz integration")
    c(vmax < 80.0, "no velocity blow-up (max |qvel| < 80)")
    return c.ok


def report_topple(model, data):
    """Informational: how long pure joint-PD holds before the uncontrolled
    base pendulum diverges. Motivates the Milestone 2 standing controller."""
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(model, data, kid)
    data.qpos[2] -= 0.020
    mujoco.mj_forward(model, data)
    q_adr, v_adr = actuator_qv_adr(model)
    kp, kd = gains(model)
    tau_lim = model.actuator_forcerange[:, 1].copy()
    q_star = data.qpos[q_adr].copy()
    t_topple = None
    for k in range(int(10.0 / model.opt.timestep)):
        tau = kp * (q_star - data.qpos[q_adr]) - kd * data.qvel[v_adr]
        data.ctrl[:] = np.clip(tau, -tau_lim, tau_lim)
        mujoco.mj_step(model, data)
        if t_topple is None and base_tilt_deg(data) > 30.0:
            t_topple = k * model.opt.timestep
            break
    print("\n[i] INFO  time-to-topple under pure joint PD (no balance): "
          + (f"{t_topple:.2f} s" if t_topple else "> 10 s")
          + "   -> Milestone 2 adds the standing/balance controller")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="BR-1 Milestone 1 model check")
    ap.add_argument("--seconds", type=float, default=3.0, help="duration of the numeric test")
    args = ap.parse_args()

    if not SCENE.exists():
        print(f"ERROR: model not found: {SCENE}")
        sys.exit(2)

    model, data = load()
    total = report_structure(model)

    ok = check_structure(model, total)
    model, data = load()
    ok &= check_contact(model, data, total)
    model, data = load()
    ok &= check_numeric(model, data, args.seconds)
    model, data = load()
    report_topple(model, data)

    print("\n" + ("MILESTONE 1: ALL CHECKS PASSED" if ok else "MILESTONE 1: SOME CHECKS FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
