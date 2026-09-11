"""
Milestone 2 verification: control/pd_controller.py + control/actuator.py +
control/balance_controller.py.

CHECKS
  1. Posture-only PD (no balance) topples within ~1.5-2.5 s -- confirms the
     M1 finding still holds and that the standing controller is doing real
     work, not riding on a model that was secretly already stable.
  2. StandingBalanceController holds for 10 s with no disturbance: small
     tilt, base height near nominal, both feet loaded.
  3. Push recovery at two conservative magnitudes (well inside the ~0.3 m/s
     fore-aft / ~0.2 m/s lateral envelope measured during tuning): must not
     fall.
  4. Informational: coarse max-recoverable-push scan in both directions.

USAGE
  python scripts/test_controller.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import mujoco

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
SCENE = REPO / "mujoco" / "scene.xml"

from control.actuator import ActuatorInterface          # noqa: E402
from control.pd_controller import PDController          # noqa: E402
from control.balance_controller import StandingBalanceController, POSTURE_GAINS  # noqa: E402


class Checklist:
    def __init__(self):
        self.ok = True

    def __call__(self, cond, msg):
        cond = bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
        self.ok &= cond


def load():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    return model, mujoco.MjData(model)


def nominal_qpos(model, act: ActuatorInterface):
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    return model.key_qpos[kid][act.qpos_adr].copy()


def base_tilt_deg(data) -> float:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, data.qpos[3:7])
    return float(np.degrees(np.arccos(np.clip(R.reshape(3, 3)[2, 2], -1.0, 1.0))))


def reset_standing(model, data):
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(model, data, kid)
    data.qpos[2] -= 0.020
    mujoco.mj_forward(model, data)


# --------------------------------------------------------------------------- #
def test_posture_only_topples(model, data) -> tuple[bool, float]:
    print("\n[1] SANITY: posture-only PD (no balance) must still topple")
    act = ActuatorInterface(model)
    q_nom = nominal_qpos(model, act)
    kp, kd = act.gains_by_suffix(POSTURE_GAINS)
    pd = PDController(kp, kd, act.torque_limit)
    reset_standing(model, data)
    t_topple = None
    for k in range(int(6.0 / model.opt.timestep)):
        q, qd = act.joint_state(data)
        act.apply(data, pd.compute(q, qd, q_nom))
        mujoco.mj_step(model, data)
        if t_topple is None and base_tilt_deg(data) > 30.0:
            t_topple = k * model.opt.timestep
            break
    print(f"  time-to-topple (posture-only): {t_topple}")
    c = Checklist()
    c(t_topple is not None and t_topple < 3.0, "posture-only PD topples within 3 s (as in M1)")
    return c.ok, t_topple


def run_balance(model, data, seconds, push_xy=None, push_time=1.5):
    act = ActuatorInterface(model)
    q_nom = nominal_qpos(model, act)
    ctrl = StandingBalanceController(model, q_nom)
    reset_standing(model, data)
    ctrl.reset(data)
    dt = model.opt.timestep
    pushed = False
    max_tilt = 0.0
    for k in range(int(seconds / dt)):
        tau = ctrl.compute(data, dt)
        act.apply(data, tau)
        if push_xy is not None and not pushed and k * dt >= push_time:
            data.qvel[0] += push_xy[0]
            data.qvel[1] += push_xy[1]
            pushed = True
        mujoco.mj_step(model, data)
        max_tilt = max(max_tilt, base_tilt_deg(data))
        if max_tilt > 45.0:
            return False, max_tilt, k * dt
    return True, max_tilt, seconds


def test_static_hold(model, data) -> bool:
    print("\n[2] STANDING BALANCE CONTROLLER: static hold, 10 s, no disturbance")
    ok_run, max_tilt, _ = run_balance(model, data, seconds=10.0)
    lt = float(data.sensor("L_touch").data[0])
    rt = float(data.sensor("R_touch").data[0])
    print(f"  final base height : {data.qpos[2]:.3f} m")
    print(f"  max tilt over run : {max_tilt:.2f} deg")
    print(f"  final foot GRF    : L={lt:.1f} N  R={rt:.1f} N")
    c = Checklist()
    c(ok_run, "did not fall in 10 s")
    c(max_tilt < 10.0, "max tilt stayed under 10 deg")
    c(data.qpos[2] > 0.75, "base height held near nominal (> 0.75 m)")
    c(lt > 40.0 and rt > 40.0, "both feet still loaded")
    return c.ok


def test_push_recovery(model, data) -> bool:
    print("\n[3] PUSH RECOVERY  (conservative magnitudes, well inside the tuned envelope)")
    c = Checklist()
    for label, push in [("forward +0.15 m/s", (0.15, 0.0)), ("lateral +0.15 m/s", (0.0, 0.15))]:
        ok_run, max_tilt, t_end = run_balance(model, data, seconds=6.0, push_xy=push)
        print(f"  {label:20s} -> {'recovered' if ok_run else f'FELL at t={t_end:.2f}s'}"
              f"  (max tilt {max_tilt:.1f} deg)")
        c(ok_run, f"recovers from {label}")
    return c.ok


def report_push_envelope(model, data):
    print("\n[i] INFO  coarse max-recoverable-push scan")
    for axis, name in [((1, 0), "fore-aft"), ((0, 1), "lateral")]:
        last_ok = 0.0
        for mag in np.arange(0.05, 0.55, 0.05):
            push = (axis[0] * mag, axis[1] * mag)
            ok_run, _, _ = run_balance(model, data, seconds=5.0, push_xy=push)
            if ok_run:
                last_ok = mag
            else:
                break
        print(f"  {name:10s}: recovers up to ~{last_ok:.2f} m/s CoM velocity kick")


# --------------------------------------------------------------------------- #
def main():
    if not SCENE.exists():
        print(f"ERROR: model not found: {SCENE}")
        sys.exit(2)

    model, data = load()
    ok1, _ = test_posture_only_topples(model, data)

    model, data = load()
    ok2 = test_static_hold(model, data)

    model, data = load()
    ok3 = test_push_recovery(model, data)

    model, data = load()
    report_push_envelope(model, data)

    ok = ok1 and ok2 and ok3
    print("\n" + ("MILESTONE 2: ALL CHECKS PASSED" if ok else "MILESTONE 2: SOME CHECKS FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
