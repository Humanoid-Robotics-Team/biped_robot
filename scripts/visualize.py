"""
Interactive viewer for BR-1.

  python scripts/visualize.py             # standing balance controller (M2, default)
                                          # -> stands indefinitely; try dragging
                                          #    the robot with the mouse (ctrl+
                                          #    right-click-drag applies a force
                                          #    in the MuJoCo viewer) to test push
                                          #    recovery interactively.
  python scripts/visualize.py --posture   # posture-only PD, no balance (M1
                                          # behaviour) -> topples after ~1.8 s
  python scripts/visualize.py --limp      # zero control -> collapses immediately

Controls in the viewer window: drag = orbit, right-drag = pan, scroll = zoom,
ctrl + right-drag on the robot = apply a push force, space = pause,
right-arrow = step while paused. Close the window to exit.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
SCENE = REPO / "mujoco" / "scene.xml"

from control.actuator import ActuatorInterface                     # noqa: E402
from control.pd_controller import PDController                     # noqa: E402
from control.balance_controller import StandingBalanceController, POSTURE_GAINS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--posture", action="store_true", help="posture-only PD (M1, will topple)")
    mode.add_argument("--limp", action="store_true", help="zero control")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(model, data, kid)
    data.qpos[2] -= 0.020
    mujoco.mj_forward(model, data)

    act = ActuatorInterface(model)
    q_nom = model.key_qpos[kid][act.qpos_adr].copy()

    if args.limp:
        controller = None
    elif args.posture:
        kp, kd = act.gains_by_suffix(POSTURE_GAINS)
        pd = PDController(kp, kd, act.torque_limit)
        controller = lambda d, dt: pd.compute(*act.joint_state(d), q_nom)  # noqa: E731
    else:
        bal = StandingBalanceController(model, q_nom)
        bal.reset(data)
        controller = lambda d, dt: bal.compute(d, dt)  # noqa: E731

    dt = model.opt.timestep
    with mujoco.viewer.launch_passive(model, data) as viewer:
        wall = time.perf_counter()
        while viewer.is_running():
            if controller is not None:
                act.apply(data, controller(data, dt))
            else:
                data.ctrl[:] = 0.0
            mujoco.mj_step(model, data)
            viewer.sync()

            wall += dt
            slack = wall - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            else:
                wall = time.perf_counter()


if __name__ == "__main__":
    main()
