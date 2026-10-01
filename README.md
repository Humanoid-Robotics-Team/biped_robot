# BR-1 — Bipedal Humanoid: Simulation + RL Locomotion Framework

First-generation simulated biped (**BR-1**) built in MuJoCo, trained with
reinforcement learning to stand, step, and walk — designed from day 1 to
transfer onto real hardware.

- **Robot:** 1.39 m, 32 kg, 12 actuated DOF (6 per leg) + 6-DOF floating base.
- **Actuation:** torque-source motors + a Python PD position loop (Option C).
- **Policy rate:** 50 Hz over 500 Hz physics (decimation 10).
- Full spec with confidence tags: [`config/robot.yaml`](config/robot.yaml).

## Status

| Milestone | Description | State |
|-----------|-------------|-------|
| M0 | Dev environment | verified |
| M1 | Static model: XML, joints, mass, collision, stable sim | verified |
| M2 | Low-level PD + standing balance controller | verified |
| M3 | Gymnasium environment: obs/action/reward/termination | verified |
| **M4** | **Standing RL policy (PPO)** | **achieved — EXP-002, 0% fall rate on 30 held-out episodes** |
| M5+ | Stepping → walking → robust → sim-to-real | not started |

**Best policy so far:** `policies/EXP-002_final.zip` — trained 2M PPO steps,
held-out eval: 0/30 episodes fell (full 10s survival), mean tilt 2.84°. See
[`experiments/EXP-registry.md`](experiments/EXP-registry.md) for the full
run history, including EXP-001's reward-normalization bug and its fix.
Watch it: `python scripts/visualize_policy.py --model policies/EXP-002_final.zip`

## Repository layout

```
config/     robot.yaml (spec), training.yaml (RL config, added at M3)
mujoco/     robot.xml (model), scene.xml (model + floor + lights) <- load this
robot/      sensors.py (noise model); kinematics.py/dynamics.py added when M5 needs them
env/        biped_env.py, observations.py, rewards.py, termination.py (M3 — done)
control/    pd_controller.py, actuator.py, balance_controller.py (M2 — done)
training/   train.py, evaluate.py, callbacks.py (added M4)
scripts/    test_robot.py, visualize.py, test_controller.py, test_env.py
tests/      pytest model/reward/obs checks
experiments/EXP-registry.md — one row per training run, append only
```

## Setup (Milestone 0)

Windows PowerShell, from the repo root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

GPU (CUDA 12.1) PyTorch, then the rest:

```powershell
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

CPU-only is fine through Milestone 6 — just `pip install -r requirements.txt`.

Verify:

```powershell
python -c "import mujoco, gymnasium, stable_baselines3, torch; print('mujoco', mujoco.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"
```

## Verify the model (Milestone 1)

```powershell
python scripts/test_robot.py            # structure + contact/GRF + numeric robustness
pytest tests/test_model.py              # fast model checks
```

Expected: `total mass 32.000 kg`, `nu 12 / nv 18 / nq 19`, all checks PASS.
The report also logs "time-to-topple under pure joint PD" (~1.8 s) — that's
expected, not a failure: joint PD holds posture, not balance. See Milestone 2.

## Verify the standing controller (Milestone 2)

```powershell
python scripts/test_controller.py       # posture-topple sanity + 10s hold + push recovery
pytest                                   # full suite (model + controller)
python scripts/visualize.py             # watch it stand indefinitely
python scripts/visualize.py --posture   # compare: posture-only PD, topples ~1.8s
```

Expected: static hold passes (tilt < 10°, both feet loaded, 10 s no fall),
push-recovery passes at 0.15 m/s in both directions, and the info line
reports the measured envelope (~0.25 m/s recoverable in this build). In the
viewer, ctrl+right-click-drag on the robot applies a push you can feel it
resist (or fall to, if you push hard enough — that boundary is real, not a
bug: a fixed-footprint ankle strategy is physically limited by the foot's
support polygon; bigger disturbances need Milestone 7/8 or a learned policy).

## Verify the RL environment (Milestone 3)

```powershell
python scripts/test_env.py     # gymnasium API, determinism, reward sanity, robustness
pytest                          # full suite: model + controller + env
```

Expected: all checks pass, including a zero-action episode that topples at
~1.8 s (the env is not secretly stabilizing anything — Milestone 4's job is
to train a policy that keeps the standing controller's stability without
hard-coding it).

Environment summary: `obs` is 47-dim (projected gravity, base angular
velocity, velocity command, joint positions/velocities relative to nominal,
previous action, gait-phase clock — see `env/observations.py`); `action` is
a 12-dim residual on the nominal joint pose, scaled ±0.30 rad and converted
to torque by the same `PDController` from Milestone 2 (`env/biped_env.py`).
`config/training.yaml` is the curriculum knob: at `curriculum_stage: stand`
all velocity commands are pinned to 0, so `lin_vel_track` simply rewards not
drifting. Later stages widen the command ranges — no environment code
changes.
