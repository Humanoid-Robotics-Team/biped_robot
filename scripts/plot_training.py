"""
Plot training curves for a run (Part 18): episode reward, episode length,
and fall rate vs. training steps -- parsed from the SB3 console log and the
EvalCallback's evaluations.npz.

  python scripts/plot_training.py --exp-id EXP-002
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]


def parse_rollout_series(console_log: Path):
    """Pulls (total_timesteps, ep_len_mean, ep_rew_mean, fall_rate, mean_tilt_deg)
    from each 'rollout/' block SB3 prints to the console log."""
    text = console_log.read_text(encoding="utf-8", errors="ignore")
    blocks = re.findall(
        r"fall_rate\s*\|\s*([\d.]+).*?mean_tilt_deg\s*\|\s*([\d.]+).*?"
        r"ep_len_mean\s*\|\s*([\d.]+).*?ep_rew_mean\s*\|\s*([\d.-]+).*?"
        r"total_timesteps\s*\|\s*(\d+)",
        text, flags=re.S,
    )
    rows = [(int(ts), float(elen), float(erew), float(fr), float(tilt))
            for fr, tilt, elen, erew, ts in blocks]
    rows.sort(key=lambda r: r[0])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    args = ap.parse_args()

    console_log = REPO / "logs" / f"{args.exp_id}_console.log"
    eval_npz = REPO / "logs" / args.exp_id / "evaluations.npz"
    if not console_log.exists():
        print(f"ERROR: {console_log} not found")
        sys.exit(2)

    rollout = parse_rollout_series(console_log)
    ts = [r[0] for r in rollout]
    ep_len = [r[1] for r in rollout]
    ep_rew = [r[2] for r in rollout]
    fall_rate = [r[3] for r in rollout]
    tilt = [r[4] for r in rollout]

    eval_ts, eval_rew, eval_len = None, None, None
    if eval_npz.exists():
        d = np.load(eval_npz)
        eval_ts = d["timesteps"]
        eval_rew = d["results"].mean(axis=1)
        eval_len = d["ep_lengths"].mean(axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle(f"{args.exp_id} training curves")

    ax = axes[0, 0]
    ax.plot(ts, ep_rew, color="tab:blue", alpha=0.6, label="rollout (stochastic)")
    if eval_ts is not None:
        ax.plot(eval_ts, eval_rew, color="tab:orange", label="eval (deterministic)")
    ax.set_title("Episode Reward vs Training Steps")
    ax.set_xlabel("timesteps"); ax.set_ylabel("mean episode reward"); ax.legend()

    ax = axes[0, 1]
    ax.plot(ts, ep_len, color="tab:blue", alpha=0.6, label="rollout")
    if eval_ts is not None:
        ax.plot(eval_ts, eval_len, color="tab:orange", label="eval")
    ax.set_title("Episode Length vs Training Steps")
    ax.set_xlabel("timesteps"); ax.set_ylabel("mean episode length (steps)"); ax.legend()

    ax = axes[1, 0]
    ax.plot(ts, fall_rate, color="tab:red")
    ax.set_title("Fall Rate vs Training Steps (rollout)")
    ax.set_xlabel("timesteps"); ax.set_ylabel("fall rate"); ax.set_ylim(-0.05, 1.05)

    ax = axes[1, 1]
    ax.plot(ts, tilt, color="tab:green")
    ax.set_title("Mean Tilt vs Training Steps (rollout)")
    ax.set_xlabel("timesteps"); ax.set_ylabel("tilt (deg)")

    fig.tight_layout()
    out = REPO / "logs" / f"{args.exp_id}_curves.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
