"""
Evaluate a trained BR-1 policy (Part 17 gait/stability metrics).

  python training/evaluate.py --model policies/EXP-001_final.zip --episodes 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from env.biped_env import BipedEnv  # noqa: E402


def evaluate(model_path: str, episodes: int, seed: int = 1234, deterministic: bool = True) -> dict:
    model = PPO.load(model_path)
    env = BipedEnv(seed=seed)

    ep_rewards, ep_lengths_s = [], []
    falls = 0
    tilt_all: list[float] = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total_r = 0.0
        steps = 0
        termination_reason = ""
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, terminated, truncated, info = env.step(action)
            total_r += r
            steps += 1
            tilt_all.append(info["tilt_deg"])
            termination_reason = info["termination_reason"]
            done = terminated or truncated
        ep_rewards.append(total_r)
        ep_lengths_s.append(steps * env.dt_ctrl)
        if termination_reason.startswith("fell"):
            falls += 1

    result = {
        "mean_reward": float(np.mean(ep_rewards)),
        "std_reward": float(np.std(ep_rewards)),
        "mean_episode_length_s": float(np.mean(ep_lengths_s)),
        "fall_rate": falls / episodes,
        "mean_tilt_deg": float(np.mean(tilt_all)),
        "episodes": episodes,
    }

    print("=" * 60)
    print(f"EVALUATION  {model_path}  ({episodes} episodes)")
    print("=" * 60)
    print(f"  mean episode reward : {result['mean_reward']:.2f} +/- {result['std_reward']:.2f}")
    print(f"  mean episode length : {result['mean_episode_length_s']:.2f} s")
    print(f"  fall rate           : {result['fall_rate'] * 100:.1f} %  ({falls}/{episodes})")
    print(f"  mean tilt            : {result['mean_tilt_deg']:.2f} deg")
    print("=" * 60)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--stochastic", action="store_true", help="sample actions instead of argmax-mean")
    args = ap.parse_args()
    evaluate(args.model, args.episodes, args.seed, deterministic=not args.stochastic)


if __name__ == "__main__":
    main()
