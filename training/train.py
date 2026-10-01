"""
Milestone 4: train a standing policy with PPO.

  python training/train.py --exp-id EXP-001 --timesteps 300000 --n-envs 8

Reads algorithm/env hyperparameters from config/training.yaml (ppo.*);
CLI flags override the file for quick experiments without editing it.

Logs TensorBoard to          logs/<exp-id>/
Periodic checkpoints to      checkpoints/<exp-id>/
Best-eval + final model to   policies/<exp-id>_best/  and  policies/<exp-id>_final.zip
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from env.biped_env import BipedEnv  # noqa: E402
from training.callbacks import EpisodeStatsCallback  # noqa: E402


def make_env(seed: int):
    def _init():
        return Monitor(BipedEnv(seed=seed))
    return _init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="EXP-001")
    ap.add_argument("--timesteps", type=int, default=None,
                     help="override config/training.yaml ppo.total_timesteps")
    ap.add_argument("--n-envs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    with open(REPO / "config" / "training.yaml") as f:
        cfg = yaml.safe_load(f)
    ppo_cfg = cfg["ppo"]

    n_envs = args.n_envs or ppo_cfg["n_envs"]
    seed = args.seed if args.seed is not None else ppo_cfg["seed"]
    total_timesteps = args.timesteps or ppo_cfg["total_timesteps"]

    log_dir = REPO / "logs" / args.exp_id
    ckpt_dir = REPO / "checkpoints" / args.exp_id
    policy_dir = REPO / "policies"
    for d in (log_dir, ckpt_dir, policy_dir):
        d.mkdir(parents=True, exist_ok=True)

    env_fns = [make_env(seed + i) for i in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns) if n_envs > 1 else DummyVecEnv(env_fns)
    eval_env = DummyVecEnv([make_env(seed + 1000)])

    model = PPO(
        "MlpPolicy", vec_env, verbose=1, seed=seed,
        n_steps=ppo_cfg["n_steps"], batch_size=ppo_cfg["batch_size"],
        n_epochs=ppo_cfg["n_epochs"], gamma=ppo_cfg["gamma"],
        gae_lambda=ppo_cfg["gae_lambda"], learning_rate=ppo_cfg["learning_rate"],
        clip_range=ppo_cfg["clip_range"], ent_coef=ppo_cfg["ent_coef"],
        tensorboard_log=str(log_dir),
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    callbacks = CallbackList([
        EpisodeStatsCallback(),
        CheckpointCallback(save_freq=max(50_000 // n_envs, 1), save_path=str(ckpt_dir),
                            name_prefix=args.exp_id),
        EvalCallback(eval_env, best_model_save_path=str(policy_dir / f"{args.exp_id}_best"),
                     log_path=str(log_dir), eval_freq=max(10_000 // n_envs, 1),
                     n_eval_episodes=5, deterministic=True),
    ])

    print(f"[{args.exp_id}] training {total_timesteps} steps, {n_envs} envs, seed {seed}")
    model.learn(total_timesteps=total_timesteps, callback=callbacks, tb_log_name=args.exp_id)

    final_path = policy_dir / f"{args.exp_id}_final.zip"
    model.save(str(final_path))
    print(f"[{args.exp_id}] saved final model -> {final_path}")


if __name__ == "__main__":
    main()
