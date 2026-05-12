"""
train_rl.py
-----------
PPO training for phase-1 single-player return policy.

The agent's action is a small residual added to ClassicalRobotPlayer's joint
targets, so training only learns the deviations that improve on the classical
baseline.

Usage
-----
    python train_rl.py                       # default 2M steps, 16 envs
    python train_rl.py --timesteps 100000    # quick smoke test
    python train_rl.py --n-envs 8            # match physical core count
    python train_rl.py --algo sac --n-envs 1

Logs and checkpoints land in runs/<algo>_<timestamp>/.
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
    VecNormalize,
)

from rl_env import TableTennisRLEnv


def make_env(rank: int, seed: int = 0):
    def _thunk():
        env = TableTennisRLEnv(gui=False, seed=seed + rank)
        env = Monitor(env)
        return env

    return _thunk


def build_vec_env(n_envs: int, seed: int):
    if n_envs == 1:
        vec = DummyVecEnv([make_env(0, seed=seed)])
    else:
        vec = SubprocVecEnv([make_env(i, seed=seed) for i in range(n_envs)])
    vec = VecMonitor(vec)
    vec = VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.0)
    return vec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load", type=str, default=None,
                        help="Path to a saved model.zip to continue training from.")
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()

    if args.algo == "sac" and args.n_envs > 4:
        print(f"[warn] SAC is off-policy; n_envs={args.n_envs} buys little. "
              f"Consider --n-envs 1.")

    run_name = args.name or f"{args.algo}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logging to {run_dir.resolve()}")

    train_env = build_vec_env(args.n_envs, seed=args.seed)
    eval_env = build_vec_env(1, seed=args.seed + 10_000)
    eval_env.training = False
    eval_env.norm_reward = False

    if args.algo == "ppo":
        if args.load:
            model = PPO.load(args.load, env=train_env, tensorboard_log=str(run_dir))
        else:
            model = PPO(
                "MlpPolicy",
                train_env,
                learning_rate=2e-4,
                n_steps=1024,
                batch_size=512,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.0,
                policy_kwargs=dict(
                    net_arch=[256, 256],
                    log_std_init=-1.5,
                ),
                tensorboard_log=str(run_dir),
                verbose=1,
                seed=args.seed,
                device="cpu",
            )
    else:
        if args.load:
            model = SAC.load(args.load, env=train_env, tensorboard_log=str(run_dir))
        else:
            model = SAC(
                "MlpPolicy",
                train_env,
                learning_rate=3e-4,
                buffer_size=200_000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1,
                policy_kwargs=dict(net_arch=[256, 256]),
                tensorboard_log=str(run_dir),
                verbose=1,
                seed=args.seed,
            )

    ckpt_cb = CheckpointCallback(
        save_freq=max(50_000 // max(args.n_envs, 1), 1_000),
        save_path=str(run_dir / "checkpoints"),
        name_prefix="model",
        save_vecnormalize=True,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best"),
        log_path=str(run_dir / "eval"),
        eval_freq=max(20_000 // max(args.n_envs, 1), 1_000),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    print(f"Training {args.algo.upper()} for {args.timesteps:,} steps "
          f"on {args.n_envs} env(s) ...")
    t0 = time.time()
    model.learn(
        total_timesteps=args.timesteps,
        callback=[ckpt_cb, eval_cb],
        tb_log_name="train",
        progress_bar=True,
    )
    elapsed = time.time() - t0
    print(f"Done in {elapsed/60:.1f} min "
          f"({args.timesteps/elapsed:.0f} steps/sec).")

    final_path = run_dir / "model.zip"
    model.save(final_path)
    train_env.save(str(run_dir / "vecnormalize.pkl"))
    print(f"Saved final model to {final_path}")


if __name__ == "__main__":
    main()
