from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from RobotNavEnv import RobotNavEnv
import torch
import os
import numpy as np
from callbacks import CurriculumCallback
import wandb
from wandb.integration.sb3 import WandbCallback

def setup_cuda():
    """
    Set up CUDA devices and return the appropriate device for training.
    Returns:
        torch.device: The device to use for training
    """
    if torch.cuda.is_available():
        n_cuda_devices = torch.cuda.device_count()
        print(f"Found {n_cuda_devices} CUDA device(s)")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        device = torch.device("cuda:9")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("No CUDA devices available, using CPU")
        device = torch.device("cpu")

    return device


def make_env(difficulty=0, eval_env=False):
    def _init():
        env = RobotNavEnv(render_mode=None, difficulty=difficulty)
        env = Monitor(env)
        return env

    return _init


def train_robot(config=None):
    # Initialize wandb
    run = wandb.init(
        project="robot-navigation",
        config={
            "algorithm": "SAC",
            "learning_rate": 3e-4,
            "batch_size": 256,
            "buffer_size": 1000000,
            "learning_starts": 10000,
            "total_timesteps": 2_000_000,
            "initial_difficulty": 0,
            "max_difficulty": 5,
        }
    )
    setup_cuda()
    # Create log directory with wandb run name
    log_dir = f"logs/{run.name}/"
    os.makedirs(log_dir, exist_ok=True)

    # Create environments
    env = DummyVecEnv([make_env(difficulty=0)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    eval_env = DummyVecEnv([make_env(difficulty=0, eval_env=True)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)

    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{log_dir}/best_model",
        log_path=f"{log_dir}/results",
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    # Initialize the fixed curriculum callback
    curriculum_callback = CurriculumCallback(
        eval_env=eval_env,
        difficulty_threshold=0.8,
        check_freq=10000,
        verbose=1
    )

    wandb_callback = WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"{log_dir}/checkpoints",
        verbose=2,
    )

    callback = CallbackList([
        eval_callback,
        curriculum_callback,
        wandb_callback
    ])

    # Initialize model
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=wandb.config.learning_rate,
        batch_size=wandb.config.batch_size,
        buffer_size=wandb.config.buffer_size,
        learning_starts=wandb.config.learning_starts,
        ent_coef='auto',
        tensorboard_log=f"{log_dir}/tensorboard/"
    )

    # Train the model
    try:
        model.learn(
            total_timesteps=wandb.config.total_timesteps,
            callback=callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
    finally:
        model.save(f"{log_dir}/final_model")
        env.save(f"{log_dir}/vec_normalize.pkl")
        wandb.finish()


if __name__ == "__main__":
    wandb.login()
    train_robot()