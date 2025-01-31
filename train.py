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
    if torch.cuda.is_available():
        n_cuda_devices = torch.cuda.device_count()
        print(f"Found {n_cuda_devices} CUDA device(s)")
        device = torch.device("cuda:0")  # Use first available GPU
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("No CUDA devices available, using CPU")
        device = torch.device("cpu")
    return device

def make_env(difficulty=0, eval_env=False):
    def _init():
        env = RobotNavEnv(
            render_mode=None,
            difficulty=difficulty
        )
        env = Monitor(env)
        return env
    return _init

def train_robot(config=None):
    # Initialize wandb with improved defaults
    run = wandb.init(
        project="robot-navigation",
        config={
            "algorithm": "SAC",
            "learning_rate": 1e-4,  # Reduced learning rate for stability
            "batch_size": 512,      # Increased batch size
            "buffer_size": 2000000, # Larger replay buffer
            "learning_starts": 25000, # More initial random actions
            "train_freq": 1,
            "gradient_steps": 1,
            "ent_coef": "auto",
            "total_timesteps": 4_000_000,  # Doubled training time
            "initial_difficulty": 0,
            "max_difficulty": 5,
            "policy_kwargs": {
                "net_arch": {
                    "pi": [512, 512, 256],  # Wider policy network
                    "qf": [512, 512, 256]   # Wider Q-function network
                }
            }
        }
    )

    # Setup device
    device = setup_cuda()

    # Create log directory
    log_dir = f"logs/{run.name}/"
    os.makedirs(log_dir, exist_ok=True)

    # Create environments
    env = DummyVecEnv([make_env(difficulty=0)])
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.,
        clip_reward=10.
    )

    # Create evaluation environment at slightly higher difficulty
    eval_env = DummyVecEnv([make_env(difficulty=0, eval_env=True)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.,
        clip_reward=10.,
        training=False  # Don't update normalization stats during eval
    )

    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{log_dir}/best_model",
        log_path=f"{log_dir}/results",
        eval_freq=10000,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )

    curriculum_callback = CurriculumCallback(
        eval_env=eval_env,
        difficulty_threshold=0.7,  # Reduced threshold
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

    # Initialize model with improved parameters
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=wandb.config.learning_rate,
        batch_size=wandb.config.batch_size,
        buffer_size=wandb.config.buffer_size,
        learning_starts=wandb.config.learning_starts,
        train_freq=wandb.config.train_freq,
        gradient_steps=wandb.config.gradient_steps,
        ent_coef=wandb.config.ent_coef,
        policy_kwargs=wandb.config.policy_kwargs,
        device=device,
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