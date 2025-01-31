import glob
from datetime import datetime

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
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
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()

        torch.backends.cudnn.benchmark = True
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory Usage: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
        return device
    return torch.device("cpu")


def make_env(difficulty=0, eval_env=False, seed=0):
    def _init():
        env = RobotNavEnv(
            render_mode=None,
            difficulty=difficulty
        )
        env.reset(seed=seed)
        env = CustomMonitor(env)
        return env
    return _init


def train_robot(config=None):
    # Initialize wandb with improved config
    run = wandb.init(
        project="robot-navigation",
        config={
            "algorithm": "SAC",
            "learning_rate": 1e-4,
            "batch_size": 1024,
            "buffer_size": 1000000,
            "learning_starts": 25000,
            "train_freq": 1,
            "gradient_steps": 2,
            "ent_coef": "auto",
            "total_timesteps": 4_000_000,
            "gradient_clip": 0.5,  # Added gradient clipping
            "policy_kwargs": {
                "net_arch": {
                    "pi": [512, 512, 256],
                    "qf": [512, 512, 256]
                },
                "optimizer_kwargs": {
                    "weight_decay": 1e-5
                }
            },
            "checkpoint_freq": 100000,  # Save every 100k steps
            "max_no_improvement_evals": 10,  # Early stopping patience
        }
    )

    # Setup device with proper error handling
    device = setup_cuda()

    # Create log directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"logs/{run.name}_{timestamp}/"
    os.makedirs(log_dir, exist_ok=True)

    # Create and normalize environments
    env, eval_env = create_envs(log_dir)

    # Setup callbacks with early stopping
    callbacks = setup_callbacks(env, eval_env, log_dir, run.config)

    # Initialize or load model
    model = load_or_create_model(env, device, log_dir, run.config)

    # Train with proper error handling and cleanup
    try:
        model.learn(
            total_timesteps=run.config.total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
    except Exception as e:
        print(f"\nError during training: {e}")
        raise
    finally:
        # Save final model and cleanup
        cleanup_training(model, env, log_dir)

def create_envs(log_dir):
    """Create training and eval environments with proper normalization."""
    env = DummyVecEnv([make_env(difficulty=0, seed=i) for i in range(4)])
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.,
        clip_reward=10.,
        gamma=0.99
    )

    eval_env = DummyVecEnv([make_env(difficulty=0, eval_env=True)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=env.norm_obs,
        norm_reward=env.norm_reward,
        gamma=env.gamma,
        clip_obs=env.clip_obs,
        clip_reward=env.clip_reward,
        training=False
    )

    # Share normalization statistics
    eval_env.obs_rms = env.obs_rms
    eval_env.ret_rms = env.ret_rms

    return env, eval_env


def setup_callbacks(env, eval_env, log_dir, config):
    """Setup training callbacks with early stopping."""
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
        difficulty_threshold=0.75,
        difficulty_decrease_threshold=0.4,
        check_freq=20000,
        min_episodes=50,
        verbose=1
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=config.checkpoint_freq,
        save_path=f"{log_dir}/checkpoints",
        name_prefix="model",
        save_vecnormalize=True
    )

    wandb_callback = WandbCallback(
        gradient_save_freq=100,
        model_save_freq=config.checkpoint_freq,
        model_save_path=f"{log_dir}/wandb_checkpoints",
        verbose=2
    )

    early_stopping = EarlyStoppingCallback(
        eval_env=eval_env,
        max_no_improvement=config.max_no_improvement_evals,
        eval_freq=10000,
        verbose=1
    )

    return CallbackList([
        eval_callback,
        curriculum_callback,
        checkpoint_callback,
        wandb_callback,
        early_stopping
    ])


def load_or_create_model(env, device, log_dir, config):
    """Create new model or load from checkpoint if available."""
    checkpoint_path = find_latest_checkpoint(f"{log_dir}/checkpoints")

    if checkpoint_path:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        try:
            model = SAC.load(
                checkpoint_path,
                env=env,
                device=device,
                custom_objects={
                    "learning_rate": config.learning_rate,
                    "batch_size": config.batch_size,
                    "buffer_size": config.buffer_size
                }
            )
            if os.path.exists(f"{checkpoint_path}_optimizer.pth"):
                model.policy.optimizer.load_state_dict(torch.load(f"{checkpoint_path}_optimizer.pth"))
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Creating new model instead.")
            model = create_new_model(env, device, config,log_dir)
    else:
        model = create_new_model(env, device, config,log_dir)

    return model


def create_new_model(env, device, config,log_dir):
    """Create a new SAC model with proper configuration."""
    return SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        buffer_size=config.buffer_size,
        learning_starts=config.learning_starts,
        train_freq=config.train_freq,
        gradient_steps=config.gradient_steps,
        ent_coef=config.ent_coef,
        policy_kwargs=config.policy_kwargs,
        device=device,
        tensorboard_log=f"{log_dir}/tensorboard/",
    )


def find_latest_checkpoint(checkpoint_dir):
    """Find the most recent checkpoint in the directory."""
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = glob.glob(os.path.join(checkpoint_dir, "model_*.zip"))
    if not checkpoints:
        return None

    return max(checkpoints, key=os.path.getctime)


def cleanup_training(model, env, log_dir):
    """Proper cleanup of training resources."""
    try:
        # Save final model state
        model.save(f"{log_dir}/final_model")

        # Only save optimizer state if available
        if hasattr(model.policy, "optimizer"):
            torch.save(
                model.policy.optimizer.state_dict(),
                f"{log_dir}/final_model_optimizer.pth"
            )

        # Save normalization statistics
        env.save(f"{log_dir}/vec_normalize.pkl")

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Close environments
        env.close()

    except Exception as e:
        print(f"Error during cleanup: {e}")
    finally:
        # Ensure wandb is properly closed
        wandb.finish()


class EarlyStoppingCallback(EvalCallback):  # Inherit from EvalCallback
    def __init__(
            self,
            eval_env,
            max_no_improvement=5,
            eval_freq=10000,
            verbose=0
    ):
        # Initialize parent EvalCallback
        super().__init__(
            eval_env,
            best_model_save_path=None,
            log_path=None,
            eval_freq=eval_freq,
            n_eval_episodes=10,
            deterministic=True,
            verbose=verbose
        )
        self.max_no_improvement = max_no_improvement
        self.no_improvement_count = 0
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        super()._on_step()
        if self.last_mean_reward > self.best_mean_reward:
            self.best_mean_reward = self.last_mean_reward
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        if self.no_improvement_count >= self.max_no_improvement:
            if self.verbose > 0:
                print(f"Early stopping triggered after {self.no_improvement_count} evaluations without improvement.")
            return False
        return True

class CustomMonitor(Monitor):
    def __init__(self, env, filename=None, allow_early_resets=False, reset_keywords=()):
        super().__init__(env, filename, allow_early_resets, reset_keywords)
        self.episode_info_buffer = []

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        if terminated or truncated:
            self.episode_info_buffer.append(info.copy())
        return observation, reward, terminated, truncated, info

    def get_episode_info_buffer(self):
        buffer = self.episode_info_buffer.copy()
        self.episode_info_buffer = []
        return buffer

if __name__ == "__main__":
    wandb.login()
    train_robot()