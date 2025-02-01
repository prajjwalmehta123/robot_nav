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
        env = CustomMonitor(env,allow_early_resets=True)
        return env
    return _init


def train_robot(config=None):
    # Initialize wandb with improved config
    run = wandb.init(
        project="robot-navigation",
        config={
            "algorithm": "SAC",
            "learning_rate": 3e-4,
            "batch_size": 256,
            "buffer_size": 1000000,
            "learning_starts": 5000,
            "train_freq": 1,
            "gradient_steps": 1,
            "ent_coef": "auto",
            "total_timesteps": 4_000_000,
            "tau": 0.02,
            "gamma": 0.98,
            "policy_kwargs": {
                "net_arch": {
                    "pi": [256, 256],
                    "qf": [256, 256]
                },
                "optimizer_kwargs": {
                    "weight_decay": 1e-5,
                    "eps": 1e-5
                }
            },
            "checkpoint_freq": 50000,
            "max_no_improvement_evals": 50,
        }
    )

    # Setup device
    device = setup_cuda()

    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"logs/{run.name}_{timestamp}/"
    os.makedirs(log_dir, exist_ok=True)

    # Create environments
    env, eval_env = create_envs(log_dir, run.config)

    # Setup callbacks
    callbacks = setup_callbacks(env, eval_env, log_dir, run.config)

    # Initialize or load model
    model = load_or_create_model(env, device, log_dir, run.config)

    try:
        model.learn(
            total_timesteps=run.config.total_timesteps,
            callback=callbacks,
            progress_bar=True,
            tb_log_name="SAC"
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
    except Exception as e:
        print(f"\nError during training: {e}")
        raise
    finally:
        cleanup_training(model, env, log_dir)

def create_envs(log_dir, config):
    """Create training and eval environments with improved normalization."""
    env = DummyVecEnv([make_env(difficulty=0, seed=i) for i in range(2)])
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.,
        clip_reward=10.,
        gamma=config.gamma,
        epsilon=1e-8
    )

    eval_env = DummyVecEnv([make_env(difficulty=0, eval_env=True)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        gamma=config.gamma,
        training=False
    )

    # Share normalization statistics
    eval_env.obs_rms = env.obs_rms
    eval_env.ret_rms = env.ret_rms

    return env, eval_env


def setup_callbacks(env, eval_env, log_dir, config):
    """Setup training callbacks with improved parameters."""
    # Evaluation callback with more lenient parameters
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{log_dir}/best_model",
        log_path=f"{log_dir}/results",
        eval_freq=5000,         # More frequent evaluation
        n_eval_episodes=5,      # Fewer evaluation episodes
        deterministic=True,
        render=False
    )

    # Modified curriculum callback
    curriculum_callback = CurriculumCallback(
        eval_env=eval_env,
        difficulty_threshold=0.6,
        difficulty_decrease_threshold=0.3,
        check_freq=10000,           # More frequent checks
        min_episodes=20,            # Fewer required episodes
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
        model_save_path=f"{log_dir}/wandb_checkpoints",
        verbose=2
    )

    return CallbackList([
        eval_callback,
        curriculum_callback,
        checkpoint_callback,
        wandb_callback
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


def create_new_model(env, device, config, log_dir):
    """Create a new SAC model with proper configuration."""
    return SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        buffer_size=config.buffer_size,
        learning_starts=config.learning_starts,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        tau=0.02,
        gamma=0.98,
        policy_kwargs={
            "net_arch": dict(pi=[256, 256], qf=[256, 256]),
            "optimizer_kwargs": dict(
                weight_decay=1e-5,
                eps=1e-5
            )
        },
        device=device,
        tensorboard_log=f"{log_dir}/tensorboard/"
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
        try:
            import pybullet as p
            if p.isConnected():
                p.disconnect()
        except Exception as e:
            print(f"Warning: PyBullet cleanup error: {e}")
        # Close environments
        try:
            env.close()
        except Exception as e:
            print(f"Warning: Environment cleanup error: {e}")

    except Exception as e:
        print(f"Error during cleanup: {e}")
    finally:
        # Ensure wandb is properly closed
        wandb.finish()


class EarlyStoppingCallback(EvalCallback):
    def __init__(
            self,
            eval_env,
            max_no_improvement=30,
            min_evals=10,
            eval_freq=20000,
            verbose=0
    ):
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
        self.min_evals = min_evals
        self.no_improvement_count = 0
        self.best_mean_reward = -np.inf
        self.eval_count = 0

    def _on_step(self) -> bool:
        continue_training = super()._on_step()

        if continue_training is False:
            return False

        if self.eval_count < self.min_evals:
            self.eval_count += 1
            return True

        if self.last_mean_reward > self.best_mean_reward:
            self.best_mean_reward = self.last_mean_reward
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        if self.no_improvement_count >= self.max_no_improvement:
            if self.verbose > 0:
                print(f"Early stopping triggered after {self.no_improvement_count} "
                      f"evaluations without improvement.")
            return False
        return True

class CustomMonitor(Monitor):
    def __init__(self, env, filename=None, allow_early_resets=True, reset_keywords=()):
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