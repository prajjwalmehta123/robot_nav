from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from RobotNavEnv import RobotNavEnv
import os
import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback


class CustomWandbCallback:
    def __init__(self, check_freq=1000):
        self.check_freq = check_freq
        self.n_calls = 0

    def __call__(self, locals_, globals_):
        self.n_calls += 1
        if self.n_calls % self.check_freq == 0:
            # Log episode statistics
            if locals_['infos']:
                episode_info = locals_['infos'][0]
                wandb.log({
                    "train/episode_reward": episode_info.get('episode', {}).get('r', 0),
                    "train/episode_length": episode_info.get('episode', {}).get('l', 0),
                    "current_difficulty": locals_['self'].env.get_attr('difficulty')[0],
                })
        return True


class CurriculumCallback:
    def __init__(self, eval_env, difficulty_threshold=0.8, check_freq=10000):
        self.eval_env = eval_env
        self.difficulty_threshold = difficulty_threshold
        self.check_freq = check_freq
        self.n_calls = 0

    def __call__(self, locals_, globals_):
        self.n_calls += 1
        if self.n_calls % self.check_freq == 0:
            mean_success = np.mean([info['is_success'] for info in locals_['infos']])

            if mean_success > self.difficulty_threshold:
                current_difficulty = self.eval_env.get_attr('difficulty')[0]
                new_difficulty = min(current_difficulty + 1, 5)
                print(f"\nIncreasing difficulty to {new_difficulty}")
                self.eval_env.set_attr('difficulty', new_difficulty)

                # Log difficulty change to wandb
                wandb.log({
                    "curriculum/difficulty": new_difficulty,
                    "curriculum/success_rate": mean_success
                })

        return True


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

    curriculum_callback = CurriculumCallback(eval_env)
    wandb_callback = WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"{log_dir}/checkpoints",
        verbose=2,
    )
    custom_wandb_callback = CustomWandbCallback()

    callback = CallbackList([
        eval_callback,
        curriculum_callback,
        wandb_callback,
        custom_wandb_callback
    ])
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