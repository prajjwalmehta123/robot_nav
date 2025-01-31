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
        env = Monitor(env)
        return env
    return _init

def train_robot(config=None):
    # Initialize wandb with improved defaults
    run = wandb.init(
        project="robot-navigation",
        config={
            "algorithm": "SAC",
            "learning_rate": 3e-4,
            "batch_size": 2048,
            "buffer_size": 1000000,
            "learning_starts": 25000,
            "train_freq": 1,
            "gradient_steps": 1,
            "ent_coef": "auto",
            "total_timesteps": 4_000_000,
            "policy_kwargs": {
                "net_arch": {
                    "pi": [512, 256],
                    "qf": [512, 256]
                },
                "optimizer_kwargs": {
                    "weight_decay": 1e-5
                }
            }
        }
    )

    # Setup device
    device =  setup_cuda()

    # Create log directory
    log_dir = f"logs/{run.name}/"
    os.makedirs(log_dir, exist_ok=True)

    # Create environments
    env = DummyVecEnv([make_env(difficulty=0, seed=i) for i in range(4)])
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.,
        clip_reward=10.
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
    eval_env.obs_rms = env.obs_rms
    eval_env.ret_rms = env.ret_rms

    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{log_dir}/best_model",
        log_path=f"{log_dir}/results",
        eval_freq=10000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        callback_after_eval= lambda locals_, globals_: wandb.log({
            "eval/mean_reward": locals_["mean_reward"],
            "eval/success_rate": np.mean([ep_info["is_success"] for ep_info in locals_["eval_info"]]),
            "eval/episode_length": np.mean([ep_info["episode_length"] for ep_info in locals_["eval_info"]]),
            "eval/collision_rate": np.mean([ep_info["collision"] for ep_info in locals_["eval_info"]]),
            "eval/min_distance": np.mean([ep_info["distance_to_target"] for ep_info in locals_["eval_info"]])
        })
    )

    curriculum_callback = CurriculumCallback(
        eval_env=eval_env,
        difficulty_threshold=0.7,
        check_freq=10000,
        verbose=1
    )

    wandb_callback = WandbCallback(
        gradient_save_freq=100,
        model_save_freq=10000,
        model_save_path=f"{log_dir}/checkpoints",
        verbose=2,
    )

    callback = CallbackList([
        eval_callback,
        curriculum_callback,
        wandb_callback
    ])

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