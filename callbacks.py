from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import wandb
from typing import List, Dict


class CurriculumCallback(BaseCallback):
    def __init__(self, eval_env, difficulty_threshold=0.8, check_freq=10000, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.difficulty_threshold = difficulty_threshold
        self.check_freq = check_freq
        self.n_calls = 0
        self.ep_info_buffer = []

    def _init_callback(self) -> None:
        """Initialize callback-specific variables."""
        self.ep_info_buffer = []

    def _on_step(self) -> bool:
        self.n_calls += 1

        rewards = [rewards for rewards in self.training_env.get_attr('rewards')]
        for env_rewards in rewards:
            if len(env_rewards) > 0:
                self.ep_info_buffer.append({'is_success': env_rewards[-1] > 0})

        if self.n_calls % self.check_freq == 0 and len(self.ep_info_buffer) > 10:
            recent_episodes = self.ep_info_buffer[-10:]
            success_rate = np.mean([info['is_success'] for info in recent_episodes])

            if success_rate > self.difficulty_threshold:
                difficulty = self.training_env.get_attr('difficulty')[0]
                new_difficulty = min(difficulty + 0.5, 5)

                if self.verbose > 0:
                    print(f"\nIncreasing difficulty to {new_difficulty}")

                self.eval_env.set_attr('difficulty', new_difficulty)
                self.training_env.set_attr('difficulty', new_difficulty)

                wandb.log({
                    "curriculum/difficulty": new_difficulty,
                    "curriculum/success_rate": success_rate,
                    "curriculum/episode_count": len(self.ep_info_buffer),
                    "training/mean_reward": np.mean(
                        [r for rewards in self.training_env.get_attr('rewards') for r in rewards]),
                    "training/success_rate": success_rate,
                    "training/episode_length": np.mean(
                        [len(rewards) for rewards in self.training_env.get_attr('rewards')]),
                    "training/collision_rate": np.mean([info['collision'] for info in recent_episodes]),
                    "training/min_distance": np.mean([info['distance_to_target'] for info in recent_episodes])
                })

            self.ep_info_buffer = []

        return True