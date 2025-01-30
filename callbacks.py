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
        """
        Check if we should increase difficulty based on success rate.
        """
        self.n_calls += 1

        # Get episode info
        if len(self.training_env.get_attr('rewards')[0]) > 0:
            self.ep_info_buffer.append({
                'is_success': self.training_env.get_attr('rewards')[0][-1] > 0
            })

        if self.n_calls % self.check_freq == 0:
            # Calculate success rate from recent episodes
            if len(self.ep_info_buffer) > 10:
                recent_episodes = self.ep_info_buffer[-10:]
                success_rate = np.mean([info['is_success'] for info in recent_episodes])

                if success_rate > self.difficulty_threshold:
                    current_difficulty = self.eval_env.get_attr('difficulty')[0]
                    new_difficulty = min(current_difficulty + 1, 5)

                    if self.verbose > 0:
                        print(f"\nIncreasing difficulty to {new_difficulty}")

                    self.eval_env.set_attr('difficulty', new_difficulty)

                    # Log to wandb
                    wandb.log({
                        "curriculum/difficulty": new_difficulty,
                        "curriculum/success_rate": success_rate
                    })

                # Clear buffer after checking
                self.ep_info_buffer = []

        return True