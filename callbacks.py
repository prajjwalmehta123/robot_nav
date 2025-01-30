from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import wandb


class CurriculumCallback(BaseCallback):
    def __init__(self, eval_env, difficulty_threshold=0.8, check_freq=10000, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.difficulty_threshold = difficulty_threshold
        self.check_freq = check_freq
        self.n_calls = 0

    def _init_callback(self) -> None:
        """Initialize callback-specific variables."""
        pass

    def _on_step(self) -> bool:
        """
        Check if we should increase difficulty based on success rate.
        """
        self.n_calls += 1

        if self.n_calls % self.check_freq == 0:
            # Get episode info from the last few episodes
            if len(self.model.ep_info_buffer) > 0:
                ep_info = self.model.ep_info_buffer[-1]
                success_rate = np.mean([info['is_success']
                                        for info in self.model.ep_info_buffer[-10:]
                                        if 'is_success' in info
                                        ])

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

        return True