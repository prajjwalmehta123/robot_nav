from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import wandb
from typing import List, Dict


class CurriculumCallback(BaseCallback):
    def __init__(
            self,
            eval_env,
            difficulty_threshold=0.8,
            difficulty_decrease_threshold=0.3,
            check_freq=10000,
            min_episodes=30,  # Minimum episodes before difficulty change
            verbose=0
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.difficulty_threshold = difficulty_threshold
        self.difficulty_decrease_threshold = difficulty_decrease_threshold
        self.check_freq = check_freq
        self.min_episodes = min_episodes
        self.n_calls = 0
        self.ep_info_buffer = []
        self.last_difficulty_change = 0
        self.consecutive_failures = 0

    def _init_callback(self) -> None:
        self.ep_info_buffer = []

    def _on_step(self) -> bool:
        self.n_calls += 1
        for env_idx in range(self.training_env.num_envs):
            try:
                ep_infos = self.training_env.env_method("get_episode_info_buffer", indices=env_idx)
                for info in ep_infos[0]:
                    self.ep_info_buffer.append({
                        'is_success': info.get('is_success', False),
                        'collision': info.get('collision', False),
                        'distance_to_target': info.get('distance_to_target', 0.0),
                        'timesteps': self.n_calls - self.last_difficulty_change
                    })
            except AttributeError:
                if self.verbose > 0:
                    print("Warning: Failed to collect episode info - check Monitor wrapper")
                continue

        # Check if we should evaluate difficulty adjustment
        if (self.n_calls % self.check_freq == 0 and
                len(self.ep_info_buffer) >= self.min_episodes):

            # Calculate statistics over a larger window
            recent_episodes = self.ep_info_buffer[-self.min_episodes:]
            success_rate = np.mean([info['is_success'] for info in recent_episodes])
            collision_rate = np.mean([info['collision'] for info in recent_episodes])
            avg_distance = np.mean([info['distance_to_target'] for info in recent_episodes])

            current_difficulty = self.training_env.get_attr('difficulty')[0]
            new_difficulty = current_difficulty

            # Difficulty increase logic
            if success_rate >= self.difficulty_threshold:
                new_difficulty = min(current_difficulty + 0.5, 5.0)
                self.consecutive_failures = 0

            # Difficulty decrease logic
            elif success_rate <= self.difficulty_decrease_threshold:
                self.consecutive_failures += 1
                if self.consecutive_failures >= 2:  # Require multiple failures
                    new_difficulty = max(current_difficulty - 0.5, 0.0)
                    self.consecutive_failures = 0

            # Apply difficulty change if needed
            if new_difficulty != current_difficulty:
                self._log_difficulty_change(
                    current_difficulty,
                    new_difficulty,
                    success_rate,
                    collision_rate,
                    avg_distance
                )
                self.eval_env.set_attr('difficulty', new_difficulty)
                self.training_env.set_attr('difficulty', new_difficulty)
                self.last_difficulty_change = self.n_calls

            # Clear buffer but keep recent episodes
            self.ep_info_buffer = self.ep_info_buffer[-self.min_episodes:]

        return True

    def _log_difficulty_change(
            self,
            old_diff,
            new_diff,
            success_rate,
            collision_rate,
            avg_distance
    ):
        """Log detailed metrics about difficulty changes."""
        print(f"\nBuffer size: {len(self.ep_info_buffer)}")
        print(f"Recent success rates: {[ep['is_success'] for ep in self.ep_info_buffer[-5:]]}")
        if self.verbose > 0:
            change_type = "Increasing" if new_diff > old_diff else "Decreasing"
            print(f"\n{change_type} difficulty from {old_diff:.1f} to {new_diff:.1f}")
            print(f"Success rate: {success_rate:.2%}")
            print(f"Collision rate: {collision_rate:.2%}")
            print(f"Average distance to target: {avg_distance:.2f}")

        wandb.log({
            "curriculum/difficulty": new_diff,
            "curriculum/old_difficulty": old_diff,
            "curriculum/success_rate": success_rate,
            "curriculum/collision_rate": collision_rate,
            "curriculum/avg_distance": avg_distance,
            "curriculum/consecutive_failures": self.consecutive_failures,
            "curriculum/episodes_at_difficulty": len(self.ep_info_buffer),
            "curriculum/timesteps_at_difficulty": self.n_calls - self.last_difficulty_change
        })