from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from RobotNavEnv import RobotNavEnv
import numpy as np
import time
import os
import torch
import wandb


def make_eval_env(difficulty=0, render=False):
    """Create an evaluation environment."""
    env = RobotNavEnv(
        render_mode="human" if render else None,
        difficulty=difficulty
    )
    return env


def evaluate_policy(
        model_path,
        vec_normalize_path,
        n_eval_episodes=20,
        render=True,
        difficulties=[0, 1, 2, 3, 4, 5]
):
    """
    Evaluate a trained model across different difficulty levels.
    """
    # Initialize wandb
    run = wandb.init(
        project="robot-navigation",
        name="evaluation",
        config={
            "n_eval_episodes": n_eval_episodes,
            "difficulties": difficulties,
            "model_path": model_path,
            "evaluation_type": "curriculum_testing",
            "render_enabled": render
        },
        tags=["evaluation", "curriculum"]
    )

    # Load the trained model
    model = SAC.load(model_path)
    print(f"\nLoaded model from: {model_path}")

    # Initialize arrays for tracking metrics
    all_metrics = {
        'success_rates': [],
        'collision_rates': [],
        'timeout_rates': [],
        'avg_rewards': [],
        'avg_episode_lengths': [],
        'avg_target_distances': [],
        'avg_obstacle_distances': [],
        'avg_velocities': [],
        'avg_smoothness': []
    }

    all_stats = {}

    for difficulty in difficulties:
        print(f"\nEvaluating at difficulty level {difficulty}")

        # Create and wrap environment
        env = make_eval_env(difficulty=difficulty, render=render)
        eval_env = DummyVecEnv([lambda: env])
        eval_env = VecNormalize.load(vec_normalize_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False

        # Statistics tracking
        episode_stats = {
            'rewards': [],
            'lengths': [],
            'success': 0,
            'collision': 0,
            'timeout': 0,
            'min_target_distances': [],
            'min_obstacle_distances': [],
            'average_velocities': [],
            'action_smoothness': []
        }

        for episode in range(n_eval_episodes):
            print(f"\nDifficulty {difficulty} - Episode {episode + 1}/{n_eval_episodes}")

            obs = eval_env.reset()[0]
            episode_reward = 0
            episode_length = 0
            min_target_distance = float('inf')
            min_obstacle_distance = float('inf')
            total_velocity = 0
            action_changes = []
            prev_action = np.zeros(2)

            done = False
            while not done:
                # Get original observation for tracking
                unwrapped_obs = eval_env.get_original_obs()
                lidar_data = unwrapped_obs[0][:-2]
                target_pos = unwrapped_obs[0][-2:]

                # Track minimum distances
                min_obstacle_distance = min(min_obstacle_distance, np.min(lidar_data))
                current_target_distance = np.linalg.norm(target_pos)
                min_target_distance = min(min_target_distance, current_target_distance)

                # Get action
                action, _ = model.predict(obs, deterministic=True)
                if isinstance(action, np.ndarray) and action.shape == (2,):
                    action = np.array([action])

                # Track action smoothness
                action_changes.append(np.linalg.norm(action[0] - prev_action))
                total_velocity += np.abs(action[0][0])
                prev_action = action[0]

                # Execute action
                next_obs, reward, terminated, info = eval_env.step(action)

                # Update episode tracking
                episode_reward += float(reward) if np.isscalar(reward) else reward[0]
                episode_length += 1
                obs = next_obs
                done = terminated[0] if isinstance(terminated, (list, np.ndarray)) else terminated

                if render:
                    time.sleep(0.01)

            # Record episode statistics
            episode_stats['rewards'].append(episode_reward)
            episode_stats['lengths'].append(episode_length)
            episode_stats['min_target_distances'].append(min_target_distance)
            episode_stats['min_obstacle_distances'].append(min_obstacle_distance)
            episode_stats['average_velocities'].append(total_velocity / episode_length)
            episode_stats['action_smoothness'].append(np.mean(action_changes))

            # Determine episode outcome
            if done:
                if episode_reward > 0:
                    episode_stats['success'] += 1
                    print(f"Success! Final distance to target: {min_target_distance:.2f}")
                else:
                    episode_stats['collision'] += 1
                    print(f"Collision! Minimum obstacle distance: {min_obstacle_distance:.2f}")
            else:
                episode_stats['timeout'] += 1
                print(f"Timeout! Closest approach to target: {min_target_distance:.2f}")

        # Calculate metrics for current difficulty
        success_rate = episode_stats['success'] / n_eval_episodes * 100
        collision_rate = episode_stats['collision'] / n_eval_episodes * 100
        timeout_rate = episode_stats['timeout'] / n_eval_episodes * 100
        avg_reward = np.mean(episode_stats['rewards'])
        avg_episode_length = np.mean(episode_stats['lengths'])
        avg_target_distance = np.mean(episode_stats['min_target_distances'])
        avg_obstacle_distance = np.mean(episode_stats['min_obstacle_distances'])
        avg_velocity = np.mean(episode_stats['average_velocities'])
        avg_smoothness = np.mean(episode_stats['action_smoothness'])

        # Store metrics
        all_metrics['success_rates'].append(success_rate)
        all_metrics['collision_rates'].append(collision_rate)
        all_metrics['timeout_rates'].append(timeout_rate)
        all_metrics['avg_rewards'].append(avg_reward)
        all_metrics['avg_episode_lengths'].append(avg_episode_length)
        all_metrics['avg_target_distances'].append(avg_target_distance)
        all_metrics['avg_obstacle_distances'].append(avg_obstacle_distance)
        all_metrics['avg_velocities'].append(avg_velocity)
        all_metrics['avg_smoothness'].append(avg_smoothness)

        # Log current difficulty metrics
        wandb.log({
            'difficulty': difficulty,
            'success_rate': success_rate,
            'collision_rate': collision_rate,
            'timeout_rate': timeout_rate,
            'avg_reward': avg_reward,
            'avg_episode_length': avg_episode_length,
            'avg_target_distance': avg_target_distance,
            'avg_obstacle_distance': avg_obstacle_distance,
            'avg_velocity': avg_velocity,
            'action_smoothness': avg_smoothness,
            'evaluation_step': difficulty  # Use difficulty as step counter
        })

        # Print statistics
        print(f"\nResults for Difficulty {difficulty}:")
        print(f"Success Rate: {success_rate:.2f}%")
        print(f"Collision Rate: {collision_rate:.2f}%")
        print(f"Timeout Rate: {timeout_rate:.2f}%")
        print(f"Average Episode Reward: {avg_reward:.2f}")
        print(f"Average Episode Length: {avg_episode_length:.2f}")

        # Store full stats
        all_stats[difficulty] = episode_stats

        # Cleanup
        eval_env.close()

    # Create summary plots
    for metric, values in all_metrics.items():
        data = [[x, y] for (x, y) in zip(difficulties, values)]
        table = wandb.Table(data=data, columns=["difficulty", metric])
        wandb.log({f"summary_{metric}": wandb.plot.line(table, "difficulty", metric,
                                                        title=f"{metric.replace('_', ' ').title()} vs Difficulty")})

    # Log overall performance
    wandb.run.summary.update({
        "overall_success_rate": np.mean(all_metrics['success_rates']),
        "overall_collision_rate": np.mean(all_metrics['collision_rates']),
        "overall_timeout_rate": np.mean(all_metrics['timeout_rates']),
        "best_success_rate": max(all_metrics['success_rates']),
        "worst_success_rate": min(all_metrics['success_rates'])
    })

    wandb.finish()
    return all_stats


def main():
    # Setup CUDA if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Get model and normalization files
    model_path = "models/final_model.zip"
    vec_normalize_path = "models/vec_normalize.pkl"

    if not os.path.exists(model_path) or not os.path.exists(vec_normalize_path):
        raise FileNotFoundError("Could not find model or normalization files")

    print(f"Using model: {model_path}")
    print(f"Using normalization: {vec_normalize_path}")

    try:
        # Run evaluation
        stats = evaluate_policy(
            model_path=model_path,
            vec_normalize_path=vec_normalize_path,
            n_eval_episodes=20,
            render=False,
            difficulties=[0, 1, 2, 3, 4, 5]
        )
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise


if __name__ == "__main__":
    main()