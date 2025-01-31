from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize
from RobotNavEnv import RobotNavEnv
import numpy as np
import time


def evaluate_policy(model_path, vec_normalize_path, n_eval_episodes=20, render=True, difficulty=1):
    # Create environment
    env = RobotNavEnv(render_mode="human" if render else None, difficulty=difficulty)
    # Load the saved VecNormalize statistics
    env = VecNormalize.load(vec_normalize_path, env)
    # Disable training mode for normalization statistics
    env.training = False
    # Do not update normalization running averages
    env.norm_reward = False

    # Load the trained model
    model = SAC.load(model_path)

    # Statistics tracking
    episode_stats = {
        'rewards': [],
        'lengths': [],
        'success': 0,
        'collision': 0,
        'timeout': 0,
        'min_target_distances': [],
        'collision_times': [],
        'min_obstacle_distances': [],
        'average_velocities': [],
        'action_smoothness': []
    }

    for episode in range(n_eval_episodes):
        print(f"\nEpisode {episode + 1}/{n_eval_episodes}")
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        min_target_distance = float('inf')
        min_obstacle_distance = float('inf')
        total_velocity = 0
        action_changes = []
        prev_action = np.zeros(2)

        done = False
        while not done:
            # Track minimum distances
            lidar_data = obs[:-2]  # First 360 values are LiDAR
            target_pos = obs[-2:]  # Last 2 values are relative target position

            min_obstacle_distance = min(min_obstacle_distance, np.min(lidar_data))
            current_target_distance = np.linalg.norm(target_pos)
            min_target_distance = min(min_target_distance, current_target_distance)

            # Get and track action
            action, _ = model.predict(obs, deterministic=True)
            action_changes.append(np.linalg.norm(action - prev_action))
            total_velocity += np.abs(action[0])  # Track linear velocity
            prev_action = action

            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated

            if render:
                time.sleep(0.01)

        # Record episode statistics
        episode_stats['rewards'].append(episode_reward)
        episode_stats['lengths'].append(episode_length)
        episode_stats['min_target_distances'].append(min_target_distance)
        episode_stats['min_obstacle_distances'].append(min_obstacle_distance)
        episode_stats['average_velocities'].append(total_velocity / episode_length)
        episode_stats['action_smoothness'].append(np.mean(action_changes))

        if terminated:
            if episode_reward > 0:  # Successful completion
                episode_stats['success'] += 1
                print(f"Success! Final distance to target: {min_target_distance:.2f}")
            else:  # Collision
                episode_stats['collision'] += 1
                episode_stats['collision_times'].append(episode_length)
                print(f"Collision! Minimum obstacle distance: {min_obstacle_distance:.2f}")
        else:  # Timeout
            episode_stats['timeout'] += 1
            print(f"Timeout! Closest approach to target: {min_target_distance:.2f}")

    # Print detailed statistics
    print("\nDetailed Evaluation Results:")
    print(f"Success Rate: {episode_stats['success'] / n_eval_episodes * 100:.2f}%")
    print(f"Collision Rate: {episode_stats['collision'] / n_eval_episodes * 100:.2f}%")
    print(f"Timeout Rate: {episode_stats['timeout'] / n_eval_episodes * 100:.2f}%")
    print(f"Average Episode Reward: {np.mean(episode_stats['rewards']):.2f}")
    print(f"Average Episode Length: {np.mean(episode_stats['lengths']):.2f}")
    print(f"Average Minimum Target Distance: {np.mean(episode_stats['min_target_distances']):.2f}")
    print(f"Average Minimum Obstacle Distance: {np.mean(episode_stats['min_obstacle_distances']):.2f}")
    print(f"Average Velocity: {np.mean(episode_stats['average_velocities']):.2f}")
    print(f"Average Action Smoothness: {np.mean(episode_stats['action_smoothness']):.2f}")

    if episode_stats['collision_times']:
        print(f"Average Time to Collision: {np.mean(episode_stats['collision_times']):.2f}")

    env.close()
    return episode_stats


if __name__ == "__main__":
    model_path = "logs/best_model/best_model.zip"
    vec_normalize_path = "logs/vec_normalize.pkl"

    try:
        # Run evaluation at a moderate difficulty
        stats = evaluate_policy(
            model_path=model_path,
            vec_normalize_path=vec_normalize_path,
            n_eval_episodes=20,
            render=True,
            difficulty=1  # Start with lower difficulty
        )
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")