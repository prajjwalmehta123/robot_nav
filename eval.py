from stable_baselines3 import SAC
from RobotNavEnv import RobotNavEnv
import numpy as np
import time


def evaluate_policy(model_path, n_eval_episodes=10, render=True):
    # Create environment
    env = RobotNavEnv(render_mode="human" if render else None, difficulty=5)  # Max difficulty

    # Load the trained model
    model = SAC.load(model_path)

    # Statistics tracking
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    collision_count = 0

    for episode in range(n_eval_episodes):
        print(f"\nEpisode {episode + 1}/{n_eval_episodes}")
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)

            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            if render:
                time.sleep(0.01)  # Slow down rendering for visualization

        # Record episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if terminated:
            if reward > 0:  # Successful completion
                success_count += 1
                print("Episode succeeded!")
            else:  # Collision
                collision_count += 1
                print("Episode failed (collision)!")
        elif truncated:
            print("Episode truncated (timeout)!")

        print(f"Episode reward: {episode_reward:.2f}")
        print(f"Episode length: {episode_length}")

    # Print final statistics
    print("\nEvaluation Results:")
    print(f"Success Rate: {success_count / n_eval_episodes * 100:.2f}%")
    print(f"Collision Rate: {collision_count / n_eval_episodes * 100:.2f}%")
    print(f"Average Episode Reward: {np.mean(episode_rewards):.2f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.2f}")

    env.close()


def main():
    # Path to your trained model
    model_path = "logs/best_model/best_model.zip"

    try:
        # Run evaluation
        evaluate_policy(
            model_path=model_path,
            n_eval_episodes=10,
            render=True
        )
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")


if __name__ == "__main__":
    main()