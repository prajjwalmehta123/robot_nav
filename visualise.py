from stable_baselines3 import SAC
from RobotNavEnv import RobotNavEnv
import numpy as np
import pybullet as p
import time


class NavigationVisualizer:
    def __init__(self, model_path):
        self.env = RobotNavEnv(render_mode="human", difficulty=5)
        self.model = SAC.load(model_path)

        # Visualization parameters
        self.path_points = []
        self.path_line_ids = []
        self.lidar_line_ids = []

    def visualize_path(self):
        # Remove old path lines
        for line_id in self.path_line_ids:
            p.removeUserDebugItem(line_id)
        self.path_line_ids.clear()

        # Draw new path lines
        for i in range(len(self.path_points) - 1):
            start = self.path_points[i]
            end = self.path_points[i + 1]

            line_id = p.addUserDebugLine(
                start, end,
                lineColorRGB=[0, 1, 0],  # Green color
                lineWidth=2,
                lifeTime=0  # Persistent until removed
            )
            self.path_line_ids.append(line_id)

    def visualize_lidar(self, lidar_data, robot_pos):
        # Remove old LIDAR lines
        for line_id in self.lidar_line_ids:
            p.removeUserDebugItem(line_id)
        self.lidar_line_ids.clear()

        # Draw new LIDAR lines
        angles = np.linspace(0, 2 * np.pi, len(lidar_data))
        for angle, distance in zip(angles, lidar_data):
            end_x = robot_pos[0] + distance * np.cos(angle)
            end_y = robot_pos[1] + distance * np.sin(angle)

            line_id = p.addUserDebugLine(
                [robot_pos[0], robot_pos[1], robot_pos[2]],
                [end_x, end_y, robot_pos[2]],
                lineColorRGB=[1, 0, 0],  # Red color
                lineWidth=1,
                lifeTime=0.1  # Short lifetime for dynamic update
            )
            self.lidar_line_ids.append(line_id)

    def run_episode(self):
        obs, _ = self.env.reset()
        self.path_points.clear()
        done = False
        episode_reward = 0

        while not done:
            # Get robot position
            robot_pos, _ = p.getBasePositionAndOrientation(self.env.robot_id)
            self.path_points.append(robot_pos)

            # Visualize path and LIDAR
            self.visualize_path()
            self.visualize_lidar(obs[:-2], robot_pos)  # obs[:-2] contains LIDAR data

            # Get action from model
            action, _ = self.model.predict(obs, deterministic=True)

            # Execute action
            obs, reward, terminated, truncated, _ = self.env.step(action)
            episode_reward += reward
            done = terminated or truncated

            time.sleep(0.01)  # Slow down visualization

        return episode_reward, terminated

    def run_visualization(self, n_episodes=5):
        for episode in range(n_episodes):
            print(f"\nStarting Episode {episode + 1}/{n_episodes}")
            reward, success = self.run_episode()
            print(f"Episode finished with reward: {reward:.2f}")
            print("Status:", "Success" if success and reward > 0 else "Failure")

            if episode < n_episodes - 1:
                time.sleep(1)  # Pause between episodes

        self.env.close()


def main():
    model_path = "logs/best_model/best_model.zip"
    visualizer = NavigationVisualizer(model_path)

    try:
        visualizer.run_visualization()
    except KeyboardInterrupt:
        print("\nVisualization interrupted by user")
        visualizer.env.close()


if __name__ == "__main__":
    main()