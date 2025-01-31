import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time


class RobotNavEnv(gym.Env):
    def __init__(self, render_mode=None,difficulty=0):
        super().__init__()
        self.action_space = spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1, 1]),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=np.array([0] * 360 + [-5, -5]),
            high=np.array([5] * 360 + [5, 5]),
            dtype=np.float32
        )
        self.render_mode = render_mode
        self.client = None
        self._setup_simulation()
        self.max_steps = 1000
        self.current_step = 0
        self.target_pos = None
        self.robot_id = None
        self.obstacle_ids = []
        self.debug_items = []
        self.left_wheel_joints = [6, 7]
        self.right_wheel_joints = [2, 3]
        self.current_action = np.zeros(2)
        self.prev_action = np.zeros(2)
        self.difficulty = difficulty

    def _setup_simulation(self):
        try:
            if self.client is not None:
                try:
                    p.disconnect(self.client)
                except p.error.ExperimentalFeatureException:
                    pass
            self.client = p.connect(p.GUI if self.render_mode == "human" else p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81)
            p.setRealTimeSimulation(0)

        except p.error.ExperimentalFeatureException as e:
            raise RuntimeError(f"Failed to setup PyBullet simulation: {e}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # Reset simulation
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF("r2d2.urdf", basePosition=[0, 0, 0.5])
        self.target_pos = self._generate_target_position()
        self._add_obstacles()

        if self.render_mode == "human":
            p.resetDebugVisualizerCamera(
                cameraDistance=5,
                cameraYaw=0,
                cameraPitch=-45,
                cameraTargetPosition=[0, 0, 0]
            )
            p.addUserDebugLine(
                [self.target_pos[0], self.target_pos[1], 0],
                [self.target_pos[0], self.target_pos[1], 1],
                [1, 0, 0],
                lineWidth=2
            )
        for _ in range(100):
            p.stepSimulation()
        observation = self._get_observation()
        info = {}

        return observation, info

    def step(self, action):
        self.prev_action = self.current_action
        self.current_action = action
        self.current_step += 1
        linear_vel, angular_vel = action
        self._apply_action(linear_vel, angular_vel)
        for _ in range(10):
            p.stepSimulation()
            if self.render_mode == "human":
                time.sleep(0.001)
        observation = self._get_observation()
        reward = self._compute_reward()
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_steps

        robot_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        distance_to_target = np.linalg.norm(
            np.array([robot_pos[0], robot_pos[1]]) - self.target_pos
        )

        info = {
            'is_success': distance_to_target < 0.5,
            'collision': self._check_collision(),
            'distance_to_target': distance_to_target
        }
        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        lidar_data = self._get_lidar_data()
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        relative_target = np.array([
            self.target_pos[0] - robot_pos[0],
            self.target_pos[1] - robot_pos[1]
        ])
        return np.concatenate([lidar_data, relative_target])

    def _get_lidar_data(self):
        ray_from = []
        ray_to = []
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        for angle in np.linspace(0, 2 * np.pi, 360):
            dx = 5 * np.cos(angle)
            dy = 5 * np.sin(angle)
            ray_from.append([robot_pos[0], robot_pos[1], robot_pos[2]])
            ray_to.append([robot_pos[0] + dx, robot_pos[1] + dy, robot_pos[2]])
        if self.render_mode == "human":
            for i in range(0, len(ray_from), 45):
                p.addUserDebugLine(ray_from[i], ray_to[i], [0, 1, 0], lineWidth=1, lifeTime=0.1)
        results = p.rayTestBatch(ray_from, ray_to)
        distances = np.array([res[2] * 5 for res in results])
        return distances

    def _compute_reward(self):
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        distance_to_target = np.linalg.norm(
            np.array([robot_pos[0], robot_pos[1]]) - self.target_pos
        )

        lidar_data = self._get_lidar_data()
        min_obstacle_distance = np.min(lidar_data)

        # Base rewards
        distance_reward = -distance_to_target
        obstacle_reward = 0

        # Obstacle sensitivity increases with difficulty
        safe_distance = 1.0 - (self.difficulty * 0.1)
        if min_obstacle_distance < safe_distance:
            obstacle_reward = -1.0 * (safe_distance - min_obstacle_distance)

        if self._check_collision():
            return -150 * (1 + self.difficulty * 0.3)

        if distance_to_target < 0.5:
            return 100 + (self.difficulty * 20)

        total_reward = distance_reward + obstacle_reward

        action_smoothness = -np.sum(np.abs(self.current_action - self.prev_action))
        smoothness_factor = 0.3 * (1 + self.difficulty * 0.2)
        total_reward += smoothness_factor * action_smoothness

        return total_reward

    @property
    def difficulty(self):
        return self._difficulty

    @difficulty.setter
    def difficulty(self, value):
        self._difficulty = max(0, min(5, value))


    def _check_collision(self):
        # Get all contact points
        contact_points = p.getContactPoints(self.robot_id)

        for contact in contact_points:
            # contact[2] is the body ID of the other object
            # contact[9] is the normal force
            if contact[2] != 0:  # 0 is the ground plane
                return True

        return False

    def _is_terminated(self):
        # Immediately terminate on collision
        if self._check_collision():
            print("Collision detected! Episode terminated.")
            return True

        # Check for success (reaching target)
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        distance_to_target = np.linalg.norm(
            np.array([robot_pos[0], robot_pos[1]]) - self.target_pos
        )

        if distance_to_target < 0.5:
            print("Target reached! Episode terminated.")
            return True

        return False

    def _generate_target_position(self):
        max_distance = 2 + self.difficulty
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(1, max_distance)
        x = distance * np.cos(angle)
        y = distance * np.sin(angle)
        return np.array([x, y])

    def _apply_action(self, linear_vel, angular_vel):
        max_speed = 40  # Increased max speed
        wheel_distance = 0.2

        # Modified differential drive equations
        left_wheel = linear_vel * max_speed - angular_vel * max_speed
        right_wheel = linear_vel * max_speed + angular_vel * max_speed

        # Apply to both front and back wheels with increased force
        for joint in self.left_wheel_joints:
            p.setJointMotorControl2(
                self.robot_id, joint, p.VELOCITY_CONTROL,
                targetVelocity=left_wheel, force=15.0
            )

        for joint in self.right_wheel_joints:
            p.setJointMotorControl2(
                self.robot_id, joint, p.VELOCITY_CONTROL,
                targetVelocity=right_wheel, force=15.0
            )

    def _add_obstacles(self):
        for obs_id in self.obstacle_ids:
            p.removeBody(obs_id)
        self.obstacle_ids.clear()
        n_obstacles = int(self.difficulty) + 1
        min_spacing = max(1.0, 1.8 - (self.difficulty * 0.1))
        obstacles = []
        max_attempts =100
        for _ in range(n_obstacles):
            valid_position = False
            attempts = 0
            while not valid_position and attempts < max_attempts:
                attempts += 1
                pos = np.random.uniform(-3, 3, size=3)
                pos[2] = 0.5
                if self._is_valid_obstacle_position(pos, obstacles, min_spacing):
                    valid_position = True
                    obstacles.append(pos)
                    scale = min(1.0, 0.5 + (self.difficulty * 0.1))
                    obs_id = p.loadURDF("cube.urdf", basePosition=pos, globalScaling=scale)
                    self.obstacle_ids.append(obs_id)

            if attempts >= max_attempts:
                print(f"Warning: Could not place obstacle {len(obstacles) + 1}/{n_obstacles}")

    def _is_valid_obstacle_position(self, pos, obstacles, min_spacing):
        robot_pos = np.array(p.getBasePositionAndOrientation(self.robot_id)[0])
        if np.linalg.norm(pos[:2] - robot_pos[:2]) < min_spacing:
            return False
        if np.linalg.norm(pos[:2] - self.target_pos) < min_spacing:
            return False
        for obs_pos in obstacles:
            if np.linalg.norm(pos[:2] - obs_pos[:2]) < min_spacing:
                return False
        return True


    def close(self):
        if self.client is not None:
            try:
                # Clean up debug items
                for item in self.debug_items:
                    p.removeUserDebugItem(item)
                self.debug_items.clear()

                # Remove all objects
                for obs_id in self.obstacle_ids:
                    p.removeBody(obs_id)
                if self.robot_id is not None:
                    p.removeBody(self.robot_id)

                # Disconnect client
                p.disconnect(self.client)
                self.client = None

            except p.error.ExperimentalFeatureException as e:
                print(f"Warning: Error during cleanup: {e}")


def test_environment():
    env = RobotNavEnv(render_mode="human")

    print("\nTesting reset...")
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"LiDAR readings: {obs[:-2].shape}")
    print(f"Target position: {obs[-2:]}")

    print("\nTesting steps...")
    for i in range(200):
        # Simple test motion: forward then turn
        if i < 50:
            action = np.array([0.5, 0.0])  # Move forward
        elif i < 100:
            action = np.array([0.3, 1.0])  # Sharp right turn
        elif i < 150:
            action = np.array([0.3, -1.0])  # Sharp left turn
        else:
            action = np.array([0.5, 0.0])  # Move forward again

        obs, reward, terminated, truncated, info = env.step(action)

        if i % 20 == 0:  # Print less frequently
            print(f"\nStep {i}:")
            print(f"Action taken: {action}")
            print(f"Reward: {reward}")
            print(f"Min LiDAR distance: {np.min(obs[:-2]):.2f}")

        if terminated or truncated:
            print("Episode finished!")
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    test_environment()