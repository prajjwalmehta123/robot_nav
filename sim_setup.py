import pybullet as p
import pybullet_data
import numpy as np

# Initialize
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.80)


def get_lidar(robot_id, num_rays=360, max_range=5):
    ray_from = []
    ray_to = []
    robot_pos, robot_orn = p.getBasePositionAndOrientation(robot_id)
    for angle in np.linspace(0, 2 * np.pi, num_rays):
        dx = max_range * np.cos(angle)
        dy = max_range * np.sin(angle)
        ray_from.append(robot_pos)
        ray_to.append([robot_pos[0] + dx, robot_pos[1] + dy, robot_pos[2]])
    results = p.rayTestBatch(ray_from, ray_to)
    distances = np.array([res[2] * max_range for res in results])
    return distances

plane_id = p.loadURDF("plane.urdf")
robot_id = p.loadURDF("r2d2.urdf", basePosition=[0, 0, 0.1])

for _ in range(5):
    obstacle_pos = np.random.uniform(-3, 3, size=3)
    obstacle_pos[2] = 0.5  # Set z-coordinate to 0.5 (half cube height)
    p.loadURDF("cube.urdf", basePosition=obstacle_pos, globalScaling=0.5)


p.resetDebugVisualizerCamera(
    cameraDistance=3,
    cameraYaw=45,
    cameraPitch=-30,
    cameraTargetPosition=[0, 0, 0]
)

try:
    while True:
        lidar_data = get_lidar(robot_id)
        print(f"LiDAR ranges: Min={np.min(lidar_data):.2f}, Max={np.max(lidar_data):.2f}")
        p.stepSimulation()

except KeyboardInterrupt:
    pass

p.disconnect()
