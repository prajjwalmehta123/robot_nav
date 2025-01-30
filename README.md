# Robot Navigation with Reinforcement Learning

This project implements a reinforcement learning system for autonomous robot navigation using PyBullet physics simulation. The robot learns to navigate through environments with obstacles while reaching target positions using LiDAR-based sensing.

## Features

- Physics-based robot simulation using PyBullet
- Custom OpenAI Gym environment for robot navigation
- LiDAR-based obstacle detection and avoidance
- Soft Actor-Critic (SAC) implementation using Stable-Baselines3
- Curriculum learning with increasing difficulty levels
- Weights & Biases integration for experiment tracking
- Comprehensive visualization tools for debugging and demonstration

## Project Structure

- `RobotNavEnv.py`: Custom Gym environment implementation
- `train.py`: Main training script with curriculum learning and W&B integration
- `eval.py`: Policy evaluation script
- `visualize.py`: Visualization tools for trained policies
- `sim_setup.py`: Basic simulation setup and testing
- `sweep.yml`: Hyperparameter optimization configuration

## Requirements

```
gymnasium
numpy
pybullet
stable-baselines3
wandb
```

## Environment Description

The environment consists of:
- R2D2 robot with differential drive control
- 360-degree LiDAR sensor
- Randomly placed obstacles
- Target position marked with a red line

### Observation Space
- 360 LiDAR readings (distances from 0 to 5 meters)
- 2D relative target position

### Action Space
- 2D continuous action space: [linear_velocity, angular_velocity]
- Values normalized between -1 and 1

### Reward Structure
- Distance-based reward for approaching target
- Penalty for obstacle proximity (scales with difficulty)
- Large positive reward for reaching target
- Large negative reward for collisions
- Action smoothness reward

## Training

The training system features:
- Curriculum learning with 6 difficulty levels (0-5)
- Automatic difficulty progression based on success rate
- Integration with Weights & Biases for experiment tracking
- Hyperparameter optimization using W&B sweeps

To start training:
```bash
python train.py
```

For hyperparameter optimization:
```bash
wandb sweep sweep.yml
wandb agent [sweep_id]
```

## Evaluation

To evaluate a trained policy:
```bash
python eval.py
```

The evaluation script provides:
- Success rate statistics
- Collision rate analysis
- Average episode rewards and lengths
- Optional visualization of the robot's behavior

## Visualization

The visualization tool provides:
- Real-time path tracking
- LiDAR data visualization
- Episode success/failure statistics

To run the visualization:
```bash
python visualize.py
```

## Curriculum Learning

The environment difficulty increases progressively based on:
- Number of obstacles (2 to 7)
- Obstacle size (0.5 to 1.0 scale)
- Target distance (2 to 7 meters)
- Reward/penalty scaling
- Safe distance thresholds

## Performance Metrics

The system tracks:
- Episode rewards
- Success rates
- Collision rates
- Average episode lengths
- Learning curves
- Policy performance across difficulty levels

## Dependencies

- Python 3.7+
- PyBullet for physics simulation
- Stable-Baselines3 for RL algorithms
- Weights & Biases for experiment tracking
- Gymnasium for environment interface
- NumPy for numerical computations

## Future Improvements

- Dynamic obstacle movement
- More complex environment geometries
- Multi-robot scenarios
- Additional sensor modalities
- Real-robot deployment capabilities