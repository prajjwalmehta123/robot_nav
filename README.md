# 🤖 Autonomous Robot Navigation with Deep RL

A robust reinforcement learning system that teaches robots to navigate complex environments autonomously. The project uses PyBullet physics simulation and implements the Soft Actor-Critic (SAC) algorithm to train a robot to reach target positions while avoiding obstacles.

## 🌟 Key Features

- **Physics-Based Simulation**: Realistic robot dynamics using PyBullet
- **Advanced Sensing**: 360° LiDAR-based obstacle detection
- **Intelligent Learning**: SAC implementation with automatic curriculum progression
- **Performance Tracking**: Comprehensive metrics and visualization through Weights & Biases
- **Adaptable Difficulty**: Six progressive difficulty levels for robust learning
- **Real-Time Visualization**: Debug tools and performance monitoring

## 🚀 Quick Start

### Prerequisites

```bash
pip install gymnasium numpy pybullet stable-baselines3 wandb
```

### Training

1. Start a new training session:
```bash
python train.py
```

2. Run hyperparameter optimization:
```bash
wandb sweep sweep.yml
wandb agent [sweep_id]
```

3. Monitor training progress in real-time through Weights & Biases dashboard.

### Evaluation

Test your trained model:
```bash
python eval.py
```

## 🎮 Environment Details

### Robot Configuration
- Platform: R2D2-style robot with differential drive
- Sensors: 360° LiDAR (range: 0-5 meters)
- Control: Continuous action space for linear and angular velocities

### Training Process
The environment implements curriculum learning with 6 difficulty levels (0-5):

| Level | Features |
|-------|----------|
| 0 | Basic navigation, few obstacles |
| 1 | Increased obstacle count |
| 2 | Larger obstacles, longer distances |
| 3 | Complex obstacle arrangements |
| 4 | Tight navigation constraints |
| 5 | Maximum difficulty with all challenges |

### Reward Structure
- **Positive Rewards**:
  - Progress toward target
  - Reaching final destination
  - Maintaining safe distances from obstacles
  - Smooth motion control

- **Negative Rewards**:
  - Collisions with obstacles
  - Excessive proximity to obstacles
  - Jerky movements

## 📊 Performance Metrics

The system tracks multiple performance indicators:
- Success rate per difficulty level
- Collision frequency
- Average episode length
- Navigation efficiency
- Learning curve progression

## 🛠️ Project Structure

```
├── RobotNavEnv.py      # Core environment implementation
├── train.py            # Training orchestration
├── eval.py            # Evaluation and metrics
├── callbacks.py       # Training callbacks and monitoring
└── sweep.yml         # Hyperparameter configuration
```

## 🔧 Customization

### Environment Parameters
- `difficulty`: Controls obstacle count, size, and placement (0-5)
- `render_mode`: "human" for visualization, None for training
- `max_steps`: Maximum episode duration

### Training Configuration
- Learning rate and batch size
- Network architecture
- Reward scaling
- Curriculum progression thresholds

## 📈 Results

The Performance of the RL model is not something which is great, as the difficulty of the environment increases,
the success rate drops drastically. This is not ideal, and it suggest that my jump from 0 to 1 difficulty is too drastic, 

Wandb Link - https://wandb.ai/prajjwalmehta123/robot-navigation?nw=nwuserpm8607


![Results](../analysis.png)


## Key Future TODO:
- [ ] Implement curriculum learning with smaller difficulty increments
- [ ] Add intermediate reward shaping for better obstacle avoidance
- [ ] Consider adjusting the reward structure to better balance exploration and safety
- [ ] The robot might benefit from a more conservative initial approach with gradual speed increases

## 🙏 Acknowledgments

Built with:
- [PyBullet](https://pybullet.org/) - Physics simulation
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - RL algorithms
- [Weights & Biases](https://wandb.ai/) - Experiment tracking
- [Gymnasium](https://gymnasium.farama.org/) - Environment framework