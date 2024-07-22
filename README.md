# RL_MUJUCO

## Files and Directories

### config.ini
Configuration file for setting parameters and hyperparameters for the training process.

### main.py
The main script to start training and evaluating the agents.

### agents
Contains implementations of various reinforcement learning algorithms.
- `__init__.py`: Initialization file for the agents module.
- `ddpg.py`: Implementation of the Deep Deterministic Policy Gradient (DDPG) algorithm.
- `Hebbianppo.py`: Implementation of the Hebbian Proximal Policy Optimization (PPO) algorithm.
- `ppo.py`: Implementation of the Proximal Policy Optimization (PPO) algorithm.
- `sac.py`: Implementation of the Soft Actor-Critic (SAC) algorithm.

### log
Contains logs of training runs for different environments and algorithms.

### model_weights
Contains saved model weights for different environments and algorithms.

### runs
Contains TensorBoard log files for visualizing training metrics.

### utils
Contains utility scripts for noise generation and other helper functions.
- `__init__.py`: Initialization file for the utils module.
- `noise.py`: Functions for noise generation used in exploration.
- `utils.py`: Miscellaneous utility functions.

## Usage

1. **Setup**: Ensure you have the required dependencies installed.
   ```bash
   pip install -r requirements.txt
   
## Implement
**You can write the code in the terminal and execute it**
```
python main.py --env_name Ant-v4 --train True --tensorboard True --algo ppo  
```
## Reference
= **https://github.com/seolhokim/Mujoco-Pytorch.git**
