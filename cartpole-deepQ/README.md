<h3 align="center">
  <img src="assets/cartpole_icon_web.png" width="300">
</h3>

# Cartpole

Reinforcement Learning solution of the [OpenAI's Cartpole](https://gym.openai.com/envs/CartPole-v0/).

## DQN
Standard DQN with Experience Replay.

### Hyperparameters:

* GAMMA = 0.995
* LEARNING_RATE = 0.001
* MEMORY_SIZE = 1000000
* BATCH_SIZE = 32
* EXPLORATION_MAX = 1.0
* EXPLORATION_MIN = 0.1
* EXPLORATION_DECAY = 0.995
* TARGET = 450
* RENDER = False
* NTIMES = 4

### Model structure:

1. Dense layer - input: **4**, output: **24**, activation: **relu**
2. Dense layer - input **24**, output: **2**, activation: **linear**

* **MSE** loss function
* **Adam** optimizer