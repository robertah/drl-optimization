# An Exploration of Optimization Alternatives for Deep Reinforcement Learning

# WORKING IN PROGRESS

### Project for Deep Learning course - ETH Zurich - Fall 2018

**Goal**: We analyze different optimization approaches and examine their performances in different DRL applications, aiming at understanding why and how they perform differently. In particular, we focus our analysis on gradient-based approaches and (gradient-free) evolution-based optimization methods.

| **Environment**                 |   CartPole-v1   |   BipedalWalker-v2   |
|---------------------------------|-----------------|----------------------|
| **Gradient-based optimization** |   Q learning    |   TD3                |
| **Gradient-free optimization**  |   genetic       |   genetic            |


***

## Getting started

### Requirements
Create a virtual environment and install all required packages:

`conda create --name deep-learning python=3.6`

`source activate deep-learning`

`pip install -r requirements.txt`

### Configuration file
In [`config.yml`](config.yml), one can choose which OpenAI Gym environment and optimization algorithm to use (all available possibilities are listed on top).
It contains also optimization algorithms' parameters we used to train agents.

For each environment, we defined a specific neural network architecture in [`src/config/models.yml`](src/config/models.yml).

### Train agents
`python src/main.py`

Please, make sure that you have set the desired environment and optimization algorithm in [`config.yml`](config.yml), before start training.

If you are using a machine without a display, please run the following instead:

`xvfb-run -s "-screen 0 1400x900x24" python src/main.py`


### Results analysis
The analysis of the different DRL optimization algorithms can be found in `notebooks/FILE`. 


## Project directory
``` bash
.
├── config.yml         # configuration file
├── src
│   ├── config         # configuration loading package
│   ├── A2C            # A2C package
│   ├── DDPG           # Deep Deterministic Policy Gradients package
│   ├── TD3            # TD3 package
│   ├── GA             # Genetic Algorithm package
│   ├── CMA_ES         # Covariance Matrix Adapatation ES package
│   └── population     # population package for evolutionary algorithms
├── main.py            # main 
├── optimizers.py      # base gradient-free optimizer
├── visualization.py   # visualization for analysis 
├── utils.py           # helper functions
├── notebooks          # notebooks with results analysis
├── results            # folder containing results after training
│   ├── scores
│   └── weights
├── runs.yml           # runs log file
└── requirements.txt   # list of all packages used

```

**Note**: the code has been tested on the following machines: macOS and Ubuntu computers, Leonhard cluster and Google Cloud Platform virtual machine.
