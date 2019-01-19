# An Exploration of Optimization Alternatives for Deep Reinforcement Learning

# WORKING IN PROGRESS

### Project for Deep Learning course - ETH Zurich - Fall 2018

**Goal**: We analyze different optimization approaches and examine their performances in different DRL applications, aiming at understanding why and how they perform differently. In particular, we focus our analysis on gradient-based approaches and (gradient-free) evolution-based optimization methods.

| **Environment**                 |   CartPole-v1        |   BipedalWalker-v2        |
|---------------------------------|----------------------|---------------------------|
| **Gradient-based optimization** |   DQN  &ast;         |   TD3 &ast;&ast;          |
| **Gradient-free optimization**  |   GA, ES  &ast;      |   GA  &ast;&ast;&ast;     |

&ast; feed-forward neural network consisting of 1 hidden layer with 24 units

&ast;&ast; feed-forward neural networks consisting of 2 hidden layers with 512, 256 units

&ast;&ast;&ast; feed-forward neural networks consisting of 3 hidden layers with 128, 128, 3 units

<p align="center">
  <img src="/results/bipedalwalker_td3/agent.gif" width="60%"/>
</p>

***

## Getting started

### Requirements
Create a virtual environment and install all required packages:

``` bash

conda create --name deep-learning python=3.6

source activate deep-learning

pip install -r requirements.txt
``` 


### Configuration file
In [`config.yml`](config.yml), one can choose which OpenAI Gym environment and optimization algorithm to use (all available possibilities are listed on top). For example:
``` yaml

environment:
  name: 'CartPole-v1'
  animate: False

algorithm: 'ga' 

``` 
For each environment, we defined a specific neural network architecture for evolutionary algorithms in [`src/config/models.yml`](src/config/models.yml).

It contains also optimization algorithms' parameters we used to train agents.


### Train agents
Please, make sure that you have set the desired environment and optimization algorithm in [`config.yml`](config.yml), before start training.

``` bash
python src/main.py
``` 

If you are using a machine without a display, please run the following instead:

``` bash
xvfb-run -s "-screen 0 1400x900x24" python src/main.py
```


### Results analysis
The analysis of the different DRL optimization algorithms can be found in `notebooks/FILE`. 


## Project directory
``` bash
.
├── config.yml                # configuration file
├── src
│   ├── config                # configuration loading package
│   ├── A2C                   # A2C package
│   ├── DDPG                  # Deep Deterministic Policy Gradients package
│   ├── TD3                   # TD3 package
│   ├── GA                    # Genetic Algorithm package
│   ├── CMA_ES                # Covariance Matrix Adapatation ES package
│   ├── population            # population package for evolutionary algorithms
│   ├── main.py               # main 
│   ├── optimizers.py         # base gradient-free optimizer
│   ├── loss_analysis.py      # functions for loss analysis 
│   ├── visualization.py      # visualization for analysis 
│   └── utils.py              # helper functions
├── notebooks                 # notebooks with results analysis
├── results                   # folder containing results after training
│   ├── scores
│   └── weights
├── runs.yml                  # runs log file
└── requirements.txt          # list of all packages used

```

**Note**: the code has been tested on the following machines: macOS and Ubuntu computers, Google Cloud Platform virtual machine, and partially on ETH Leonhard cluster.
