import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


ENV_NAME = "CartPole-v1"

GAMMA = 0.995
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 32

EXPLORATION_MAX = 1
EXPLORATION_MIN = 0.1
EXPLORATION_DECAY = 0.995
TARGET = 450
RENDER = False
NTIMES = 4 


class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX
        self.observation_space = observation_space
        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = self.create_model()
        self.model_q = self.create_model()
        self.model_q.set_weights(self.model.get_weights())

    def create_model(self):
        model = Sequential()
        model.add(Dense(24, input_shape=(self.observation_space,), activation="relu" , kernel_initializer='he_uniform'))
        #model.add(Dense(24, activation="relu"))
        model.add(Dense(self.action_space, activation="linear", kernel_initializer='he_uniform'))
        model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    #Action for evaluation has no exploration
    def act_eval(self, state):
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model_q.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.model_q.set_weights(self.model.get_weights())
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


def train():
    env = gym.make(ENV_NAME)
    env._max_episode_steps = 500
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0
    score = []
    agents_weights = []
    condition = True
    while condition:
        run += 1
        score_times = []
        #Train the agent in the first episode and evaluate in NTIMES-1
        for i in range(NTIMES):
            step = 0
            state = env.reset()
            state = np.reshape(state, [1, observation_space])
            done = False
            #Train the model
            if i == 0:
                while (not done):
                    step += 1
                    action = dqn_solver.act(state)
                    agents_weights.append(dqn_solver.model_q.get_weights())
                    state_next, reward, done, info = env.step(action)
                    reward = reward if not done else -reward
                    state_next = np.reshape(state_next, [1, observation_space])
                    dqn_solver.remember(state, action, reward, state_next, done)
                    state = state_next
                    dqn_solver.experience_replay()
            else:
                #Evaluate
                while (not done):
                    step += 1
                    if RENDER: 
                        env.render()
                    action = dqn_solver.act_eval(state)
                    state_next, reward, done, info = env.step(action)
                    reward = reward if not done else -reward
                    state_next = np.reshape(state_next, [1, observation_space])
                    state = state_next

                    if done:
                        score_times.append(step)


        print ("Run: " + str(run) + " score: " + str(np.mean(score_times)))
        # If the trained agent reached the target check its stability over 100 episodes and exit
        if np.mean(score_times) > TARGET:
            evaluation = []
            for i in range(100):
                step = 0
                state = env.reset()
                state = np.reshape(state, [1, observation_space])
                done = False
                while (not done):
                    step += 1
                    action = dqn_solver.act_eval(state)
                    state_next, reward, done, info = env.step(action)
                    state_next = np.reshape(state_next, [1, observation_space])
                    state = state_next

                    if done:
                        evaluation.append(step)
            if np.mean(evaluation) > TARGET:
                np.save('weights', np.array(agents_weights))
                np.save('scores', np.array(score))
                condition = False
