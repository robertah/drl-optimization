import gym
from gym.spaces.discrete import Discrete
import numpy as np
from keras.layers import Dense
from keras.models import Sequential

from config import ENVIRONMENT, RANDOM_SEED


class Agent:

    def __init__(self, weights=None):
        self.env = gym.make(ENVIRONMENT.name)
        # In case of CartPole-v1, maximum length of episode is 500
        self.max_time = ENVIRONMENT.max_time
        self.env._max_episode_steps = 2000  # TODO cos'è?
        # score_logger = ScoreLogger('CartPole-v1')
        # get size of state and action from environment
        if isinstance(self.env.observation_space, Discrete):
            self.state_size = self.env.observation_space.n
        else:
            self.state_size = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, Discrete):
            self.action_size = self.env.action_space.n
        else:
            self.action_size = self.env.action_space.shape[0]
        if RANDOM_SEED:
            self.env.seed(RANDOM_SEED)
        self.load_model = False  # TODO not used
        # get size of state and action
        self.value_size = 1  # TODO not used
        # create model for policy network
        self.model = self.build_model()
        if weights is not None:
            self.model.set_weights(weights)

    def build_model(self):
        model = Sequential()
        model.add(Dense(ENVIRONMENT.hidden_units[0], input_dim=self.state_size, activation='relu',
                        kernel_initializer='glorot_uniform'))
        if len(ENVIRONMENT.hidden_units) > 1:
            for i in range(1, len(ENVIRONMENT.hidden_units)):
                model.add(Dense(ENVIRONMENT.hidden_units[i], activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Dense(self.action_size, activation='softmax', kernel_initializer='he_uniform'))
        # model.summary()
        return model

    def get_action(self, state):
        policy = self.model.predict(state, batch_size=1).flatten()
        # secondo me non ha molto senso fare una multinomial perchè ora non stiamo trainando niente quindi è solo una cosa deterministica che sceglie l'azione in base alla policy, dall'esploration non trarrebbe effettivamente nessun vantaggio se non eventualmente un punteggio più alto dovuto alla casualità del sampling e che quindi non rispecchia proprimanete la policy e lo stesso per un punteggio più basso.
        # per questo prenderei soltanto l'azione con la probabilità maggiore
        # return np.random.choice(self.action_size, 1, p=policy)[0]
        return np.argmax(policy)

    def run_agent(self, render=ENVIRONMENT.animate):
        scores = []
        n_times = 8

        for i in range(n_times):
            done = False
            score = 0
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            # print("intial state: ",state)
            while (not done) and (score < self.max_time):
                if render:
                    self.env.render()

                action = self.get_action(state)
                next_state, reward, done, info = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                score += reward
                state = next_state

                if done:
                    scores.append(score)
        return np.mean(scores)
