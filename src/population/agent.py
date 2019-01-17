import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.initializers import RandomNormal, RandomUniform
from config import ENVIRONMENT, POPULATION


class Agent:

    def __init__(self, weights=None):
        self.env = ENVIRONMENT
        self.model = self.build_model()
        if weights is not None:
            self.model.set_weights(weights)

    def build_model(self):
        '''
        For the bipedal walker I had to use tanh because the output has to be a 4-dim vector with entries between [-1,1]
        (not sure if it is the best setting, the important thing is that the entries have to be positive and negative,
        the gym code already put action >1 or < -1 equal to 1 and -1 )
        :param environment:
        :return: model
        '''
        model = Sequential()
        model.add(Dense(
            units=self.env.hidden_units[0],
            input_dim=self.env.state_size,
            activation=self.env.activations[0],
            kernel_initializer='he_uniform'
        )
        )
        if len(self.env.hidden_units) > 1:
            for i in range(1, len(self.env.hidden_units)):
                model.add(Dense(
                    units=self.env.hidden_units[i],
                    activation=self.env.activations[i],
                    kernel_initializer='he_uniform'
                )
                )

        model.add(Dense(self.env.action_size, activation='softmax', kernel_initializer='he_uniform'))
        # model.summary()
        return model

    def get_action(self, state):
        policy = self.model.predict(state, batch_size=1).flatten()
        if self.env.policy == 'normal':
            return policy
        elif self.env.policy == 'argmax':
            return np.argmax(policy)
        # return policy

    def run_agent(self, render=None):
        if render is None:
            render = self.env.animate
        scores = []
        for i in range(POPULATION.n_runs):
            done = False
            score = 0
            state = self.env.env.reset()
            state = np.reshape(state, [1, self.env.state_size])

            while (not done) and (score < self.env.max_time):
                if render:
                    self.env.env.render()

                action = self.get_action(state)
                next_state, reward, done, info = self.env.env.step(action)
                next_state = np.reshape(next_state, [1, self.env.state_size])
                score += reward
                state = next_state

                if done:
                    scores.append(score)
        return np.mean(scores)
