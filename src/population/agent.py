import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.initializers import RandomNormal, RandomUniform

WEIGHT_INITIALIZER = RandomUniform(-1, 1)


class Agent:

    def __init__(self, environment_config, weights=None):
        self.env = environment_config
        self.model = self.build_model()
        if weights is not None:
            self.model.set_weights(weights)

    # def build_model(self):
    #     model = Sequential()
    #     model.add(Dense(ENVIRONMENT.hidden_units[0], input_dim=self.state_size, activation='relu',
    #                     kernel_initializer='glorot_uniform'))
    #     if len(ENVIRONMENT.hidden_units) > 1:
    #         for i in range(1, len(ENVIRONMENT.hidden_units)):
    #             model.add(Dense(ENVIRONMENT.hidden_units[i], activation='relu', kernel_initializer='glorot_uniform'))
    #     model.add(Dense(self.action_size, activation='softmax', kernel_initializer='he_uniform'))
    #     # model.summary()
    #     return model

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
            kernel_initializer=WEIGHT_INITIALIZER
        )
        )
        if len(self.env.hidden_units) > 1:
            for i in range(1, len(self.env.hidden_units)):
                model.add(Dense(
                    units=self.env.hidden_units[i],
                    activation=self.env.activations[i],
                    kernel_initializer=WEIGHT_INITIALIZER
                )
                )

        model.add(Dense(self.env.action_size, activation='softmax', kernel_initializer=WEIGHT_INITIALIZER))
        # model.summary()
        return model

    def get_action(self, state):
        policy = self.model.predict(state, batch_size=1).flatten()
        # secondo me non ha molto senso fare una multinomial perchè ora non stiamo trainando niente quindi è solo una cosa deterministica che sceglie l'azione in base alla policy, dall'esploration non trarrebbe effettivamente nessun vantaggio se non eventualmente un punteggio più alto dovuto alla casualità del sampling e che quindi non rispecchia proprimanete la policy e lo stesso per un punteggio più basso.
        # per questo prenderei soltanto l'azione con la probabilità maggiore
        # return np.random.choice(self.action_size, 1, p=policy)[0]
        if self.env.policy == 'normal':
            return policy
        elif self.env.policy == 'argmax':
            return np.argmax(policy)
        # return policy

    def run_agent(self, render=None):
        if render is None:
            render = self.env.animate
        scores = []
        for i in range(self.env.n_runs):
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
                # The following if condition is usefull to kill the agent if it is stuck in a position
                # if np.array_equal(np.around(next_state, 3), np.around(state, 3)):
                #     return score

                state = next_state

                if done:
                    scores.append(score)
        return np.mean(scores)
