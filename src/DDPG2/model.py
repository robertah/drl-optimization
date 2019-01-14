import numpy as np
from keras.layers import Dense, Input, concatenate
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.losses import mean_squared_error

GAMMA = 0.99


class Actor(Sequential):

    def __init__(self, state_size, action_size, learning_rate):
        super(Actor, self).__init__()
        self.add(Dense(input_dim=state_size, units=256, activation='relu'))
        self.add(Dense(units=128, activation='relu'))
        self.add(Dense(units=64, activation='relu'))
        self.add(Dense(units=action_size, activation='tanh'))

    def optimize(self, critic, state):
        predicted_action = self.predict(state)
        loss = -1 * np.sum(critic.predict([state, predicted_action]))

        """
        HO CALCOLATO LA LOSS, MI SERVE OTTIMIZZARE I PARAMETRI DELL'ACTOR
        """


class Critic(Model):

    def __init__(self, state_size, action_size, learning_rate):
        S = Input(shape=[state_size])
        w1 = Dense(256, activation='relu')(S)
        h1 = Dense(128, activation='linear')(w1)

        A = Input(shape=[action_size])
        a1 = Dense(128, activation='linear')(A)

        h2 = concatenate([h1, a1])

        h3 = Dense(64, activation='relu')(h2)
        value = Dense(1, activation='linear')(h3)

        super(Critic, self).__init__(input=[S, A], output=value)
        # model = Model(input=[S, A], output=value)
        # adam = Adam(lr=learning_rate)
        # self.compile(loss='mse', optimizer=adam)

    def optimize(self, target_actor, target_critic, s1, a1, r1, s2):
        a2 = target_actor.predict(s2)
        next_val = target_critic.predict([s2, a2])

        y_expected = r1 + GAMMA * next_val
        y_predicted = self.predict([s1, a1])

        loss = mean_squared_error(y_predicted, y_expected)
        """
        HO CALCOLATO LA LOSS, MI SERVE OTTIMIZZARE I PARAMETRI DEL CRITIC
        """


def update_target(target, model, tau):
    """
    Update target networks:
        target = tau * model + (1 - tau) * target

    :param target: target network (i.e. target_actor or target_critic
    :type target: keras.Model
    :param model: model network (i.e. actor or critic)
    :type model: keras.Model
    :param tau: target update coefficient (tau << 1)
    :type tau: float
    :return: updated target network
    """
    model_weights = model.layers.get_weights()
    target_weights = target.layers.get_weights()
    for i, m_weights in enumerate(model_weights):
        target_weights[i] = tau * m_weights + (1 - tau) * target_weights[i]
    target.set_weights(target_weights)
    return target
