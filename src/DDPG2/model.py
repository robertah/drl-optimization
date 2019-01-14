import numpy as np
from keras.layers import Dense, Input, concatenate
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.losses import mean_squared_error

GAMMA = 0.99

class Actor(Sequential):

    def __init__(self, state_size, action_size, learning_rate):
        super(Actor, self).__init__()
        self.build_model(state_size, action_size, learning_rate)

    def build_model(self, state_size, action_size, learning_rate):
        self.add(Dense(input_dim=state_size, units=256, activation='relu'))
        self.add(Dense(units=128, activation='relu'))
        self.add(Dense(units=64, activation='relu'))
        self.add(Dense(units=action_size, activation='tanh'))


    def optimize(self, critic, state):
        predicted_action = self.predict(state)
        # loss_actor = -1 * torch.sum(self.critic.forward(s1, pred_a1))
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
        adam = Adam(lr=learning_rate)
        self.compile(loss='mse', optimizer=adam)

    def optimize(self, target_actor, target_critic, s1, a1, r1, s2):
        # a2 = self.target_actor.forward(s2).detach()
        a2 = target_actor.predict(s2)

        # next_val = torch.squeeze(self.target_critic.forward(s2, a2).detach())
        next_val = target_critic.predict([s2, a2])

        # y_exp = r + gamma*Q'( s2, pi'(s2))
        y_expected = r1 + GAMMA * next_val
        # y_pred = Q( s1, a1)
        # y_predicted = torch.squeeze(self.critic.forward(s1, a1))
        y_predicted = self.predict([s1, a1])

        loss = mean_squared_error(y_predicted, y_expected)
        """
        HO CALCOLATO LA LOSS, MI SERVE OTTIMIZZARE I PARAMETRI DEL CRITIC
        """



def update_target(target, model, tau):
    """
    y = TAU*x + (1 - TAU)*y
    :param target:
    :param model:
    :param tau:
    :return:
    """
    model_weights = model.layers.get_weights()
    target_weights = target.layers.get_weights()
    for i, m_weights in enumerate(model_weights):
        target_weights[i] = tau * m_weights + (1 - tau) * target_weights[i]
    target.set_weights(target_weights)
    return target
