from keras.models import Sequential
from keras.layers import Dense, Input, merge, Activation, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.initializers import RandomUniform, VarianceScaling
import keras.backend as K
import tensorflow as tf


var_scaling = VarianceScaling()
init = RandomUniform(-0.003, 0.003)


class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size

        K.set_session(sess)

        # Now create the model
        self.model, self.action, self.state = self.create_critic_network(state_size, action_size)
        self.target_model, self.target_action, self.target_state = self.create_critic_network(state_size, action_size)
        self.action_grads = tf.gradients(self.model.output, self.action)  # GRADIENTS for policy update
        self.sess.run(tf.initialize_all_variables())
        self.model.summary()

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU) * critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, state_size, action_size):
        # S = Input(shape=[state_size])
        # A = Input(shape=[action_size])
        # w1 = Dense(256, activation='relu')(S)
        # a1 = Dense(128, activation='linear')(A)
        # h1 = Dense(128, activation='linear')(w1)
        # h2 = merge([h1, a1], mode='sum')
        # h3 = Dense(64, activation='relu')(h2)
        # value = Dense(1, activation='linear')(h3)

        S = Input(shape=[state_size])
        w1 = Dense(256, activation='relu', kernel_initializer=var_scaling)(S)
        h1 = Dense(128, activation='linear', kernel_initializer=var_scaling)(w1)

        A = Input(shape=[action_size])
        a1 = Dense(128, activation='linear', kernel_initializer=var_scaling)(A)

        h2 = concatenate([h1, a1])

        h3 = Dense(128, activation='relu', kernel_initializer=var_scaling)(h2)
        value = Dense(1, activation='linear', kernel_initializer=init)(h3)

        model = Model(input=[S, A], output=value)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S
