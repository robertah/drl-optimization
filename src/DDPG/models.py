from keras.layers import Dense, Input, concatenate
import keras.backend as K
import tensorflow as tf
from keras.initializers import RandomUniform, VarianceScaling
from keras.layers import Dense, Input, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

# var_scaling = VarianceScaling()
unif_init = 'lecun_uniform'
init = RandomUniform(-0.003, 0.003)


class Actor:

    def __init__(self, sess, state_size, action_size, batch_size, tau, learning_rate, hidden_units, activations):
        self.sess = sess
        self.batch_size = batch_size
        self.tau = tau
        self.lr = learning_rate

        K.set_session(sess)

        # Now create the model
        self.model, self.weights, self.state = self.build_model(state_size, action_size, hidden_units, activations)
        self.target_model, self.target_weights, self.target_state = self.build_model(state_size, action_size,
                                                                                     hidden_units, activations)
        self.action_gradient = tf.placeholder(tf.float32, [None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(learning_rate).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())
        # self.model.summary()

    def build_model(self, state_size, action_dim, hidden_units, activations):
        state = Input(shape=[state_size])

        h0 = Dense(256, activation='relu', kernel_initializer=unif_init)(state)
        # h1 = Dense(128, activation='relu', kernel_initializer=var_scaling)(h0)
        # h2 = Dense(64, activation='relu', kernel_initializer=var_scaling)(h1)

        action = Dense(action_dim, activation='tanh', kernel_initializer=init)(h0)

        model = Model(inputs=state, outputs=action)
        return model, model.trainable_weights, state

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def update_target(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.tau * actor_weights[i] + (1 - self.tau) * actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)


class Critic:

    def __init__(self, sess, state_size, action_size, batch_size, tau, learning_rate):
        self.sess = sess
        self.batch_size = batch_size
        self.tau = tau
        self.lr = learning_rate
        self.action_size = action_size

        K.set_session(sess)

        self.model, self.state, self.action = self.build_model(state_size, action_size)
        self.target_model, self.target_state, self.target_action = self.build_model(state_size, action_size)
        self.action_grads = tf.gradients(self.model.output, self.action)
        self.sess.run(tf.global_variables_initializer())
        # self.model.summary()

    def build_model(self, state_size, action_size):
        state = Input(shape=[state_size])
        h1 = Dense(256, activation='relu', kernel_initializer=unif_init)(state)
        # h1 = Dense(128, activation='relu')(w1)

        action = Input(shape=[action_size])
        # a1 = Dense(128, activation='relu', kernel_initializer=var_scaling)(action)

        h2 = concatenate([h1, action])

        h3 = Dense(256, activation='relu', kernel_initializer=unif_init)(h2)

        h4 = Dense(128, activation='relu', kernel_initializer=unif_init)(h3)
        value = Dense(1, activation='linear', kernel_initializer=init)(h4)

        model = Model(inputs=[state, action], outputs=value)
        adam = Adam(lr=self.lr, decay=0.0001)
        model.compile(loss='mse', optimizer=adam)
        return model, state, action

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def update_target(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.tau * critic_weights[i] + (1 - self.tau) * critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)
