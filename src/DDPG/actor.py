import keras.backend as K
import tensorflow as tf
from keras.layers import Dense, Input
from keras.models import Model


class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):

        print(state_size, action_size)
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        # Now create the model
        self.model, self.weights, self.state = self.create_actor_network(state_size, action_size)
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size)
        self.action_gradient = tf.placeholder(tf.float32, [None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.initialize_all_variables())
        self.model.summary()

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU) * actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    @staticmethod
    def create_actor_network(state_size, action_dim):
        state = Input(shape=[state_size])
        h0 = Dense(256, activation='relu')(state)
        h1 = Dense(128, activation='relu')(h0)
        h2 = Dense(64, activation='relu')(h1)
        action = Dense(action_dim, activation='tanh')(h2)

        model = Model(input=state, output=action)
        return model, model.trainable_weights, state
