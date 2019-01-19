import numpy as np
import tensorflow as tf


class Actor:

    def __init__(self, sess, state_size, action_size, action_high, action_low, learning_rate, tau, batch_size,
                 name='actor'):
        """
        Initialize TD3 Actor class

        :param sess: tensorflow session
        :param state_size: environment's state size
        :param action_size: environment's action size
        :param action_high: environment's highest action value
        :param action_low: environment's lowest action value
        :param learning_rate: learning rate
        :param tau: polyak averaging for target network
        :param batch_size: learning batch size
        :param name: namescope
        """

        self.sess = sess
        self.state_size = state_size
        self.action_size = action_size
        self.action_high = action_high
        self.action_low = action_low
        self.learning_rate = learning_rate
        self.grad_norm_clip = 5
        self.tau = tau
        self.batch_size = batch_size

        # generate actor neural network
        with tf.variable_scope(name):
            self.state, self.action = self.build_actor()
        self.params = tf.trainable_variables(scope=name)

        # generate target actor network
        with tf.variable_scope(name + '_target'):
            self.target_state, self.target_action = self.build_actor()
        self.target_params = tf.trainable_variables(scope=name + '_target')

        # set optimizer
        (self.update_target_op, self.action_gradient, self.train_op) = self.create_actor_ops()

    def build_actor(self):
        """
        Build actor network, with two hidden layers (512, 256) with relu non-linearity

        :return: inputs and clipped actions
        """
        s_inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.state_size])
        x = tf.layers.dense(inputs=s_inputs, units=512, activation=tf.nn.tanh, name="actor_hidden1")
        x = tf.layers.dense(inputs=x, units=256, activation=tf.nn.tanh, name="actor_hidden2")
        # x = tf.layers.dense(inputs=s_inputs, units=128, activation=tf.nn.tanh)
        # x = tf.layers.dense(inputs=x, units=128, activation=tf.nn.tanh)
        # x = tf.layers.dense(inputs=x, units=32, activation=tf.nn.tanh)
        actions = tf.layers.dense(inputs=x, units=self.action_size, activation=tf.nn.tanh)
        scaled_actions = actions * (self.action_high - self.action_low) + self.action_low
        return s_inputs, scaled_actions

    def create_actor_ops(self):
        """
        Actor optimization

        :return:
        """
        # update target network with polyak averaging
        target_update_op = tf.group([x.assign(tf.multiply(y, self.tau) + tf.multiply(x, 1. - self.tau)) for x, y in
                                     zip(self.target_params, self.params)])
        # update policy
        grad_phi = tf.placeholder(tf.float32, [None, self.action_size])
        actor_gradients = tf.gradients(self.action, self.params, -grad_phi)
        clipped_grad = map(lambda x: tf.clip_by_norm(tf.div(x, self.batch_size), self.grad_norm_clip), actor_gradients)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_op = optimizer.apply_gradients(zip(clipped_grad, self.params))
        return target_update_op, grad_phi, train_op

    def train(self, s, a_grad):
        """
        Train actor

        :param s: states
        :param a_grad: action gradients
        :return: hidden layers' weights
        """
        self.sess.run(self.train_op, feed_dict={self.state: s, self.action_gradient: a_grad})
        with tf.variable_scope("actor/actor_hidden1", reuse=True):
            w1 = tf.get_variable("kernel")
        with tf.variable_scope("actor/actor_hidden2", reuse=True):
            w2 = tf.get_variable("kernel")
        return w1, w2

    def get_action(self, state):
        """
        Get predicted action

        :param state: input state
        :return: predicted action
        """
        if np.ndim(state) == 1:
            state = np.expand_dims(state, axis=0)
        return self.sess.run(self.action, feed_dict={self.state: state})

    def get_target_action(self, state):
        """
        Get predicted target action

        :param state: input state
        :return:
        """
        if np.ndim(state) == 1:
            state = np.expand_dims(state, axis=0)
        return self.sess.run(self.target_action, feed_dict={self.target_state: state})

    def update_target_network(self):
        self.sess.run(self.update_target_op)


class Critic:

    def __init__(self, sess, state_size, action_size, learning_rate, tau, gamma, name='critic'):
        """
        Initialize TD3 Actor class

        :param sess: tensorflow session
        :param state_size: environment's state size
        :param action_size: environment's action size
        :param learning_rate: learning rate
        :param tau: polyak averaging for target network
        :param name: namescope
        """
        self.sess = sess
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        with tf.variable_scope(name):
            (self.state, self.action, self.q_value) = self.build_critic()
        self.params = tf.trainable_variables(scope=name)
        with tf.variable_scope(name + '_target'):
            (self.target_state, self.target_action, self.target_q_value) = self.build_critic()
        self.target_params = tf.trainable_variables(scope=name + '_target')

        (self.update_target_op, self.y_phi, self.train_op, self.action_grad) = self.create_critic_ops()

    def build_critic(self):
        """
        Build critic network, with two hidden layers (512, 256) with relu non-linearity

        :return: inputs states, actions, q values
        """
        s_inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.state_size])
        a_inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.action_size])
        inputs = tf.concat(values=[s_inputs, a_inputs], axis=1)
        x = tf.layers.dense(inputs=inputs, units=512, activation=tf.nn.relu)
        x = tf.layers.dense(inputs=x, units=256, activation=tf.nn.relu)
        # x = tf.layers.dense(inputs=inputs, units=128, activation=tf.nn.relu)
        # x = tf.layers.dense(inputs=x, units=128, activation=tf.nn.relu)
        # x = tf.layers.dense(inputs=x, units=32, activation=tf.nn.relu)
        q_value = tf.squeeze(tf.layers.dense(inputs=x, units=1))
        return s_inputs, a_inputs, q_value

    def create_critic_ops(self):
        """
        Critic optimization

        :return:
        """
        # update target network with polyak averaging
        target_update_op = tf.group([x.assign(tf.multiply(y, self.tau) + tf.multiply(x, 1. - self.tau)) for x, y in
                                     zip(self.target_params, self.params)])

        # update q function
        y_phi = tf.placeholder(dtype=tf.float32, shape=[None])
        loss = tf.reduce_mean(tf.losses.mean_squared_error(y_phi, self.q_value))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_op = optimizer.minimize(loss, var_list=self.params)
        action_grads = tf.gradients(self.q_value, self.action)
        return target_update_op, y_phi, train_op, action_grads

    def train(self, s, a, y):
        """
        Train critic

        :param s: state
        :param a: action
        :param y: target
        """
        return self.sess.run(self.train_op, feed_dict={
            self.state: s,
            self.action: a,
            self.y_phi: y
        })

    def get_qval(self, state, action):
        """
        Get Q value

        :param state:
        :param action:
        :return: Q value
        """
        if np.ndim(state) == 1:
            state = np.expand_dims(state, axis=0)
        if np.ndim(action) == 1:
            action = np.expand_dims(action, axis=0)
        return self.sess.run(self.q_value, feed_dict={
            self.state: state,
            self.action: action
        })

    def get_target_qval(self, state, action):
        """
        Get target Q value

        :param state: state
        :param action: action
        :return: target Q value
        """
        if np.ndim(state) == 1:
            state = np.expand_dims(state, axis=0)
        if np.ndim(action) == 1:
            action = np.expand_dims(action, axis=0)
        return self.sess.run(self.target_q_value, feed_dict={
            self.target_state: state,
            self.target_action: action
        })

    def get_action_gradients(self, state, actions):
        """
        Get gradients
        :param state:
        :param actions:
        :return:
        """
        return self.sess.run(self.action_grad, feed_dict={
            self.state: state,
            self.action: actions
        })

    def update_target_network(self):
        """
        Update target network
        """
        self.sess.run(self.update_target_op)
