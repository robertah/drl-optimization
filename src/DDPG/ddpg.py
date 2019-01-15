from datetime import datetime

import numpy as np
import tensorflow as tf
from keras import backend as K

from config import LOGGER
from .models import Actor, Critic
from .noise import OrnsteinUhlenbeckNoise
from .replay_buffer import ReplayBuffer


class DDPG:

    def __init__(self, environment, ddpg_config):
        self.environment = environment

        self.n_episodes = ddpg_config.n_episodes

        self.batch_size = ddpg_config.batch_size
        self.gamma = ddpg_config.gamma
        self.tau = ddpg_config.tau
        self.actor_lr = ddpg_config.actor_lr
        self.critic_lr = ddpg_config.critic_lr

        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        K.set_session(sess)

        self.actor = Actor(sess, self.environment.state_size, self.environment.action_size, self.batch_size,
                           self.tau, self.actor_lr, self.environment.hidden_units, self.environment.activations)

        self.critic = Critic(sess, self.environment.state_size, self.environment.action_size, self.batch_size,
                             self.tau, self.critic_lr)

        self.buffer = ReplayBuffer(ddpg_config.buffer_size)
        self.noise = OrnsteinUhlenbeckNoise(self.environment.env.action_space)

        # models' weights path
        self.actor_weights = ddpg_config.actor_weights
        self.critic_weights = ddpg_config.critic_weights
        self.target_actor_weights = ddpg_config.target_actor_weights
        self.target_critic_weights = ddpg_config.target_critic_weights

    def get_params(self):
        params = {
            'n_episodes': self.n_episodes,
            'batch_size': self.batch_size,
            'gamma': self.gamma,
            'tau': self.tau,
            'actor_lr': self.actor_lr,
            'critic_lr': self.critic_lr
        }
        return params

    def run(self, train=True):

        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

        step = 0

        self.load_weights()

        try:

            for i in range(self.n_episodes):

                # get initial state
                state = self.environment.env.reset()

                total_reward = 0

                for j in range(self.environment.max_time):
                    # loss = 0

                    self.environment.env.render()

                    action = self.actor.model.predict(state.reshape(1, state.shape[0]))
                    action = self.noise.get_noisy_action(action, j)

                    new_state, reward, done, info = self.environment.env.step(action[0])

                    self.buffer.add(state, action[0], reward, new_state, done)

                    s, a, r, new_s, d = self.buffer.get_batch(self.batch_size)

                    target_q = self.critic.target_model.predict([new_s, self.actor.target_model.predict(new_s)])

                    y = r
                    for k in range(len(d)):
                        # if d[k]:
                        #     y[k] = r[k]
                        # else:
                        #     y[k] = r[k] + self.gamma * target_q[k]
                        y[k] = r[k] + self.gamma * target_q[k]

                    if train:
                        # loss += self.critic.model.train_on_batch([s, a], y)
                        self.critic.model.train_on_batch([s, a], y)
                        a_grads = self.actor.model.predict(s)
                        grads = self.critic.gradients(s, a_grads)
                        self.actor.train(s, grads)

                        self.actor.update_target()
                        self.critic.update_target()

                    total_reward += reward

                    if i + 1 % 200 == 0:
                        LOGGER.log(environment=self.environment.name,
                                   timestamp=timestamp,
                                   algorithm=self.__class__.__name__,
                                   parameters=self.get_params(),
                                   episodes=i,
                                   score=total_reward)
                        self.save_weights()

                    if done or np.array_equal(np.around(new_state, 3), np.around(state, 3)):
                        previous_reward = total_reward
                        break

                    state = new_state

                    # print("Episode", i, "Step", step, "Action", action, "Reward", reward, "Loss", loss)

                    step += 1

                print("Episode: {:<5d}  Total Reward: {:<+10.3f}  Total Steps: {:<10d} "
                      " Replay Buffer size: {}".format(i, total_reward, step, self.buffer.n_experiences))

        except KeyboardInterrupt:
            if i >= 100:
                print("Saving weights...")
                LOGGER.log(environment=self.environment.name,
                           timestamp=timestamp,
                           algorithm=self.__class__.__name__,
                           parameters=self.get_params(),
                           episodes=i,
                           score=previous_reward)
                self.save_weights()

    def load_weights(self):
        try:
            self.actor.model.load_weights(self.actor_weights)
            self.critic.model.load_weights(self.critic_weights)
            self.actor.target_model.load_weights(self.target_actor_weights)
            self.critic.target_model.load_weights(self.target_critic_weights)
            print("Weights loaded successfully.")
        except (FileNotFoundError, OSError):
            print("Pre-trained weights not found.")

    def save_weights(self):
        self.actor.model.save_weights(self.actor_weights)
        self.critic.model.save_weights(self.critic_weights)
        self.actor.target_model.save(self.target_actor_weights)
        self.critic.target_model.save(self.target_critic_weights)
