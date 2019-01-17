import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from gym import wrappers

from config import ENVIRONMENT as env_cfg
from config import TD3_Config as td3_cfg
from config import LOGGER
from .td3 import TD3
from copy import deepcopy


class Agent:

    def __init__(self):
        self.agent = None

    def train(self):
        """Train."""

        # log_dir = os.path.join(config.job_dir, 'log')

        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        model_path = os.path.join(td3_cfg.models_path, '{}-{}'.format(env_cfg.name, timestamp), 'model.ckpt')
        results_path = os.path.join(td3_cfg.models_path, '{}-{}'.format(env_cfg.name, timestamp), 'results.npy')
        video_dir = os.path.join(td3_cfg.models_path, '{}-{}'.format(env_cfg.name, timestamp), 'video')

        env = env_cfg.env

        if td3_cfg.record_videos:
            env = wrappers.Monitor(env, video_dir, video_callable=lambda ep: ep % td3_cfg.record_videos == 0)

        rewards = []

        with tf.Session() as sess:

            self.agent = TD3(sess)

            saver = tf.train.Saver()
            # tf.logging.info('Start to train {} ...'.format(config.agent))
            init = tf.global_variables_initializer()
            sess.run(init)
            # writer = tf.summary.FileWriter(log_dir, sess.graph)

            self.agent.initialize()
            global_step = 0

            for i in range(td3_cfg.n_episodes):

                s = env.reset()
                ep_reward = 0
                ep_steps = 0
                noises = []
                actions = []
                done = False

                while not done:

                    env.render()

                    if ep_steps < 5:
                        action = self.agent.random_action()
                    else:
                        action, action_org, noise = self.agent.action_with_noise(s)
                        noises.append(noise)
                        actions.append(action_org)
                    action = action.squeeze()

                    s2, r, done, info = env.step(action.tolist())
                    ep_reward += r
                    ep_steps += 1
                    global_step += 1
                    self.agent.store_experience(s, action, r, done, s2)

                    flipped_s = reverse_obs(s)
                    flipped_s2 = reverse_obs(s2)
                    flipped_a = reverse_act(action)
                    self.agent.store_experience(flipped_s, flipped_a, r, done, flipped_s2)

                    self.agent.train(global_step)
                    s = s2

                    if done:
                        count = i + 1

                        if count % td3_cfg.test_every == 0:
                            eval_ep_reward, eval_ep_steps = self.evaluate(env)
                            print("Episode: {:<10d}  Reward: {:<+10.3f}  Total Steps: {:10d}".format(count, ep_reward,
                                                                                                     global_step))
                            rewards.append(eval_ep_reward)
                            saver.save(sess, model_path, global_step=count)
                            np.save(results_path, rewards)

        LOGGER.log(environment=env_cfg.name,
                   timestamp=timestamp,
                   algorithm=self.agent.__class__.__name__,
                   parameters=vars(td3_cfg),
                   total_steps=global_step,
                   score=ep_reward)
        env.close()

    def evaluate(self, env):
        """

        :param env:
        :return:
        """
        tf.logging.info('Testing ...')
        s = env.reset()
        ep_reward = 0
        ep_steps = 0
        done = False

        while not done:
            if ep_steps < 5:
                action = self.agent.random_action()
            else:
                action = self.agent.action(s)
            s2, r, done, info = env.step(action.squeeze().tolist())
            ep_reward += r
            ep_steps += 1
            s = s2
        return ep_reward, ep_steps


def reverse_obs(states):
    """

    :param states:
    :return:
    """
    mirror_states = deepcopy(states)
    tmp = deepcopy(mirror_states[4:9])
    mirror_states[4:9] = mirror_states[9:14]
    mirror_states[9:14] = tmp
    return mirror_states


def reverse_act(action):
    """

    :param action:
    :return:
    """
    mirror_actions = deepcopy(action)
    tmp = deepcopy(mirror_actions[:2])
    mirror_actions[:2] = mirror_actions[2:]
    mirror_actions[2:] = tmp
    return mirror_actions
