import os
from copy import deepcopy
from datetime import datetime

import numpy as np
import tensorflow as tf
from gym import wrappers

from config import ENVIRONMENT as env_cfg
from config import LOGGER
from config import TD3_Config as td3_cfg
from loss_analysis import compute_distance_episodes
from .td3 import TD3


class Agent:

    def __init__(self):
        self.agent = None

    def train(self):
        """Train."""

        # log_dir = os.path.join(config.job_dir, 'log')

        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        model_path = os.path.join(td3_cfg.models_path, '{}-{}'.format(env_cfg.name, timestamp), 'model.ckpt')
        results_path = os.path.join(td3_cfg.models_path, '{}-{}'.format(env_cfg.name, timestamp), 'results.npy')
        distances_path = os.path.join(td3_cfg.models_path, '{}-{}'.format(env_cfg.name, timestamp), 'distances.npy')
        weights_path = os.path.join(td3_cfg.models_path, '{}-{}'.format(env_cfg.name, timestamp),
                                    'weights_init_fin.npy')
        video_dir = os.path.join(td3_cfg.models_path, '{}-{}'.format(env_cfg.name, timestamp), 'video')

        env = env_cfg.env

        if td3_cfg.record_videos:
            env = wrappers.Monitor(env, video_dir, video_callable=lambda ep: ep % td3_cfg.record_videos == 0)

        rewards = []
        distances_consecutive = np.zeros(2, dtype=np.ndarray)
        distances_init = np.zeros(2, dtype=np.ndarray)

        with tf.Session() as sess:

            self.agent = TD3(sess)

            saver = tf.train.Saver()
            # tf.logging.info('Start to train {} ...'.format(config.agent))
            init = tf.global_variables_initializer()
            sess.run(init)
            # writer = tf.summary.FileWriter(log_dir, sess.graph)

            self.agent.initialize()
            global_step = 0

            weights_init = get_actor_weights(sess)
            weights_old = weights_init

            try:

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

                        temp = self.agent.train(global_step)
                        if temp:
                            weights = temp

                        s = s2

                        if done:
                            count = i + 1
                            rewards.append(ep_reward)

                            weights = get_actor_weights(sess)
                            for iw, w in enumerate(weights):
                                con, init = compute_distance_episodes(weights_init[iw], weights_old[iw], weights[iw])
                                distances_consecutive[iw] = np.append(distances_consecutive[iw], con)
                                distances_init[iw] = np.append(distances_init[iw], init)
                            weights_old = weights

                            if count % td3_cfg.test_every == 0:
                                eval_ep_reward, eval_ep_steps = self.evaluate(env)
                                print(
                                    "Episode: {:<10d} Evaluation Reward: {:<+10.3f}  "
                                    "Total Training Steps: {:10d}".format(count, eval_ep_reward, global_step))
                            if count % td3_cfg.save_every == 0:
                                print("Saving results...")
                                saver.save(sess, model_path, global_step=count)
                                np.save(results_path, rewards)
                                np.save(distances_path, np.vstack((distances_consecutive, distances_init)))
                                np.save(weights_path, np.append(weights_init, weights))
            except KeyboardInterrupt as e:
                print("Training interrupted.")

            print('Total steps:', global_step)
            print("Saving results...")
            LOGGER.log(environment=env_cfg.name,
                       timestamp=timestamp,
                       algorithm=self.agent.__class__.__name__,
                       parameters=vars(td3_cfg),
                       total_steps=global_step,
                       score=ep_reward)
            saver.save(sess, model_path, global_step=count)
            np.save(results_path, rewards)
            np.save(distances_path, np.vstack((distances_consecutive, distances_init)))
            np.save(weights_path, np.append(weights_init, weights))

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


def get_actor_weights(session):
    with tf.variable_scope("actor/actor_hidden1", reuse=True):
        w1 = tf.get_variable("kernel")
    with tf.variable_scope("actor/actor_hidden2", reuse=True):
        w2 = tf.get_variable("kernel")
    w1, w2 = w1.eval(session=session), w2.eval(session=session)
    return w1, w2
