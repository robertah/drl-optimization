# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import gym
from gym import wrappers

import tensorflow as tf
import numpy as np

import pickle

from .td3 import TD3
from .util import *

from config import ENVIRONMENT
from config import TD3_Config as conf
import os
from datetime import datetime

LR_C = 2e-4
LR_A = 1e-4
GAMMA = 0.99
TAU = 0.001
NOISE_STD = 0.1
BUFFER_SIZE = 1000000
WARMUP_BUFFER = 10000
BATCH_SIZE = 32
N_EPISODES = 5000
TEST_STEPS = 100


# def build_summaries():
#     """Training and evaluation summaries."""
#     # train summaries
#     episode_reward = tf.placeholder(dtype=tf.float32, shape=[])
#     summary_vars = [episode_reward]
#     with tf.name_scope('Training'):
#         reward = tf.summary.scalar("Reward", episode_reward)
#     summary_ops = tf.summary.merge([reward])
#     # eval summary
#     eval_episode_reward = tf.placeholder(dtype=tf.float32, shape=[])
#     eval_summary_vars = [eval_episode_reward]
#     with tf.name_scope('Evaluation'):
#         eval_reward = tf.summary.scalar("EvalReward", eval_episode_reward)
#     eval_summary_ops = tf.summary.merge([eval_reward])
#
#     return summary_ops, summary_vars, eval_summary_ops, eval_summary_vars


training = False

def log_metrics(sess, writer, summary_ops, summary_vals, metrics, test=False):
    """Log metrics."""
    ep_cnt, ep_r, steps, actions, noises = metrics
    if test:
        tf.logging.info(
            '[TEST] Episode: {:d} | Reward: {:.2f} | AvgReward: {:.2f} | '
            'Steps: {:d}'.format(ep_cnt, ep_r, ep_r / steps, steps))
    else:
        aa = np.array(actions).mean(axis=0).squeeze()
        nn = np.array(noises).mean(axis=0).squeeze()
        tf.logging.info(
            '| Episode: {:d} | Reward: {:.2f} | AvgReward: {:.2f} | '
            'Steps: {:d} | AvgAction: {} | AvgNoise: {}'.format(
                ep_cnt, ep_r, ep_r / steps, steps, aa, nn))
    summary_str = sess.run(summary_ops, feed_dict={summary_vals[0]: ep_r})
    writer.add_summary(summary_str, ep_cnt)
    writer.flush()


def test(env, agent=None):
    """Test the trained agent"""
    tf.logging.info('Testing ...')
    s = env.reset()
    ep_reward = 0
    ep_steps = 0
    done = False

    while not done:
        if ep_steps < 5:
            action = agent.random_action(s)
        else:
            action = agent.action(s)
        s2, r, done, info = env.step(action.squeeze().tolist())
        ep_reward += r
        ep_steps += 1
        s = s2
    return ep_reward, ep_steps


def train():
    """Train."""

    # log_dir = os.path.join(config.job_dir, 'log')

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    model_path = os.path.join(conf.models_path, '{}'.format(timestamp), 'model.ckpt')
    results_path = os.path.join(conf.models_path, '{}'.format(timestamp), 'results.npy')
    video_dir = os.path.join(conf.models_path, '{}'.format(timestamp), 'video')

    rewards = []
    record_video = True
    env = ENVIRONMENT.env
    if record_video:
        eval_interval = TEST_STEPS
        env = wrappers.Monitor(env, video_dir, video_callable=lambda ep: ep % 2 == 0)

    # (summary_ops, summary_vars, eval_summary_ops, eval_summary_vars) = build_summaries()

    with tf.Session() as sess:

        agent = TD3(env, sess)

        saver = tf.train.Saver(max_to_keep=50)
        # tf.logging.info('Start to train {} ...'.format(config.agent))
        init = tf.global_variables_initializer()
        sess.run(init)
        # writer = tf.summary.FileWriter(log_dir, sess.graph)

        agent.initialize()
        global_step = 0
        for i in range(N_EPISODES+1):

            s = env.reset()
            ep_reward = 0
            ep_steps = 0
            noises = []
            actions = []
            done = False

            while not done:

                # env.render()

                if ep_steps < 5:
                    action = agent.random_action(s)
                else:
                    action, action_org, noise = agent.action_with_noise(s)
                    noises.append(noise)
                    actions.append(action_org)
                action = action.squeeze()

                s2, r, done, info = env.step(action.tolist())
                ep_reward += r
                ep_steps += 1
                global_step += 1
                agent.store_experience(s, action, r, done, s2)

                # mirror observations and actions
                flipped_s = reverse_obs(s)
                flipped_s2 = reverse_obs(s2)
                flipped_a = reverse_act(action)
                agent.store_experience(flipped_s, flipped_a, r, done, flipped_s2)

                agent.train(global_step)
                s = s2

                if done:
                    ep_cnt = i + 1
                    # log_metrics(sess,
                    #             writer,
                    #             summary_ops,
                    #             summary_vars,
                    #             metrics=(ep_cnt,
                    #                      ep_reward,
                    #                      ep_steps,
                    #                      actions,
                    #                      noises))
                    if ep_cnt % 100 == 0:
                        eval_ep_reward, eval_ep_steps = test(env, agent)
                        rewards.append(eval_ep_reward)
                        eval_ep_cnt = ep_cnt / 100
                        # log_metrics(sess,
                        #             writer,
                        #             eval_summary_ops,
                        #             eval_summary_vars,
                        #             metrics=(eval_ep_cnt,
                        #                      eval_ep_reward,
                        #                      eval_ep_steps,
                        #                      None,
                        #                      None),
                        #             test=True)
                        saver.save(sess, model_path, global_step=ep_cnt)
                        np.save(results_path, rewards)
                        # tf.logging.info('Model saved to {}'.format(ckpt_path))
    env.close()

