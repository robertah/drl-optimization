import numpy as np
import tensorflow as tf
import json

from .replay_buffer import ReplayBuffer
from .actor import ActorNetwork
from .critic import CriticNetwork
from .noise import OrnsteinUhlenbeckNoise

from config import ENVIRONMENT


def get_noisy_action(action, noise):
    return action + noise.noise()


def playGame(train_indicator=1):  # 1 means Train, 0 means simply Run
    BUFFER_SIZE = 1000000
    BATCH_SIZE = 128
    GAMMA = 0.99
    TAU = 0.001  # Target Network HyperParameters
    LRA = 0.001  # Learning rate for Actor
    LRC = 0.001  # Lerning rate for Critic

    action_size = ENVIRONMENT.action_size  # Steering/Acceleration/Brake
    state_size = ENVIRONMENT.state_size  # of sensors input

    np.random.seed(1337)

    vision = False

    # EXPLORE = 100000.
    episode_count = 5000
    max_steps = 2000
    reward = 0
    done = False
    step = 0
    # epsilon = 1
    indicator = 0

    noise = OrnsteinUhlenbeckNoise(action_size)

    # Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    actor = ActorNetwork(sess, state_size, action_size, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_size, action_size, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)  # Create replay buffer

    for i in range(episode_count):

        print("Episode : " + str(i) + " Replay Buffer " + str(buff.n_experiences))
        state = ENVIRONMENT.env.reset()

        total_reward = 0

        for j in range(max_steps):
            loss = 0
            # epsilon -= 1.0 / EXPLORE

            action = actor.model.predict(state.reshape(1, state.shape[0]))

            action = get_noisy_action(action, noise)

            new_state, reward, done, info = ENVIRONMENT.env.step(action[0])

            buff.add(state, action[0], reward, new_state, done)

            s, a, r, new_s, d = buff.get_batch(BATCH_SIZE)

            y_t = r
            target_q_values = critic.target_model.predict([new_s, actor.target_model.predict(new_s)])

            for k in range(len(d)):
                if d[k]:
                    y_t[k] = r[k]
                else:
                    y_t[k] = r[k] + GAMMA * target_q_values[k]

            if train_indicator:
                loss += critic.model.train_on_batch([s, a], y_t)
                a_for_grad = actor.model.predict(s)
                grads = critic.gradients(s, a_for_grad)
                actor.train(s, grads)

                actor.target_train()
                critic.target_train()

            total_reward += reward

            if np.array_equal(np.around(new_state, 3), np.around(state, 3)):
               break

            state = new_state

            # print("Episode", i, "Step", step, "Action", action, "Reward", reward, "Loss", loss)

            step += 1
            if done:
                break

        print("TOTAL REWARD @ " + str(i) + "-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
