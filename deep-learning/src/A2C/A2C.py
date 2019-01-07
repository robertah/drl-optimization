import sys
import gym
import numpy as np
import matplotlib.pyplot as plt
from .agent import A2CAgent

EPISODES = 1000


def run_agent_a2c():
    # In case of CartPole-v1, maximum length of episode is 500
    env = gym.make('CartPole-v1')
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # make A2C agent
    agent = A2CAgent(state_size, action_size)

    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        if e == 0:
            print(state)

        while not done:
            if agent.render:
                env.render()

            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            # if an action make the episode end, then gives penalty of -100
            reward = reward if not done or score == 499 else -100

            agent.train_model(state, action, reward, next_state, done)

            score += reward
            state = next_state

            if done:
                # every episode, plot the play time
                score = score if score == 500.0 else score + 100
                scores.append(score)
                episodes.append(e)
                print(agent.history)
                plt.plot(agent.history)
                plt.show()
                #pylab.plot(episodes, scores, 'b')
                #pylab.savefig("./save_graph/cartpole_a2c.png")
                print("episode:", e, "  score:", score)

                # if the mean of scores of last 10 episode is bigger than 490
                # stop training
                if np.mean(scores[-min(10, len(scores)):]) > 200:
                    plt.plot(agent.history)
                    plt.show()
                    sys.exit()

        # save the model
        # if e % 50 == 0:
        #     agent.actor.save_weights("./save_model/cartpole_actor.h5")
    # agent.critic.save_weights("./save_model/cartpole_critic.h5")
    return scores
