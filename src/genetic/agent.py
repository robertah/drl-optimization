# -*- coding: utf-8 -*-
import gym
from gym.spaces.discrete import Discrete
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
import tensorflow as tf
import keras.backend as K
from config import ENVIRONMENT, RANDOM_SEED

class Environment():
    '''
    To separate agents and environment is usefull in order to initialize the env just once and perform agents on it.
    '''
    def __init__(self):
        self.env = gym.make(ENVIRONMENT.name)
        # In case of CartPole-v1, maximum length of episode is 500
        #self.env._max_episode_steps = ENVIRONMENT.max_time
        # score_logger = ScoreLogger('CartPole-v1') n7
        # get size of state and action from environment
        if isinstance(self.env.observation_space, Discrete):
            self.state_size = self.env.observation_space.n
        else:
            self.state_size = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, Discrete):
            self.action_size = self.env.action_space.n
        else:
            self.action_size = self.env.action_space.shape[0]
        if RANDOM_SEED:
            self.env.seed(RANDOM_SEED)

class Agent():

    def __init__(self, weights=None):
        self.session = tf.Session(graph=tf.Graph())

        # create model for policy network
        #self.model = self.build_model(environment)
        #if weights is not None:
        #    self.model.set_weights(weights)


    '''
        def build_model(self):
            model = Sequential()
            model.add(Dense(ENVIRONMENT.hidden_units[0], input_dim=self.state_size, activation='relu',
                            kernel_initializer='he_uniform'))
            if len(ENVIRONMENT.hidden_units) > 1:
                for i in range(1, len(ENVIRONMENT.hidden_units)):
                    model.add(Dense(ENVIRONMENT.hidden_units[i], activation='relu', kernel_initializer='he_uniform'))
            model.add(Dense(self.action_size, activation='softmax', kernel_initializer='he_uniform'))
            # model.summary()
            return model
    
        def get_action(self, state):
            policy = self.model.predict(state, batch_size=1).flatten()
            # secondo me non ha molto senso fare una multinomial perch ora non stiamo trainando niente quindi  solo una cosa deterministica che sceglie l'azione in base alla policy, dall'esploration non trarrebbe effettivamente nessun vantaggio se non eventualmente un punteggio più alto dovuto alla casualità del sampling e che quindi non rispecchia proprimanete la policy e lo stesso per un punteggio più basso.
            # per questo prenderei soltanto l'azione con la probabilità maggiore
            # return np.random.choice(self.action_size, 1, p=policy)[0]
            return np.argmax(policy)
    '''
    #sto usando i seguenti model e get_aciton per il bipedal:

    def build_model(self, environment):
        '''
        For the bipedal walker I had to use tanh because the output has to be a 4-dim vector with entries between [-1,1]
        (not sure if it is the best setting, the important thing is that the entries have to be positive and negative,
        the gym code already put action >1 or < -1 equal to 1 and -1 )
        :param environment:
        :return: model
        '''
        #graph = tf.Graph()
        #with tf.Session(graph = graph) as sess:
        #sess = tf.Session(graph = graph)

        #K.set_session(sess)
        model = Sequential()
        model.add(Dense(ENVIRONMENT.hidden_units[0], input_dim = environment.state_size, activation='tanh',
                        kernel_initializer='he_uniform'))
        if len(ENVIRONMENT.hidden_units) > 1:
            for i in range(1, len(ENVIRONMENT.hidden_units)):
                model.add(Dense(ENVIRONMENT.hidden_units[i], activation='tanh', kernel_initializer='he_uniform'))
        model.add(Dense(environment.action_size, activation='tanh', kernel_initializer='he_uniform'))
        # model.summary()
        #sess.run(tf.global_variables_initializer())
        return model


    def get_action(self, state):
        policy = self.model.predict(state, batch_size=1).flatten()
        # secondo me non ha molto senso fare una multinomial perch ora non stiamo trainando niente quindi  solo una cosa deterministica che sceglie l'azione in base alla policy, dall'esploration non trarrebbe effettivamente nessun vantaggio se non eventualmente un punteggio più alto dovuto alla casualità del sampling e che quindi non rispecchia proprimanete la policy e lo stesso per un punteggio più basso.
        # per questo prenderei soltanto l'azione con la probabilità maggiore
        # return np.random.choice(self.action_size, 1, p=policy)[0]
        #print(policy)
        return policy/np.linalg.norm(policy)
        #return policy
    '''
    def run_agent(self, environment, render=ENVIRONMENT.animate):
        done = False
        score = 0
        state = environment.env.reset()
        state = np.reshape(state, [1, environment.state_size])
        # print("intial state: ",state)
        while (not done) and (score < ENVIRONMENT.max_time):
            if render:
                environment.env.render()

            action = self.get_action(state)
            next_state, reward, done, info = environment.env.step(action)
            next_state = np.reshape(next_state, [1, environment.state_size])
            score += reward
            #The following if condition is usefull to kill the agent if it is stuck in a position
            if np.array_equal(np.around(next_state,2),np.around(state,2)):
                done = True
                return score
            state = next_state

            if done:
                return score
    '''
    def run_agent(self, environment, render=ENVIRONMENT.animate):
        scores = []
        n_times = 1
        #graph = tf.Graph()
        #with tf.Session(graph = graph) as sess:
        #sess = tf.Session(graph = graph)
        with self.session as sess:
            K.set_session(sess)
            self.model = self.build_model(environment)
            for i in range(n_times):
                done = False
                score = 0
                state = environment.env.reset()
                state = np.reshape(state, [1, environment.state_size])
                # print("intial state: ",state)
                while (not done) and (score < ENVIRONMENT.max_time):
                    if render:
                        environment.env.render()

                    action = self.get_action(state)
                    next_state, reward, done, info = environment.env.step(action)
                    next_state = np.reshape(next_state, [1, environment.state_size])
                    score += reward
                    # The following if condition is usefull to kill the agent if it is stuck in a position
                    if np.array_equal(np.around(next_state, 3), np.around(state, 3)):
                        done = True
                        return score
                    state = next_state

                    if done:
                        scores.append(score)
        return np.mean(scores)
