import random
import time
from collections import deque

import gym
import numpy as np
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam

np.set_printoptions(precision=3, suppress=True, threshold=10000, linewidth=250)


class DQNAgent:

    def __init__(self, state_dim, action_size, gamma=0.99):
        self.state_dim = state_dim
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = gamma

        self.batch_size = 64
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995

        self.q_model = self._build_model()
        self.q_model.compile(loss='mse', optimizer=Adam(lr=0.001))
        self.q_model.predict_one = lambda x: self.q_model.predict(np.array([x]))[0]
        self.target_q_model = self._build_model()
        self.target_q_model.predict_one = lambda x: self.target_q_model.predict(np.array([x]))[0]

        self.update_target_q_weights()  # target Q network 의 파라미터를 Q-newtork 에서 복사

    def _build_model(self):
        model = Sequential([
            Dense(64, input_dim=self.state_dim, activation='relu'),
            Dense(64, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        return model

    def update_target_q_weights(self):
        #################################
        # TODO:
        pass
        #################################

    def act(self, state):
        # epsilon-greedy policy
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, self.action_size)
        else:
            q_values = self.q_model.predict_one(state)
            return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(agent.memory) < self.batch_size:
            return
        mini_batch = random.sample(self.memory, self.batch_size)
        input_state_batch, target_q_values_batch = [], []

        for state, action, reward, next_state, done in mini_batch:
            q_values = self.q_model.predict_one(state)

            #################################
            # TODO:
            if done:
                q_values[action] = 0
            else:
                q_values[action] = 0
            ##################################

            input_state_batch.append(state)
            target_q_values_batch.append(q_values)

        # Q-network 학습
        self.q_model.fit(np.array(input_state_batch), np.array(target_q_values_batch), batch_size=self.batch_size, epochs=1)

    def update_epsilon(self):
        self.epsilon = np.max([self.epsilon * self.epsilon_decay, self.epsilon_min])


""" Load environment """
env_name = 'CartPole-v0'
# env_name = 'MyPendulum-v0'

env = gym.make(env_name)
env.T = env.R = None

state_dim = env.observation_space.shape[0]
action_size = env.action_space.n

agent = DQNAgent(state_dim, action_size, gamma=0.99)

for episode in range(5000):
    state = env.reset()

    episode_reward = 0.
    for t in range(10000):
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)

        # Replay buffer 에 (s,a,r,s') 저장
        agent.remember(state, action, reward, next_state, done)

        episode_reward += reward
        print("[epi=%4d,t=%4d] state=%4s / action=%s / reward=%7.4f / next_state=%4s / Q[s]=%s" % (episode, t, state, action, reward, next_state, agent.q_model.predict_one(state)))
        if episode % 100 == 0 or episode > 3000:
            env.render()
            time.sleep(0.01)

        if done:
            break
        state = next_state

    # 에피소드가 끝날 때마다 Q-network 을 학습하고, epsilon 을 점차 낮춘다.
    agent.train()
    agent.update_epsilon()

    # 에피소드마다 10번마다 target network 의 파라미터를 현재 Q-network 파라미터로 갱신해준다.
    if episode % 10 == 0:
        agent.update_target_q_weights()

    print('[%4d] Episode reward=%.4f / epsilon=%f' % (episode, episode_reward, agent.epsilon))
time.sleep(10)
