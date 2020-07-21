import numpy as np
import argparse
import gym
import tensorflow as tf
import pybulletgym


class Policy(tf.layers.Layer):
    """ NN-based policy approximation """
    def __init__(self, obs_dim, act_dim):
        """
        Args:
            obs_dim: num observation dimensions (int)
            act_dim: num action dimensions (int)
        """
        super(Policy, self).__init__()
        self.obs_ph = tf.keras.layers.Input(obs_dim, name='obs')  # [batch_size, obs_dim]

        ### Set hid_size freely,
        hid1_size = 64 ### YOUR IMPLETETAION PART ###
        hid2_size = 64 ### YOUR IMPLETETAION PART ###
        # 2 hidden layers with tanh activations
        h1 = tf.keras.layers.Dense(hid1_size)(self.obs_ph) ### YOUR IMPLETETAION PART ###
        h2 = tf.keras.layers.Dense(hid2_size)(h1) ### YOUR IMPLETETAION PART ###
        means = tf.keras.layers.Dense(act_dim)(h2) ### YOUR IMPLETETAION PART ###
        log_vars = self.add_weight('logvars', (act_dim), initializer=tf.constant_initializer(-1.0))
        batch_size = tf.shape(self.obs_ph)[0]
        self.sampled_act = means + tf.exp(log_vars) * tf.random_normal(shape=(batch_size, act_dim,))

    def sample(self, obs):
        sess = tf.keras.backend.get_session()
        return sess.run(self.sampled_act, feed_dict={self.obs_ph: np.array([obs])})[0]


def init_gym(env_name, animate=False):

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    if animate:
        env.render()

    return env, obs_dim, act_dim


if __name__ == "__main__":

    env_name = 'MountainCarContinuous-v0'

    """
    STEP 2
    Run one episode with random policy, Store observations, actions, rewards in the list
    """
    env, obs_dim, act_dim = init_gym(env_name, animate=True)
    policy = Policy(obs_dim, act_dim)

    obs = env.reset()
    observes, actions, rewards= [], [], []
    done = False
    while not done:
        ########################
        # YOUR IMPLETETAION PART
        pass
        ########################

    accumulated_reward = np.sum(rewards)
    print("======================================")
    print(f"During one episodes, The accumulated Rewards: {accumulated_reward}")
    print("======================================")
    del env

    """
    STEP3
    """
    env, _, _ = init_gym(env_name, animate=False)
    episodes = 10

    total_reward = 0
    returns = []
    for e in range(episodes):
        obs = env.reset()
        done = False
        accumulated_reward = 0
        # For one episode
        for t in range(10000):
            ########################
            # YOUR IMPLETETAION PART
            accumulated_reward += 0
            ########################

            if done:
                break
        returns.append(accumulated_reward)

    avg_returns = np.mean(returns)
    print("======================================")
    print(f"During 10 episodes, The average of accumulated Rewards: {avg_returns}")
    print("======================================")
