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
        """
        STEP 1
        Gaussian NN Policy 
        using tf.keras.layers.Dense
        """
        ########################
        # YOUR IMPLEMENTATION PART

        ### Set hid_size freely,
        hid1_size = 64
        hid2_size = 64
        # 2 hidden layers with tanh activations
        h1 = tf.keras.layers.Dense(hid1_size)(self.obs_ph)
        h2 = tf.keras.layers.Dense(hid2_size)(h1)
        means = tf.keras.layers.Dense(act_dim)(h2)
        ########################

        log_vars = self.add_weight('logvars', (act_dim), initializer=tf.constant_initializer(-1.0))
        batch_size = tf.shape(self.obs_ph)[0]
        self.sampled_act = means + tf.exp(log_vars) * tf.random_normal(shape=(batch_size, act_dim,))

    def sample(self, obs):
        sess = tf.keras.backend.get_session()
        return sess.run(self.sampled_act, feed_dict={self.obs_ph: np.array([obs])})[0]


if __name__ == "__main__":

    env_name = 'MountainCarContinuous-v0'

    """
    STEP 2
    Run one episode with random policy, Store observations, actions, rewards in the list
    """
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    policy = Policy(obs_dim, act_dim)

    obs = env.reset()
    observes, actions, rewards= [], [], []
    done = False
    while not done:
        ########################
        # YOUR IMPLEMENTATION PART
        act = policy.sample(obs)
        obs, reward, done, _ = env.step(act)
        observes.append(obs)
        actions.append(act)
        rewards.append(reward)
        ########################

    accumulated_reward = np.sum(rewards)
    print("======================================")
    print(f"During one episodes, The accumulated Rewards: {accumulated_reward}")
    print("======================================")

    """
    STEP3
    """
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
            # YOUR IMPLEMENTATION PART
            act = policy.sample(obs)
            obs, reward, _ = env.step(act)
            accumulated_reward += reward
            ########################

            if done:
                break
        returns.append(accumulated_reward)

    avg_returns = np.mean(returns)
    print("======================================")
    print(f"During 10 episodes, The average of accumulated Rewards: {avg_returns}")
    print("======================================")
