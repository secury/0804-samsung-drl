import numpy as np
import argparse
import gym
import tensorflow as tf
import pybulletgym

class Policy(object):
    """ NN-based policy approximation """
    def __init__(self, sess, obs_dim, act_dim, clipping=0.2):
        """
        Args:
            obs_dim: num observation dimensions (int)
            act_dim: num action dimensions (int)
        """
        epochs = 20
        obs_ph = tf.placeholder(tf.float32, (None, obs_dim), 'obs')
        act_ph = tf.placeholder(tf.float32, (None, act_dim), 'act')
        advantages_ph = tf.placeholder(tf.float32, (None,), 'advantages')
        old_log_vars_ph = tf.placeholder(tf.float32, (act_dim,), 'old_log_vars')
        old_means_ph = tf.placeholder(tf.float32, (None, act_dim), 'old_means')

        with tf.compat.v1.variable_scope("policy_nn"):

            ### Set hid_size freely,
            hid1_size = None ### YOUR IMPLETETAION PART ###
            hid3_size = None ### YOUR IMPLETETAION PART ###
            hid2_size = None ### YOUR IMPLETETAION PART ###
            # 3 hidden layers with tanh activations
            h1 = None ### YOUR IMPLETETAION PART ###
            h2 = None ### YOUR IMPLETETAION PART ###
            h3 = None ### YOUR IMPLETETAION PART ###
            means = None ### YOUR IMPLETETAION PART ###
            log_vars = tf.get_variable('logvars', (act_dim), tf.float32,
                                       tf.constant_initializer(-1.0))


        def sample_action(obs):
            sampled_act = means + tf.exp(log_vars / 2.0) * tf.random_normal(shape=(act_dim,))
            feed_dict = {obs_ph: obs}
            return sess.run(sampled_act, feed_dict=feed_dict)


        self.sample = sample_action


def init_gym(env_name, animate=False):

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    if animate:
        env.render()

    return env, obs_dim, act_dim


if __name__ == "__main__":

    env_name = 'InvertedPendulumPyBulletEnv-v0'

    ### FIXME STEP 2 ###
    ### FIXME Run one episode with random policy, Store observations, actions, rewards in the list
    env, obs_dim, act_dim = init_gym(env_name, animate=True)
    sess = tf.Session()
    policy = Policy(sess, obs_dim, act_dim)
    sess.run(tf.compat.v1.initializers.global_variables())

    obs = env.reset()
    observes, actions, rewards= [], [], []
    done = False
    while not done:
        ### YOUR IMPLETETAION PART ###

    accumulated_reward = sum(rewards)
    print("="*50)
    print("During one episodes, The accumulated Rewards: {}".format(accumulated_reward))
    print("="*50)
    del env

    ### FIXME STEP3
    ### FIXME Run 10 episodes, each trajectory is sotred in a dictonary as the form {'observe':, 'actions':, 'rewards' :}
    ### FIXME Store 10 trajectories  in list []
    env, _, _ = init_gym(env_name, animate=False)
    episodes = 10
    trajectories = []
    for e in range(episodes):
        obs = env.reset()
        done = False
        observes, actions, rewards = [], [], []
        while not done:
            # For one episode
            ### YOUR IMPLETETAION PART ###


            trajectory = {'observes': None, ### YOUR IMPLETETAION PART ###
                          'actions': None, ### YOUR IMPLETETAION PART ###
                          'rewards': None, ### YOUR IMPLETETAION PART ###
                          }


        trajectories.append(trajectory)


    avg_returns = None ### YOUR IMPLETETAION PART ###
    print("=" * 50)
    print("During 10 episodes, The average of accumulated Rewards: {}".format(avg_returns))
    print("=" * 50)