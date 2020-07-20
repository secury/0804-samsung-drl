import gym
import numpy as np
import tensorflow as tf

from value_function import NNValueFunction
import scipy.signal
from utils import Logger, Scaler
from datetime import datetime
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

        logp = -0.5 * tf.reduce_sum(log_vars)
        logp += -0.5 * tf.reduce_sum(tf.square(act_ph - means) /
                                     tf.exp(log_vars), axis=1)
        logp_old = -0.5 * tf.reduce_sum(old_log_vars_ph)
        logp_old += -0.5 * tf.reduce_sum(tf.square(act_ph - old_means_ph) /
                                         tf.exp(old_log_vars_ph), axis=1)

        pg_ratio = None ### YOUR IMPLETETAION PART ###
        clipped_pg_ratio = None ### YOUR IMPLETETAION PART ###
        surrogate_loss = None ### YOUR IMPLETETAION PART ###
        loss = -tf.reduce_mean(surrogate_loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(loss)

        def sample_action(obs):
            sampled_act = means + tf.exp(log_vars / 2.0) * tf.random_normal(shape=(act_dim,))
            feed_dict = {obs_ph: obs}
            return sess.run(sampled_act, feed_dict=feed_dict)

        def update(observes, actions, advantages, logger):
            feed_dict = {obs_ph: observes, act_ph: actions, advantages_ph: advantages}
            old_means_np, old_log_vars_np = sess.run([means, log_vars], feed_dict)
            feed_dict[old_log_vars_ph] = old_log_vars_np
            feed_dict[old_means_ph] = old_means_np
            for e in range(epochs):
                sess.run(train_op, feed_dict)
                policy_loss = sess.run(loss, feed_dict)

            logger.log({'PolicyLoss': policy_loss})

        self.sample = sample_action
        self.update = update

def run_episode(env, policy, scaler, animate=True):

    obs = env.reset()
    observes, actions, rewards, unscaled_obs = [], [], [], []
    done = False
    step = 0.0
    scale, offset = scaler.get()
    scale[-1] = 1.0  # don't scale time step feature
    offset[-1] = 0.0  # don't offset time step feature
    while not done:
        if animate:
            env.render()
        obs = obs.astype(np.float32).reshape((1, -1))
        obs = np.append(obs, [[step]], axis=1)  # add time step feature
        unscaled_obs.append(obs)
        obs = (obs - offset) * scale  # center and scale observations
        observes.append(obs)
        action = policy.sample(obs).reshape((1, -1)).astype(np.float32)
        actions.append(action)
        obs, reward, done, _ = env.step(np.squeeze(action, axis=0))
        if not isinstance(reward, float):
            reward = np.asscalar(reward)
        rewards.append(reward)
        step += 1e-3  # increment time step feature

    return (np.concatenate(observes), np.concatenate(actions),
            np.array(rewards, dtype=np.float64), np.concatenate(unscaled_obs))


def run_policy(env, policy, scaler, logger, episodes):

    total_steps = 0
    trajectories = []
    for e in range(episodes):
        observes, actions, rewards, unscaled_obs = run_episode(env, policy, scaler)
        total_steps += observes.shape[0]
        trajectory = {'observes': observes,
                      'actions': actions,
                      'rewards': rewards,
                      'unscaled_obs': unscaled_obs}
        trajectories.append(trajectory)
    unscaled = np.concatenate([t['unscaled_obs'] for t in trajectories])
    scaler.update(unscaled)  # update running statistics for scaling observations
    logger.log({'_MeanReward': np.mean([t['rewards'].sum() for t in trajectories]),
                'Steps': total_steps})

    return trajectories


def discount(x, gamma):
    """ Calculate discounted forward sum of a sequence at each point """
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]

def build_train_set(trajectories, val_func, gamma, lam):

    for trajectory in trajectories:
        observes = trajectory['observes']
        values = val_func.predict(observes)

        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        disc_sum_rew = discount(rewards, gamma)
        trajectory['disc_sum_rew'] = disc_sum_rew

        tds = rewards - values + np.append(values[1:] * gamma, 0)
        advantages = discount(tds, gamma * lam)
        trajectory['advantages'] = advantages

    observes = np.concatenate([t['observes'] for t in trajectories])
    actions = np.concatenate([t['actions'] for t in trajectories])
    disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in trajectories])
    advantages = np.concatenate([t['advantages'] for t in trajectories])
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

    return observes, actions, advantages, disc_sum_rew


if __name__ == "__main__":

    env_name = 'InvertedPendulumPyBulletEnv-v0'
    num_episodes = 1000
    gamma = 0.995
    lam = 0.98
    batch_size = 5
    hid1_mult = 10

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    obs_dim += 1 # add 1 to obs dimension for time step feature (see run_episode())
    act_dim = env.action_space.shape[0]
    sess = tf.Session()
    policy = Policy(sess, obs_dim, act_dim)
    val_func = NNValueFunction(sess, obs_dim, hid1_mult)
    sess.run(tf.compat.v1.initializers.global_variables())

    now = datetime.utcnow().strftime("%b-%d_%H:%M:%S")  # create unique directories
    logger = Logger(logname=env_name, now=now)

    scaler = Scaler(obs_dim)

    run_policy(env, policy, scaler, logger, episodes=5)
    episode = 0
    while episode < num_episodes:
        trajectories = run_policy(env, policy, scaler, logger, episodes=batch_size)
        episode += len(trajectories)
        observes, actions, advantages, disc_sum_rew = build_train_set(trajectories, val_func, gamma, lam)
        policy.update(observes, actions, advantages, logger)
        val_func.fit(observes, disc_sum_rew, logger)
        logger.log({
            '_Episode': episode,
        })
        logger.write(display=True)
