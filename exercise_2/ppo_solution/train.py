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

    def __init__(self, obs_dim, act_dim, clipping=0.2):
        """
        Args:
            obs_dim: num observation dimensions (int)
            act_dim: num action dimensions (int)
        """
        self.sess = tf.keras.backend.get_session()
        self.epochs = 20

        self.obs_ph = tf.keras.layers.Input(obs_dim, name='obs')
        self.act_ph = tf.keras.layers.Input(act_dim, name='act')
        self.advantages_ph = tf.keras.layers.Input((None,), name='advantages')
        self.old_log_vars_ph = tf.keras.layers.Input(act_dim, name='old_log_vars')
        self.old_means_ph = tf.keras.layers.Input(act_dim, name='old_means')

        ### Set hid_size freely,
        hid1_size = obs_dim * 5
        hid3_size = act_dim * 10
        hid2_size = int(np.sqrt(hid1_size * hid3_size))

        # 3 hidden layers with tanh activations
        h1 = tf.keras.layers.Dense(hid1_size, activation="tanh", name="h1")(self.obs_ph)
        h2 = tf.keras.layers.Dense(hid2_size, activation="tanh", name="h2")(h1)
        h3 = tf.keras.layers.Dense(hid3_size, activation="tanh", name="h3")(h2)
        self.means = tf.keras.layers.Dense(act_dim, name="means", activation="linear")(h3)
        self.log_vars = tf.get_variable('logvars', (act_dim), tf.float32,
                                   tf.constant_initializer(-1.0))

        logp = -0.5 * tf.reduce_sum(self.log_vars)
        logp += -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.means) /
                                     tf.exp(self.log_vars), axis=1)
        logp_old = -0.5 * tf.reduce_sum(self.old_log_vars_ph)
        logp_old += -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.old_means_ph) /
                                         tf.exp(self.old_log_vars_ph), axis=1)

        """
        STEP 4
        The Clipped Surrogate Objective Function
        """
        ########################
        # YOUR IMPLEMENTATION PART
        pg_ratio = tf.exp(logp - logp_old)
        clipped_pg_ratio = tf.clip_by_value(pg_ratio, 1 - clipping, 1 + clipping)
        surrogate_loss = tf.minimum(self.advantages_ph * pg_ratio,
                                    self.advantages_ph * clipped_pg_ratio)
        ########################
        self.loss = -tf.reduce_mean(surrogate_loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        self.train_op = optimizer.minimize(self.loss)

    def sample(self, obs):
        sampled_act = self.means + tf.exp(self.log_vars / 2.0) * tf.random_normal(shape=(act_dim,))
        feed_dict = {self.obs_ph: obs}
        return self.sess.run(sampled_act, feed_dict=feed_dict)

    def update(self, observes, actions, advantages, logger):
        feed_dict = {self.obs_ph: observes, self.act_ph: actions, self.advantages_ph: [advantages]}
        old_means_np, old_log_vars_np = self.sess.run([self.means, self.log_vars], feed_dict)
        feed_dict[self.old_log_vars_ph] = [old_log_vars_np]
        feed_dict[self.old_means_ph] = old_means_np
        for e in range(self.epochs):
            self.sess.run(self.train_op, feed_dict)
            policy_loss = self.sess.run(self.loss, feed_dict)

        logger.log({'PolicyLoss': policy_loss})



def run_episode(env, policy, scaler):

    obs = env.reset()
    observes, actions, rewards, unscaled_obs = [], [], [], []
    done = False
    step = 0.0
    scale, offset = scaler.get()
    scale[-1] = 1.0  # don't scale time step feature
    offset[-1] = 0.0  # don't offset time step feature
    while not done:

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

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    obs_dim += 1 # add 1 to obs dimension for time step feature (see run_episode())
    act_dim = env.action_space.shape[0]
    # sess = tf.Session()
    policy = Policy(obs_dim, act_dim)
    val_func = NNValueFunction(obs_dim)
    # sess.run(tf.compat.v1.initializers.global_variables())

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
