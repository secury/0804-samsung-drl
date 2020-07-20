import numpy as np
import argparse
import gym
import tensorflow as tf
import pybulletgym
import pdb

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
            hid1_size = obs_dim * 5
            hid3_size = act_dim * 10
            hid2_size = int(np.sqrt(hid1_size * hid3_size))

            # 3 hidden layers with tanh activations
            h1 = tf.keras.layers.Dense(hid1_size, activation="tanh", name="h1")(obs_ph)
            h2 = tf.keras.layers.Dense(hid2_size, activation="tanh", name="h2")(h1)
            h3 = tf.keras.layers.Dense(hid3_size, activation="tanh", name="h3")(h2)
            means = tf.keras.layers.Dense(act_dim, name="means", activation="linear")(h3)
            log_vars = tf.get_variable('logvars', (act_dim), tf.float32,
                                       tf.constant_initializer(-1.0))

        logp = -0.5 * tf.reduce_sum(log_vars)
        logp += -0.5 * tf.reduce_sum(tf.square(act_ph - means) /
                                         tf.exp(log_vars), axis=1)
        logp_old = -0.5 * tf.reduce_sum(old_log_vars_ph)
        logp_old += -0.5 * tf.reduce_sum(tf.square(act_ph - old_means_ph) /
                                             tf.exp(old_log_vars_ph), axis=1)

        pg_ratio = tf.exp(logp - logp_old)
        clipped_pg_ratio = tf.clip_by_value(pg_ratio, 1 - clipping, 1 + clipping)
        surrogate_loss = tf.minimum(advantages_ph * pg_ratio,
                                        advantages_ph * clipped_pg_ratio)
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
        obs = obs.astype(np.float32).reshape((1, -1))
        action = policy.sample(obs).reshape((1, -1)).astype(np.float32)
        actions.append(action)
        obs, reward, done, _ = env.step(action)
        rewards.append(reward)

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
            obs = obs.astype(np.float32).reshape((1, -1))
            observes.append(obs)
            action = policy.sample(obs).reshape((1, -1)).astype(np.float32)

            actions.append(action)
            obs, reward, done, _ = env.step(action)
            rewards.append(reward)

            trajectory = {'observes': np.concatenate(observes),
                          'actions': np.concatenate(actions),
                          'rewards': np.array(rewards, dtype=np.float64)
                          }


        trajectories.append(trajectory)

    returns = 0
    for trajectory in trajectories:
        returns += sum(trajectory['rewards'])

    avg_returns = returns/len(trajectories)
    print("=" * 50)
    print("During 10 episodes, The average of accumulated Rewards: {}".format(avg_returns))
    print("=" * 50)