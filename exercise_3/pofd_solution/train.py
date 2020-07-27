import time
import os

import gym
import numpy as np
import tensorflow as tf

from value_function import NNValueFunction
import scipy.signal
from utils import Logger, Scaler
from matplotlib import pyplot as plt, animation
import pybulletgym
import exercise_3.envs

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class DiscriminatorNetwork(tf.keras.layers.Layer):
    def __init__(self):
        super(DiscriminatorNetwork, self).__init__()

        self._l1 = tf.keras.layers.Dense(32, activation='tanh')
        self._l2 = tf.keras.layers.Dense(32, activation='tanh')
        self._out = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, **kwargs):
        inputs = tf.concat(inputs, axis=1)
        h = self._l1(inputs)
        h = self._l2(h)
        return self._out(h)  # batch_size * 1


class Discriminator:
    def __init__(self, env_name, obs_dim, act_dim, learning_rate=1e-2):
        # Load demo data
        traj_data = np.load('../demo/{}.npz'.format(env_name), allow_pickle=True)
        self._demo_obs, self._demo_act = traj_data['obs'], traj_data['act']

        # define placeholders
        self._agent_obs_ph = tf.keras.layers.Input(obs_dim, name='agent_obs', dtype=tf.float32)
        self._agent_act_ph = tf.keras.layers.Input(act_dim, name='agent_act', dtype=tf.float32)
        self._demo_obs_ph = tf.keras.layers.Input(obs_dim, name='demo_obs', dtype=tf.float32)
        self._demo_act_ph = tf.keras.layers.Input(act_dim, name='demo_act', dtype=tf.float32)

        # create network
        self._discriminator_network = DiscriminatorNetwork()
        agent_logits = self._discriminator_network([self._agent_obs_ph, self._agent_act_ph])
        demo_logits = self._discriminator_network([self._demo_obs_ph, self._demo_act_ph])

        #############################################
        # (1) 빈칸 채우기: agent_loss와 demo_loss를 작성 #
        #############################################
        agent_loss = -tf.reduce_mean(tf.log(agent_logits + 1e-8))
        demo_loss = -tf.reduce_mean(tf.log(1. - demo_logits + 1e-8))
        self._total_loss = agent_loss + demo_loss
        #############################################

        #####################################################################
        # (2) 빈칸 채우기: Demo 활용 reward 값 생성 operator 작성                  #
        #                                                                   #
        # self._reward_op                                                   #
        #   : agent의 (s, a)에 대해 reward로 활용할 discriminator output을        #
        #    계산하기 위한 operator. 현재 코드에서는 -log(D(s, a)) 값을 생성해야 함.     #
        #                                                                   #
        # self._reward_op는 discriminator의 get_reward(...) 함수에서 사용.       #
        # - 해당 함수 참고 요망                                                  #
        #####################################################################
        self._reward_op = -tf.log(agent_logits + 1e-8)
        #####################################################################

        self._train_op = tf.train.AdamOptimizer(learning_rate).minimize(self._total_loss)

        self._sess = tf.keras.backend.get_session()
        self._sess.run(tf.global_variables_initializer())

    def train(self, agent_obs, agent_act, scaler, batch_size=256, epoch=30, logger=None):
        scale, offset = scaler.get()
        scale[-1] = 1.0  # don't scale time step feature
        offset[-1] = 0.0  # don't offset time step feature

        loss = None
        for _ in range(epoch):
            agent_batch_idx = np.random.randint(len(agent_obs), size=batch_size)
            if len(self._demo_obs) > batch_size:
                demo_batch_idx = np.random.randint(len(self._demo_obs), size=batch_size)
            else:
                demo_batch_idx = np.arange(0, len(self._demo_obs))

            feed_dict = {
                self._agent_obs_ph: agent_obs[agent_batch_idx],
                self._agent_act_ph: agent_act[agent_batch_idx],
                self._demo_obs_ph: (self._demo_obs[demo_batch_idx] - offset) * scale,
                self._demo_act_ph: self._demo_act[demo_batch_idx],
            }

            _, loss = self._sess.run([self._train_op, self._total_loss], feed_dict)

        if logger:
            logger.log({
                'DiscriminatorLoss': loss,
            })

    def get_rewards(self, agent_obs, agent_act):
        """
        주어진 agent의 (s, a) 데이터 각각에 대해 discriminator 기반 reward를 리턴.

        - run_episode() 함수 내에서 학습에 활용할 reward 계산 시, 본 함수를 호출하여 사용.
        - 학습에 활용하는 reward = 환경에서 주어지는 reward + reward_lambda * (-log(D(s, a))) 임.
        """
        feed_dict = {
            self._agent_obs_ph: agent_obs,
            self._agent_act_ph: agent_act,
        }

        reward = self._sess.run(self._reward_op, feed_dict)
        reward = reward.reshape((-1,))

        return reward


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
        self.sampled_act = self.means + tf.exp(self.log_vars / 2.0) * tf.random_normal(shape=(act_dim,))

        logp = -0.5 * tf.reduce_sum(self.log_vars)
        logp += -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.means) /
                                     tf.exp(self.log_vars), axis=1)
        logp_old = -0.5 * tf.reduce_sum(self.old_log_vars_ph)
        logp_old += -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.old_means_ph) /
                                         tf.exp(self.old_log_vars_ph), axis=1)


        pg_ratio = tf.exp(logp - logp_old)
        clipped_pg_ratio = tf.clip_by_value(pg_ratio, 1 - clipping, 1 + clipping)
        surrogate_loss = tf.minimum(self.advantages_ph * pg_ratio,
                                    self.advantages_ph * clipped_pg_ratio)
        self.loss = -tf.reduce_mean(surrogate_loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        self.train_op = optimizer.minimize(self.loss)

    def sample(self, obs):
        feed_dict = {self.obs_ph: obs}
        return self.sess.run(self.sampled_act, feed_dict=feed_dict)

    def update(self, observes, actions, advantages, logger):
        feed_dict = {self.obs_ph: observes, self.act_ph: actions, self.advantages_ph: [advantages]}
        old_means_np, old_log_vars_np = self.sess.run([self.means, self.log_vars], feed_dict)
        feed_dict[self.old_log_vars_ph] = [old_log_vars_np]
        feed_dict[self.old_means_ph] = old_means_np
        for e in range(self.epochs):
            self.sess.run(self.train_op, feed_dict)
            policy_loss = self.sess.run(self.loss, feed_dict)

        logger.log({'PolicyLoss': policy_loss})


def run_episode(env, policy, scaler, animate=False, sleep=None, discriminator=None, reward_lambda=1e-1):
    obs = env.reset()
    observes, actions, env_rewards, unscaled_obs = [], [], [], []
    done = False
    step = 0.0
    scale, offset = scaler.get()
    scale[-1] = 1.0  # don't scale time step feature
    offset[-1] = 0.0  # don't offset time step feature
    while not done:
        if animate:
            env.render()
            if sleep:
                time.sleep(sleep)

        obs = obs.astype(np.float32).reshape((1, -1))
        obs = np.append(obs, [[step]], axis=1)  # add time step feature
        unscaled_obs.append(obs)
        obs = (obs - offset) * scale  # center and scale observations
        observes.append(obs)
        action = policy.sample(obs).reshape((1, -1)).astype(np.float32)
        actions.append(action)
        obs, env_reward, done, _ = env.step(np.squeeze(action, axis=0))
        if not isinstance(env_reward, float):
            env_reward = np.asscalar(env_reward)
        env_rewards.append(env_reward)
        step += 1e-3  # increment time step feature

    observes, actions, env_rewards, unscaled_obs = np.concatenate(observes), np.concatenate(actions), np.array(
        env_rewards, dtype=np.float64), np.concatenate(unscaled_obs)

    if discriminator:
        ##################################################################
        discriminator_rewards = discriminator.get_rewards(observes, actions)
        total_rewards = env_rewards + reward_lambda * discriminator_rewards
        ##################################################################
    else:
        total_rewards = env_rewards

    return (observes, actions, total_rewards, unscaled_obs, env_rewards)


def run_policy(env, policy, scaler, logger, episodes, discriminator=None, reward_lambda=1e-1):
    total_steps = 0
    trajectories = []
    for e in range(episodes):
        observes, actions, rewards, unscaled_obs, env_rewards = run_episode(env, policy, scaler,
                                                                            discriminator=discriminator,
                                                                            reward_lambda=reward_lambda)
        total_steps += observes.shape[0]
        trajectory = {'observes': observes,
                      'actions': actions,
                      'rewards': rewards,
                      'unscaled_obs': unscaled_obs,
                      'env_rewards': env_rewards}
        trajectories.append(trajectory)
    unscaled = np.concatenate([t['unscaled_obs'] for t in trajectories])
    scaler.update(unscaled)  # update running statistics for scaling observations
    logger.log({'_MeanReward': np.mean([t['rewards'].sum() for t in trajectories]),
                'EnvReward': np.mean([t['env_rewards'].sum() for t in trajectories]),
                'DemoReward': np.mean([t['rewards'].sum() for t in trajectories]) - np.mean(
                    [t['env_rewards'].sum() for t in trajectories]),
                'TotalReward:': np.mean([t['rewards'].sum() for t in trajectories]),
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
    ######################################################
    # 도메인 별 학습 환경 설정                                #
    ######################################################
    # [InvertedPendulumPyBulletSparseEnv-v0]             #
    # - 최대 reward: 500                                  #
    # - 설정 >> reward_lambda = 0.1, num_episodes = 1000  #
    #                                                    #
    # [InvertedDoublePendulumPyBulletSparseEnv-v0]       #
    # - 최대 reward: 1000                                 #
    # - 설정 >> reward_lambda = 0.1, num_episodes = 5000  #
    ######################################################

    env_name = 'InvertedPendulumPyBulletSparseEnv-v0'
    # env_name = 'InvertedDoublePendulumPyBulletSparseEnv-v0'

    # POfD 파라미터
    use_demo = True # True: POfD, False: PPO
    reward_lambda = 0.1

    # PPO 파라미터
    num_episodes = 5000
    gamma = 0.995
    lam = 0.98
    batch_size = 20

    # 학습 환경 생성
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    obs_dim += 1  # add 1 to obs dimension for time step feature (see run_episode())
    act_dim = env.action_space.shape[0]

    # visualize 환경 생성
    viz_env = gym.make(env_name)
    viz_env.render()

    # policy 및 value-funciton 네트워크 생성
    policy = Policy(obs_dim, act_dim)
    val_func = NNValueFunction(obs_dim)

    # POfD를 위한 discriminator 생성
    if use_demo:
        discriminator = Discriminator(env_name, obs_dim, act_dim)
    else:
        discriminator = None

    # logger 및 observation 정규화 객체 생성
    logger = Logger()
    scaler = Scaler(obs_dim)

    # PPO 학습 시작
    run_policy(env, policy, scaler, logger, episodes=5)
    episode = 0
    while episode < num_episodes:
        # 행동궤적 생성
        trajectories = run_policy(env, policy, scaler, logger, episodes=batch_size, discriminator=discriminator,
                                  reward_lambda=reward_lambda)
        episode += len(trajectories)
        observes, actions, advantages, disc_sum_rew = build_train_set(trajectories, val_func, gamma, lam)

        # policy 및 value-function 학습
        policy.update(observes, actions, advantages, logger)
        val_func.fit(observes, disc_sum_rew, logger)

        # POfD의 경우 discriminator 학습
        if discriminator:
            discriminator.train(observes, actions, scaler, logger=logger)

        logger.log({
            '_Episode': episode,
        })
        logger.write(display=True)

        # visualize current policy
        for _ in range(1):
            run_episode(viz_env, policy, scaler, animate=True, sleep=0.01)
