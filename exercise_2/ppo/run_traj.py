import numpy as np
from policy import Policy
from utils import Logger, Scaler
from datetime import datetime
import os
import argparse
import gym
import signal
import pybulletgym

def init_gym(env_name, animate=True):
    """
    Initialize gym environment, return dimension of observation
    and action spaces.

    Args:
        env_name: str environment name (e.g. "Humanoid-v1")

    Returns: 3-tuple
        gym environment (object)
        number of observation dimensions (int)
        number of action dimensions (int)
    """
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    if animate:
        env.render()

    return env, obs_dim, act_dim


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Train policy on OpenAI Gym environment '
                                                  'using Proximal Policy Optimizer'))
    parser.add_argument('--env_name', type=str, default='InvertedPendulumPyBulletEnv-v0', help='OpenAI Gym environment name')

    args = parser.parse_args()

    ### FIXME STEP 2 ###
    ### FIXME Run one episode with random policy, Store observations, actions, rewards in the list
    env, obs_dim, act_dim = init_gym(args.env_name, animate=True)
    policy = Policy(obs_dim, act_dim, kl_targ=0.03, hid1_mult=5, policy_logvar=1.0, clipping_range=[0.2]*2)
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
    env, _, _ = init_gym(args.env_name, animate=False)
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