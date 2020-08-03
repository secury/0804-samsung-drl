from gym.envs.registration import register

register(
    id='InvertedPendulumPyBulletSparseEnv-v0',
    entry_point='envs.inverted_pendulum_sparse_env:InvertedPendulumBulletSparseEnv',
    max_episode_steps=1000,
    reward_threshold=950.0,
)

register(
    id='InvertedDoublePendulumPyBulletSparseEnv-v0',
    entry_point='envs.inverted_double_pendulum_sparse_env:InvertedDoublePendulumBulletSparseEnv',
    max_episode_steps=1000,
    reward_threshold=9100.0,
)
