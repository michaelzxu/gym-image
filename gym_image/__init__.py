from gym.envs.registration import register

register(
    id='ObjectCount-v0',
    entry_point='gym_image.envs:ObjectCountEnv',
)
