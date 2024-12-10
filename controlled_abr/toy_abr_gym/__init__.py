from gym.envs.registration import register
from toy_abr_gym import abr_wrapper, abr

register(
    id='toy-abr-gym-base-v0',
    entry_point='toy_abr_gym.abr:ToyABREnv')
register(
    id='toy-abr-gym-framestack-v0',
    entry_point='toy_abr_gym.abr_wrapper:ToyABRWrapper')