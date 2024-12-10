from gymnasium import register
from abr_gym import abr_wrapper, abr, pensieve_wrapper, abr_no_stack

register(
    id='abr-gym-base-v0',
    entry_point='abr_gym.abr:ABRSimEnv')
register(
    id='abr-gym-framestack-v0',
    entry_point='abr_gym.abr_wrapper:ABRWrapper')
register(
    id='abr-gym-pensieve-v0',
    entry_point='abr_gym.pensieve_wrapper:PensieveWrapper')
register(
    id='abr-gym-no-framestack-v0',
    entry_point='abr_gym.abr_no_stack:ABRNoFramestack')