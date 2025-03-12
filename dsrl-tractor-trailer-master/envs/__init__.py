from gymnasium.envs.registration import register

register(
    id='TractorTrailer-v0',
    entry_point='envs.tractor_trailer:TractorTrailer',
)