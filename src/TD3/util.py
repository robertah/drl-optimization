import copy

def reverse_obs(obs):
    """Given observation from BipedalWalker-v2, flip to duplicate.
    obs[:4]   - body observations
    obs[4:9]  - leg 1
    obs[9:14] - leg 2
    """
    mirror_obs = copy.deepcopy(obs)
    tmp = copy.deepcopy(mirror_obs[4:9])
    mirror_obs[4:9] = mirror_obs[9:14]
    mirror_obs[9:14] = tmp
    return mirror_obs


def reverse_act(action):
    """Given action from BipedalWalker-v2, flip to duplicate.
    action[:2] - leg 1
    action[2:] - leg 2
    """
    mirror_act = copy.deepcopy(action)
    tmp = copy.deepcopy(mirror_act[:2])
    mirror_act[:2] = mirror_act[2:]
    mirror_act[2:] = tmp
    return mirror_act
