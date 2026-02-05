# import numpy as np
#
#
# def obs_to_state(prev_state: np.ndarray, obs: np.ndarray) -> np.ndarray:
#     obs = np.transpose(obs, (2, 0, 1))
#     return np.concatenate((prev_state, np.expand_dims(obs, 1)), axis=1)[:, 1:, :, :]
