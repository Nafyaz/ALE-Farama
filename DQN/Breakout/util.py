import gymnasium as gym
import numpy as np

from DQNAgent import DQNAgent


def obs_to_state(prev_state: np.ndarray, obs: np.ndarray) -> np.ndarray:
    obs = np.transpose(obs, (2, 0, 1))
    return np.concatenate((prev_state[3:, :, :], obs), axis=0)


def record_video(
    agent: DQNAgent,
    env_name: str,
    stack_size: int,
    step: int,
    num_episodes: int,
):
    video_env = gym.make(env_name, render_mode="rgb_array")
    video_env = gym.wrappers.RecordVideo(
        video_env,
        f"videos/step_{step}",
        episode_trigger=lambda _: True,
        name_prefix=f"step-{step}",
    )

    for _ in range(num_episodes):
        obs, _ = video_env.reset()
        state = obs_to_state(np.zeros((stack_size * 3, 210, 160)), obs)
        done = False

        while not done:
            action = agent.select_action(state, eval_mode=True)
            obs, _, terminated, truncated, _ = video_env.step(action)
            done = terminated or truncated

    video_env.close()
    print(f"Video saved to videos/step_{step}.mp4")
