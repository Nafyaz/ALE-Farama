import gymnasium as gym
import numpy as np

from DQNAgent import DQNAgent


def make_train_env(env_name: str) -> gym.Env:
    return gym.make(env_name)


def make_eval_env(env_name: str, step: int) -> gym.Env:
    env = gym.make(env_name, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        f"videos/step_{step}",
        episode_trigger=lambda _: True,
        name_prefix=f"step-{step}",
    )
    return env


def obs_to_state(prev_state: np.ndarray, obs: np.ndarray) -> np.ndarray:
    obs = np.transpose(obs, (2, 0, 1))
    return np.concatenate((prev_state, np.expand_dims(obs, 1)), axis=1)[:, 1:, :, :]


def record_video(
    agent: DQNAgent,
    env_name: str,
    state_shape: tuple,
    step: int,
    num_episodes: int,
):
    video_env = make_eval_env(env_name, step)

    for _ in range(num_episodes):
        obs, _ = video_env.reset()
        state = obs_to_state(np.zeros(state_shape), obs)
        done = False
        total_rewards = 0
        total_steps = 0
        while not done:
            action = agent.select_action(state, eval_mode=True)
            obs, reward, terminated, truncated, _ = video_env.step(action)
            total_rewards += reward
            total_steps += 1

            if total_steps > 1000:
                break

            done = terminated or truncated

    video_env.close()
    print(
        f"Video saved to videos/step_{step}.mp4 with total rewards {total_rewards} and total steps {total_steps}",
    )
