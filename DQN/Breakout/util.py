import gymnasium as gym
import numpy as np

def obs_to_state(prev_state, obs):
    obs = np.transpose(obs, (2, 0, 1))
    return np.concatenate((prev_state[3:,:,:], obs), axis=0)

def record_video(agent, env_name, stack_size, step, num_episodes):
    video_env = gym.make(env_name, render_mode="rgb_array")
    video_env = gym.wrappers.RecordVideo(
        video_env,
        f"videos/step_{step}",
        episode_trigger=lambda x: True,
        name_prefix=f"step-{step}"
    )

    for _ in range(num_episodes):
        obs, info = video_env.reset()
        state = obs_to_state(np.zeros((stack_size*3, 210, 160)), obs)
        done = False

        while not done:
            action = agent.select_action(state, eval_mode=True)
            obs, reward, terminated, truncated, info = video_env.step(action)
            done = terminated or truncated

    video_env.close()
    print(f"Video saved to videos/step_{step}.mp4")