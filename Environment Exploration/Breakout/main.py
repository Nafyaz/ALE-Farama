import gymnasium as gym
import ale_py

env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
obs, info = env.reset()

for step in range(10000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    # env.render()

    if terminated:
        print(f"terminated at {step}")
        break
    if truncated:
        print(f"truncated at {step}")
        break
