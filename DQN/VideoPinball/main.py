import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

env = gym.make('ALE/VideoPinball-v5', render_mode="human")
obs, info = env.reset()

for _ in range(1000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    env.render()
env.close()