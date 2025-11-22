import ale_py
import gymnasium as gym
import mlflow
import numpy as np

from DQNAgent import DQNAgent
from ReplayBuffer import ReplayBuffer
from util import obs_to_state, record_video


def train():
    # TODO: Why do we need to register?
    gym.register_envs(ale_py)

    env_name = "ALE/Breakout-v5"
    stack_size = 4
    num_steps = 500
    batch_size = 64
    replay_buffer_size = 1000
    target_update_freq = 10
    learning_rate = 1e-3
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    video_freq = 50
    experiment_name = "DQN-Breakout"

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "env_name": env_name,
                "num_steps": num_steps,
                "batch_size": batch_size,
                "replay_buffer_size": replay_buffer_size,
                "target_update_freq": target_update_freq,
                "learning_rate": learning_rate,
                "gamma": gamma,
                "epsilon": epsilon,
                "epsilon_min": epsilon_min,
                "epsilon_decay": epsilon_decay,
            },
        )

        env = gym.make(env_name)
        action_dim = env.action_space.n

        replay_buffer = ReplayBuffer(replay_buffer_size)
        agent = DQNAgent(
            stack_size,
            action_dim,
            replay_buffer,
            learning_rate,
            gamma,
            epsilon,
            epsilon_min,
            epsilon_decay,
        )

        episode_rewards = []
        best_reward = -float("inf")

        print(f"Starting training on {env_name}...")
        print(f"Observation dimension: {env.observation_space.shape}")
        print(f"Action dimension: {action_dim}")

        obs, _ = env.reset()
        state = obs_to_state(np.zeros((stack_size * 3, 210, 160)), obs)

        episode_reward = 0
        for step in range(num_steps):
            action = agent.select_action(state, eval_mode=False)
            print(action)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = obs_to_state(state, next_obs)

            replay_buffer.push(state, action, reward, terminated, next_state)

            state = next_state
            episode_reward += reward

            if terminated or truncated:
                episode_rewards.append(episode_reward)
                best_reward = max(best_reward, episode_reward)
                mlflow.log_metrics(
                    {
                        "episode_reward": episode_reward,
                        "best_reward": best_reward,
                        "epsilon": agent.epsilon,
                    },
                )
                obs, _ = env.reset()
                state = obs_to_state(np.zeros((stack_size * 3, 210, 160)), obs)
                episode_reward = 0

            if len(replay_buffer) >= batch_size:
                loss = agent.update(batch_size)
                mlflow.log_metrics({"loss": loss})

            if step % target_update_freq == 0:
                agent.update_target()

            agent.decay_epsilon()

            if step % video_freq == 0:
                record_video(agent, env_name, stack_size, step, 1)

        env.close()
