from collections import deque

import ale_py
import gymnasium as gym
import mlflow
from gymnasium.wrappers import (
    FrameStackObservation,
    GrayscaleObservation,
    ResizeObservation,
)

from DQNAgent import DQNAgent
from FireWrapper import FireWrapper
from ReplayBuffer import ReplayBuffer


def train():
    # TODO: Why do we need to register?
    gym.register_envs(ale_py)

    env_name = "ALE/Breakout-v5"
    experiment_name = "DQN-Breakout"

    num_steps = 10_000_000
    stack_size = 4
    frame_size = (84, 84)
    batch_size = 128
    replay_buffer_size = 100_000
    learning_starts = 50_000
    learning_rate = 1e-4
    gamma = 0.99

    train_frequency = 4
    target_update_freq = 1_000
    checkpoint_freq = 100_000
    video_length = 2000

    episode_length_buffer_size = 100

    mlflow.set_tracking_uri("http://localhost:5000/")
    mlflow.set_experiment(experiment_name)

    # mlflow.config.enable_system_metrics_logging()
    # mlflow.config.set_system_metrics_sampling_interval(1)

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
            },
        )

        env = gym.make(env_name, render_mode="rgb_array")
        env = FireWrapper(env)
        env = gym.wrappers.RecordVideo(
            env,
            f"videos/{env_name}-training",
            step_trigger=lambda x: (x + 1) % checkpoint_freq == 0,
            video_length=video_length,
            name_prefix="training",
        )
        env = ResizeObservation(env, frame_size)
        env = GrayscaleObservation(env)
        env = FrameStackObservation(env, stack_size)

        action_dim = env.action_space.n
        state_shape = (stack_size, *frame_size)

        replay_buffer = ReplayBuffer(replay_buffer_size)
        agent = DQNAgent(
            state_shape,
            action_dim,
            replay_buffer,
            learning_rate,
            gamma,
        )

        episode_legths = deque(maxlen=episode_length_buffer_size)
        episode_rewards = deque(maxlen=episode_length_buffer_size)

        state, _ = env.reset()
        episode_length = 0
        episode_reward = 0

        for step in range(num_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_length += 1
            episode_reward += reward

            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state

            if done:
                episode_legths.append(episode_length)
                episode_length = 0

                episode_rewards.append(episode_reward)
                episode_reward = 0

                mlflow.log_metrics(
                    {
                        "trailing_average_episode_length": sum(episode_legths)
                        / len(episode_legths),
                        "trailing_average_episode_rewards": sum(episode_rewards)
                        / len(episode_rewards),
                    },
                    step=step,
                )
                state, _ = env.reset()

            if (
                len(replay_buffer) >= learning_starts
                and (step + 1) % train_frequency == 0
            ):
                loss = agent.update(batch_size)

            if (step + 1) % target_update_freq == 0:
                agent.update_target()

            if (step + 1) % checkpoint_freq == 0:
                mlflow.pytorch.log_model(agent.model, name=f"checkpoint_{step + 1}")

        mlflow.pytorch.log_model(agent.model, name=f"checkpoint_{step + 1}")

        env.close()
