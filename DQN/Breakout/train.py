from collections import deque

import ale_py
import gymnasium as gym
import mlflow
from gymnasium.wrappers import (
    FrameStackObservation,
    GrayscaleObservation,
    ResizeObservation,
)
from tqdm import trange

from DQNAgent import DQNAgent
from FireWrapper import FireWrapper
from ReplayBuffer import ReplayBuffer


def train():
    # TODO: Why do we need to register?
    gym.register_envs(ale_py)

    env_name = "ALE/Breakout-v5"
    experiment_name = "DQN-Breakout"

    num_steps = 100000
    stack_size = 4
    frame_size = (84, 84)
    batch_size = 32
    replay_buffer_size = 10000
    learning_rate = 1e-3
    gamma = 0.99

    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.99995

    target_update_freq = 100
    checkpoint_freq = 10000
    rewards_buffer_size = 1000

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
                "epsilon": epsilon,
                "epsilon_min": epsilon_min,
                "epsilon_decay": epsilon_decay,
            },
        )

        env = gym.make(env_name, render_mode="rgb_array")
        env = FireWrapper(env)
        env = gym.wrappers.RecordVideo(
            env,
            f"videos/{env_name}-training",
            step_trigger=lambda x: x % checkpoint_freq == 0,
            video_length=1000,
            name_prefix="training",
        )
        env = ResizeObservation(env, frame_size)
        env = GrayscaleObservation(env)
        env = FrameStackObservation(env, stack_size)

        obs_dim = env.observation_space.shape
        action_dim = env.action_space.n
        state_shape = (stack_size, *frame_size)

        print(f"Starting training on {env_name}...")
        print(f"Observation dimension: {obs_dim}")
        print(f"Action dimension: {action_dim}")
        print(f"State shape: {state_shape}")

        replay_buffer = ReplayBuffer(replay_buffer_size)
        agent = DQNAgent(
            state_shape,
            action_dim,
            replay_buffer,
            learning_rate,
            gamma,
            epsilon,
            epsilon_min,
            epsilon_decay,
        )

        state, info = env.reset()
        rewards = deque(maxlen=rewards_buffer_size)

        for step in trange(num_steps):
            action = agent.select_action(state, eval_mode=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)

            replay_buffer.push(state, action, reward, terminated, next_state)

            state = next_state

            if terminated or truncated:
                mlflow.log_metrics(
                    {"episode_frame_number": info["episode_frame_number"]},
                    step=step,
                )
                state, info = env.reset()

            if len(replay_buffer) >= batch_size:
                loss = agent.update(batch_size)
                mlflow.log_metrics(
                    {
                        "loss": loss,
                    },
                    step=step,
                )

            mlflow.log_metrics(
                {
                    "trailing average reward": sum(rewards) / len(rewards),
                    "epsilon": agent.epsilon,
                },
                step=step,
            )

            if (step + 1) % target_update_freq == 0:
                agent.update_target()

            agent.decay_epsilon()

            if (step + 1) % checkpoint_freq == 0:
                mlflow.pytorch.log_model(agent.model, name=f"checkpoint_{step + 1}")

        mlflow.pytorch.log_model(agent.model, name=f"checkpoint_{step + 1}")

        env.close()
