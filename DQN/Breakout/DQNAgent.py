import mlflow
import numpy as np
import torch
from torch import optim

from DQN import DQN
from ReplayBuffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        state_shape: tuple,
        action_dim: int,
        replay_buffer: ReplayBuffer,
        learning_rate: float,
        gamma: float,
        model_location: str = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.action_dim = action_dim
        self.replay_buffer = replay_buffer
        self.gamma = gamma

        if model_location is not None:
            self.model = mlflow.pytorch.load_model(model_location).to(self.device)
        else:
            self.model = DQN(state_shape, action_dim).to(self.device)

        self.target_model = DQN(state_shape, action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def select_action(self, state: np.ndarray) -> int:
        with torch.no_grad():
            state = torch.as_tensor(
                state,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            dist = torch.distributions.Categorical(logits=self.model(state))

            return dist.sample().item()

    def update(self, batch_size: int) -> int:
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            batch_size,
        )

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = torch.nn.functional.mse_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
