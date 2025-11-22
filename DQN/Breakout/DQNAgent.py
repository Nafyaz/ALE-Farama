import numpy as np
import torch
from torch import optim

from DQN import DQN
from ReplayBuffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        stack_size: int,
        action_dim: int,
        replay_buffer: ReplayBuffer,
        learning_rate: float,
        gamma: float,
        epsilon: float,
        epsilon_min: float,
        epsilon_decay: float,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.action_dim = action_dim
        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.model = DQN(stack_size, action_dim).to(self.device)
        self.target_model = DQN(stack_size, action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def select_action(self, state: np.ndarray, eval_mode: bool):
        if not eval_mode and torch.rand(1).item() < self.epsilon:
            return torch.randint(0, self.action_dim, (1,)).item()

        with torch.no_grad():
            state = torch.tensor(
                state,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            return self.model(state).argmax().item()

    def update(self, batch_size: int):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            batch_size,
        )
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

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

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
