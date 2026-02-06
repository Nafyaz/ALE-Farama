import random
from collections import deque

import torch


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def push(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        action = torch.tensor(action, dtype=torch.int64, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        next_state = torch.tensor(
            next_state,
            dtype=torch.float32,
            device=self.device,
        )
        done = torch.tensor(done, dtype=torch.float32, device=self.device)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch, strict=True)
        return (
            torch.stack(state),
            torch.stack(action),
            torch.stack(reward),
            torch.stack(next_state),
            torch.stack(done),
        )

    def __len__(self) -> int:
        return len(self.buffer)
