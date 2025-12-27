import torch
from torch import nn


class DQN(nn.Module):
    def __init__(self, input_shape: tuple, action_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(1, 8, 8), stride=4),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(1, 4, 4), stride=2),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy_input = torch.zeros((1, *input_shape))
            flatten_dim = self.features(dummy_input).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flatten_dim, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255
        x = self.features(x)
        return self.fc(x)
