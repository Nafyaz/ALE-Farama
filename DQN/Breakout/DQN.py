import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, stack_size, action_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(stack_size*3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 22 * 16, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, x):
        return self.network(x)