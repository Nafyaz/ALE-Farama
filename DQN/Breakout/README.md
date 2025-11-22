# Breakout with DQN

TODO: Find and fix memory leakage
TODO: Manage artifactory for mlflow and recorded videos
TODO: Do 3D convolution
TODO: Do Preprocessing

1. `train.py` is not concerned with pytorch, or how the model is implemented. It only cares about how to interact with
   environment.
2. `DQNAgent` is the bridge between environment and model. It is responsible for selecting actions, updating the model
   etc.
3. `DQN` is the model. It knows only torch.
4. `ReplayBuffer` is for storage of states.
5. obs/Observation: direct output of environment.
6. state: processed observation. 