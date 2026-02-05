from gymnasium import Wrapper


class FireWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.lives = 0

    def step(self, action):
        obs, rewards, terminated, truncated, info = self.env.step(action)

        if info["lives"] != self.lives:
            obs, rewards, terminated, truncated, info = self.env.step(1)
            self.lives = info["lives"]

        return obs, rewards, terminated, truncated, info
