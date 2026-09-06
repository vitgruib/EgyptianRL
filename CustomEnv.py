from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from Game import Game, Deck, Model
from stable_baselines3.common.env_checker import check_env

verbosity = 3
seed = 42  # TODO make this actually do something


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    def __init__(self):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.MultiBinary(1)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Dict(
            {
                "pile": spaces.MultiDiscrete([14] * 52),
                "pile_size": spaces.Discrete(53),
                "deck_size": spaces.Discrete(53),
            }
        )
        self.game = None
        self.steps = 0

    def _get_obs(self):
        return {
            "pile": np.array(list(self.game.pile) + [0] * (52 - len(self.game.pile))),
            "pile_size": len(self.game.pile),
            "deck_size": len(self.game.decks[self.game.rl]),
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        self.game = Game([Model() for i in range(2)], verbosity=verbosity)
        self.steps = 0

        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: to preslap?

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # preslap, turning multibinary into just a binary
        self.game.move(action[0])
        self.steps += 1

        # Check if game terminated
        terminated = self.game.terminated
        truncated = self.steps > 1000
        # Simple reward structure: +1 for reaching target, 0 otherwise
        # Alternative: could give small negative rewards for each step to encourage efficiency
        reward = self.game.reward
        self.game.reward = 0

        observation = self._get_obs()
        info = {}

        return observation, reward, terminated, truncated, info


if __name__ == "__main__":
    env = CustomEnv()
    # It will check your custom environment and output additional warnings if needed
    check_env(env)
    observation = env.reset()
    episode_over = False
    total_reward = 0
    print("episode started")
    while not episode_over:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        episode_over = terminated or truncated
    print("episode end")
    env.close()
