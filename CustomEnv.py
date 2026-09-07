from typing import Optional
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import FlattenObservation
from Game import Game, Model
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

verbosity = 0
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
                "pile": spaces.MultiDiscrete([14] * 4),
                "pile_size": spaces.Discrete(53),
                "deck_size": spaces.Discrete(53),
            }
        )
        self.game = None
        self.steps = 0

    def _get_obs(self):
        top_cards = list(self.game.pile)[-4:]
        return {
            "pile": np.array(top_cards + [0] * (4 - len(top_cards))),
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
        info = dict(self.game.info)

        return observation, reward, terminated, truncated, info


class LogEveryNStepsCallback(BaseCallback):
    """Prints progress every n_steps environment timesteps."""

    def __init__(self, n_steps: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.n_steps = n_steps
        self._last_logged = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_logged >= self.n_steps:
            self._last_logged = self.num_timesteps
            print(f"timesteps: {self.num_timesteps}")
        return True


if __name__ == "__main__":
    env = Monitor(FlattenObservation(CustomEnv()))
    model = PPO(
        "MlpPolicy", env, verbose=1, tensorboard_log="./ppo_egyptianrl_tensorboard/"
    )
    model.learn(total_timesteps=25000, callback=LogEveryNStepsCallback(1000))
    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
