from typing import Optional
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from Game import Game, Model
from stable_baselines3.common.callbacks import BaseCallback

from deck_configs import resolve_deck

verbosity = 0
seed = 42  # TODO make this actually do something


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    # Hard cap on moves per episode. Every loop that advances game state is
    # bounded by this so an episode can never run (or hang) unboundedly.
    MAX_STEPS = 1000
    # Long-horizon reward: a large bonus/penalty based on who actually ends
    # the episode with more cards, so the objective is "win the game," not
    # just "grab high-value piles." Scaled well above per-card reward values
    # (max 6 per card) so it dominates the episode's cumulative reward.
    TERMINAL_WIN_BONUS = 50.0

    def __init__(self, deck="full"):
        super().__init__()
        # deck controls the game's difficulty/size (see deck_configs.py) --
        # observation space is derived from it so a simplified deck also
        # means a smaller, easier-to-learn input.
        deck_config = resolve_deck(deck)
        self.deck_ranks = deck_config["ranks"]
        self.deck_copies = deck_config["copies"]
        self.max_rank = max(self.deck_ranks)
        self.total_cards = len(self.deck_ranks) * self.deck_copies

        self.action_space = spaces.MultiBinary(1)
        self.observation_space = spaces.Dict(
            {
                "pile": spaces.MultiDiscrete([self.max_rank + 1] * 4),
                "pile_size": spaces.Discrete(self.total_cards + 1),
                "deck_size": spaces.Discrete(self.total_cards + 1),
                # face_counter[0]: turns left on the face-card challenge, clipped to
                # [-1, 3] and shifted to [0, 4] (0 = no challenge in progress).
                # face_counter[1]: 0 = no owner, 1 = rl agent owns it, 2 = opponent does.
                "face_counter": spaces.MultiDiscrete([5, 3]),
            }
        )
        self.game = None
        self.steps = 0

    def _get_obs(self):
        top_cards = list(self.game.pile)[-4:]
        remaining, owner = self.game.face_counter
        remaining_enc = min(max(remaining, -1), 3) + 1
        if owner == -1:
            owner_enc = 0
        elif owner == self.game.rl:
            owner_enc = 1
        else:
            owner_enc = 2
        return {
            "pile": np.array(top_cards + [0] * (4 - len(top_cards))),
            "pile_size": len(self.game.pile),
            "deck_size": len(self.game.decks[self.game.rl]),
            "face_counter": np.array([remaining_enc, owner_enc]),
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

        self.game = Game(
            [Model() for i in range(2)],
            verbosity=verbosity,
            deck_ranks=self.deck_ranks,
            deck_copies=self.deck_copies,
        )
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
        info = dict(self.game.info)

        # Fast-forward through the opponent's turn(s): they don't take our
        # action, so they aren't a decision point for the policy and
        # shouldn't be surfaced as a separate env.step(). Bounded by
        # MAX_STEPS so this can never spin past the episode's own budget.
        while (
            not self.game.terminated
            and self.game.active_turn != self.game.rl
            and self.steps < self.MAX_STEPS
        ):
            self.game.move()
            self.steps += 1

        terminated = self.game.terminated
        truncated = self.steps >= self.MAX_STEPS
        reward = self.game.reward
        self.game.reward = 0

        if terminated or truncated:
            opponent = 1 - self.game.rl
            card_margin = len(self.game.decks[self.game.rl]) - len(self.game.decks[opponent])
            if card_margin > 0:
                reward += self.TERMINAL_WIN_BONUS
            elif card_margin < 0:
                reward -= self.TERMINAL_WIN_BONUS

        observation = self._get_obs()

        return observation, reward, terminated, truncated, info


class PreslapLoggingCallback(BaseCallback):
    """Logs the RL agent's own preslap decisions to TensorBoard.

    SB3's Monitor only auto-logs episode reward/length; it never forwards
    custom `info` keys. This reads `pile_top_type`/`preslap` off each step's
    info (tracked on every decision, not just preslapped ones) and records,
    once per rollout:
      - custom/preslap_rate: overall P(preslap), unconditional baseline.
      - custom/preslap_rate_given_card_type/{type}: P(preslap | that card
        type is on top of the pile) -- how often you actually slap it each
        time it appears.
      - custom/preslap_lift_given_card_type/{type}: the conditional rate
        divided by the overall rate, i.e. how much more/less than baseline
        you slap that type (1.0 = no different from baseline, >1 = slap it
        more than average, <1 = less).
    """

    CARD_TYPES = ["Ace", "Jack", "Queen", "King", "Number"]

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._reset_counts()

    def _reset_counts(self):
        self.step_count = 0
        self.preslap_count = 0
        self.type_seen = {t: 0 for t in self.CARD_TYPES}
        self.type_preslapped = {t: 0 for t in self.CARD_TYPES}

    def _on_rollout_start(self) -> None:
        self._reset_counts()

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            self.step_count += 1
            preslapped = info.get("preslap", False)
            if preslapped:
                self.preslap_count += 1
            top_type = info.get("pile_top_type")
            if top_type:
                self.type_seen[top_type] += 1
                if preslapped:
                    self.type_preslapped[top_type] += 1
        return True

    def _on_rollout_end(self) -> None:
        if not self.step_count:
            return
        overall_rate = self.preslap_count / self.step_count
        self.logger.record("custom/preslap_rate", overall_rate)
        for card_type in self.CARD_TYPES:
            seen = self.type_seen[card_type]
            if not seen:
                continue
            conditional_rate = self.type_preslapped[card_type] / seen
            self.logger.record(f"custom/preslap_rate_given_card_type/{card_type}", conditional_rate)
            if overall_rate:
                self.logger.record(
                    f"custom/preslap_lift_given_card_type/{card_type}",
                    conditional_rate / overall_rate,
                )


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


# Training entrypoint lives in train.py, which takes deck/architecture as
# CLI flags -- `python3 train.py --deck simple --net-arch 128,128`.
