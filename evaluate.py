"""Evaluate a trained PPO EgyptianRL model against fixed baselines.

Runs many episodes per policy and compares win rate (did the RL seat end the
episode with more cards than the opponent) and mean episode reward, each with
a 95% confidence interval, so a real edge can be told apart from noise.
"""

import argparse
import json
import math
import os

import numpy as np
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO

from CustomEnv import CustomEnv
from deck_configs import DECK_PRESETS


def resolve_model_deck(model_path, cli_deck):
    """Prefer the deck recorded in a train.py sidecar JSON over --deck, so
    evaluating a model always uses the environment it was actually trained
    on. Falls back to --deck for older checkpoints with no sidecar."""
    sidecar = os.path.splitext(model_path)[0] + ".json"
    if os.path.exists(sidecar):
        with open(sidecar) as f:
            config = json.load(f)
        deck = config.get("deck", cli_deck)
        print(f"Found {sidecar}: evaluating with deck={deck!r} (from training config)")
        return deck
    print(f"No sidecar config found for {model_path}; assuming deck={cli_deck!r}")
    return cli_deck


def run_episodes(env, action_fn, n_episodes):
    rewards = np.zeros(n_episodes)
    wins = np.zeros(n_episodes)
    lengths = np.zeros(n_episodes)
    for i in range(n_episodes):
        obs, info = env.reset()
        ep_reward = 0.0
        ep_len = 0
        terminated = truncated = False
        while not (terminated or truncated):
            action = action_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_len += 1
        game = env.unwrapped.game
        opponent = 1 - game.rl
        rewards[i] = ep_reward
        wins[i] = 1.0 if len(game.decks[game.rl]) > len(game.decks[opponent]) else 0.0
        lengths[i] = ep_len
    return rewards, wins, lengths


def summarize(name, rewards, wins, lengths):
    n = len(rewards)
    win_rate = wins.mean()
    win_ci = 1.96 * math.sqrt(win_rate * (1 - win_rate) / n)
    reward_mean = rewards.mean()
    reward_ci = 1.96 * rewards.std(ddof=1) / math.sqrt(n)
    print(
        f"{name:<16} win_rate={win_rate:5.3f} +/- {win_ci:.3f}   "
        f"reward={reward_mean:7.2f} +/- {reward_ci:5.2f}   "
        f"ep_len={lengths.mean():6.1f}   n={n}"
    )
    return {"win_rate": win_rate, "win_ci": win_ci, "reward": reward_mean, "reward_ci": reward_ci}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to a saved PPO model, e.g. ppo_full_64x64.zip")
    parser.add_argument("--deck", default="full", choices=list(DECK_PRESETS), help="Fallback deck if the model has no train.py sidecar JSON")
    parser.add_argument("--episodes", type=int, default=1500, help="Episodes per policy")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    deck = resolve_model_deck(args.model, args.deck)

    print(f"Evaluating over {args.episodes} episodes per policy (95% CI shown)\n")

    results = {}

    # Fixed baselines don't need obs, so they run on the plain (unwrapped) env,
    # but it must be the same deck the trained model saw for a fair comparison.
    baseline_env = CustomEnv(deck=deck)
    baselines = {
        "never_preslap": lambda obs: np.array([0]),
        "always_preslap": lambda obs: np.array([1]),
        "random": lambda obs: np.array([rng.integers(0, 2)]),
    }
    for name, action_fn in baselines.items():
        rewards, wins, lengths = run_episodes(baseline_env, action_fn, args.episodes)
        results[name] = summarize(name, rewards, wins, lengths)

    # Trained policy needs the same FlattenObservation wrapper used in training.
    model = PPO.load(args.model)
    trained_env = FlattenObservation(CustomEnv(deck=deck))
    trained_action = lambda obs: model.predict(obs, deterministic=True)[0]
    rewards, wins, lengths = run_episodes(trained_env, trained_action, args.episodes)
    results["trained_ppo"] = summarize("trained_ppo", rewards, wins, lengths)

    print("\nTrained vs. best baseline per metric (non-overlapping 95% CI => likely a real difference):")
    baseline_names = [k for k in results if k != "trained_ppo"]
    trained = results["trained_ppo"]
    for metric, ci_key in (("win_rate", "win_ci"), ("reward", "reward_ci")):
        best_name = max(baseline_names, key=lambda k: results[k][metric])
        best = results[best_name]
        gap = trained[metric] - best[metric]
        trained_lo, trained_hi = trained[metric] - trained[ci_key], trained[metric] + trained[ci_key]
        best_lo, best_hi = best[metric] - best[ci_key], best[metric] + best[ci_key]
        overlap = not (trained_hi < best_lo or best_hi < trained_lo)
        verdict = "no significant difference" if overlap else (
            "trained_ppo genuinely higher" if gap > 0 else "trained_ppo genuinely lower"
        )
        print(f"  [{metric}] best baseline: {best_name} ({metric}={best[metric]:.3f})")
        print(f"    trained_ppo - best baseline = {gap:+.3f}   CIs overlap: {overlap}   ({verdict})")


if __name__ == "__main__":
    main()
