"""Train a PPO agent on EgyptianRL.

Deck and network architecture are CLI flags, not hardcoded, so alternating
between variants is just a different command line:

    python3 train.py --deck full   --net-arch 64,64
    python3 train.py --deck simple --net-arch 128,128 --timesteps 1000000

Each run saves ppo_<run-name>.zip plus a ppo_<run-name>.json sidecar
recording the deck/architecture used, so evaluate.py can reconstruct a
matching environment automatically instead of guessing.
"""

import argparse
import json

from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from CustomEnv import CustomEnv, LogEveryNStepsCallback, PreslapLoggingCallback
from deck_configs import DECK_PRESETS


def parse_net_arch(spec):
    return [int(x) for x in spec.split(",") if x]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deck", default="full", choices=list(DECK_PRESETS), help="Deck preset (see deck_configs.py)")
    parser.add_argument("--net-arch", default="64,64", help="Comma-separated hidden layer sizes, shared by policy and value nets")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--run-name", default=None, help="Defaults to <deck>_<net-arch>; used for model filename and TensorBoard run name")
    parser.add_argument("--log-interval", type=int, default=10_000)
    args = parser.parse_args()

    net_arch = parse_net_arch(args.net_arch)
    run_name = args.run_name or f"{args.deck}_{args.net_arch.replace(',', 'x')}"

    env = Monitor(FlattenObservation(CustomEnv(deck=args.deck)))
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./ppo_egyptianrl_tensorboard/",
        ent_coef=args.ent_coef,
        policy_kwargs=dict(net_arch=net_arch),
    )
    model.learn(
        total_timesteps=args.timesteps,
        callback=[LogEveryNStepsCallback(args.log_interval), PreslapLoggingCallback()],
        tb_log_name=run_name,
    )

    model_path = f"ppo_{run_name}"
    model.save(model_path)
    with open(f"{model_path}.json", "w") as f:
        json.dump(
            {
                "deck": args.deck,
                "net_arch": net_arch,
                "timesteps": args.timesteps,
                "ent_coef": args.ent_coef,
            },
            f,
            indent=2,
        )
    print(f"Training complete, model saved to {model_path}.zip (config: {model_path}.json)")


if __name__ == "__main__":
    main()
