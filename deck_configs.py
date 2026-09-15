"""Named deck presets.

train.py and evaluate.py both resolve decks through here, so adding a new
difficulty variant is a one-line addition to DECK_PRESETS -- no changes
needed in Game.py or CustomEnv.py.
"""

DECK_PRESETS = {
    "full": {"ranks": list(range(1, 14)), "copies": 4},  # standard 52-card deck
    "simple": {"ranks": list(range(1, 11)), "copies": 4},  # 40 cards, no face cards
}


def resolve_deck(name_or_config):
    """Accepts a preset name (str) or an explicit {"ranks": [...], "copies": n} dict."""
    if isinstance(name_or_config, str):
        try:
            return dict(DECK_PRESETS[name_or_config])
        except KeyError:
            raise ValueError(f"Unknown deck preset {name_or_config!r}; choices: {list(DECK_PRESETS)}")
    return dict(name_or_config)
