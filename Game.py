from collections import deque
import random
from typing import Optional
import numpy as np

CARD_TYPES = {1: "Ace", 11: "Jack", 12: "Queen", 13: "King"}


def card_type(card):
    if not card:
        return None
    return CARD_TYPES.get(card, "Number")


class Game:
    def __init__(self, models, verbosity=3, deck_ranks=None, deck_copies=4):
        # RL
        # verbosity 1 prints the ending, 2 prints every slap, 3 every card
        self.verbosity = verbosity
        self.rl = 0  # the id of the rl agent
        self.terminated = False
        self.reward = 0
        self.info = {}
        # game
        self.n = len(models)
        self.active_turn = 0  # 0 is going to be the player
        self.pile = deque()
        self.face_counter = [-1, -1]  # remaining turns, owner
        # players
        self.decks = Deck(ranks=deck_ranks, copies=deck_copies).deal(self.n)
        self.models = models
        for model in models:
            model.connect(self)
        # play
        self.play()

    def play(self):
        guard = 0
        while self.active_turn != self.rl:
            self.move()
            guard += 1
            if guard > 10000:
                self.terminated = True
                break

    def move(self, preslap=False):
        curr_model = self.models[self.active_turn]
        if self.active_turn != self.rl:
            preslap = curr_model.move()
        # Top-of-pile card at decision time, tracked regardless of whether we
        # preslap, so callers can compute P(preslap | card type) instead of
        # only ever seeing the card when a preslap happened.
        pile_top_card = self.pile[-1] if len(self.pile) else 0
        self.info = {
            "preslap": bool(preslap),
            "pile_top_card": pile_top_card,
            "pile_top_type": card_type(pile_top_card),
        }
        card = self.draw()
        if self.terminated:
            return
        self.pile.append(card)
        self.face_counter[0] -= 1
        # check slap
        valid = self.check_valid()
        if self.verbosity > 2:
            val = card
            if val == 10:
                val = "T"
            elif val == 11:
                val = "J"
            elif val == 12:
                val = "Q"
            elif val == 13:
                val = "K"
            print(val, end="")
        if valid:
            speeds = [model.speed for model in self.models]
            if preslap:
                speeds[self.active_turn] = curr_model.preslap
            tot = sum(speeds)
            speeds = [speed / tot for speed in speeds]
            # we set the turn to the winner first, then reset
            self.active_turn = np.random.choice(self.n, p=speeds)
            self.reset(valid)
            return
        elif preslap:
            card = self.draw()
            if self.terminated:
                return
            self.pile.appendleft(card)
            if self.verbosity >= 3:
                print("X", end="")
        # face card
        if card > 10:
            self.face_counter = [card - 10, self.active_turn]
        elif self.face_counter[0] == 0:
            self.active_turn = self.face_counter[1]
            self.reset("faces")

        # proceed
        self.active_turn += 1
        self.active_turn %= self.n

    def reset(self, win_type):
        while len(self.pile):
            card = self.pile.pop()
            self.decks[self.active_turn].append(card)
            if self.active_turn == self.rl:
                self.reward += self.cardValue(card)
            else:
                self.reward -= self.cardValue(card)
        self.face_counter = [-1, -1]
        if self.verbosity >= 2:
            print(
                f". P{self.active_turn} won off {win_type}, score: {[len(deck) for deck in self.decks]}"
            )

    def check_valid(self):
        if len(self.pile) > 1 and self.pile[-1] == self.pile[-2]:
            return "Double"
        if len(self.pile) > 1 and self.pile[-1] * self.pile[-2] == 156:
            return "Marriage"
        if len(self.pile) > 2 and self.pile[-1] == self.pile[-3]:
            return "Sandwich"
        return ""

    def draw(self):
        if not len(self.decks[self.active_turn]) or self.terminated:
            if self.verbosity >= 1:
                print(f". Game ended: {[len(deck) for deck in self.decks]}")
            self.terminated = True
            return
        card = self.decks[self.active_turn].popleft()
        if self.active_turn == self.rl:
            self.reward -= self.cardValue(card) / 20
        return card

    def cardValue(self, card):
        if card <= 10:
            return 1
        elif card == 11:
            return 3
        elif card == 12:
            return 2.5
        elif card == 13:
            return 2


class Deck:
    def __init__(self, ranks=None, copies=4):
        ranks = list(ranks) if ranks is not None else list(range(1, 14))
        self.mainDeck = ranks * copies

    def deal(self, n):
        random.shuffle(self.mainDeck)
        per_player = len(self.mainDeck) // n
        return [
            deque(self.mainDeck[(i) * per_player : (i + 1) * per_player])
            for i in range(n)
        ]


class Model:
    def __init__(self, behavior=0):
        self.behavior = behavior
        if behavior == 0:
            self.speed = 1
            self.preslap = 5

    def connect(self, game):
        self.game = game

    def move(self, info=None):
        """decides whether or not to preslap and card"""
        if self.behavior == 0:
            return False  # preslap and then what you drew


if __name__ == "__main__":
    game = Game([Model(), Model()])
    game.play()
