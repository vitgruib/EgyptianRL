from Deck import Deck
from Model import Model
from collections import deque
import numpy as np
class Game:
    def __init__(self, models, verbosity=0):
        self.n = len(models)
        self.verbosity = verbosity
        self.models = models
        self.active_turn = 0
        self.pile = deque()
        self.face_counter = [-1, -1] #remaining turns, owner
        #verbosity 1 prints the ending, 2 prints every slap, 3 every card
        self.play()

    def play(self):
        while(len(self.models[self.active_turn].deck)):
            self.move()
        if self.verbosity >= 1:
            print(f". Game ended: {[len(model.deck) for model in self.models]}")

    def move(self):
        curr_model = self.models[self.active_turn]
        preslap, card = curr_model.move(self.pile)
        self.pile.append(card)
        self.face_counter[0] -= 1
        #check slap
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
            print(val, end = "")
        if valid:
            speeds = [model.speed for model in self.models]
            if preslap:
                speeds[self.active_turn] = curr_model.preslap
            tot = sum(speeds)
            speeds = [speed / tot for speed in speeds]
            #we set the turn to the winner first, then reset
            self.active_turn = np.random.choice(self.n, p=speeds)
            self.reset(valid)
            return
        elif preslap:
            self.pile.appendleft(curr_model.burn())
        #face card
        if card > 10:
            self.face_counter = [card - 10, self.active_turn]
        elif self.face_counter[0] == 0:
            self.active_turn = self.face_counter[1]
            self.reset("faces")

        #proceed
        self.active_turn += 1
        self.active_turn %= self.n
            
    def reset(self, win_type):
        while(len(self.pile)):
            self.models[self.active_turn].deck.append(self.pile.pop())
        self.face_counter = [-1, -1]
        if self.verbosity >= 2:
            print(f". P{self.active_turn} won off {win_type}, score: {[len(model.deck) for model in self.models]}")

    def check_valid(self):
        if len(self.pile) > 1 and self.pile[-1] == self.pile[-2]:
            return "Double"
        if len(self.pile) > 1 and self.pile[-1] * self.pile[-2] == 156:
            return "Marriage"
        if len(self.pile) > 2 and self.pile[-1] == self.pile[-3]:
            return "Sandwich"
        return ""

if __name__ == "__main__":
    n = 2
    decks = Deck().deal(n)
    game = Game([Model(decks[i]) for i in range(n)], 2)
