import random
from collections import deque
class Deck:
    def __init__(self):
        self.mainDeck = [x for x in range(1, 14)] * 4

    def deal(self, n):
        random.shuffle(self.mainDeck)
        return [deque(self.mainDeck[(i) * (52 // n):(i + 1) * (52 // n)]) for i in range(n)]


    