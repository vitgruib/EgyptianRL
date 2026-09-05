from collections import deque
class Model:
    def __init__(self, deck, behavior):
        self.deck = deck
        self.behavior = behavior
        if behavior == "base":
            self.speed = 1
            self.preslap = 5
        elif behavior == "rl":
            #TODO: make rl shit happen
            pass
    def move(self, info):
        """decides whether or not to preslap and card"""
        return False, self.draw()
    def burn(self):
        return self.draw()
    def draw(self):
        return self.deck.popleft()

