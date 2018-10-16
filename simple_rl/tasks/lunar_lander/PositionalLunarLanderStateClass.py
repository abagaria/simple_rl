# Python imports.
from __future__ import print_function

# Other imports.
from simple_rl.mdp.StateClass import State

class PositionalLunarLanderState(State):
    def __init__(self, x, y, is_terminal=False):
        self.x = x
        self.y = y

        State.__init__(self, data=[x, y], is_terminal=is_terminal)

    def __hash__(self):
        return hash(tuple(self.data))

    def __str__(self):
        return "(x: {}, y: {})".format(*tuple(self.data), self.is_terminal())

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return isinstance(other, PositionalLunarLanderState) and self.data == other.data

    def __ne__(self, other):
        return not self == other

    def is_out_of_frame(self):
        return self.x < -1. or self.x > 1.