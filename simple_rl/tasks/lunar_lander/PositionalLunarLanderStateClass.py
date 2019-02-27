# Python imports.
from __future__ import print_function
import numpy as np

# Other imports.
from simple_rl.mdp.StateClass import State

class PositionalLunarLanderState(State):
    def __init__(self, x, y, ydot, theta, left_leg_on_ground, right_leg_on_ground, is_terminal=False):
        self.x = x
        self.y = y
        self.ydot = ydot
        self.theta = theta
        self.left_leg_on_ground = left_leg_on_ground
        self.right_leg_on_ground = right_leg_on_ground

        State.__init__(self, data=[x, y, ydot, theta, left_leg_on_ground, right_leg_on_ground], is_terminal=is_terminal)

    def get_position(self):
        return np.array([self.x, self.y, self.ydot, self.theta])

    def __hash__(self):
        return hash(tuple(self.data))

    def __str__(self):
        return "(x: {}, y: {}, ydot: {}, theta: {}, ll: {}, rl: {}, term: {})".format(*tuple(self.data), self.is_terminal())

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return isinstance(other, PositionalLunarLanderState) and self.data == other.data and \
               self.is_terminal() == other.is_terminal()

    def __ne__(self, other):
        return not self == other

    def is_out_of_frame(self):
        return self.x < -1. or self.x > 1.