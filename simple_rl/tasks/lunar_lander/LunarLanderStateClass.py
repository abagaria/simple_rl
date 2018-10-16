# Python imports.
from __future__ import print_function

# Other imports.
from simple_rl.mdp.StateClass import State

class LunarLanderState(State):
    def __init__(self, x, y, xdot, ydot, theta, theta_dot, left_leg_on_ground, right_leg_on_ground, is_terminal=False):
        self.x = x
        self.y = y
        self.xdot = xdot
        self.ydot = ydot
        self.theta = theta
        self.theta_dot = theta_dot
        self.left_leg_on_ground = left_leg_on_ground
        self.right_leg_on_ground = right_leg_on_ground

        State.__init__(self, data=[x, y, xdot, ydot, theta, theta_dot, left_leg_on_ground, right_leg_on_ground],
                       is_terminal=is_terminal)

    def __hash__(self):
        return hash(tuple(self.data))

    def __str__(self):
        return "(x: {}, y: {}, xdot: {}, ydot: {}, theta: {}, tdot: {}, lleg: {}, rleg: {}, term: {})".format(
            *tuple(self.data), self.is_terminal()
        )

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return isinstance(other, LunarLanderState) and self.data == other.data

    def __ne__(self, other):
        return not self == other

    def is_out_of_frame(self):
        return self.x < -1. or self.x > 1.