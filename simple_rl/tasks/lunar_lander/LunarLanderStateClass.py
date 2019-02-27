# Python imports.
from __future__ import print_function

# Other imports.
from simple_rl.mdp.StateClass import State
from simple_rl.tasks.lunar_lander.PositionalLunarLanderStateClass import PositionalLunarLanderState

class LunarLanderState(State):
    def __init__(self, x, y, xdot, ydot, theta, theta_dot, left_leg_on_ground, right_leg_on_ground,
                 is_goal_state, is_terminal=False):
        self.x = x
        self.y = y
        self.xdot = xdot
        self.ydot = ydot
        self.theta = theta
        self.theta_dot = theta_dot
        self.left_leg_on_ground = left_leg_on_ground
        self.right_leg_on_ground = right_leg_on_ground
        self.is_goal_state = is_goal_state

        State.__init__(self, data=[x, y, xdot, ydot, theta, theta_dot, left_leg_on_ground, right_leg_on_ground],
                       is_goal_state=is_goal_state, is_terminal=is_terminal)

    def convert_to_continuous_state(self):
        return ContinuousLunarLanderState(self.x, self.y, self.xdot, self.ydot, self.theta, self.theta_dot,
                                          self.is_goal_state, self.is_terminal())

    def state_space_size(self):
        return len(self.data)

    def __hash__(self):
        return hash(tuple(self.data))

    def __str__(self):
        return "(x: {}, y: {}, xdot: {}, ydot: {}, theta: {}, tdot: {}, lleg: {}, rleg: {}, goal: {}, term: {})".format(
            *tuple(self.data), self.is_goal_state, self.is_terminal()
        )

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return isinstance(other, LunarLanderState) and self.data == other.data

    def __ne__(self, other):
        return not self == other

    def is_out_of_frame(self):
        return self.x < -1. or self.x > 1.


class ContinuousLunarLanderState(State):
    def __init__(self, x, y, xdot, ydot, theta, theta_dot, is_goal_state, is_terminal=False):
        self.x = x
        self.y = y
        self.xdot = xdot
        self.ydot = ydot
        self.theta = theta
        self.theta_dot = theta_dot
        self.is_goal_state = is_goal_state

        State.__init__(self, data=[x, y, xdot, ydot, theta, theta_dot],
                       is_goal_state=is_goal_state, is_terminal=is_terminal)

    def state_space_size(self):
        return len(self.data)

    def __hash__(self):
        return hash(tuple(self.data))

    def __str__(self):
        return "(x: {}, y: {}, xdot: {}, ydot: {}, theta: {}, tdot: {}, goal: {}, term: {})".format(
            *tuple(self.data), self.is_goal_state, self.is_terminal()
        )

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return isinstance(other, ContinuousLunarLanderState) and self.data == other.data

    def __ne__(self, other):
        return not self == other
