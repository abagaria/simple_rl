# Python imports.
from __future__ import print_function
import numpy as np

# Other imports.
from simple_rl.mdp.StateClass import State

class AcrobotState(State):
    def __init__(self, cos_theta_1, sin_theta_1, cos_theta_2, sin_theta_2, theta_1_dot, theta_2_dot, is_terminal=False):
        self.cos_theta_1 = cos_theta_1
        self.sin_theta_1 = sin_theta_1
        self.cos_theta_2 = cos_theta_2
        self.sin_theta_2 = sin_theta_2
        self.theta_1_dot = theta_1_dot
        self.theta_2_dot = theta_2_dot

        State.__init__(self, data=[cos_theta_1, sin_theta_1, cos_theta_2, sin_theta_2, theta_1_dot, theta_2_dot],
                       is_terminal=is_terminal)

    @staticmethod
    def wrap(x, m, M):
        """
        <Taken from gym>
        :param x: a scalar
        :param m: minimum possible value in range
        :param M: maximum possible value in range
        Wraps ``x`` so m <= x <= M; but unlike ``bound()`` which
        truncates, ``wrap()`` wraps x around the coordinate system defined by m,M.\n
        For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.
        """
        diff = M - m
        while x > M:
            x = x - diff
        while x < m:
            x = x + diff
        return x

    @staticmethod
    def acrobot_acos(cos_theta):
        theta = np.arccos(cos_theta)
        theta_wrapped = AcrobotState.wrap(theta, -np.pi, np.pi)
        return theta_wrapped

    def state_space_size(self):
        return len(self.data)

    def __hash__(self):
        return hash(tuple(self.data))

    def __str__(self):
        return "(cos(t1): {}, sin(t1): {}, cos(t2): {}, sin(t2): {}, t1dot: {}, t2dot: {}, term: {})".format(
            *tuple(self.data), self.is_terminal()
        )

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return isinstance(other, AcrobotState) and self.data == other.data

    def __ne__(self, other):
        return not self == other

    def is_out_of_frame(self):
        return False