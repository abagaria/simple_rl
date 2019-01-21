# Python Imports.
from __future__ import print_function
import copy
import numpy as np

# Other imports.
from rlpy.Domains.Pinball import Pinball
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.pinball.PinballStateClass import PinballState
from simple_rl.abstraction.action_abs.PredicateClass import Predicate
import pdb
import numpy as np

class PinballMDP(MDP):
    """ Class for pinball domain. """

    def __init__(self, noise=0., episode_length=1000, reward_scale=10000., goal_predicate=None, render=False):
        self.domain = Pinball(noise=noise, episodeCap=episode_length) #, configuration="/home/akhil/git-repos/rlpy/rlpy/Domains/PinballConfigs/pinball_hard_single.cfg")
        self.render = render
        self.reward_scale = reward_scale

        # Each observation from domain.step(action) is a tuple of the form reward, next_state, is_term, possible_actions
        # s0 returns initial state, is_terminal, possible_actions
        init_observation = self.domain.s0()
        init_state = tuple(init_observation[0])

        actions = self.domain.actions
        self.goal_predicate = goal_predicate if goal_predicate is not None else self.default_goal_predicate()

        MDP.__init__(self, actions, self._transition_func, self._reward_func, init_state=PinballState(*init_state))

    def _reward_func(self, state, action):
        """
        Args:
            state (PinballState)
            action (int): number between 0 and 4 inclusive

        Returns:
            next_state (PinballState)
        """
        reward, obs, done, possible_actions = self.domain.step(action)

        if self.render:
            self.domain.showDomain(action)

        self.next_state = PinballState(*tuple(obs), is_terminal=done)

        # if self.default_goal_predicate().is_true(self.next_state):
        #     print()
        #     print(80 * "=")
        #     print("Hit goal state!")
        #     print(80 * "=")

        # current_position = np.array([state.x, state.y])
        # goal_position = np.array(self.domain.environment.target_pos)
        #
        # return -np.linalg.norm(current_position - goal_position)
        return np.clip(reward, -0.001, 10.)


    def _transition_func(self, state, action):
        return self.next_state

    def reset(self):
        init_observation = self.domain.s0()
        init_state = tuple(init_observation[0])
        self.init_state = PinballState(*init_state)
        cur_state = copy.deepcopy(self.init_state)
        self.set_current_state(cur_state)

    def set_current_state(self, new_state):
        self.cur_state = new_state
        self.domain.state = new_state.features()

    def default_goal_predicate(self):
        """
        We will pass a reference to the PinballModel function that indicates
        when the ball hits its target.
        Returns:
            goal_predicate (Predicate)
        """
        return Predicate(func=lambda s: self.domain.environment.episode_ended())

    @staticmethod
    def is_primitive_action(action):
        assert action >= 0, "Action cannot be negative {}".format(action)
        return action < 5

    def __str__(self):
        return "RlPy_Pinball_Domain"

    def __repr__(self):
        return str(self)

