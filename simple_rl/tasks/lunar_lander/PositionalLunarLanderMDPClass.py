# Python imports.
from __future__ import print_function
import numpy as np
import pdb

# Other imports.
import gym
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.lunar_lander.PositionalLunarLanderStateClass import PositionalLunarLanderState
from simple_rl.abstraction.action_abs.PredicateClass import Predicate

class PositionalLunarLanderMDP(MDP):
    """ Class for Lunar Lander MDP. """

    def __init__(self, goal_predicate=None, render=False):
        """
        Args:
            goal_predicate (Predicate): f(s) -> {0, 1}
            render (bool)
        """
        self.env_name = "LunarLander-v2"
        self.env = gym.make(self.env_name)
        self.goal_predicate = goal_predicate if goal_predicate is not None else self.default_goal_predicate()
        self.render = render

        # Each observation from env.step(action) is a tuple of the form state, reward, done, {}
        # But env.reset() just returns a state
        init_observation = self.env.reset()
        init_state = init_observation[0], init_observation[1]

        MDP.__init__(self, list(range(self.env.action_space.n)), self._transition_func, self._reward_func,
                     init_state=PositionalLunarLanderState(*init_state))

    def _reward_func(self, state, action):
        """
        Args:
            state (LunarLanderState)
            action (int): number between 0 and 3 inclusive

        Returns:
            next_state (LunarLanderState)
        """
        obs, reward, is_terminal, info = self.env.step(action)

        if self.render:
            self.env.render()

        truncated_observation = obs[0], obs[1]
        self.next_state = PositionalLunarLanderState(*truncated_observation, is_terminal=is_terminal)

        return reward

    # Assuming that _reward_func(s, a) is called before _transition_func(s, a)
    def _transition_func(self, state, action):
        return self.next_state

    @staticmethod
    def default_goal_predicate():
        return Predicate(
            func=lambda s: (-0.2 < s.x < 0.2)
                           and (-0.1 < s.y < 0.1)
                           and s.is_terminal()
                         )

    def reset(self):
        self.env.reset()

    def __str__(self):
        return "positional-gym-" + str(self.env_name)

    @staticmethod
    def get_discretized_positional_state_space():
        x_range = np.arange(-1., 1.1, 0.1)
        y_range = np.arange(0., 1.05, 0.05)
        return [PositionalLunarLanderState(x, y) for x in x_range for y in y_range]
