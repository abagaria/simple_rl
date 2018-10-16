# Python imports.
from __future__ import print_function
import pdb

# Other imports.
import gym
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.lunar_lander.LunarLanderStateClass import LunarLanderState
from simple_rl.abstraction.action_abs.PredicateClass import Predicate

class LunarLanderMDP(MDP):
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
        init_observation = self.env.reset()
        init_state = tuple(init_observation)

        MDP.__init__(self, list(range(self.env.action_space.n)), self._transition_func, self._reward_func,
                     init_state=LunarLanderState(*init_state))

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

        self.next_state = LunarLanderState(*tuple(obs), is_terminal=is_terminal)

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
        return "gym-" + str(self.env_name)
