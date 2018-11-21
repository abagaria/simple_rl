# Python imports.
from __future__ import print_function
import copy

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
        obs, _, is_terminal, info = self.env.step(action)

        if self.render:
            self.env.render()

        self.next_state = LunarLanderState(*tuple(obs), is_terminal=is_terminal)

        # Sparse reward function
        if self.positive_reward_predicate().is_true(self.next_state):
            print("\nHit goal state, getting 100. points!")
            self.next_state.is_terminal = True
            return 100.
        elif self.negative_reward_predicate().is_true(self.next_state):
            # print("\nHit bad termination, getting -100 points =(")
            return -100.
        else:
            return 0.

    # Assuming that _reward_func(s, a) is called before _transition_func(s, a)
    def _transition_func(self, state, action):
        return self.next_state

    # TODO: Incorporate this to test with sparse reward lunar lander
    @staticmethod
    def positive_reward_predicate():
        return Predicate(
            func=lambda s: abs(s.x) < 0.2
                       and abs(s.y) < 0.01
                       and abs(s.xdot) < 0.1
                       and abs(s.ydot) < 0.01
                       and abs(s.theta) < 0.1
        )

    @staticmethod
    def negative_reward_predicate():
        return Predicate(
            func=lambda s: s.is_terminal()
                       and not LunarLanderMDP.positive_reward_predicate().is_true(s)
        )

    def reset(self):
        init_observation = self.env.reset()
        init_state = tuple(init_observation)
        self.init_state = copy.deepcopy(LunarLanderState(*init_state))
        
        super(LunarLanderMDP, self).reset()

    def __str__(self):
        return "gym-" + str(self.env_name)
