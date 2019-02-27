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

    def __init__(self, render=False):
        """
        Args:
            render (bool)
        """
        self.env_name = "LunarLander-v2"
        self.env = gym.make(self.env_name)
        self.render = render

        # Each observation from env.step(action) is a tuple of the form state, reward, done, {}
        init_observation = self.env.reset()
        init_state = tuple(init_observation)

        MDP.__init__(self, list(range(self.env.action_space.n)), self._transition_func, self._reward_func,
                     init_state=LunarLanderState(*init_state, is_goal_state=False, is_terminal=False))

    def _reward_func(self, state, action):
        """
        Args:
            state (LunarLanderState)
            action (int): number between 0 and 3 inclusive

        Returns:
            next_state (LunarLanderState)
        """
        assert self.is_primitive_action(action), "Can only implement primitive actions to the MDP"
        obs, _, _, info = self.env.step(action)

        reward = 0.
        done = False
        goal = False


        if self.env.env.game_over or abs(obs[0]) >= 1.:
            done = True
            goal = False
            reward = -10
        if not self.env.env.lander.awake:
            done = True
            goal = True  # TODO: Currently marking landing as a lower reward goal state
            reward = +5
        if (not self.env.env.lander.awake) and (abs(obs[0]) <= 0.2):
            done = True
            goal = True
            reward = +10

        if self.render:
            self.env.render()

        self.next_state = LunarLanderState(*tuple(obs), is_goal_state=goal, is_terminal=done)

        if reward > 0:
            print("s':{}, r: {}".format(self.next_state, reward))

        return reward

    # Assuming that _reward_func(s, a) is called before _transition_func(s, a)
    def _transition_func(self, state, action):
        return self.next_state

    @staticmethod
    def is_goal_state(state):
        return state.is_goal_state

    @staticmethod
    def is_primitive_action(action):
        assert action >= 0, "action cannot be negative {}".format(action)
        return action < 4

    def reset(self):
        init_observation = self.env.reset()
        init_state = tuple(init_observation)
        self.init_state = copy.deepcopy(LunarLanderState(*init_state, is_goal_state=False, is_terminal=False))

        super(LunarLanderMDP, self).reset()

    def __str__(self):
        return "gym-" + str(self.env_name)
