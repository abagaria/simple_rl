# Python Imports.
from __future__ import print_function
import copy
import numpy as np
import time

# Other imports.
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.acrobot.AcrobotStateClass import AcrobotState
import pdb
import numpy as np
import gym

class AcrobotMDP(MDP):
    """ Class for acrobot domain. """

    def __init__(self, seed, render=False):
        self.env_name = "Acrobot-v1"
        self.env = gym.make(self.env_name)

        self.env.seed(seed)
        np.random.seed(seed)

        self.render = render

        init_observation = self.env.reset()
        init_state = tuple(init_observation)

        actions = list(range(self.env.action_space.n))

        MDP.__init__(self, actions, self._transition_func, self._reward_func,
                     init_state=AcrobotState(*init_state))

    def _reward_func(self, state, action, option_idx=None):
        """
        Args:
            state (AcrobotState)
            action (int): number between 0 and 4 inclusive
            option_idx (int): was the action selected based on an option policy

        Returns:
            reward (float)
        """
        assert self.is_primitive_action(action), "Can only implement primitive actions to the MDP"
        obs, reward, done, info = self.env.step(action)

        if self.render:
            self.env.render()

        self.next_state = AcrobotState(*tuple(obs))

        if reward == 0:
            assert done == 1, "Expect done to be true when terminal, done = {}".format(done)
            self.next_state.set_terminal(True)
        else:
            self.next_state.set_terminal(False)

        return reward


    def _transition_func(self, state, action):
        return self.next_state

    def execute_agent_action(self, action, option_idx=None):
        """
        Args:
            action (str)
            option_idx (int): given if action came from an option policy

        Returns:
            (tuple: <float,State>): reward, State

        Summary:
            Core method of all of simple_rl. Facilitates interaction
            between the MDP and an agent.
        """
        reward = self.reward_func(self.cur_state, action, option_idx)
        next_state = self.transition_func(self.cur_state, action)
        self.cur_state = next_state

        if self.next_state.is_terminal():
            print("GOAL")
            time.sleep(3)

        return reward, next_state

    def reset(self):
        init_observation = self.env.reset()
        init_state = tuple(init_observation)
        self.init_state = AcrobotState(*init_state)
        self.cur_state = copy.deepcopy(self.init_state)

    # @staticmethod
    # def is_goal_state(state):
    #     theta_1_wrapped = AcrobotState.acrobot_acos(state.cos_theta_1)
    #     theta_2_wrapped = AcrobotState.acrobot_acos(state.cos_theta_2)
    #
    #     return bool(-np.cos(theta_1_wrapped) - np.cos(theta_2_wrapped + theta_1_wrapped) > 1.)

    @staticmethod
    def is_primitive_action(action):
        assert action >= 0, "Action cannot be negative {}".format(action)
        return action < 3

    def __str__(self):
        return self.env_name

    def __repr__(self):
        return str(self)
