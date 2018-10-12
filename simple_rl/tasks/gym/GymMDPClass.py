'''
GymMDPClass.py: Contains implementation for MDPs of the Gym Environments.
'''

# Python imports.
from __future__ import print_function
import random
import sys
import os
import random

# Other imports.
import gym
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.gym.GymStateClass import GymState
from simple_rl.abstraction.action_abs.PredicateClass import Predicate

class GymMDP(MDP):
    ''' Class for Gym MDPs '''

    def __init__(self, discretized_actions=None, subgoal_predicate=None, env_name='CartPole-v0', render=False):
        '''
        Args:
            discretized_actions (np.ndarray)
            subgoal_predicate (Predicate)
            env_name (str)
            render (bool)
        '''
        actions = range(self.env.action_space.n) if discretized_actions is None else discretized_actions

        self.subgoal_predicate = subgoal_predicate
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.render = render
        MDP.__init__(self, actions, self._transition_func, self._reward_func, init_state=GymState(self.env.reset()))

    # TODO Akhil: Querying the reward / transition function should not cause the agent to act in the real world
    def _reward_func(self, state, action):
        '''
        Args:
            state (AtariState)
            action (str)

        Returns
            (float)
        '''
        obs, reward, is_terminal, info = self.env.step([action])

        if self.render:
            print('Torque={}\tNewState={}\tReward={}'.format(action, obs, reward))
            self.env.render()

        if self.subgoal_predicate is not None:
            self.next_state = GymState(obs, is_terminal=self.subgoal_predicate.is_true(obs))
            return 0. if self.subgoal_predicate.is_true(obs) else reward
        else:
            self.next_state = GymState(obs, is_terminal=is_terminal)
            return reward

    def _transition_func(self, state, action):
        '''
        Args:
            state (AtariState)
            action (str)

        Returns
            (State)
        '''
        return self.next_state

    def reset(self):
        self.env.reset()

    def __str__(self):
        return "gym-" + str(self.env_name)
