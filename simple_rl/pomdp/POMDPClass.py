from simple_rl.pomdp.BeliefUpdaterClass import BeliefUpdater
from simple_rl.mdp.MDPClass import MDP

from collections import defaultdict

class POMDP(MDP):
    ''' Abstract class for a Partially Observable Markov Decision Process. '''

    def __init__(self, actions, observations, transition_func, reward_func, observation_func,
                 init_belief, belief_updater_type='discrete', gamma=0.99, step_cost=0):
        '''
        In addition to the input parameters needed to define an MDP, the POMDP
        definition requires an observation function, a way to update the belief
        state and an initial belief.
        Args:
            actions (list)
            observations (list)
            transition_func: T(s, a) -> s'
            reward_func: R(s, a) -> float
            observation_func: O(s, a) -> z
            init_belief (defaultdict): initial probability distribution over states
            belief_updater_type (str): discrete/kalman/particle
            gamma (float)
            step_cost (int)
        '''
        self.observations = observations
        self.observation_func = observation_func
        self.curr_belief = init_belief

        # init_belief_state = BeliefState(data=init_belief.values())
        sampled_init_state = max(init_belief, key=init_belief.get)
        MDP.__init__(self, actions, transition_func, reward_func, sampled_init_state, gamma, step_cost)

        self.belief_updater = BeliefUpdater(self, transition_func, reward_func, observation_func, belief_updater_type)
        self.belief_updater_func = self.belief_updater.updater

    def get_curr_belief(self):
        return self.curr_belief

    def get_observation_func(self):
        '''
        Returns:
            observation_function: O(s, a) -> o
        '''
        return self.observation_func

    def execute_agent_action(self, action):
        '''
        Args:
            action (str)

        Returns:
            reward (float)
            next_belief (defaultdict)
        '''
        reward = self.get_simulated_reward(action)
        observation = self.get_simulated_observation(action)
        new_belief = self.belief_updater_func(self.curr_belief, action, observation)

        # True underlying state inside the POMDP unobserved by any solver
        _next_state = self.transition_func(self.cur_state, action)
        self.cur_state = _next_state
        self.curr_belief = new_belief

        return reward, observation, new_belief

    def get_simulated_reward(self, action):
        '''
        Reward given by the simulated environment when the agent takes action from unobserved current state.
        Args:
            belief (defaultdict)
            action (str)

        Returns:
            reward (float)
        '''
        reward = self.reward_func(self.cur_state, action)
        return reward

    def get_simulated_observation(self, action):
        '''
        Observation given by the simulated environment when the agent takes action from unobserved current state.
        Args:
            belief (defaultdict)
            action (str)

        Returns:
            observation (str)
        '''
        observation = self.observation_func(self.cur_state, action)
        return observation
