# Python imports.
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pdb
from collections import deque, defaultdict
from copy import deepcopy
from IPython import embed

# Other imports.
from simple_rl.abstraction.action_abs.PredicateClass import Predicate

from simple_rl.agents.AgentClass import Agent
from simple_rl.agents import QLearningAgent
from simple_rl.tasks import GymMDP

class SkillChaining(object):
    def __init__(self, mdp, overall_goal_predicate, rl_agent=None, buffer_length=20):
        """
        Args:
            mdp (MDP): Underlying domain we have to solve
            overall_goal_predicate (Predicate)
            rl_agent (Agent): RL agent used to determine the policy for each option
            buffer_length (int): size of the circular buffer used as experience buffer
        """
        self.mdp = mdp
        self.original_actions = deepcopy(mdp.actions)
        self.overall_goal_predicate = overall_goal_predicate
        self.global_solver = rl_agent if rl_agent is not None else QLearningAgent(mdp.get_actions())
        self.buffer_length = buffer_length

        self.trained_options = []
        self.untrained_options = []

    def skill_chaining(self, num_episodes=15, num_steps=25):
        from simple_rl.abstraction.action_abs.OptionClass import Option
        pdb.set_trace()
        goal_option = Option(init_predicate=None, term_predicate=self.overall_goal_predicate, overall_mdp=self.mdp,
                             actions=self.original_actions, policy={}, name='overall_goal_policy', term_prob=0.)
        self.untrained_options.append(goal_option)

        for episode in range(num_episodes):
            self.mdp.reset()
            state = self.mdp.init_state
            reward = 0
            last_n_states = deque([], maxlen=self.buffer_length)
            for step in range(num_steps):
                action = self.global_solver.act(state, reward)
                if isinstance(action, Option):
                    reward, next_state = action.execute_option_in_mdp(state, self.mdp, verbose=True)
                else:
                    reward, next_state = self.mdp.execute_agent_action(action)
                last_n_states.append(next_state)

                for untrained_option in self.untrained_options: # type: Option
                    if untrained_option.is_term_true(next_state) and len(last_n_states) == self.buffer_length:
                        # Train the initiation set classifier for the option
                        untrained_option.initiation_data = last_n_states
                        untrained_option.train_initiation_classifier()
                        init_predicate = Predicate(func=lambda s: untrained_option.initiation_classifier.predict([s])[0])
                        untrained_option.init_predicate = init_predicate

                        # Update the solver of the untrained option on all the states in its experience
                        action.learn_policy_from_experience()

                        # Add the trained option to the action set of the global solver
                        self.untrained_options.remove(untrained_option)
                        self.trained_options.append(untrained_option)
                        self.mdp.actions.append(untrained_option)

                # Create options whose goal predicate is the same as the initiation predicate of someone else
                for trained_option in self.trained_options: # type: Option
                    if trained_option.is_init_true(next_state):
                        untrained_option = Option(init_predicate=None, term_predicate=trained_option.init_predicate, policy={},
                                                  overall_mdp=self.mdp, name="option_{}".format(np.random.randint(0, 100)), term_prob=0.)
                        self.untrained_options.append(untrained_option)

def construct_pendulum_domain(predicate):
    # Pendulum domain parameters
    max_torque = 2.

    # Construct GymMDP wrapping around Gym's Pendulum MDP
    discretized_actions = np.arange(-max_torque, max_torque + 0.1, 0.1)
    return GymMDP(discretized_actions, subgoal_predicate=predicate, env_name='Pendulum-v0', render=True)

if __name__ == '__main__':
    target_cos_theta, cos_theta_tol = 1., 0.001
    goal_pred = Predicate(lambda s: (abs(s[0] - target_cos_theta) < cos_theta_tol))
    overall_mdp = construct_pendulum_domain(goal_pred)
    chainer = SkillChaining(overall_mdp, goal_pred)
    chainer.skill_chaining()