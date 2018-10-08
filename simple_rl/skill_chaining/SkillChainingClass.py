# Python imports.
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pdb
from collections import deque, defaultdict
from copy import deepcopy

# Other imports.
from simple_rl.abstraction.action_abs.PredicateClass import Predicate

from simple_rl.agents.AgentClass import Agent
from simple_rl.agents import QLearningAgent
from simple_rl.tasks import GymMDP, GridWorldMDP

class SkillChaining(object):
    def __init__(self, mdp, overall_goal_predicate, rl_agent=None, buffer_length=40):
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
        self.global_solver = rl_agent if rl_agent is not None else QLearningAgent(mdp.get_actions(), name="GlobalSolver")
        self.buffer_length = buffer_length

        self.trained_options = []

    def skill_chaining(self, num_episodes=100, num_steps=500):
        from simple_rl.abstraction.action_abs.OptionClass import Option
        goal_option = Option(init_predicate=None, term_predicate=self.overall_goal_predicate, overall_mdp=self.mdp,
                             init_state=self.mdp.init_state, actions=self.original_actions, policy={},
                             name='overall_goal_policy', term_prob=0.)

        # Pointer to the current option:
        # 1. This option has the termination set which defines our current goal trigger
        # 2. This option has an untrained initialization set, which we need to train from experience
        untrained_option = goal_option

        for episode in range(num_episodes):

            print '-------------'
            print 'Episode = {}'.format(episode)
            print '-------------'
            self.mdp.reset()
            state = deepcopy(self.mdp.init_state)
            reward = 0
            last_n_states = deque([], maxlen=self.buffer_length)

            for step in range(num_steps):
                if reward > 0: print "Global-Q-Learning act() with R={}".format(reward)
                action = self.global_solver.act(state, reward)

                # if isinstance(action, Option):
                #     print "\n\nWatchOut: global solver is using option {}\n\n".format(action.name)
                #     reward, next_state = action.execute_option_in_mdp(state, self.mdp, verbose=True)
                # else: # Primitive action
                reward, next_state = self.mdp.execute_agent_action(action)
                if reward > 0: print "Global MDP: R={}\tS'={}".format(reward, next_state)

                last_n_states.append(next_state)
                state = next_state

                if untrained_option.is_term_true(next_state) and len(last_n_states) == self.buffer_length:

                    # Train the initiation set classifier for the option
                    untrained_option.initiation_data = last_n_states
                    untrained_option.train_initiation_classifier()

                    # Update the solver of the untrained option on all the states in its experience
                    untrained_option.learn_policy_from_experience()

                    # Add the trained option to the action set of the global solver
                    if untrained_option not in self.trained_options:
                        self.trained_options.append(untrained_option)
                    if untrained_option not in self.mdp.actions:
                        self.mdp.actions.append(untrained_option)

                    # Create new option whose termination is the initiation of the option we just trained
                    name = "option_{}".format(str(len(self.trained_options)))

                    # Using the global init_state as the init_state for all child options
                    untrained_option = untrained_option.create_child_option(init_state=deepcopy(self.mdp.init_state),
                                                                            actions=self.original_actions,
                                                                            new_option_name=name)

                    # Reset the agent so that we don't keep moving around the initiation set of the trained option
                    break

def construct_pendulum_domain():
    # Overall goal predicate in the Pendulum domain
    target_cos_theta, cos_theta_tol = 1., 0.001
    predicate = Predicate(lambda s: (abs(s[0] - target_cos_theta) < cos_theta_tol))

    # Pendulum domain parameters
    max_torque = 2.

    # Construct GymMDP wrapping around Gym's Pendulum MDP
    discretized_actions = np.arange(-max_torque, max_torque + 0.1, 0.1)
    return GymMDP(discretized_actions, subgoal_predicate=predicate, env_name='Pendulum-v0', render=True)

def construct_grid_world_mdp():
    predicate = Predicate(func=lambda s: s.x == 10 and s.y == 10, name='OverallOption_GoalPredicate')
    return GridWorldMDP(10, 10, goal_predicate=predicate)

if __name__ == '__main__':
    overall_mdp = construct_grid_world_mdp()
    chainer = SkillChaining(overall_mdp, overall_mdp.goal_predicate)
    chainer.skill_chaining()