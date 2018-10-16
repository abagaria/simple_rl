# Python imports.
from __future__ import print_function
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pdb
from collections import deque, defaultdict
from copy import deepcopy

# Other imports.
from simple_rl.abstraction.action_abs.PredicateClass import Predicate
from simple_rl.abstraction.action_abs.OptionClass import Option
from simple_rl.agents.AgentClass import Agent
from simple_rl.agents import QLearningAgent
from simple_rl.tasks import GymMDP, GridWorldMDP
from simple_rl.tasks.lunar_lander.LunarLanderMDPClass import LunarLanderMDP

class SkillChaining(object):
    def __init__(self, mdp, overall_goal_predicate, rl_agent=None, buffer_length=40, subgoal_reward=1., subgoal_hits=10):
        """
        Args:
            mdp (MDP): Underlying domain we have to solve
            overall_goal_predicate (Predicate)
            rl_agent (Agent): RL agent used to determine the policy for each option
            buffer_length (int): size of the circular buffer used as experience buffer
            subgoal_reward (float): Hitting a subgoal must yield a supplementary reward to enable local policy
            subgoal_hits (int): number of times the RL agent has to hit the goal of an option o to learn its I_o, Beta_o
        """
        self.mdp = mdp
        self.original_actions = deepcopy(mdp.actions)
        self.overall_goal_predicate = overall_goal_predicate
        self.global_solver = rl_agent if rl_agent is not None else QLearningAgent(mdp.get_actions(), name="GlobalSolver")
        self.buffer_length = buffer_length
        self.subgoal_reward = subgoal_reward
        self.num_goal_hits_before_training = subgoal_hits

        self.trained_options = []

    def _train_untrained_option(self, untrained_option):
        """
        Train the current untrained option and initialize a new one to target.
        Args:
            untrained_option (Option)

        Returns:
            new_untrained_option (Option)
        """
        print("Training the initiation set and policy for {}.".format(untrained_option.name))
        # Train the initiation set classifier for the option
        untrained_option.train_initiation_classifier()

        # Update the solver of the untrained option on all the states in its experience
        untrained_option.learn_policy_from_experience()

        # The max of the qvalues sampled during the transitions that triggered the option's target event
        # is used to initialize the solver of the new untrained option.
        max_qvalue = 0.  # max([untrained_option.solver.get_max_q_value(s) for s in untrained_option.initiation_data])

        # Add the trained option to the action set of the global solver
        if untrained_option not in self.trained_options:
            self.trained_options.append(untrained_option)
        if untrained_option not in self.mdp.actions:
            self.mdp.actions.append(untrained_option)

        # Create new option whose termination is the initiation of the option we just trained
        name = "option_{}".format(str(len(self.trained_options)))

        print("Creating {}".format(name))

        # Using the global init_state as the init_state for all child options
        new_untrained_option = untrained_option.create_child_option(init_state=deepcopy(self.mdp.init_state),
                                                                actions=self.original_actions,
                                                                new_option_name=name,
                                                                default_q=max_qvalue)
        return new_untrained_option

    def _get_trained_options(self, new_option_just_created):
        trained_options = []
        if new_option_just_created and len(self.trained_options) > 1:
            trained_options = self.trained_options[:-2]
        elif not new_option_just_created:
            trained_options = self.trained_options
        return trained_options

    def skill_chaining(self, num_episodes=50, num_steps=1000):
        from simple_rl.abstraction.action_abs.OptionClass import Option
        goal_option = Option(init_predicate=None, term_predicate=self.overall_goal_predicate, overall_mdp=self.mdp,
                             init_state=self.mdp.init_state, actions=self.original_actions, policy={},
                             name='overall_goal_policy', term_prob=0.)

        # Pointer to the current option:
        # 1. This option has the termination set which defines our current goal trigger
        # 2. This option has an untrained initialization set and policy, which we need to train from experience
        untrained_option = goal_option

        for episode in range(num_episodes):

            print('-------------')
            print('Episode = {}'.format(episode))
            print('-------------')
            self.mdp.reset()
            reset_agent = False
            state = deepcopy(self.mdp.init_state)
            reward = 0
            experience_buffer = deque([], maxlen=self.buffer_length)
            state_buffer = deque([], maxlen=self.buffer_length)

            for step in range(num_steps):
                action = self.global_solver.act(state, reward)

                # if isinstance(action, Option):
                #     print "\n\nWatchOut: global solver is using option {}\n\n".format(action.name)
                #     reward, next_state = action.execute_option_in_mdp(state, self.mdp, verbose=True)
                # else: # Primitive action
                reward, next_state = self.mdp.execute_agent_action(action)

                experience = state, action, reward, next_state
                experience_buffer.append(experience)
                state_buffer.append(state)

                if untrained_option.is_term_true(next_state) and len(experience_buffer) == self.buffer_length:
                    print("Entered the goal state for option {}.".format(untrained_option))

                    # If we hit a subgoal, modify the last experience to reflect the augmented reward
                    if untrained_option != goal_option:
                        experience_buffer[-1] = (state, action, reward + self.subgoal_reward, next_state)

                    untrained_option.num_goal_hits += 1
                    untrained_option.add_initiation_experience(state_buffer)
                    untrained_option.add_experience_buffer(experience_buffer)

                    if untrained_option.num_goal_hits >= self.num_goal_hits_before_training:
                        untrained_option = self._train_untrained_option(untrained_option)

                    reset_agent = True

                for trained_option in self._get_trained_options(new_option_just_created=reset_agent): # type: Option
                    trained_option.maybe_update_policy(experience)

                state = next_state

                # Reset the agent so that we don't keep moving around the initiation set of the trained option
                # TODO: Won't have to use reset_agent to break if I always take an option when I enter its initiation set
                # TODO: But what if that state has 2 options you can take from it?
                if reset_agent or state.is_out_of_frame() or state.is_terminal():
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
    return GridWorldMDP(10, 10, goal_predicate=predicate, slip_prob=0.2)

def contruct_lunar_lander_mdp():
    predicate = LunarLanderMDP.default_goal_predicate()
    return LunarLanderMDP(goal_predicate=predicate, render=True)

def construct_positional_lunar_lander_mdp():
    from simple_rl.tasks.lunar_lander.PositionalLunarLanderMDPClass import PositionalLunarLanderMDP
    predicate = PositionalLunarLanderMDP.default_goal_predicate()
    return PositionalLunarLanderMDP(goal_predicate=predicate, render=True)

if __name__ == '__main__':
    overall_mdp = construct_positional_lunar_lander_mdp()
    chainer = SkillChaining(overall_mdp, overall_mdp.goal_predicate)
    chainer.skill_chaining()