# Python imports.
from __future__ import print_function

import sys
sys.path = [""] + sys.path

import matplotlib
matplotlib.use('TkAgg')
from collections import deque
from copy import deepcopy
import torch
import _pickle as pickle
import pdb
import argparse
import sys

# Other imports.
from simple_rl.mdp.StateClass import State
from simple_rl.abstraction.action_abs.PredicateClass import Predicate
from simple_rl.abstraction.action_abs.OptionClass import Option
from simple_rl.agents.func_approx.TorchDQNAgentClass import DQNAgent
from simple_rl.tasks.lunar_lander.LunarLanderMDPClass import LunarLanderMDP
from simple_rl.skill_chaining.skill_chaining_utils import *
from simple_rl.skill_chaining.create_pre_trained_options import *

class SkillChaining(object):
    def __init__(self, mdp, overall_goal_predicate, rl_agent, pretrained_options=[],
                 buffer_length=25, subgoal_reward=2000.0, subgoal_hits=3):
        """
        Args:
            mdp (MDP): Underlying domain we have to solve
            overall_goal_predicate (Predicate)
            rl_agent (DQNAgent): RL agent used to determine the policy for each option
            pretrained_options (list): options obtained from a previous run of the skill chaining algorithm
            buffer_length (int): size of the circular buffer used as experience buffer
            subgoal_reward (float): Hitting a subgoal must yield a supplementary reward to enable local policy
            subgoal_hits (int): number of times the RL agent has to hit the goal of an option o to learn its I_o, Beta_o
        """
        self.mdp = mdp
        self.original_actions = deepcopy(mdp.actions)
        self.overall_goal_predicate = overall_goal_predicate
        self.global_solver = rl_agent
        self.buffer_length = buffer_length
        self.subgoal_reward = subgoal_reward
        self.num_goal_hits_before_training = subgoal_hits

        self.trained_options = []

        # If we are given pretrained options, we will just use them as trained options
        if len(pretrained_options) > 0:
            self.trained_options = pretrained_options

    def _train_untrained_option(self, untrained_option):
        """
        Train the current untrained option and initialize a new one to target.
        Args:
            untrained_option (Option)

        Returns:
            new_untrained_option (Option)
        """
        print("\nTraining the initiation set and policy for {}.".format(untrained_option.name))
        # Train the initiation set classifier for the option
        untrained_option.train_initiation_classifier()

        # Update the solver of the untrained option on all the states in its experience
        untrained_option.initialize_option_policy()

        # Add the trained option to the action set of the global solver
        if untrained_option not in self.trained_options:
            self.trained_options.append(untrained_option)

        # Augment the global DQN with the newly trained option
        num_actions = len(self.mdp.actions) + len(self.trained_options)
        num_state_dimensions = self.mdp.init_state.state_space_size()
        new_global_agent = DQNAgent(num_state_dimensions, num_actions, len(self.original_actions), self.trained_options, seed=0, name=self.global_solver.name)
        new_global_agent.replay_buffer = self.global_solver.replay_buffer

        new_global_agent.policy_network.initialize_with_smaller_network(self.global_solver.policy_network)
        new_global_agent.target_network.initialize_with_smaller_network(self.global_solver.target_network)

        self.global_solver = new_global_agent

        # Update the global solver of all previously trained options
        for trained_option in self.trained_options:
            trained_option.global_solver = new_global_agent

        # Create new option whose termination is the initiation of the option we just trained
        name = "option_{}".format(str(len(self.trained_options)))

        print("Creating {}".format(name))

        # plot_initiation_examples(untrained_option)
        plot_one_class_initiation_classifier(untrained_option)

        # Using the global init_state as the init_state for all child options
        new_untrained_option = untrained_option.create_child_option(init_state=deepcopy(self.mdp.init_state),
                                                                actions=self.original_actions,
                                                                new_option_name=name,
                                                                global_solver=self.global_solver,
                                                                buffer_length=self.buffer_length,
                                                                num_subgoal_hits=self.num_goal_hits_before_training)
        return new_untrained_option

    def take_action(self, state, current_option):
        """

        Args:
            state (State)
            current_option (Option)

        Returns:
            experience (tuple): (s, a, r, s')
        """
        # if current_option:
        #     action, reward, next_state = current_option.execute_option_in_mdp(state, self.mdp)
        #     return state, action, reward, next_state

        action = self.global_solver.act(state.features(), self.global_solver.epsilon)
        option_chosen_action = action
        if self.mdp.is_primitive_action(action):
            reward, next_state = self.mdp.execute_agent_action(action)
        else: # Selected option
            option_idx = action - len(self.mdp.actions)
            selected_option = self.trained_options[option_idx] # type: Option
            # We are going to use the primitive action chosen by the option's DQN solver as our current experience
            # because this experience is used to initialize the policy of an untrained option
            option_chosen_action, reward, next_state = selected_option.execute_option_in_mdp(state, self.mdp)
        self.global_solver.step(state.features(), action, reward, next_state.features(), next_state.is_terminal())
        self.global_solver.update_epsilon()
        return state, option_chosen_action, reward, next_state

    def find_option_for_state(self, state):
        """
        If state is in the initiation set of a trained option, return that trained option
        Args:
            state (State)

        Returns:
            trained_option (Option)
        """
        for option in self.trained_options:
            if option.is_init_true(state):
                return option
        return None

    @staticmethod
    def get_next_state_from_experience(experience):
        assert isinstance(experience[-1], State), "Expected last element to be next_state, got {}".format(experience[-1])
        return experience[-1]

    @staticmethod
    def get_reward_from_experience(experience):
        assert isinstance(experience[2], float) or isinstance(experience[2], int), "Expected 3rd element to be reward (float), got {}".format(experience[2])
        return float(experience[2])

    def skill_chaining(self, num_episodes=250, num_steps=1000):
        from simple_rl.abstraction.action_abs.OptionClass import Option
        goal_option = Option(init_predicate=None, term_predicate=self.overall_goal_predicate, overall_mdp=self.mdp,
                             init_state=self.mdp.init_state, actions=self.original_actions, policy={},
                             name='overall_goal_policy', term_prob=0., global_solver=self.global_solver,
                             buffer_length=self.buffer_length, num_subgoal_hits_required=self.num_goal_hits_before_training)

        # Pointer to the current option:
        # 1. This option has the termination set which defines our current goal trigger
        # 2. This option has an untrained initialization set and policy, which we need to train from experience
        untrained_option = goal_option

        # For logging purposes
        per_episode_scores = []
        last_10_scores = deque(maxlen=10)

        for episode in range(num_episodes):

            self.mdp.reset()
            score = 0.
            uo_episode_terminated = False
            state = deepcopy(self.mdp.init_state)
            experience_buffer = deque([], maxlen=self.buffer_length)
            state_buffer = deque([], maxlen=self.buffer_length)

            # for _ in range(num_steps):
            while not state.is_terminal():
                current_option = self.find_option_for_state(state) # type: Option
                experience = self.take_action(state, current_option)

                experience_buffer.append(experience)
                state_buffer.append(state)

                # Update current_state = next_state
                state = self.get_next_state_from_experience(experience)
                score += self.get_reward_from_experience(experience)

                if untrained_option.is_term_true(state) and len(experience_buffer) == self.buffer_length and not uo_episode_terminated and len(self.trained_options) < 6:
                    uo_episode_terminated = True
                    untrained_option.num_goal_hits += 1
                    print("\nHit the termination condition of {} {} times so far".format(untrained_option, untrained_option.num_goal_hits))

                    # Augment the most recent experience with the subgoal reward
                    experience_buffer[-1] = (experience[0], experience[1], experience[2] + self.subgoal_reward, experience[3])
                    untrained_option.add_initiation_experience(state_buffer)
                    untrained_option.add_experience_buffer(experience_buffer)

                    if untrained_option.num_goal_hits >= self.num_goal_hits_before_training:
                        untrained_option = self._train_untrained_option(untrained_option)

                if state.is_out_of_frame() or state.is_terminal():
                    break

            last_10_scores.append(score)
            per_episode_scores.append(score)

            if self._log_dqn_status(episode, last_10_scores):
                break

        return per_episode_scores


    def _log_dqn_status(self, episode, last_10_scores):
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(last_10_scores)), end="")
        if episode % 10 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(last_10_scores)))

        return False

    def save_all_dqns(self):
        torch.save(self.global_solver.policy_network.state_dict(), 'global_dqn.pth')
        for option in self.trained_options: # type: Option
            torch.save(option.solver.policy_network.state_dict(), '{}_dqn.pth'.format(option.name))

    def save_all_initiation_classifiers(self):
        for option in self.trained_options:
            with open("{}_svm.pkl".format(option.name), "wb+") as _f:
                pickle.dump(option.initiation_classifier, _f)

    def perform_experiments(self):
        for option in self.trained_options:
            # plot_initiation_set(option)
            visualize_option_policy(option)
            visualize_option_starting_and_ending_points(option)
            plot_replay_buffer_size(option)
            visualize_replay_buffer(option)

    def trained_forward_pass(self, verbose=True):
        """
        Called when skill chaining has finished training: execute options when possible and then atomic actions
        Returns:
            overall_reward (float): score accumulated over the course of the episode.
            verbose (bool): if True, then will print out which option/action is being executed
        """
        self.mdp.reset()
        state = deepcopy(self.mdp.init_state)
        overall_reward = 0.
        self.mdp.render = True
        while not state.is_terminal():
            current_option = self.find_option_for_state(state)
            if current_option:
                action = current_option.solver.act(state.features(), eps=0.)
                if verbose: print("Taking {}".format(current_option.name))
            else:
                action = self.global_solver.act(state.features(), eps=0.)
                if verbose: print("Taking action {}".format(action))
            sys.stdout.flush()
            reward, next_state = self.mdp.execute_agent_action(action)
            overall_reward += reward
            state = next_state

        self.mdp.render = False
        print()

        # If it is a Gym environment, explicitly close it. RlPy domains don't need this
        if hasattr(self.mdp, "env"):
            self.mdp.env.close()

        return overall_reward


def construct_lunar_lander_mdp():
    predicate = LunarLanderMDP.default_goal_predicate()
    mdp = LunarLanderMDP(goal_predicate=predicate, render=False)
    mdp.env.seed(0)
    return mdp

def construct_pinball_mdp():
    from simple_rl.tasks.pinball.PinballMDPClass import PinballMDP
    mdp = PinballMDP(noise=0., episode_length=1000, render=False)
    return mdp

if __name__ == '__main__':
    overall_mdp = construct_pinball_mdp()
    state_space_size = overall_mdp.init_state.state_space_size()
    solver = DQNAgent(state_space_size, len(overall_mdp.actions), len(overall_mdp.actions), [], seed=0, name="GlobalDQN")
    buffer_len = 30

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=bool, help="whether or not to load pretrained options", default=False)
    args = parser.parse_args()

    if args.pretrained:
        loader = PretrainedOptionsLoader(overall_mdp, solver, buffer_length=buffer_len)
        pretrained_options = loader.get_pretrained_options()
        print("Running skill chaining with pretrained options: {}".format(pretrained_options))
        chainer = SkillChaining(overall_mdp, overall_mdp.goal_predicate, rl_agent=solver, buffer_length=buffer_len,
                                pretrained_options=pretrained_options)
        episodic_scores = chainer.skill_chaining()
    else:
        print("Training skill chaining agent from scratch with a buffer length of ", buffer_len)
        print("MDP InitState = ", overall_mdp.init_state)
        print("MDP GoalPosition = ", overall_mdp.domain.environment.target_pos)
        chainer = SkillChaining(overall_mdp, overall_mdp.goal_predicate, rl_agent=solver, buffer_length=buffer_len)
        episodic_scores = chainer.skill_chaining()
        chainer.save_all_dqns()
        chainer.save_all_initiation_classifiers()
        chainer.perform_experiments()
