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
                 buffer_length=40, subgoal_reward=20.0, subgoal_hits=10):
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

        # TODO: Eventually need to add trained options to the DQN
        # if untrained_option not in self.mdp.actions:
        #     self.mdp.actions.append(untrained_option)

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
        if current_option:
            action, reward, next_state = current_option.execute_option_in_mdp(state, self.mdp)
            return state, action, reward, next_state

        action = self.global_solver.act(state.features(), self.global_solver.epsilon)
        reward, next_state = self.mdp.execute_agent_action(action)
        self.global_solver.step(state.features(), action, reward, next_state.features(), next_state.is_terminal())
        self.global_solver.update_epsilon()
        return state, action, reward, next_state

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

    def skill_chaining(self, num_episodes=600, num_steps=1000):
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
        last_100_scores = deque(maxlen=100)

        for episode in range(num_episodes):

            self.mdp.reset()
            score = 0.
            uo_episode_terminated = False
            state = deepcopy(self.mdp.init_state)
            experience_buffer = deque([], maxlen=self.buffer_length)
            state_buffer = deque([], maxlen=self.buffer_length)

            for _ in range(num_steps):
                current_option = self.find_option_for_state(state) # type: Option
                experience = self.take_action(state, current_option)

                experience_buffer.append(experience)
                state_buffer.append(state)

                # Update current_state = next_state
                state = self.get_next_state_from_experience(experience)
                score += self.get_reward_from_experience(experience)

                if untrained_option.is_term_true(state) and len(experience_buffer) == self.buffer_length and not uo_episode_terminated and len(self.trained_options) < 3:
                    uo_episode_terminated = True
                    untrained_option.num_goal_hits += 1
                    untrained_option.add_initiation_experience(state_buffer)
                    untrained_option.add_experience_buffer(experience_buffer)

                    if untrained_option.num_goal_hits >= self.num_goal_hits_before_training:
                        untrained_option = self._train_untrained_option(untrained_option)

                if state.is_out_of_frame() or state.is_terminal():
                    break

            last_100_scores.append(score)
            per_episode_scores.append(score)

            if self._log_dqn_status(episode, last_100_scores):
                break

        return per_episode_scores


    def _log_dqn_status(self, episode, last_100_scores):
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(last_100_scores)), end="")
        if episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(last_100_scores)))

        if np.mean(last_100_scores) >= 200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode - 100,
                                                                                         np.mean(last_100_scores)))
            return True

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
            # plot_replay_buffer_size(option)

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
    return LunarLanderMDP(goal_predicate=predicate, render=False)

if __name__ == '__main__':
    overall_mdp = construct_lunar_lander_mdp()
    environment = overall_mdp.env
    environment.seed(0) # TODO: Set this seed so that we can compare between runs
    solver = DQNAgent(environment.observation_space.shape[0], environment.action_space.n, 0)
    buffer_len = 40

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
        chainer = SkillChaining(overall_mdp, overall_mdp.goal_predicate, rl_agent=solver, buffer_length=buffer_len)
        episodic_scores = chainer.skill_chaining()
        chainer.save_all_dqns()
        chainer.save_all_initiation_classifiers()
        chainer.perform_experiments()
