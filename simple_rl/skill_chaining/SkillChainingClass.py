# Python imports.
from __future__ import print_function

import sys
sys.path = [""] + sys.path

import matplotlib
matplotlib.use('TkAgg')
from collections import deque, defaultdict
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
                 buffer_length=25, subgoal_reward=5000.0, subgoal_hits=3, max_num_options=4):
        """
        Args:
            mdp (MDP): Underlying domain we have to solve
            overall_goal_predicate (Predicate)
            rl_agent (DQNAgent): RL agent used to determine the policy for each option
            pretrained_options (list): options obtained from a previous run of the skill chaining algorithm
            buffer_length (int): size of the circular buffer used as experience buffer
            subgoal_reward (float): Hitting a subgoal must yield a supplementary reward to enable local policy
            subgoal_hits (int): number of times the RL agent has to hit the goal of an option o to learn its I_o, Beta_o
            max_num_options (int): Maximum number of options that the skill chaining agent can create
        """
        self.mdp = mdp
        self.original_actions = deepcopy(mdp.actions)
        self.overall_goal_predicate = overall_goal_predicate
        self.global_solver = rl_agent
        self.buffer_length = buffer_length
        self.subgoal_reward = subgoal_reward
        self.num_goal_hits_before_training = subgoal_hits
        self.max_num_options = max_num_options

        self.trained_options = []

        # If we are given pretrained options, we will just use them as trained options
        if len(pretrained_options) > 0:
            self.trained_options = pretrained_options

        # Debug variables
        self.global_execution_states = []
        self.num_option_executions = defaultdict(lambda : [])
        self.option_rewards = defaultdict(lambda : [])
        self.option_qvalues = defaultdict(lambda : [])

    def _augment_agent_with_new_option(self, newly_trained_option):
        """
        Train the current untrained option and initialize a new one to target.
        Args:
            newly_trained_option (Option)
        """
        # Add the trained option to the action set of the global solver
        if newly_trained_option not in self.trained_options:
            self.trained_options.append(newly_trained_option)

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

    def make_off_policy_updates_for_options(self, state, action, reward, next_state):
        for option in self.trained_options:
            if option.is_term_true(next_state):
                option.solver.step(state.features(), action, reward + self.subgoal_reward, next_state.features(), next_state.is_terminal())
            elif option.is_init_true(state):
                option.solver.step(state.features(), action, reward, next_state.features(), next_state.is_terminal())

    def take_action(self, state, episode_option_executions):
        """
        Either take a primitive action from `state` or execute a closed-loop option policy.
        Args:
            state (State)
            episode_option_executions (defaultdict)

        Returns:
            experiences (list): list of (s, a, r, s') tuples
            reward (float): sum of all rewards accumulated while executing chosen action
            next_state (State): state we landed in after executing chosen action
        """
        # Query the global Q-function to determine optimal action from current state
        action = self.global_solver.act(state.features(), self.global_solver.epsilon)

        if self.mdp.is_primitive_action(action):
            reward, next_state = self.mdp.execute_agent_action(action)
            # self.make_off_policy_updates_for_options(state, action, reward, next_state)
            self.global_solver.step(state.features(), action, reward, next_state.features(), next_state.is_terminal())
            self.global_solver.update_epsilon()

            self.global_execution_states.append(state)
            return [(state, action, reward, next_state)], reward, next_state

        # Selected option
        option_idx = action - len(self.mdp.actions)
        selected_option = self.trained_options[option_idx] # type: Option
        option_transitions = selected_option.execute_option_in_mdp(state, self.mdp)

        option_reward = self.get_reward_from_experiences(option_transitions)
        next_state = self.get_next_state_from_experiences(option_transitions)
        # augmented_reward = max(-1, option_reward + (selected_option.is_term_true(next_state) * self.subgoal_reward))

        # Add data to train Q(s, o)
        self.global_solver.step(state.features(), action, option_reward, next_state.features(), next_state.is_terminal())

        # Debug logging
        episode_option_executions[selected_option.name] += 1
        self.option_rewards[selected_option.name].append(option_reward)
        self.option_qvalues[selected_option.name].append(self.sample_qvalue(selected_option))

        return option_transitions, option_reward, next_state

    def sample_qvalue(self, option):
        option_idx = self.trained_options.index(option)
        action_idx = len(self.mdp.actions) + option_idx
        sample_state = option.experience_buffer[5, -1].state
        return self.global_solver.get_qvalue(sample_state.features(), action_idx)

    @staticmethod
    def get_next_state_from_experiences(experiences):
        """
        Given a list of experiences, fetch the final state encountered.
        Args:
            experiences (list): list of (s, a, r, s') tuples

        Returns:
            next_state (State)
        """
        assert isinstance(experiences[-1][-1], State), "Expected last element to be next_state, got {}".format(experiences[-1][-1])
        return experiences[-1][-1]

    @staticmethod
    def get_reward_from_experiences(experiences):
        """
        Given a list of experiences, fetch the overall reward encountered.
        Args:
            experiences (list): list of (s, a, r, s') tuples

        Returns:
            total_reward (float): Sum of all the rewards encountered in `experiences`
        """
        total_reward = 0.
        for experience in experiences:
            reward = experience[2]
            assert isinstance(reward, float) or isinstance(reward, int), "Expected {} to be a float/int".format(reward)
            total_reward += reward
        return total_reward

    def should_create_more_options(self):
        start_state = deepcopy(self.mdp.init_state)
        for option in self.trained_options: # type: Option
            if option.is_init_true(start_state):
                return False
        return len(self.trained_options) < self.max_num_options

    def skill_chaining(self, num_episodes=120, num_steps=100000):
        from simple_rl.abstraction.action_abs.OptionClass import Option
        goal_option = Option(init_predicate=None, term_predicate=self.overall_goal_predicate, overall_mdp=self.mdp,
                             init_state=self.mdp.init_state, actions=self.original_actions, policy={},
                             name='overall_goal_policy', term_prob=0., global_solver=self.global_solver,
                             buffer_length=self.buffer_length,
                             num_subgoal_hits_required=self.num_goal_hits_before_training,
                             subgoal_reward=self.subgoal_reward)

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
            episode_option_executions = defaultdict(lambda : 0)

            for _ in range(num_steps):
                experiences, reward, state = self.take_action(state, episode_option_executions)
                score += reward
                for experience in experiences:
                    experience_buffer.append(experience)
                    state_buffer.append(experience[0])

                if untrained_option.is_term_true(state) and len(experience_buffer) == self.buffer_length and not uo_episode_terminated and self.should_create_more_options():
                    uo_episode_terminated = True

                    if untrained_option.train(experience_buffer, state_buffer):
                        plot_one_class_initiation_classifier(untrained_option)
                        self._augment_agent_with_new_option(untrained_option)
                        new_untrained_option = untrained_option.get_child_option(len(self.trained_options))
                        untrained_option = new_untrained_option
                        
                if state.is_out_of_frame() or state.is_terminal():
                    break

            last_10_scores.append(score)
            per_episode_scores.append(score)

            if self._log_dqn_status(episode, last_10_scores, episode_option_executions):
                break

        return per_episode_scores

    def _log_dqn_status(self, episode, last_10_scores, episode_option_executions):
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(last_10_scores)), end="")
        if episode % 10 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(last_10_scores)))
        for trained_option in self.trained_options:  # type: Option
            self.num_option_executions[trained_option.name].append(episode_option_executions[trained_option.name])
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
            visualize_global_dqn_execution_points(self.global_execution_states)

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
            action = self.global_solver.act(state.features(), eps=0.)

            sys.stdout.flush()
            if self.mdp.is_primitive_action(action):
                if verbose: print("Taking action {}".format(action))
                reward, next_state = self.mdp.execute_agent_action(action)
                overall_reward += reward
            else:
                option_idx = action - len(self.mdp.actions)
                selected_option = self.trained_options[option_idx]  # type: Option
                if verbose: print("Taking {}".format(selected_option))
                option_reward, next_state = selected_option.trained_option_execution(state, self.mdp)
                overall_reward += option_reward

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
    mdp = PinballMDP(noise=0.0, episode_length=1000, render=True)
    return mdp

if __name__ == '__main__':
    overall_mdp = construct_pinball_mdp()
    state_space_size = overall_mdp.init_state.state_space_size()
    solver = DQNAgent(state_space_size, len(overall_mdp.actions), len(overall_mdp.actions), [], seed=0, name="GlobalDQN")
    buffer_len = 20

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
