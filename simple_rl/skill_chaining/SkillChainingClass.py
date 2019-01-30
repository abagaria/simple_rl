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
                 buffer_length=25, subgoal_reward=5000.0, subgoal_hits=3, max_num_options=4, lr_decay=False,
                 enable_option_timeout=False, option_subgoal_reward_ratio=0.5, seed=0):
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
            lr_decay (bool): Whether or not to decay the global solver learning rate over time
            enable_option_timeout (bool): whether or not the option times out after some number of steps
            option_subgoal_reward_ratio (float): ratio of subgoal_reward for options to fall off a cliff
            seed (int): We are going to use the same random seed for all the DQN solvers
        """
        self.mdp = mdp
        self.original_actions = deepcopy(mdp.actions)
        self.overall_goal_predicate = overall_goal_predicate
        self.global_solver = rl_agent
        self.buffer_length = buffer_length
        self.subgoal_reward = subgoal_reward
        self.num_goal_hits_before_training = subgoal_hits
        self.max_num_options = max_num_options
        self.lr_decay = lr_decay
        self.enable_option_timeout = enable_option_timeout
        self.option_subgoal_reward_ratio = option_subgoal_reward_ratio
        self.seed = seed

        self.trained_options = []

        # If we are given pretrained options, we will just use them as trained options
        if len(pretrained_options) > 0:
            self.trained_options = pretrained_options

        self.validation_scores = []

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
        new_global_agent = DQNAgent(num_state_dimensions, num_actions, len(self.original_actions), self.trained_options,
                                    seed=self.seed, name=self.global_solver.name, eps_start=self.global_solver.epsilon,
                                    tensor_log=self.global_solver.tensor_log, use_double_dqn=self.global_solver.use_ddqn,
                                    lr=self.global_solver.learning_rate)
        new_global_agent.replay_buffer = self.global_solver.replay_buffer

        new_global_agent.policy_network.initialize_with_smaller_network(self.global_solver.policy_network)
        new_global_agent.target_network.initialize_with_smaller_network(self.global_solver.target_network)
        # new_global_agent.initialize_optimizer_with_smaller_agent(self.global_solver)

        # pdb.set_trace()

        self.global_solver = new_global_agent

        # Update the global solver of all previously trained options
        for trained_option in self.trained_options:
            trained_option.global_solver = new_global_agent

    def make_off_policy_updates_for_options(self, state, action, reward, next_state):
        for option in self.trained_options: # type: Option
            option.update_option_solver(state, action, reward, next_state)

    def take_action(self, state, step_number, episode_option_executions):
        """
        Either take a primitive action from `state` or execute a closed-loop option policy.
        Args:
            state (State)
            step_number (int): which iteration of the control loop we are on
            episode_option_executions (defaultdict)

        Returns:
            experiences (list): list of (s, a, r, s') tuples
            reward (float): sum of all rewards accumulated while executing chosen action
            next_state (State): state we landed in after executing chosen action
        """
        # Query the global Q-function to determine optimal action from current state
        action = self.global_solver.act(state.features(), train_mode=True)

        if self.mdp.is_primitive_action(action):
            reward, next_state = self.mdp.execute_agent_action(action)
            self.make_off_policy_updates_for_options(state, action, reward, next_state)
            assert self.mdp.is_primitive_action(action), "Expected primitive action, got {}".format(action)
            self.global_solver.step(state.features(), action, reward, next_state.features(), next_state.is_terminal(), num_steps=1)
            self.global_solver.update_epsilon()

            self.global_execution_states.append(state)
            return [(state, action, reward, next_state)], reward, next_state, 1

        # Selected option
        option_idx = action - len(self.mdp.actions)
        selected_option = self.trained_options[option_idx] # type: Option
        option_transitions, discounted_reward = selected_option.execute_option_in_mdp(self.mdp, step_number)

        option_reward = self.get_reward_from_experiences(option_transitions)
        next_state = self.get_next_state_from_experiences(option_transitions)

        # Add data to train Q(s, o)
        assert not self.mdp.is_primitive_action(action), "Expected an option, got {}".format(action)
        self.global_solver.step(state.features(), action, discounted_reward, next_state.features(), next_state.is_terminal(), num_steps=len(option_transitions))

        # Debug logging
        episode_option_executions[selected_option.name] += 1
        self.option_rewards[selected_option.name].append(discounted_reward)

        if not selected_option.pretrained:
            sampled_q_value = self.sample_qvalue(selected_option)
            self.option_qvalues[selected_option.name].append(sampled_q_value)
            if self.global_solver.tensor_log:
                self.global_solver.writer.add_scalar("{}_q_value".format(selected_option.name), sampled_q_value, selected_option.total_executions)

        return option_transitions, option_reward, next_state, len(option_transitions)

    def sample_qvalue(self, option):
        option_idx = self.trained_options.index(option)
        action_idx = len(self.mdp.actions) + option_idx
        sample_state = option.experience_buffer[-1][5].state
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

    def skill_chaining(self, num_episodes=251, num_steps=20000):
        from simple_rl.abstraction.action_abs.OptionClass import Option
        goal_option = Option(init_predicate=None, term_predicate=self.overall_goal_predicate, overall_mdp=self.mdp,
                             init_state=self.mdp.init_state, actions=self.original_actions, policy={},
                             name='overall_goal_policy', term_prob=0., global_solver=self.global_solver,
                             buffer_length=self.buffer_length,
                             num_subgoal_hits_required=self.num_goal_hits_before_training,
                             subgoal_reward=self.subgoal_reward, seed=self.seed, max_steps=num_steps, parent=None,
                             classifier_type="ocsvm", enable_timeout=self.enable_option_timeout,
                             negative_reward_ratio=self.option_subgoal_reward_ratio)

        # Pointer to the current option:
        # 1. This option has the termination set which defines our current goal trigger
        # 2. This option has an untrained initialization set and policy, which we need to train from experience
        untrained_option = goal_option

        # For logging purposes
        per_episode_scores = []
        per_episode_durations = []
        last_10_scores = deque(maxlen=10)
        last_10_durations = deque(maxlen=10)

        for episode in range(num_episodes):

            self.mdp.reset()
            score = 0.
            step_number = 0
            uo_episode_terminated = False
            state = deepcopy(self.mdp.init_state)
            experience_buffer = deque([], maxlen=self.buffer_length)
            state_buffer = deque([], maxlen=self.buffer_length)
            episode_option_executions = defaultdict(lambda : 0)

            # if self.lr_decay and (episode > 0) and (episode % 40 == 0):
            if self.lr_decay and (episode > 0 and episode % 40 == 0):
                self.global_solver.reduce_learning_rate()
                print("\rSetting new learning rate to {}\n\n".format(self.global_solver.learning_rate))

            while step_number < num_steps:
                experiences, reward, state, steps = self.take_action(state, step_number, episode_option_executions)
                score += reward
                step_number += steps
                for experience in experiences:
                    experience_buffer.append(experience)
                    state_buffer.append(experience[0])
                state_buffer.append(experiences[-1][-1]) # Don't forget to add the last s' to the buffer

                if untrained_option.is_term_true(state) and len(experience_buffer) == self.buffer_length and \
                        not uo_episode_terminated and self.should_create_more_options():
                    uo_episode_terminated = True

                    if untrained_option.train(experience_buffer, state_buffer):
                        if not self.global_solver.tensor_log: render_value_function(self.global_solver, torch.device("cuda"), episode=episode-1000)
                        if not self.global_solver.tensor_log: plot_one_class_initiation_classifier(untrained_option)
                        self._augment_agent_with_new_option(untrained_option)
                        if not self.global_solver.tensor_log: render_value_function(self.global_solver, torch.device("cuda"), episode=episode+1000)
                        new_untrained_option = untrained_option.get_child_option(len(self.trained_options))
                        untrained_option = new_untrained_option
                        
                if state.is_out_of_frame() or state.is_terminal():
                    break

            last_10_scores.append(score)
            last_10_durations.append(step_number)
            per_episode_scores.append(score)
            per_episode_durations.append(step_number)

            if self._log_dqn_status(episode, last_10_scores, episode_option_executions, last_10_durations):
                break

        return per_episode_scores, per_episode_durations

    def _log_dqn_status(self, episode, last_10_scores, episode_option_executions, last_10_durations):
        print('\rEpisode {}\tAverage Score: {:.2f}\tDuration: {:.2f} steps'.format(episode, np.mean(last_10_scores), np.mean(last_10_durations)), end="")
        if episode % 10 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tDuration: {:.2f} steps'.format(episode, np.mean(last_10_scores), np.mean(last_10_durations)))
        if episode > 0 and episode % 5 == 0:
            eval_score = self.trained_forward_pass(verbose=False, render=False)
            self.validation_scores.append(eval_score)
            print("\rEpisode {}\tValidation Score: {:.2f}".format(episode, eval_score))

        if not self.global_solver.tensor_log and episode % 5 == 0:
            render_value_function(self.global_solver, torch.device("cuda"), episode=episode)
        for trained_option in self.trained_options:  # type: Option
            self.num_option_executions[trained_option.name].append(episode_option_executions[trained_option.name])
            if self.global_solver.tensor_log:
                self.global_solver.writer.add_scalar("{}_executions".format(trained_option.name), episode_option_executions[trained_option.name], episode)
            # if not self.global_solver.tensor_log and episode % 5 == 0:
            #     render_value_function(trained_option.solver, torch.device("cuda"), episode=episode)
        return False

    def save_all_dqns(self):
        torch.save(self.global_solver.policy_network.state_dict(), 'global_policy_dqn.pth')
        torch.save(self.global_solver.target_network.state_dict(), 'global_target_dqn.pth')
        for option in self.trained_options: # type: Option
            torch.save(option.solver.policy_network.state_dict(), '{}_policy_dqn.pth'.format(option.name))
            torch.save(option.solver.target_network.state_dict(), '{}_target_dqn.pth'.format(option.name))

    def save_all_initiation_classifiers(self):
        for option in self.trained_options:
            with open("{}_svm.pkl".format(option.name), "wb+") as _f:
                pickle.dump(option.initiation_classifier, _f)

    def save_all_scores(self, pretrained, scores, durations):
        print("\rSaving training and validation scores..")
        with open("sc_pretrained_{}_training_scores.pkl".format(pretrained), "wb+") as _f:
            pickle.dump(scores, _f)

        with open("sc_pretrained_{}_training_durations.pkl".format(pretrained), "wb+") as _f:
            pickle.dump(durations, _f)

        with open("sc_pretrained_{}_validation_scores.pkl".format(pretrained), "wb+") as _f:
            pickle.dump(self.validation_scores, _f)

    def perform_experiments(self):
        for option in self.trained_options:
            visualize_option_replay_buffer(option)
            # plot_one_class_initiation_classifier(option)
            # visualize_option_policy(option)
            # visualize_option_starting_and_ending_points(option)
            # plot_replay_buffer_size(option)
            # visualize_replay_buffer(option)
            # visualize_global_dqn_execution_points(self.global_execution_states)

    def trained_forward_pass(self, verbose=True, max_num_steps=5000, render=True):
        """
        Called when skill chaining has finished training: execute options when possible and then atomic actions
        Returns:
            overall_reward (float): score accumulated over the course of the episode.
            verbose (bool): if True, then will print out which option/action is being executed
        """
        was_rendering = deepcopy(self.mdp.render)
        self.mdp.reset()
        state = deepcopy(self.mdp.init_state)
        overall_reward = 0.
        self.mdp.render = render
        num_steps = 0

        while not state.is_terminal() and num_steps < max_num_steps:
            action = self.global_solver.act(state.features(), train_mode=False)

            sys.stdout.flush()
            if self.mdp.is_primitive_action(action):
                if verbose: print("Taking action {}".format(action))
                reward, next_state = self.mdp.execute_agent_action(action)
                overall_reward += reward
            else:
                option_idx = action - len(self.mdp.actions)
                selected_option = self.trained_options[option_idx]  # type: Option
                if verbose: print("Taking {}".format(selected_option))
                option_reward, next_state, num_steps = selected_option.trained_option_execution(self.mdp, num_steps, max_num_steps)
                overall_reward += option_reward

            state = next_state
            num_steps += 1

        if not was_rendering:
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
    mdp = PinballMDP(noise=0.0, episode_length=20000, reward_scale=100., render=True)
    return mdp

if __name__ == '__main__':
    overall_mdp = construct_pinball_mdp()
    state_space_size = overall_mdp.init_state.state_space_size()
    random_seed = 4351
    buffer_len = 20
    sub_reward = 1.
    lr = 1e-4
    max_number_of_options = 3
    NUM_STEPS_PER_EPISODE = 20000
    solver = DQNAgent(state_space_size, len(overall_mdp.actions), len(overall_mdp.actions), [], seed=random_seed, lr=lr,
                      name="GlobalDQN", eps_start=1.0, tensor_log=False, use_double_dqn=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=bool, help="whether or not to load pretrained options", default=False)
    args = parser.parse_args()

    if args.pretrained:
        loader = PretrainedOptionsLoader(overall_mdp, solver, buffer_len, num_subgoal_hits_required=3, subgoal_reward=sub_reward,
                                         max_steps=NUM_STEPS_PER_EPISODE, seed=random_seed)
        pretrained_options = loader.get_pretrained_options()
        for pretrained_option in pretrained_options:
            solver = loader.augment_global_agent_with_option(solver, pretrained_option)
        print("Running skill chaining with pretrained options: {}".format(pretrained_options))
        chainer = SkillChaining(overall_mdp, overall_mdp.goal_predicate, rl_agent=solver, buffer_length=buffer_len,
                                seed=random_seed, subgoal_reward=sub_reward, max_num_options=0,
                                lr_decay=False, pretrained_options=pretrained_options)
        pretrained_episodic_scores, pretrained_episodic_durations = chainer.skill_chaining()
        chainer.save_all_scores(args.pretrained, pretrained_episodic_scores, pretrained_episodic_durations)
    else:
        print("Training skill chaining agent from scratch with a buffer length of {} and subgoal reward {}".format(buffer_len, sub_reward))
        print("MDP InitState = ", overall_mdp.init_state)
        print("MDP GoalPosition = ", overall_mdp.domain.environment.target_pos)
        chainer = SkillChaining(overall_mdp, overall_mdp.goal_predicate, rl_agent=solver, buffer_length=buffer_len,
                                seed=random_seed, subgoal_reward=sub_reward, max_num_options=max_number_of_options,
                                lr_decay=False)
        episodic_scores, episodic_durations = chainer.skill_chaining()
        chainer.save_all_dqns()
        chainer.save_all_initiation_classifiers()
        chainer.perform_experiments()
        chainer.save_all_scores(args.pretrained, episodic_scores, episodic_durations)
