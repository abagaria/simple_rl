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
import os

# Other imports.
from simple_rl.mdp.StateClass import State
from simple_rl.abstraction.action_abs.OptionClass import Option
from simple_rl.agents.func_approx.TorchDQNAgentClass import DQNAgent
from simple_rl.tasks.lunar_lander.LunarLanderMDPClass import LunarLanderMDP
from simple_rl.skill_chaining.skill_chaining_utils import *
from simple_rl.skill_chaining.create_pre_trained_options import *

class SkillChaining(object):
    def __init__(self, mdp, rl_agent, pretrained_options=[], buffer_length=25,
                 subgoal_reward=5000.0, subgoal_hits=5, max_num_options=4, lr_decay=False,
                 enable_option_timeout=True, intra_option_learning=False, generate_plots=False, log_dir="", seed=0):
        """
        Args:
            mdp (MDP): Underlying domain we have to solve
            rl_agent (DQNAgent): RL agent used to determine the policy for each option
            pretrained_options (list): options obtained from a previous run of the skill chaining algorithm
            buffer_length (int): size of the circular buffer used as experience buffer
            subgoal_reward (float): Hitting a subgoal must yield a supplementary reward to enable local policy
            subgoal_hits (int): number of times the RL agent has to hit the goal of an option o to learn its I_o, Beta_o
            max_num_options (int): Maximum number of options that the skill chaining agent can create
            lr_decay (bool): Whether or not to decay the global solver learning rate over time
            enable_option_timeout (bool): whether or not the option times out after some number of steps
            intra_option_learning (bool): whether or not to use intra-option learning while making SMDP updates
            generate_plots (bool): whether or not to produce plots in this run
            log_dir (os.path): directory to store all the scores for this run  
            seed (int): We are going to use the same random seed for all the DQN solvers
        """
        self.mdp = mdp
        self.original_actions = deepcopy(mdp.actions)
        self.global_solver = rl_agent
        self.buffer_length = buffer_length
        self.subgoal_reward = subgoal_reward
        self.num_goal_hits_before_training = subgoal_hits
        self.max_num_options = max_num_options
        self.lr_decay = lr_decay
        self.enable_option_timeout = enable_option_timeout
        self.enable_intra_option_learning = intra_option_learning
        self.generate_plots = generate_plots
        self.log_dir = log_dir
        self.seed = seed

        np.random.seed(seed)

        self.trained_options = []

        # If we are given pretrained options, we will just use them as trained options
        if len(pretrained_options) > 0:
            self.trained_options = pretrained_options

        self.validation_scores = []

        goal_option = Option(overall_mdp=self.mdp, name='overall_goal_policy', global_solver=self.global_solver,
                             buffer_length=self.buffer_length,
                             num_subgoal_hits_required=self.num_goal_hits_before_training,
                             subgoal_reward=self.subgoal_reward, seed=self.seed, max_steps=20000,
                             classifier_type="ocsvm", enable_timeout=self.enable_option_timeout)

        # Pointer to the current option:
        # 1. This option has the termination set which defines our current goal trigger
        # 2. This option has an untrained initialization set and policy, which we need to train from experience
        self.untrained_option = goal_option

        # Debug variables
        self.global_execution_states = []
        self.num_option_executions = defaultdict(lambda : [])
        self.option_rewards = defaultdict(lambda : [])
        self.option_qvalues = defaultdict(lambda : [])

    def create_child_option(self):
        # Create new option whose termination is the initiation of the option we just trained
        name = "option_{}".format(str(len(self.trained_options)))
        print("Creating {}".format(name))

        old_untrained_option_id = id(self.untrained_option)
        new_untrained_option = Option(self.mdp, name=name, global_solver=self.global_solver, buffer_length=self.buffer_length,
                                      num_subgoal_hits_required=self.num_goal_hits_before_training, subgoal_reward=self.subgoal_reward,
                                      seed=self.seed, parent=self.untrained_option, enable_timeout=True)

        self.untrained_option.children.append(new_untrained_option)
        self.untrained_option = new_untrained_option

        new_untrained_option_id = id(self.untrained_option)
        assert new_untrained_option_id != old_untrained_option_id, "Checking python references"
        assert id(self.untrained_option.parent) == old_untrained_option_id, "Checking python references"

    def get_init_q_value_for_new_option(self, newly_trained_option):
        """
        Sample the Q-values of transitions that triggered the option’s target event during its gestation,
        and initialize Q(s, o) to the max of these values.
        Args:
            newly_trained_option (Option)

        Returns:
            init_q_value (float)
        """
        final_transitions = newly_trained_option.get_final_transitions()
        final_state_action_pairs = [(experience.state, experience.action) for experience in final_transitions]
        q_values = [self.global_solver.get_qvalue(state.features(), action).item() for state, action in final_state_action_pairs]
        return np.max(q_values)

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

        init_q_value = self.get_init_q_value_for_new_option(newly_trained_option)
        new_global_agent.policy_network.initialize_with_smaller_network(self.global_solver.policy_network, init_q_value)
        new_global_agent.target_network.initialize_with_smaller_network(self.global_solver.target_network, init_q_value)

        self.global_solver = new_global_agent

        # Update the global solver of all previously trained options
        for trained_option in self.trained_options:
            trained_option.global_solver = new_global_agent

    def make_off_policy_updates_for_options(self, state, action, reward, next_state):
        for option in self.trained_options: # type: Option
            option.off_policy_update(state, action, reward, next_state)

    def make_smdp_update(self, state, action, total_discounted_reward, next_state, option_transitions):
        """
        Use Intra-Option Learning for sample efficient learning of the option-value function.
        Args:
            state (State): state from which we started option execution
            action (int): option taken by the global solver
            total_discounted_reward (float): cumulative reward from the overall SMDP update
            next_state (State): state we landed in after executing the option
            option_transitions (list): list of (s, a, r, s') tuples representing the trajectory during option execution
        """
        assert not self.mdp.is_primitive_action(action), "Expected an option, got {}".format(action)

        # TODO: Should we do intra-option learning only when the option was successful in reaching its subgoal?
        # TODO: (since the option could have "failed" simply because it timed out)
        # TODO: During execution, the option could have gone outside its initiation set, we shouldn't make updates for
        # TODO: those transitions
        if self.enable_intra_option_learning:
            def get_reward(transitions):
                gamma = self.global_solver.gamma
                raw_rewards = [tt[2] for tt in transitions]
                return sum([ (gamma ** idx) * rr for idx, rr in enumerate(raw_rewards)])
            for i, transition in enumerate(option_transitions):
                sub_transitions = option_transitions[i:]
                start_state = transition[0]
                discounted_reward = get_reward(sub_transitions)
                self.global_solver.step(start_state.features(), action, discounted_reward, next_state.features(),
                                        next_state.is_terminal(), num_steps=len(sub_transitions))
        else:
            self.global_solver.step(state.features(), action, total_discounted_reward, next_state.features(),
                                    next_state.is_terminal(), num_steps=len(option_transitions))

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
        self.make_smdp_update(state, action, discounted_reward, next_state, option_transitions)

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

    def number_of_states_in_term_set(self, untrained_option, trajectory):
        return sum([untrained_option.is_term_true(state) for state in trajectory])

    def skill_chaining(self, num_episodes, num_steps):

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

            while step_number < num_steps:
                experiences, reward, state, steps = self.take_action(state, step_number, episode_option_executions)
                score += reward
                step_number += steps
                for experience in experiences:
                    experience_buffer.append(experience)
                    state_buffer.append(experience[0])
                state_buffer.append(experiences[-1][-1]) # Don't forget to add the last s' to the buffer

                if self.untrained_option.is_term_true(state) and \
                    self.number_of_states_in_term_set(self.untrained_option, state_buffer) < (self.buffer_length // 4) and \
                    len(experience_buffer) == self.buffer_length and \
                    not uo_episode_terminated and self.untrained_option.get_training_phase() == "gestation":

                    uo_episode_terminated = True

                    if self.untrained_option.train(experience_buffer, state_buffer):
                        if self.generate_plots and not self.global_solver.tensor_log:
                            render_value_function(self.global_solver, torch.device("cuda"), episode=episode-1000)
                            plot_one_class_initiation_classifier(self.untrained_option)
                        self._augment_agent_with_new_option(self.untrained_option)
                        if self.generate_plots and not self.global_solver.tensor_log:
                            render_value_function(self.global_solver, torch.device("cuda"), episode=episode+1000)

                if self.untrained_option.get_training_phase() == "initiation_done" and self.should_create_more_options():
                    self.create_child_option()
                        
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
        print('\rEpisode {}\tAverage Score: {:.2f}\tDuration: {:.2f} steps\tEpsilon: {:.2f}'.format(episode, np.mean(last_10_scores), np.mean(last_10_durations), self.global_solver.epsilon), end="")
        if episode % 10 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tDuration: {:.2f} steps\tEpsilon: {:.2f}'.format(episode, np.mean(last_10_scores), np.mean(last_10_durations), self.global_solver.epsilon))
        if episode > 0 and episode % 100 == 0:
            eval_score = self.trained_forward_pass(verbose=False, render=False)
            self.validation_scores.append(eval_score)
            print("\rEpisode {}\tValidation Score: {:.2f}".format(episode, eval_score))

        if self.generate_plots and not self.global_solver.tensor_log and episode % 100 == 0:
            render_value_function(self.global_solver, torch.device("cuda"), episode=episode)
        for trained_option in self.trained_options:  # type: Option
            self.num_option_executions[trained_option.name].append(episode_option_executions[trained_option.name])
            if self.global_solver.tensor_log:
                self.global_solver.writer.add_scalar("{}_executions".format(trained_option.name), episode_option_executions[trained_option.name], episode)
            # if not self.global_solver.tensor_log and episode % 100 == 0:
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
        training_scores_file_name = "sc_pretrained_{}_training_scores_{}.pkl".format(pretrained, self.seed)
        training_durations_file_name = "sc_pretrained_{}_training_durations_{}.pkl".format(pretrained, self.seed)
        validation_scores_file_name = "sc_pretrained_{}_validation_scores_{}.pkl".format(pretrained, self.seed)

        if self.log_dir:
            training_scores_file_name = os.path.join(self.log_dir, training_scores_file_name)
            training_durations_file_name = os.path.join(self.log_dir, training_durations_file_name)
            validation_scores_file_name = os.path.join(self.log_dir, validation_scores_file_name)

        with open(training_scores_file_name, "wb+") as _f:
            pickle.dump(scores, _f)
        with open(training_durations_file_name, "wb+") as _f:
            pickle.dump(durations, _f)
        with open(validation_scores_file_name, "wb+") as _f:
            pickle.dump(self.validation_scores, _f)

    def perform_experiments(self):
        for option in self.trained_options:
            visualize_dqn_replay_buffer(option.solver)
            # plot_one_class_initiation_classifier(option)
            # visualize_option_policy(option)
            # visualize_option_starting_and_ending_points(option)
            # plot_replay_buffer_size(option)
            # visualize_replay_buffer(option)
            # visualize_global_dqn_execution_points(self.global_execution_states)
        visualize_dqn_replay_buffer(self.global_solver)
        visualize_smdp_updates(self.global_solver, self.mdp)

        for i, o in enumerate(self.trained_options):
            plt.subplot(1, len(self.trained_options), i + 1)
            plt.plot(self.option_qvalues[o.name])
            plt.title(o.name)
        plt.savefig("sampled_q_so.png")
        plt.close()

    def trained_forward_pass(self, verbose=True, max_num_steps=2500, render=True):
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

def create_log_dir(experiment_name):
    path = os.path.join(os.getcwd(), experiment_name)
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)
    return path

def construct_pinball_mdp(episode_length):
    from simple_rl.tasks.pinball.PinballMDPClass import PinballMDP
    mdp = PinballMDP(noise=0.0, episode_length=episode_length, reward_scale=1000., render=True)
    return mdp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=bool, help="whether or not to load pretrained options", default=False)
    parser.add_argument("--seed", type=int, help="Random seed for this run (default=0)", default=0)
    args = parser.parse_args()

    EXPERIMENT_NAME = "hard_pinball_sg_10"
    NUM_EPISODES = 3000
    NUM_STEPS_PER_EPISODE = 1000

    # overall_mdp = construct_pinball_mdp(NUM_STEPS_PER_EPISODE)
    overall_mdp = LunarLanderMDP(render=False)
    state_space_size = overall_mdp.init_state.state_space_size()

    random_seed = args.seed
    buffer_len = 190
    sub_reward = 10.
    lr = 1e-4
    max_number_of_options = 3
    logdir = create_log_dir(EXPERIMENT_NAME)

    solver = DQNAgent(state_space_size, len(overall_mdp.actions), len(overall_mdp.actions), [], seed=random_seed, lr=lr,
                      name="GlobalDQN", eps_start=1.0, tensor_log=False, use_double_dqn=True)

    if args.pretrained:
        loader = PretrainedOptionsLoader(overall_mdp, solver, buffer_len, num_subgoal_hits_required=3, subgoal_reward=sub_reward,
                                         max_steps=NUM_STEPS_PER_EPISODE, seed=random_seed)
        pretrained_options = loader.get_pretrained_options()
        for pretrained_option in pretrained_options:
            solver = loader.augment_global_agent_with_option(solver, pretrained_option)
        print("Running skill chaining with pretrained options: {}".format(pretrained_options))
        chainer = SkillChaining(overall_mdp, rl_agent=solver, buffer_length=buffer_len,
                                seed=random_seed, subgoal_reward=sub_reward, max_num_options=0,
                                lr_decay=False, pretrained_options=pretrained_options,
                                log_dir=logdir)
        pretrained_episodic_scores, pretrained_episodic_durations = chainer.skill_chaining(NUM_EPISODES, NUM_STEPS_PER_EPISODE)
        chainer.save_all_scores(args.pretrained, pretrained_episodic_scores, pretrained_episodic_durations)
    else:
        print("Training skill chaining agent from scratch with a buffer length of {} and subgoal reward {}".format(buffer_len, sub_reward))
        print("MDP InitState = ", overall_mdp.init_state)
        # print("MDP GoalPosition = ", overall_mdp.domain.environment.target_pos)
        chainer = SkillChaining(overall_mdp, rl_agent=solver, buffer_length=buffer_len,
                                seed=random_seed, subgoal_reward=sub_reward, max_num_options=max_number_of_options,
                                lr_decay=False, log_dir=logdir, generate_plots=True)
        episodic_scores, episodic_durations = chainer.skill_chaining(NUM_EPISODES, NUM_STEPS_PER_EPISODE)

        # Log performance metrics
        chainer.save_all_dqns()
        chainer.save_all_initiation_classifiers()
        chainer.perform_experiments()
        chainer.save_all_scores(args.pretrained, episodic_scores, episodic_durations)
