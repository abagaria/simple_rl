# Python imports.
from __future__ import print_function

import sys
sys.path = [""] + sys.path

import matplotlib
matplotlib.use('TkAgg')
from collections import deque
from copy import deepcopy
import torch
import time
import pdb

# Other imports.
from simple_rl.abstraction.action_abs.PredicateClass import Predicate
from simple_rl.abstraction.action_abs.OptionClass import Option
from simple_rl.agents.func_approx.TorchDQNAgentClass import DQNAgent
from simple_rl.tasks.lunar_lander.LunarLanderMDPClass import LunarLanderMDP
from simple_rl.skill_chaining.skill_chaining_utils import *
from simple_rl.agents.func_approx.TorchDQNAgentClass import EPS_START, EPS_DECAY, EPS_END

class SkillChaining(object):
    def __init__(self, mdp, overall_goal_predicate, rl_agent, buffer_length=40, subgoal_reward=20.0, subgoal_hits=10):
        """
        Args:
            mdp (MDP): Underlying domain we have to solve
            overall_goal_predicate (Predicate)
            rl_agent (DQNAgent): RL agent used to determine the policy for each option
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

        # Using the global init_state as the init_state for all child options
        new_untrained_option = untrained_option.create_child_option(init_state=deepcopy(self.mdp.init_state),
                                                                actions=self.original_actions,
                                                                new_option_name=name,
                                                                global_solver=self.global_solver,
                                                                buffer_length=self.buffer_length)
        return new_untrained_option

    def _derive_state_buffer(self, experience_buffer):
        input_buffer = deepcopy(experience_buffer)
        state_buffer = deque([], maxlen=self.buffer_length)
        for experience in input_buffer:
            state_buffer.append(experience[0])
        return state_buffer

    def execute_trained_option_if_possible(self, state, in_evaluation_mode):
        """
        Cycle through the list of trained options and execute one.
        Args:
            state (State)
            in_evaluation_mode (bool): False if we want to update DQNs with this call

        Returns:
            reward (float)
            next_state (State)
            experiences (deque): Queue of (s,a,r,s') tuples encountered while executing option
        """
        # If s' is in the initiation set of ANY trained option, execute the option
        for trained_option in self.trained_options:  # type: Option
            if trained_option.is_init_true(state):
                if in_evaluation_mode:
                    reward, next_state = trained_option.trained_option_execution(state, self.mdp)
                    return reward, next_state, deque([])
                reward, next_state, experiences = trained_option.execute_option_in_mdp(state, self.mdp)
                return reward, next_state, experiences
        return 0., state, deque([])

    def take_action(self, state, experience_buffer, global_solver_epsilon):
        """
        Skill Chaining agent will take an option if possible, else it will take an atomic action.
        Args:
            state (State)
            experience_buffer (deque)
            global_solver_epsilon (float)

        Returns:
            modified_experience_buffer (deque): Modified version of input buffer with new (s,a,r,s') tuples appended
            reward (float): overall reward executed either from option execution or from taking primitive action
            next_state (State): state we landed in after either executing an option or a primitive action
        """
        modified_experience_buffer = deepcopy(experience_buffer)
        option_reward, next_state, option_experiences = self.execute_trained_option_if_possible(state, False)

        if state == next_state or len(option_experiences) == 0:
            action = self.global_solver.act(state.features(), global_solver_epsilon)
            reward, next_state = self.mdp.execute_agent_action(action)
            self.global_solver.step(state.features(), action, reward, next_state.features(), next_state.is_terminal())
            experience = state, action, reward, next_state
            modified_experience_buffer.append(experience)
            return modified_experience_buffer, reward, next_state

        for option_experience in option_experiences:
            modified_experience_buffer.append(option_experience)
        return modified_experience_buffer, option_reward, next_state

    def skill_chaining(self, num_episodes=1200, num_steps=1000):
        from simple_rl.abstraction.action_abs.OptionClass import Option
        goal_option = Option(init_predicate=None, term_predicate=self.overall_goal_predicate, overall_mdp=self.mdp,
                             init_state=self.mdp.init_state, actions=self.original_actions, policy={},
                             name='overall_goal_policy', term_prob=0., global_solver=self.global_solver,
                             buffer_length=self.buffer_length)

        # Pointer to the current option:
        # 1. This option has the termination set which defines our current goal trigger
        # 2. This option has an untrained initialization set and policy, which we need to train from experience
        untrained_option = goal_option

        # For logging purposes
        per_episode_scores = []
        last_100_scores = deque(maxlen=100)
        epsilon = EPS_START

        for episode in range(num_episodes):

            self.mdp.reset()
            score, reset_agent = 0, False
            state = deepcopy(self.mdp.init_state)
            experience_buffer = deque([], maxlen=self.buffer_length)

            for _ in range(num_steps):
                experience_buffer, reward, next_state = self.take_action(state, experience_buffer, epsilon)
                state_buffer = self._derive_state_buffer(experience_buffer)
                score += reward

                if untrained_option.is_term_true(next_state) and len(experience_buffer) == self.buffer_length and len(self.trained_options) < 2:
                    untrained_option.num_goal_hits += 1
                    untrained_option.add_initiation_experience(state_buffer)
                    untrained_option.add_experience_buffer(experience_buffer)

                    if untrained_option.num_goal_hits >= self.num_goal_hits_before_training:
                        untrained_option = self._train_untrained_option(untrained_option)
                    else:
                        # TODO: Need to get rid of this reset_agent stuff
                        reset_agent = True

                for trained_option in self.trained_options: # type: Option
                    if trained_option.is_term_true(next_state):
                        # TODO: If I break after this update, o2 will be updated in the correct region
                        trained_option.update_trained_option_policy(experience_buffer)
                        break

                state = next_state

                # Reset the agent so that we don't keep moving around the initiation set of the trained option
                if reset_agent or state.is_out_of_frame() or state.is_terminal():
                    break

            last_100_scores.append(score)
            per_episode_scores.append(score)

            # Decay epsilon
            epsilon = max(EPS_END, EPS_DECAY*epsilon)

            if self._log_dqn_status(episode, last_100_scores):
                break

        else:
            torch.save(self.global_solver.policy_network.state_dict(), 'unsolved_gsolver_{}.pth'.format(time.time()))

        return per_episode_scores


    def _log_dqn_status(self, episode, last_100_scores):
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(last_100_scores)), end="")
        if episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(last_100_scores)))

        if np.mean(last_100_scores) >= 200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode - 100,
                                                                                         np.mean(last_100_scores)))
            torch.save(self.global_solver.policy_network.state_dict(), 'checkpoint_gsolver_{}.pth'.format(time.time()))
            return True

        return False

    def perform_experiments(self):
        for option in self.trained_options:
            plot_initiation_set(option)
            visualize_option_policy(option)
            visualize_option_starting_and_ending_points(option)
            visualize_reason_for_option_termination(option)
            plot_epsilon_history(option)
            plot_replay_buffer_size(option)
            plot_num_learning_updates(option)
            plot_policy_refinement_data(option)

    def trained_forward_pass(self):
        """
        Called when skill chaining has finished training: execute options when possible and then atomic actions
        Returns:
            overall_reward (float): score accumulated over the course of the episode.
        """
        self.mdp.reset()
        state = deepcopy(self.mdp.init_state)
        overall_reward = 0.
        self.mdp.render = True
        while not state.is_terminal():

            # If possible, take an option
            option_reward, next_state, _ = self.execute_trained_option_if_possible(state, in_evaluation_mode=True)
            overall_reward += option_reward

            # If we did not take an option, take a primitive action
            if next_state == state:
                action = self.global_solver.act(next_state.features(), eps=0.)
                reward, next_state = self.mdp.execute_agent_action(action)
                overall_reward += reward

            state = next_state
        self.mdp.render = False

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
    # environment.seed(0) TODO: Set this seed so that we can compare between runs
    solver = DQNAgent(environment.observation_space.shape[0], environment.action_space.n, 0)
    chainer = SkillChaining(overall_mdp, overall_mdp.goal_predicate, rl_agent=solver)
    episodic_scores = chainer.skill_chaining()
    chainer.perform_experiments()
