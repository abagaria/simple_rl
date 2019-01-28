# Python imports.
from __future__ import print_function
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import random
from sklearn import svm
import numpy as np
import pdb
from copy import deepcopy
import itertools
import torch

# Other imports.
from simple_rl.mdp.StateClass import State
from simple_rl.agents.func_approx.TorchDQNAgentClass import DQNAgent
from simple_rl.abstraction.action_abs.PredicateClass import Predicate
from simple_rl.tasks.lunar_lander.LunarLanderStateClass import LunarLanderState
from simple_rl.tasks.pinball.PinballStateClass import PinballState, PositionalPinballState
import time

class Experience(object):
	def __init__(self, s, a, r, s_prime):
		self.state = s
		self.action = a
		self.reward = r
		self.next_state = s_prime

	def serialize(self):
		return self.state, self.action, self.reward, self.next_state

	def __str__(self):
		return "s: {}, a: {}, r: {}, s': {}".format(self.state, self.action, self.reward, self.next_state)

	def __repr__(self):
		return str(self)

	def __eq__(self, other):
		return isinstance(other, Experience) and self.__dict__ == other.__dict__

	def __ne__(self, other):
		return not self == other

class Option(object):

	def __init__(self, init_predicate, term_predicate, init_state, policy, overall_mdp, actions=[], name="o",
				 term_prob=0.01, default_q=0., global_solver=None, buffer_length=40, pretrained=False,
				 num_subgoal_hits_required=3, subgoal_reward=5000., give_negative_rewards=True,
				 max_steps=20000, seed=0, parent=None, children=[], classifier_type="bocsvm"):
		'''
		Args:
			init_predicate (S --> {0,1})
			term_predicate (S --> {0,1})
			init_state (State)
			policy (S --> A)
			overall_mdp (MDP)
			actions (list)
			name (str)
			term_prob (float)
			default_q (float)
			global_solver (DQNAgent)
			buffer_length (int)
			pretrained (bool)
			num_subgoal_hits_required (int)
			subgoal_reward (float)
			give_negative_rewards (bool)
			max_steps (int)
			seed (int)
			parent (Option)
			children (list)
			classifier_type (str)
		'''
		self.init_predicate = init_predicate
		self.term_predicate = term_predicate
		self.init_state = init_state
		self.term_flag = False
		self.name = name
		self.term_prob = term_prob
		self.buffer_length = buffer_length
		self.pretrained = pretrained
		self.num_subgoal_hits_required = num_subgoal_hits_required
		self.subgoal_reward = subgoal_reward
		self.give_negatives_subgoal_rewards = give_negative_rewards
		self.max_steps = max_steps
		self.seed = seed
		self.parent = parent
		self.children = children
		self.classifier_type = classifier_type

		self.option_idx = 0 if self.parent is None else self.parent.option_idx + 1

		random.seed(seed)

		if classifier_type == "bocsvm" and parent is None:
			raise AssertionError("{}'s parent cannot be none".format(self.name))

		# if init_state.is_terminal() and not self.is_term_true(init_state):
		init_state.set_terminal(False)

		if type(policy) is defaultdict or type(policy) is dict:
			self.policy_dict = dict(policy)
			self.policy = self.policy_from_dict
		else:
			self.policy = policy

		self.solver = DQNAgent(overall_mdp.init_state.state_space_size(), len(overall_mdp.actions), len(overall_mdp.actions),
							   trained_options=[], seed=self.seed, name=name, use_double_dqn=global_solver.use_ddqn,
							   lr=global_solver.learning_rate, tensor_log=global_solver.tensor_log)
		self.global_solver = global_solver

		self.solver.policy_network.initialize_with_bigger_network(self.global_solver.policy_network)
		self.solver.target_network.initialize_with_bigger_network(self.global_solver.target_network)

		self.initiation_classifier = svm.OneClassSVM(nu=0.01, gamma="auto")

		# List of buffers: will use these to train the initiation classifier and the local policy respectively
		self.initiation_data = []
		self.experience_buffer = []

		self.overall_mdp = overall_mdp
		self.num_goal_hits = 0

		# Debug member variables
		self.total_executions = 0
		self.starting_points = []
		self.ending_points 	 = []
		self.num_states_in_replay_buffer = []
		self.num_learning_updates_dqn = []
		self.num_times_indirect_update = 0
		self.num_indirect_updates = []
		self.policy_refinement_data = []

		self.num_option_updates = 0
		self.num_successful_updates = 0
		self.num_unsuccessful_updates = 0
		self.num_executions = 0

	def __str__(self):
		return self.name

	def __repr__(self):
		return str(self)

	def __hash__(self):
		return hash(self.name)

	def __eq__(self, other):
		if not isinstance(other, Option):
			return False
		return str(self) == str(other)

	def __ne__(self, other):
		return not self == other

	def is_init_true(self, ground_state):
		if isinstance(ground_state, LunarLanderState) or isinstance(ground_state, PinballState):
			positional_state = ground_state.convert_to_positional_state()
			return self.init_predicate.is_true(positional_state)
		elif isinstance(ground_state, np.ndarray):
			positional_state = PositionalPinballState(ground_state[0], ground_state[1])
			return self.init_predicate.is_true(positional_state)
		return self.init_predicate.is_true(ground_state)

	def is_term_true(self, ground_state):
		if isinstance(ground_state, LunarLanderState) or isinstance(ground_state, PinballState):
			positional_state = ground_state.convert_to_positional_state()
			return self.term_predicate.is_true(positional_state)
		elif isinstance(ground_state, np.ndarray):
			positional_state = PositionalPinballState(ground_state[0], ground_state[1])
			return self.term_predicate.is_true(positional_state)
		return self.term_predicate.is_true(ground_state) # or self.term_flag or self.term_prob > random.random()

	def act(self, ground_state):
		return self.policy(ground_state)

	def set_policy(self, policy):
		self.policy = policy

	def set_name(self, new_name):
		self.name = new_name

	def get_child_option(self, num_trained_options):
		# Create new option whose termination is the initiation of the option we just trained
		name = "option_{}".format(str(num_trained_options))
		print("Creating {}".format(name))

		# Using the global init_state as the init_state for all child options
		untrained_option = self.create_child_option(init_state=deepcopy(self.overall_mdp.init_state),
													actions=self.overall_mdp.actions,
													new_option_name=name,
													global_solver=self.global_solver,
													buffer_length=self.buffer_length,
													num_subgoal_hits=self.num_subgoal_hits_required,
													subgoal_reward=self.subgoal_reward)
		return untrained_option

	def add_initiation_experience(self, states_queue):
		"""
		SkillChaining class will give us a queue of states that correspond to its most recently successful experience.
		Args:
			states_queue (deque)
		"""
		assert type(states_queue) == deque, "Expected initiation experience sample to be a queue"
		states = list(states_queue)

		# Convert the high dimensional states to positional states for ease of learning the initiation classifier
		positional_states = [state.convert_to_positional_state() for state in states]

		last_state = positional_states[-1]
		filtered_positional_states = list(filter(lambda s: np.linalg.norm(s.features() - last_state.features()) < 0.3, positional_states))

		self.initiation_data.append(filtered_positional_states)

	def add_experience_buffer(self, experience_queue):
		"""
		Skill chaining class will give us a queue of (sars') tuples that correspond to its most recently successful experience.
		Args:
			experience_queue (deque)
		"""
		assert type(experience_queue) == deque, "Expected initiation experience sample to be a queue"
		experiences = [Experience(*exp) for exp in experience_queue]
		self.experience_buffer.append(experiences)

	def are_states_in_sibling_initiation(self, states_queue):
		states = list(states_queue)
		siblings = self.parent.children
		for sibling in siblings: # type: Option
			if sibling != self:
				for state in states:
					if sibling.is_init_true(state):
						return True
		return False

	@staticmethod
	def _construct_feature_matrix(examples_matrix):
		states = list(itertools.chain.from_iterable(examples_matrix))
		n_samples = len(states)
		n_features = states[0].get_num_feats()
		X = np.zeros((n_samples, n_features))
		for row in range(X.shape[0]):
			X[row, :] = states[row].features()
		return X

	def sample_points_inside(self, ocsvm, num_points_to_sample):
		random_points = np.random.rand(500, 2)

		predicate = lambda s: ocsvm.predict([s])[0] == 1
		perimissable_points = list(filter(predicate, random_points))
		return np.array(perimissable_points)[-num_points_to_sample:, :]

	@staticmethod
	def _convert_spread_to_nu(spread):
		def get_slope_and_intercept(points):
			x_coords, y_coords = zip(*points)
			A = np.vstack([x_coords, np.ones(len(x_coords))]).T
			slope, intercept = np.linalg.lstsq(A, y_coords, rcond=None)[0]
			return slope, intercept

		spread_lower_limit = 0.05
		spread_uppper_limit = 0.3
		nu_lower_limit = 0.01
		nu_upper_limit = 0.5

		xy_points = [(spread_lower_limit, nu_lower_limit), (spread_uppper_limit, nu_upper_limit)]

		if spread <= spread_lower_limit:
			return nu_lower_limit
		if spread >= spread_uppper_limit:
			return nu_upper_limit

		m, c = get_slope_and_intercept(xy_points)

		return (m * spread) + c

	def train_biased_one_class_svm(self):
		positive_feature_matrix = self._construct_feature_matrix(self.initiation_data)
		std = np.std(positive_feature_matrix, axis=0)
		spread = np.sum(std)

		parent_sampled_positives = self.sample_points_inside(self.parent.initiation_classifier, num_points_to_sample=6)
		features = np.concatenate((positive_feature_matrix, parent_sampled_positives))

		nu = self._convert_spread_to_nu(spread)
		print("\nCreating BOC-SVM with nu ", nu)
		self.initiation_classifier = svm.OneClassSVM(nu=nu, gamma="scale")
		self.initiation_classifier.fit(features)

		self.init_predicate = Predicate(func=lambda s: self.initiation_classifier.predict([s.features()])[0] == 1,
										name=self.name + '_init_predicate')

	def train_one_class_svm(self):
		assert len(self.initiation_data) == self.num_subgoal_hits_required, "Expected init data to be a list of lists"
		positive_feature_matrix = self._construct_feature_matrix(self.initiation_data)
		self.initiation_classifier.fit(positive_feature_matrix)
		# The OneClassSVM.predict() returns 1 for in-class samples, and -1 for out-of-class samples
		self.init_predicate = Predicate(func=lambda s: self.initiation_classifier.predict([s.features()])[0] == 1,
										name=self.name + '_init_predicate')


	def train_initiation_classifier(self):
		if self.classifier_type == "bocsvm":
			self.train_biased_one_class_svm()
		else:
			self.train_one_class_svm()

	def compute_degree_overlap_with_parent(self):
		if self.parent is None:
			return 0.
		parent_initiation_data = list(itertools.chain.from_iterable(self.parent.initiation_data))
		num_inside, total_num = 0., 0
		for state in parent_initiation_data:
			total_num += 1
			if self.is_init_true(state):
				num_inside += 1
		return num_inside / total_num

	@staticmethod
	def get_center_of_initiation_data(initiation_data):
		initiation_states = list(itertools.chain.from_iterable(initiation_data))
		x_positions = [state.x for state in initiation_states]
		y_positions = [state.y for state in initiation_states]
		x_center = (max(x_positions) + min(x_positions)) / 2.
		y_center = (max(y_positions) + min(y_positions)) / 2.
		return np.array([x_center, y_center])

	def is_option_further_than_parent(self):
		if self.parent is None:
			return True
		parent_center_point = self.get_center_of_initiation_data(self.parent.initiation_data)
		self_center_point = self.get_center_of_initiation_data(self.initiation_data)
		return (self_center_point[1] > parent_center_point[1]) or \
			   ((parent_center_point[1] - self_center_point[1]) < 0.01)

	def is_beneficial_to_construct_option(self):
		overlap_condition = self.compute_degree_overlap_with_parent() < 0.8
		relation_condition = self.is_option_further_than_parent()
		return overlap_condition and relation_condition

	def update_option_solver(self, s, a, r, s_prime):
		assert self.overall_mdp.is_primitive_action(a), "Option solver should be over primitive actions: {}".format(a)
		if self.pretrained: return

		successful = self.is_init_true(s) and self.is_term_true(s_prime) and not self.is_term_true(s)
		failed = self.is_init_true(s) and not self.is_init_true(s_prime) and not self.is_term_true(s_prime)

		if self.give_negatives_subgoal_rewards:
			midst = self.is_init_true(s) and self.is_init_true(s_prime) and not self.is_term_true(s) and not self.is_term_true(s_prime)
		else:
			midst = self.is_init_true(s) and not self.is_term_true(s) and not self.is_term_true(s_prime)

		if successful:
			assert self.is_term_true(s_prime), "Thought {} was a successful transition".format(s_prime)
			self.num_successful_updates += 1
			self.solver.step(s.features(), a, r + self.subgoal_reward, s_prime.features(), True, num_steps=1)

		elif failed and self.give_negatives_subgoal_rewards:
			assert not self.is_term_true(s_prime), "Thought {} was supposed to be failure state".format(s_prime)
			self.num_unsuccessful_updates += 1
			self.solver.step(s.features(), a, r - (0.5*self.subgoal_reward), s_prime.features(), True, num_steps=1)

		# In the middle of executing the option
		elif midst:
			self.solver.step(s.features(), a, r, s_prime.features(), False, num_steps=1)

		self.num_option_updates += 1
		if self.solver.tensor_log:
			self.solver.writer.add_scalar("PositiveExecutions", self.num_successful_updates, self.num_option_updates)
			self.solver.writer.add_scalar("NegativeExecutions", self.num_unsuccessful_updates, self.num_option_updates)

	def initialize_option_policy(self):
		# Initialize the local DQN's policy with the weights of the global DQN
		self.solver.policy_network.initialize_with_bigger_network(self.global_solver.policy_network)
		self.solver.target_network.initialize_with_bigger_network(self.global_solver.target_network)
		self.solver.epsilon = self.global_solver.epsilon

		## $ Uncomment to plot policy initiation data
		# from simple_rl.skill_chaining.skill_chaining_utils import plot_all_trajectories_in_initiation_data
		# plot_all_trajectories_in_initiation_data(self.experience_buffer, True, True, False, self.name)

		# Fitted Q-iteration on the experiences that led to triggering the current option's termination condition
		experience_buffer = list(itertools.chain.from_iterable(self.experience_buffer))
		for experience in experience_buffer:
			state, a, r, s_prime = experience.serialize()
			self.update_option_solver(state, a, r, s_prime)

		# TODO: Experimental
		# We only see 1 positive reward transition for each trajectory (which can be as long as 15000 steps)
		# To increase the probability of samping the positive reward transition, I am adding that transition
		# multiple times to the replay buffer
		# for _ in range(2500):
		# 	experience = experience_buffer[-1]
		# 	state, action, reward, next_state = experience.serialize()
		# 	self.solver.step(state.features(), action, reward, next_state.features(), next_state.is_terminal())

	def bottom_right_states(self, states_buffer):
		states = list(states_buffer)
		for state in states:
			if state.x < 0.4 or state.y < 0.5:
				return False
		return True

	def train(self, experience_buffer, state_buffer):
		"""
		Called every time the agent hits the current option's termination set.
		Args:
			experience_buffer (deque)
			state_buffer (deque)

		Returns:
			trained (bool): whether or not we actually trained this option
		"""
		if "tree" in self.name:
			if not self.are_states_in_sibling_initiation(state_buffer) and self.bottom_right_states(state_buffer):
				self.add_initiation_experience(state_buffer)
				self.add_experience_buffer(experience_buffer)
				self.num_goal_hits += 1
		else:
			self.add_initiation_experience(state_buffer)
			self.add_experience_buffer(experience_buffer)
			self.num_goal_hits += 1

		if self.num_goal_hits >= self.num_subgoal_hits_required:
			self.train_initiation_classifier()
			# pdb.set_trace()
			if not self.is_beneficial_to_construct_option():
				self.num_goal_hits = 0
				self.initiation_data = []
				self.experience_buffer = []
				return False
			self.initialize_option_policy()
			return True
		return False

	def update_trained_option_policy(self, experience_buffer):
		"""
		If we hit the termination set of a trained option, we may want to update its policy.
		Args:
			experience_buffer (deque): (s, a, r, s')
		"""
		self.num_times_indirect_update += 1
		positive_experiences = list(experience_buffer)
		positive_states = [positive_experience[0] for positive_experience in positive_experiences]
		self.policy_refinement_data.append(positive_states)
		for experience in positive_experiences:
			state, action, reward, next_state = experience
			self.solver.step(state.features(), action, reward, next_state.features(), self.is_term_true(next_state))

	def create_child_option(self, init_state, actions, new_option_name, global_solver, buffer_length, num_subgoal_hits,
							default_q=0., pretrained=False, subgoal_reward=5000.):
		term_pred = Predicate(func=self.init_predicate.func, name=new_option_name + '_term_predicate')
		untrained_option = Option(init_predicate=None, term_predicate=term_pred, policy={}, init_state=init_state,
								  actions=actions, overall_mdp=self.overall_mdp, name=new_option_name, term_prob=0.,
								  default_q=default_q, global_solver=global_solver, buffer_length=buffer_length,
								  pretrained=pretrained, num_subgoal_hits_required=num_subgoal_hits,
								  subgoal_reward=subgoal_reward, give_negative_rewards=self.give_negatives_subgoal_rewards,
								  seed=self.seed, parent=self, children=[], classifier_type="ocsvm")
		self.children.append(untrained_option)
		return untrained_option

	def act_until_terminal(self, cur_state, transition_func):
		'''
		Summary:
			Executes the option until termination.
		'''
		if self.is_init_true(cur_state):
			cur_state = transition_func(cur_state, self.act(cur_state))
			while not self.is_term_true(cur_state):
				cur_state = transition_func(cur_state, self.act(cur_state))

		return cur_state

	def rollout(self, cur_state, reward_func, transition_func, step_cost=0):
		'''
		Summary:
			Executes the option until termination.

		Returns:
			(tuple):
				1. (State): state we landed in.
				2. (float): Reward from the trajectory.
		'''
		total_reward = 0
		if self.is_init_true(cur_state):
			# First step.
			total_reward += reward_func(cur_state, self.act(cur_state)) - step_cost
			cur_state = transition_func(cur_state, self.act(cur_state))

			# Act until terminal.
			while not self.is_term_true(cur_state):
				cur_state = transition_func(cur_state, self.act(cur_state))
				total_reward += reward_func(cur_state, self.act(cur_state)) - step_cost

		return cur_state, total_reward

	def execute_option_in_mdp(self, mdp, step_number):
		"""
		Option main control loop.

		Args:
		    mdp (MDP): environment where actions are being taken
		    step_number (int): how many steps have already elapsed in the outer control loop.

		Returns:
			option_transitions (list): list of (s, a, r, s') tuples
			discounted_reward (float): cumulative discounted reward obtained by executing the option
		"""
		state = mdp.cur_state

		if self.is_init_true(state):

			# ------------------ Debug logging for option's policy learning ------------------
			self.total_executions += 1
			self.starting_points.append(state)
			self.num_states_in_replay_buffer.append(len(self.solver.replay_buffer))
			self.num_learning_updates_dqn.append(self.solver.num_updates)
			self.num_indirect_updates.append(self.num_times_indirect_update)
			# ---------------------------------------------------------------------------------

			option_transitions = []
			discounted_reward = 0.
			self.num_executions += 1
			num_steps = 0

			while self.is_init_true(state) and not self.is_term_true(state) and \
					not state.is_terminal() and step_number < self.max_steps and num_steps < (10 * self.buffer_length):

				epsilon = self.solver.epsilon if not self.pretrained else 0.
				action = self.solver.act(state.features(), epsilon)
				reward, next_state = mdp.execute_agent_action(action, option_idx=self.option_idx)

				if not self.pretrained:
					self.update_option_solver(state, action, reward, next_state)

				# Note: We are not using the option augmented subgoal reward while making off-policy updates to global DQN
				assert mdp.is_primitive_action(action), "Option solver should be over primitive actions: {}".format(action)
				self.global_solver.step(state.features(), action, reward, next_state.features(), next_state.is_terminal(), num_steps=1)

				option_transitions.append((state, action, reward, next_state))
				discounted_reward += ((self.solver.gamma ** num_steps) * reward)
				state = next_state

				# Epsilon decay
				self.solver.update_epsilon()
				self.global_solver.update_epsilon()

				# step_number is to check if we exhaust the episodic step budget
				# num_steps is to appropriately discount the rewards during option execution
				step_number += 1
				num_steps += 1

			self.ending_points.append(state)

			if self.solver.tensor_log:
				self.solver.writer.add_scalar("ExecutionLength", len(option_transitions), self.num_executions)

			return option_transitions, discounted_reward

		raise Warning("Wanted to execute {}, but initiation condition not met".format(self))

	def trained_option_execution(self, mdp):
		state = mdp.cur_state
		score, step_number = 0., 0
		while self.is_init_true(state) and not self.is_term_true(state) and not state.is_terminal()\
				and not state.is_out_of_frame() and step_number < 5000:
			action = self.solver.act(state.features(), eps=0.)
			reward, state = mdp.execute_agent_action(action)
			score += reward
			step_number += 1
		return score, state

	def visualize_learned_policy(self, mdp, num_times=5):
		for _ in range(num_times):
			positional_state = self.sample_points_inside(self.initiation_classifier, num_points_to_sample=1)
			state = PinballState(positional_state[0][0], positional_state[0][1], 0, 0)
			mdp.render = True
			mdp.cur_state = state
			mdp.domain.state = state.features()
			mdp.execute_agent_action(4) # noop
			mdp.execute_agent_action(4) # noop
			time.sleep(0.3)
			if self.is_init_true(state) and not self.is_term_true(state):
				score, next_state = self.trained_option_execution(mdp)
				print("Success" if self.is_term_true(next_state) else "Failure")
			elif not self.is_init_true(state):
				print("{} not in {}'s initiation set".format(state, self.name))


	def policy_from_dict(self, state):
		if state not in self.policy_dict.keys():
			self.term_flag = True
			return random.choice(list(set(self.policy_dict.values())))
		else:
			self.term_flag = False
			return self.policy_dict[state]

	def term_func_from_list(self, state):
		return state in self.term_list