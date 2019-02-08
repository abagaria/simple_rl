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
from scipy.spatial import distance

# Other imports.
from simple_rl.mdp.StateClass import State
from simple_rl.agents.func_approx.TorchDQNAgentClass import DQNAgent
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

	def __init__(self, overall_mdp, name, global_solver, buffer_length=20, pretrained=False,
				 num_subgoal_hits_required=3, subgoal_reward=1., max_steps=20000, seed=0, parent=None, children=[],
				 classifier_type="ocsvm", enable_timeout=True, timeout=200):
		'''
		Args:
			overall_mdp (MDP)
			name (str)
			global_solver (DQNAgent)
			buffer_length (int)
			pretrained (bool)
			num_subgoal_hits_required (int)
			subgoal_reward (float)
			max_steps (int)
			seed (int)
			parent (Option)
			children (list)
			classifier_type (str)
			enable_timeout (bool)
			timeout (int)
		'''
		self.term_flag = False
		self.name = name
		self.buffer_length = buffer_length
		self.pretrained = pretrained
		self.num_subgoal_hits_required = num_subgoal_hits_required
		self.subgoal_reward = subgoal_reward
		self.max_steps = max_steps
		self.seed = seed
		self.parent = parent
		self.children = children
		self.classifier_type = classifier_type
		self.enable_timeout = enable_timeout
		self.timeout = timeout if enable_timeout else np.inf

		self.initiation_period = 7

		self.option_idx = 0 if self.parent is None else self.parent.option_idx + 1

		random.seed(seed)
		np.random.seed(seed)

		if classifier_type == "bocsvm" and parent is None:
			raise AssertionError("{}'s parent cannot be none".format(self.name))

		self.solver = DQNAgent(overall_mdp.init_state.state_space_size(), len(overall_mdp.actions), len(overall_mdp.actions),
							   trained_options=[], seed=self.seed, name=name, use_double_dqn=global_solver.use_ddqn,
							   lr=global_solver.learning_rate, tensor_log=global_solver.tensor_log,
							   loss_function=global_solver.loss_function, gradient_clip=global_solver.gradient_clip)
		self.global_solver = global_solver

		self.solver.policy_network.initialize_with_bigger_network(self.global_solver.policy_network)
		self.solver.target_network.initialize_with_bigger_network(self.global_solver.target_network)

		self.initiation_classifier = None

		# List of buffers: will use these to train the initiation classifier and the local policy respectively
		self.positive_examples = []
		self.negative_examples = []
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

	@staticmethod
	def _get_positional_state(ground_state):
		if isinstance(ground_state, PositionalPinballState):
			return ground_state
		if isinstance(ground_state, PinballState):
			return ground_state.convert_to_positional_state()
		elif isinstance(ground_state, np.ndarray):
			return PositionalPinballState(ground_state[0], ground_state[1])
		raise ValueError("Got state of type {}".format(type(ground_state)))

	def distance_to_closest_positive_example(self, state):
		XA = state.get_position()
		XB = self._construct_feature_matrix(self.positive_examples)
		distances = distance.cdist(XA[None, ...], XB, "euclidean")
		return np.min(distances)

	def batched_is_init_true(self, positional_state_matrix):
		assert positional_state_matrix.shape[1] == 2, "Expected columns to correspond to x, y positions"
		if self.classifier_type == "tcsvm":
			svm_predictions = self.initiation_classifier.predict(positional_state_matrix)

			positive_example_matrix = self._construct_feature_matrix(self.positive_examples)
			distance_matrix = distance.cdist(positional_state_matrix, positive_example_matrix, "euclidean")
			closest_distances = np.min(distance_matrix, axis=1)
			distance_predictions = closest_distances < 0.1

			predictions = np.logical_and(svm_predictions, distance_predictions)
			return predictions
		if self.classifier_type == "ocsvm":
			return self.initiation_classifier.predict(positional_state_matrix)
		raise NotImplementedError("Classifier type {} not supported".format(self.classifier_type))

	def is_init_true(self, ground_state):
		positional_state = self._get_positional_state(ground_state)
		svm_decision = self.initiation_classifier.predict([positional_state.features()])[0] == 1

		if self.classifier_type == "ocsvm":
			return svm_decision
		if self.classifier_type == "tcsvm":
			dist = self.distance_to_closest_positive_example(positional_state)
			return svm_decision and dist < 0.1
		raise NotImplementedError("Classifier type {} not supported".format(self.classifier_type))

	# TODO: Does it make more sense to return true for entering *any* parent's initiation set
	# e.g, if I enter my grand-parent's initiation set, should that be a terminal transition?
	# TODO: Write a batched version of this function and then use it in the DQNAgentClass
	def is_term_true(self, ground_state):
		if self.parent is not None:
			return self.parent.is_init_true(ground_state)

		# If option does not have a parent, it must be the goal option
		assert self.name == "overall_goal_policy", "{}".format(self.name)
		positional_state = self._get_positional_state(ground_state)
		return self.overall_mdp.is_goal_state(positional_state)

	def get_final_positive_examples(self):
		positive_trajectories = self.positive_examples
		return [trajectory[-1] for trajectory in positive_trajectories]

	def get_training_phase(self):
		if self.num_goal_hits < self.num_subgoal_hits_required:
			return "gestation"
		if self.num_goal_hits < (self.num_subgoal_hits_required + self.initiation_period):
			return "initiation"
		if self.num_goal_hits == (self.num_subgoal_hits_required + self.initiation_period):
			return "initiation_done"
		return "trained"

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

		self.positive_examples.append(filtered_positional_states)

	def add_experience_buffer(self, experience_queue):
		"""
		Skill chaining class will give us a queue of (sars') tuples that correspond to its most recently successful experience.
		Args:
			experience_queue (deque)
		"""
		assert type(experience_queue) == deque, "Expected initiation experience sample to be a queue"
		experiences = [Experience(*exp) for exp in experience_queue]
		self.experience_buffer.append(experiences)

	@staticmethod
	def _construct_feature_matrix(examples_matrix):
		states = list(itertools.chain.from_iterable(examples_matrix))
		n_samples = len(states)
		n_features = states[0].get_num_feats()
		X = np.zeros((n_samples, n_features))
		for row in range(X.shape[0]):
			X[row, :] = states[row].features()
		return X

	def train_one_class_svm(self):
		assert len(self.positive_examples) == self.num_subgoal_hits_required, "Expected init data to be a list of lists"
		positive_feature_matrix = self._construct_feature_matrix(self.positive_examples)

		self.initiation_classifier = svm.OneClassSVM(nu=0.01, gamma="auto")
		self.initiation_classifier.fit(positive_feature_matrix)

	def train_two_class_classifier(self):
		positive_feature_matrix = self._construct_feature_matrix(self.positive_examples)
		negative_feature_matrix = self._construct_feature_matrix(self.negative_examples)
		positive_labels = [1] * positive_feature_matrix.shape[0]
		negative_labels = [0] * negative_feature_matrix.shape[0]

		X = np.concatenate((positive_feature_matrix, negative_feature_matrix))
		Y = np.concatenate((positive_labels, negative_labels))

		# We use a 2-class balanced SVM which sets class weights based on their ratios in the training data
		self.initiation_classifier = svm.SVC(kernel="rbf", gamma="scale", class_weight="balanced")
		self.initiation_classifier.fit(X, Y)
		self.classifier_type = "tcsvm"

	def train_initiation_classifier(self):
		self.train_one_class_svm()

	@staticmethod
	def get_center_of_initiation_data(initiation_data):
		initiation_states = list(itertools.chain.from_iterable(initiation_data))
		x_positions = [state.x for state in initiation_states]
		y_positions = [state.y for state in initiation_states]
		x_center = (max(x_positions) + min(x_positions)) / 2.
		y_center = (max(y_positions) + min(y_positions)) / 2.
		return np.array([x_center, y_center])

	def off_policy_update(self, state, action, reward, next_state):
		assert self.overall_mdp.is_primitive_action(action), "option should be markov: {}".format(action)
		assert not state.is_terminal(), "Terminal state did not terminate at some point"

		# Don't make updates while walking around the termination set of an option
		if self.is_term_true(state):
			return

		# Off-policy updates for states outside tne initiation set were discarded
		# if self.is_init_true(state):
		if self.is_init_true(state) and self.is_term_true(next_state):
			self.solver.step(state.features(), action, reward + self.subgoal_reward, next_state.features(), True, num_steps=1)
		elif self.is_init_true(state):
			self.solver.step(state.features(), action, reward, next_state.features(), next_state.is_terminal(), num_steps=1)

	def update_option_solver(self, s, a, r, s_prime):
		assert self.overall_mdp.is_primitive_action(a), "Option solver should be over primitive actions: {}".format(a)
		assert not s.is_terminal(), "Terminal state did not terminate at some point"

		if self.is_term_true(s): #, "Don't call on-policy updater on {} term states: {}".format(self.name, s)
			pdb.set_trace()

		if self.pretrained:
			return

		if self.is_term_true(s_prime):
			self.num_successful_updates += 1
			self.solver.step(s.features(), a, r + self.subgoal_reward, s_prime.features(), True, num_steps=1)
		elif s_prime.is_terminal():
			print("\rWarning: {} is taking me to the goal state even though its not in its term set\n".format(self.name))
			self.solver.step(s.features(), a, r, s_prime.features(), True, num_steps=1)
		else:
			self.solver.step(s.features(), a, r, s_prime.features(), False, num_steps=1)

		self.num_option_updates += 1
		if self.solver.tensor_log:
			self.solver.writer.add_scalar("PositiveExecutions", self.num_successful_updates, self.num_option_updates)

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
			self.off_policy_update(state, a, r, s_prime)

	def train(self, experience_buffer, state_buffer):
		"""
		Called every time the agent hits the current option's termination set.
		Args:
			experience_buffer (deque)
			state_buffer (deque)

		Returns:
			trained (bool): whether or not we actually trained this option
		"""
		self.add_initiation_experience(state_buffer)
		self.add_experience_buffer(experience_buffer)
		self.num_goal_hits += 1

		if self.num_goal_hits >= self.num_subgoal_hits_required:
			self.train_initiation_classifier()
			self.initialize_option_policy()
			return True
		return False

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
		start_state = deepcopy(mdp.cur_state)
		state = mdp.cur_state

		if self.is_init_true(state):

			option_transitions = []
			visited_states = []
			discounted_reward = 0.
			self.num_executions += 1
			num_steps = 0

			while not self.is_term_true(state) and not state.is_terminal() and \
					step_number < self.max_steps and num_steps < self.timeout:

				action = self.solver.act(state.features(), train_mode=True)
				reward, next_state = mdp.execute_agent_action(action, option_idx=self.option_idx)

				if not self.pretrained:
					self.update_option_solver(state, action, reward, next_state)

				# Note: We are not using the option augmented subgoal reward while making off-policy updates to global DQN
				assert mdp.is_primitive_action(action), "Option solver should be over primitive actions: {}".format(action)
				self.global_solver.step(state.features(), action, reward, next_state.features(), next_state.is_terminal(), num_steps=1)

				option_transitions.append((state, action, reward, next_state))
				visited_states.append(state)

				discounted_reward += ((self.solver.gamma ** num_steps) * reward)
				state = next_state

				# Epsilon decay
				self.solver.update_epsilon()
				self.global_solver.update_epsilon()

				# step_number is to check if we exhaust the episodic step budget
				# num_steps is to appropriately discount the rewards during option execution
				step_number += 1
				num_steps += 1

			# Don't forget to add the terminal state to the followed trajectory
			visited_states.append(state)

			if self.get_training_phase() == "initiation":
				self.refine_initiation_set_classifier(visited_states, start_state, state, num_steps)

			if self.solver.tensor_log:
				self.solver.writer.add_scalar("ExecutionLength", len(option_transitions), self.num_executions)

			return option_transitions, discounted_reward

		raise Warning("Wanted to execute {}, but initiation condition not met".format(self))

	def refine_initiation_set_classifier(self, visited_states, start_state, final_state, num_steps):
		if self.is_term_true(final_state):  # success
			self.num_goal_hits += 1
			if not self.pretrained:
				positive_states = [start_state] + visited_states[-self.buffer_length:]
				positive_positional_states = list(map(lambda s: s.convert_to_positional_state(), positive_states))
				self.positive_examples.append(positive_positional_states)

		elif num_steps == self.timeout:
			negative_examples = [start_state.convert_to_positional_state()]
			if self.parent is not None:
				parent_sampled_negative = random.choice(self.parent.get_final_positive_examples())
				negative_examples.append(parent_sampled_negative)
			self.negative_examples.append(negative_examples)
		else:
			assert final_state.is_terminal(), "Hit else case, but {} was not terminal".format(final_state)
			print("\rWarning: Ended up in goal state when {} was not terminal {}".format(self.name, final_state))

		# Refine the initiation set classifier
		if len(self.negative_examples) > 0:
			self.train_two_class_classifier()
			from simple_rl.skill_chaining.skill_chaining_utils import plot_binary_initiation_set
			plot_binary_initiation_set(self)

	def trained_option_execution(self, mdp, outer_step_counter, episodic_budget):
		state = mdp.cur_state
		score, step_number = 0., deepcopy(outer_step_counter)
		num_steps = 0
		while not self.is_term_true(state) and not state.is_terminal()\
				and not state.is_out_of_frame() and step_number < episodic_budget and num_steps < self.timeout:
			action = self.solver.act(state.features(), train_mode=False)
			reward, state = mdp.execute_agent_action(action, option_idx=self.option_idx)
			score += reward
			step_number += 1
			num_steps += 1
		return score, state, step_number

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
				score, next_state, option_num_steps = self.trained_option_execution(mdp, 0, 2000)
				print("Success" if self.is_term_true(next_state) else "Failure")
			elif not self.is_init_true(state):
				print("{} not in {}'s initiation set".format(state, self.name))
