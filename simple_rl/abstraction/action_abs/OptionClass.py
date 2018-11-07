# Python imports.
from __future__ import print_function
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import random
from sklearn import svm
import numpy as np
import pdb

# Other imports.
from simple_rl.mdp.StateClass import State
from simple_rl.agents.func_approx.TorchDQNAgentClass import DQNAgent
from simple_rl.abstraction.action_abs.PredicateClass import Predicate
from simple_rl.tasks.lunar_lander.LunarLanderStateClass import LunarLanderState
from simple_rl.tasks.lunar_lander.PositionalLunarLanderStateClass import PositionalLunarLanderState
from simple_rl.agents.func_approx.TorchDQNAgentClass import EPS_START, EPS_END

EPS_DECAY = 0.998
GESTATION_PERIOD = 20 # episodes

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
				 term_prob=0.01, default_q=0., global_solver=None, buffer_length=40):
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
		'''
		self.init_predicate = init_predicate
		self.term_predicate = term_predicate
		self.init_state = init_state
		self.term_flag = False
		self.name = name
		self.term_prob = term_prob
		self.buffer_length = buffer_length

		# if init_state.is_terminal() and not self.is_term_true(init_state):
		init_state.set_terminal(False)

		if type(policy) is defaultdict or type(policy) is dict:
			self.policy_dict = dict(policy)
			self.policy = self.policy_from_dict
		else:
			self.policy = policy

		self.solver = DQNAgent(overall_mdp.env.observation_space.shape[0], overall_mdp.env.action_space.n, 0, name=name)
		self.global_solver = global_solver
		self.epsilon = EPS_START

		if global_solver:
			self.solver.policy_network.load_state_dict(global_solver.policy_network.state_dict())

		self.initiation_classifier = svm.SVC(kernel="rbf")

		# List of buffers: will use these to train the initiation classifier and the local policy respectively
		self.initiation_data = np.empty((buffer_length, 10), dtype=State)
		self.experience_buffer = np.empty((buffer_length, 10), dtype=Experience)
		self.new_experience_buffer = deque([], maxlen=50)

		self.overall_mdp = overall_mdp
		self.num_goal_hits = 0
		self.times_executed_since_being_trained = 0

		# Debug member variables
		self.starting_points = []
		self.ending_points 	 = []
		self.epsilon_history = []
		self.num_states_in_replay_buffer = []
		self.num_learning_updates_dqn = []
		self.num_times_indirect_update = 0
		self.num_indirect_updates = []
		self.policy_refinement_data = []

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
		if isinstance(ground_state, LunarLanderState):
			positional_state = PositionalLunarLanderState(ground_state.x, ground_state.y, ground_state.is_terminal())
			return self.init_predicate.is_true(positional_state)
		return self.init_predicate.is_true(ground_state)

	def is_term_true(self, ground_state):
		if isinstance(ground_state, LunarLanderState):
			positional_state = PositionalLunarLanderState(ground_state.x, ground_state.y, ground_state.is_terminal())
			return self.term_predicate.is_true(positional_state)
		return self.term_predicate.is_true(ground_state) # or self.term_flag or self.term_prob > random.random()

	def act(self, ground_state):
		return self.policy(ground_state)

	def set_policy(self, policy):
		self.policy = policy

	def set_name(self, new_name):
		self.name = new_name

	def add_initiation_experience(self, states_queue):
		"""
		SkillChaining class will give us a queue of states that correspond to its most recently successful experience.
		Args:
			states_queue (deque)
		"""
		assert type(states_queue) == deque, "Expected initiation experience sample to be a queue"
		states = list(states_queue)

		# Convert the high dimensional states to positional states for ease of learning the initiation classifier
		positional_states = [PositionalLunarLanderState(state.x, state.y, state.is_terminal()) for state in states]
		self.initiation_data[:, self.num_goal_hits-1] = np.asarray(positional_states)

	def add_experience_buffer(self, experience_queue):
		"""
		Skill chaining class will give us a queue of (sars') tuples that correspond to its most recently successful experience.
		Args:
			experience_queue (deque)
		"""
		assert type(experience_queue) == deque, "Expected initiation experience sample to be a queue"
		experiences = [Experience(*exp) for exp in experience_queue]
		self.experience_buffer[:, self.num_goal_hits-1] = np.asarray(experiences)

	@staticmethod
	def _construct_feature_matrix(examples_matrix):
		states = examples_matrix.reshape(-1)
		n_samples = len(states)
		n_features = states[0].get_num_feats()
		X = np.zeros((n_samples, n_features))
		for row in range(X.shape[0]):
			X[row, :] = states[row].features()
		return X

	def _split_experience_into_pos_neg_examples(self):
		num_negative_examples = (5 * self.buffer_length) // 8
		negative_experiences = self.initiation_data[:num_negative_examples, :] # 1st 25 states are negative examples
		positive_experiences = self.initiation_data[num_negative_examples:, :] # Last 15 states are positive examples
		return positive_experiences, negative_experiences

	def train_initiation_classifier(self):
		positive_examples, negative_examples = self._split_experience_into_pos_neg_examples()
		positive_feature_matrix = self._construct_feature_matrix(positive_examples)
		negative_feature_matrix = self._construct_feature_matrix(negative_examples)
		positive_labels = [1] * positive_feature_matrix.shape[0]
		negative_labels = [0] * negative_feature_matrix.shape[0]
		X = np.concatenate((positive_feature_matrix, negative_feature_matrix))
		Y = np.concatenate((positive_labels, negative_labels))

		self.initiation_classifier.fit(X, Y)
		self.init_predicate = Predicate(func=lambda s: self.initiation_classifier.predict([s.features()])[0], name=self.name+'_init_predicate')

	def learn_policy_from_experience(self):
		experience_buffer = self.experience_buffer[25:, :].reshape(-1)
		for _ in range(2):
			for experience in experience_buffer:
				state, a, r, s_prime = experience.serialize()
				self.solver.step(state.features(), a, r, s_prime.features(), s_prime.is_terminal())

	def _learn_policy_from_new_experiences(self):
		experience_buffer = self.new_experience_buffer
		for _ in range(2):
			for experience in experience_buffer:
				state, a, r, s_prime = experience
				self.solver.step(state.features(), a, r, s_prime.features(), s_prime.is_terminal())

	def expand_experience_buffer(self, experience):
		"""
		After an Option is trained, you may still enter its Initiation set. If this happens more than N
		times, then we should use that experience to update the current option's policy. This method expands
		the experience buffer of the current option every time the agent walks into its initiation set post
		initial training.
		Args:
			experience (tuple): (state, action, reward, next_state)
		"""
		self.new_experience_buffer.append(experience)

	def update_trained_option_policy(self, experience_buffer):
		"""
		If we hit the initiation set of a trained option, we may want to update its policy.
		Args:
			experience_buffer (deque): (s, a, r, s')
		"""
		self.num_times_indirect_update += 1
		positive_experiences = list(experience_buffer)[-15:]
		positive_states = [positive_experience[0] for positive_experience in positive_experiences]
		self.policy_refinement_data.append(positive_states)
		for experience in positive_experiences:
			state, action, reward, next_state = experience
			self.solver.step(state.features(), action, reward, next_state.features(), next_state.is_terminal())


	def create_child_option(self, init_state, actions, new_option_name, global_solver, buffer_length, default_q=0.):
		term_pred = Predicate(func=self.init_predicate.func, name=new_option_name + '_term_predicate')
		untrained_option = Option(init_predicate=None, term_predicate=term_pred, policy={}, init_state=init_state,
								  actions=actions, overall_mdp=self.overall_mdp, name=new_option_name, term_prob=0.,
								  default_q=default_q, global_solver=global_solver, buffer_length=buffer_length)
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

	def execute_option_in_mdp(self, state, mdp):

		if self.is_init_true(state):

			# ------------------ Debug logging for option's policy learning ------------------
			self.starting_points.append(state)
			self.epsilon_history.append(self.epsilon)
			self.num_states_in_replay_buffer.append(len(self.solver.replay_buffer))
			self.num_learning_updates_dqn.append(self.solver.num_updates)
			self.num_indirect_updates.append(self.num_times_indirect_update)
			# ---------------------------------------------------------------------------------

			self.times_executed_since_being_trained += 1
			action = self.solver.act(state.features(), eps=self.epsilon)
			reward, next_state = mdp.execute_agent_action(action)
			self.solver.step(state.features(), action, reward, next_state.features(), next_state.is_terminal())
			state = next_state

			while self.is_init_true(state) and not self.is_term_true(state) and not state.is_terminal() and not state.is_out_of_frame():

				# Perform off-policy updates on the option's DQN during the option's gestation period
				if self.times_executed_since_being_trained >= GESTATION_PERIOD:
					action = self.solver.act(state.features(), eps=self.epsilon)
				else:
					action = self.global_solver.act(state.features(), eps=self.epsilon)

				r, next_state = mdp.execute_agent_action(action)
				self.solver.step(state.features(), action, r, next_state.features(), next_state.is_terminal())

				# The global DQN is updated at each time step, even if an option is being executed
				# i.e, making off-policy updates to the global solver when we are executing an option
				self.global_solver.step(state.features(), action, r, next_state.features(), next_state.is_terminal())

				reward += r
				state = next_state

			# Epsilon decay
			self.epsilon = max(EPS_END, EPS_DECAY*self.epsilon)

			self.ending_points.append(state)

			return reward, state

		raise Warning("Wanted to execute {}, but initiation condition not met".format(self))

	def policy_from_dict(self, state):
		if state not in self.policy_dict.keys():
			self.term_flag = True
			return random.choice(list(set(self.policy_dict.values())))
		else:
			self.term_flag = False
			return self.policy_dict[state]

	def term_func_from_list(self, state):
		return state in self.term_list

	# --------------------------------------
	# Debug methods
	# --------------------------------------
	def get_initiation_set(self):
		state_space = self.overall_mdp.get_discretized_positional_state_space()
		return [(state.x, state.y) for state in state_space if self.is_init_true(state)]

	@staticmethod
	def plot_x_y_points_on_image(x_grid, y_grid, image_path="/Users/akhil/Desktop/LunarLanderImage.png"):
		plt.figure()
		im = plt.imread(image_path)
		_ = plt.imshow(im)
		x_image, y_image = Option.pos_array_to_image_array(x_grid, y_grid)
		plt.plot(x_image, y_image, marker='.', color='r', linestyle='none')
		plt.show()

	@staticmethod
	def pos_array_to_image_array(x_grid, y_grid):
		x_scale = lambda x: 600 * (x + 1)
		y_scale = lambda y: 800 * y
		x_func = np.vectorize(x_scale)
		y_func = np.vectorize(y_scale)
		x_image = x_func(x_grid)
		y_image = y_func(y_grid)
		return x_image, y_image
