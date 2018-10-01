# Python imports.
from collections import defaultdict
import random
from sklearn import svm
import numpy as np
from copy import deepcopy
import pdb

# Other imports.
from simple_rl.mdp.StateClass import State
from simple_rl.agents.QLearningAgentClass import QLearningAgent
from simple_rl.tasks.grid_world.GridWorldMDPClass import GridWorldMDP
from simple_rl.abstraction.action_abs.PredicateClass import Predicate

class Option(object):

	def __init__(self, init_predicate, term_predicate, policy, overall_mdp, actions=[], name="o", term_prob=0.01):
		'''
		Args:
			init_func (S --> {0,1})
			init_func (S --> {0,1})
			policy (S --> A)
		'''
		self.init_predicate = init_predicate
		self.term_predicate = term_predicate
		self.term_flag = False
		self.name = name
		self.term_prob = term_prob

		if type(policy) is defaultdict or type(policy) is dict:
			self.policy_dict = dict(policy)
			self.policy = self.policy_from_dict
		else:
			self.policy = policy

		self.solver = QLearningAgent(actions, name=self.name+'_q_solver')
		self.initiation_data = []
		self.initiation_classifier = svm.SVC()

		self.overall_mdp = overall_mdp
		self.subgoal_mdp = self._create_subgoal_mdp()

	def __str__(self):
		return self.name

	def __repr__(self):
		return str(self)

	def __hash__(self):
		return hash(self.name)

	def is_init_true(self, ground_state):
		return self.init_predicate.is_true(ground_state)

	def is_term_true(self, ground_state):
		return self.term_predicate.is_true(ground_state) or self.term_flag or self.term_prob > random.random()

	def act(self, ground_state):
		return self.policy(ground_state)

	def set_policy(self, policy):
		self.policy = policy

	def set_name(self, new_name):
		self.name = new_name

	# TODO
	def _create_subgoal_mdp(self):
		return GridWorldMDP(10, 10, goal_predicate=self.term_predicate)

	@staticmethod
	def _construct_feature_matrix(states):
		n_samples = len(states)
		n_features = states[0].get_num_feats()
		X = np.zeros((n_samples, n_features))
		for row in range(X.shape[0]):
			X[row, :] = states[row].features()
		return X

	@staticmethod
	def _split_experience_into_pos_neg_examples(examples):

		# Last quarter of the states in the experience buffer are treated as positive examples
		r = 0.5
		last_index = len(examples) - int(r * len(examples))
		first_index = int(r * len(examples))

		positive_examples = [examples[-i] for i in range(first_index)]
		negative_examples = [examples[i] for i in range(last_index)]

		return positive_examples, negative_examples

	def train_initiation_classifier(self):
		positive_examples, negative_examples = self._split_experience_into_pos_neg_examples(self.initiation_data)
		X = self._construct_feature_matrix(positive_examples + negative_examples)
		Y = np.array(([1] * len(positive_examples)) + ([0] * len(negative_examples)))

		self.initiation_classifier.fit(X, Y)
		self.init_predicate = Predicate(func=lambda s: self.initiation_classifier.predict([s])[0], name=self.name+'_init_predicate')

	# TODO: This is wrong. You need to call act() and execute_action() until the curr_state in the subgoal_mdp is_terminal()
	def learn_policy_from_experience(self):
		# print "{} learning policy from experience".format(self.name)
		reward, policy = 0, defaultdict()
		for state in self.initiation_data:
			if self.init_predicate.is_true(state):
				action = self.solver.act(state, reward)
				reward, _ = self.subgoal_mdp.execute_agent_action(action)
				policy[state] = action
		self.policy_dict = policy

	def create_child_option(self):
		new_option_name = "option_{}".format(np.random.randint(0, 100))
		term_pred = Predicate(func=lambda s: self.initiation_classifier.predict([s])[0],
							  name=new_option_name + '_term_predicate')
		untrained_option = Option(init_predicate=None, term_predicate=term_pred, policy={},
								  actions=self.subgoal_mdp.actions,
								  overall_mdp=self.overall_mdp, name=new_option_name, term_prob=0.)
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

	def execute_option_in_mdp(self, state, mdp, verbose=False):
		if self.is_init_true(state):
			if verbose: print("From state {}, taking option {}".format(state, self.name))
			reward, state = mdp.execute_agent_action(self.act(state))
			while not self.is_term_true(state):
				if verbose: print("from state {}, taking action {}".format(state, self.act(state)))
				r, state = mdp.execute_agent_action(self.act(state))
				reward += r
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
		initiation = []
		for state in self.overall_mdp.get_states():
			if self.init_predicate.is_true(state):
				initiation.append(state)
		return initiation

	def get_termination_set(self):
		terminal = []
		for state in self.overall_mdp.get_states():
			if self.term_predicate.is_true(state):
				terminal.append(state)
		return terminal