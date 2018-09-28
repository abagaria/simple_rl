# Python imports.
from collections import defaultdict
import random
from sklearn import svm
import numpy as np
from copy import deepcopy

# Other imports.
from simple_rl.mdp.StateClass import State
from simple_rl.agents.QLearningAgentClass import QLearningAgent

class Option(object):

	def __init__(self, init_predicate, term_predicate, policy, actions=[], name="o", term_prob=0.01):
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
		r = 0.25
		positive_examples = examples[-int(r * len(examples)):]
		negative_examples = examples[:len(examples) - int(r * len(examples))]

		return positive_examples, negative_examples

	def train_initiation_classifier(self):
		positive_examples, negative_examples = self._split_experience_into_pos_neg_examples(self.initiation_data)
		X = self._construct_feature_matrix(positive_examples + negative_examples)
		Y = np.array(([1] * len(positive_examples)) + ([0] * len(negative_examples)))

		self.initiation_classifier.fit(X, Y)

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

	def __str__(self):
		return "option." + str(self.name)