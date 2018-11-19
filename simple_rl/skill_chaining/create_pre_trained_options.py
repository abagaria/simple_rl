# Python imports.
import _pickle as pickle
import torch
import os

# Other imports
from simple_rl.agents.func_approx.TorchDQNAgentClass import DQNAgent
from simple_rl.abstraction.action_abs.OptionClass import Option
from simple_rl.abstraction.action_abs.PredicateClass import Predicate

class PretrainedOptionsLoader(object):
    def __init__(self, mdp, global_solver, buffer_length):
        self.mdp = mdp
        self.global_solver = global_solver
        self.buffer_length = buffer_length

    def create_goal_option(self, path_to_dqn, init_classifier_pickle):
        name = "overall_goal_policy"
        env = self.mdp.env
        goal_agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, 0)
        goal_agent.policy_network.load_state_dict(torch.load(path_to_dqn))
        goal_predicate = self.mdp.default_goal_predicate()
        with open(init_classifier_pickle, "rb") as _f:
            initiation_classifier = pickle.load(_f)
        init_predicate = Predicate(func=lambda s: initiation_classifier.predict([s.features()])[0],
                                   name=name + '_init_predicate')
        goal_option = Option(init_predicate=init_predicate, term_predicate=goal_predicate, overall_mdp=self.mdp,
                             init_state=self.mdp.init_state, actions=self.mdp.actions, policy={},
                             name=name, term_prob=0., global_solver=self.global_solver, buffer_length=self.buffer_length,
                             pretrained=True)
        goal_option.solver = goal_agent
        return goal_option

    # TODO: Something wrong with child option initiation set classifiers (not getting executed)
    def create_subgoal_option(self, name, previous_option, path_to_dqn, init_classifier_pickle):
        env = self.mdp.env
        agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, 0)
        agent.policy_network.load_state_dict(torch.load(path_to_dqn))
        with open(init_classifier_pickle, "rb") as _f:
            initiation_classifier = pickle.load(_f)
        init_predicate = Predicate(func=lambda s: initiation_classifier.predict([s.features()])[0],
                                   name=name + '_init_predicate')
        option = previous_option.create_child_option(self.mdp.init_state, self.mdp.actions, name, self.global_solver,
                                                     buffer_length=self.buffer_length, pretrained=True,
                                                     num_subgoal_hits=10)
        option.init_predicate = init_predicate
        option.solver = agent
        return option

    def get_pretrained_options(self):
        data_dir = os.getcwd()

        path_to_goal_dqn = os.path.join(data_dir, "overall_goal_policy_dqn.pth")
        path_to_o1_dqn = os.path.join(data_dir, "option_1_dqn.pth")
        path_to_o2_dqn = os.path.join(data_dir, "option_2_dqn.pth")

        path_to_og_init_clf_pkl = os.path.join(data_dir, "overall_goal_policy_svm.pkl")
        path_to_o1_init_clf_pkl = os.path.join(data_dir, "option_1_svm.pkl")
        path_to_o2_init_clf_pkl = os.path.join(data_dir, "option_2_svm.pkl")

        goal_option = self.create_goal_option(path_to_goal_dqn, path_to_og_init_clf_pkl)
        option_1 = self.create_subgoal_option("option_1", goal_option, path_to_o1_dqn, path_to_o1_init_clf_pkl)
        option_2 = self.create_subgoal_option("option_2", option_1, path_to_o2_dqn, path_to_o2_init_clf_pkl)

        return [goal_option, option_1, option_2]


