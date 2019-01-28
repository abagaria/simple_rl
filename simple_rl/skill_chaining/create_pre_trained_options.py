# Python imports.
import _pickle as pickle
import torch
import os
from copy import deepcopy
import pdb

# Other imports
from simple_rl.agents.func_approx.TorchDQNAgentClass import DQNAgent
from simple_rl.abstraction.action_abs.OptionClass import Option
from simple_rl.abstraction.action_abs.PredicateClass import Predicate

class PretrainedOptionsLoader(object):
    def __init__(self, mdp, global_solver, buffer_length, num_subgoal_hits_required, subgoal_reward, max_steps, seed=0):
        self.mdp = mdp
        self.global_solver = global_solver
        self.buffer_length = buffer_length
        self.num_subgoal_hits_required = num_subgoal_hits_required
        self.subgoal_reward = subgoal_reward
        self.max_steps = max_steps
        self.seed = seed

    def augment_global_agent_with_option(self, old_solver, newly_trained_option):
        trained_options = old_solver.trained_options + [newly_trained_option]
        num_actions = old_solver.num_original_actions + len(trained_options)
        num_state_dimensions = self.mdp.init_state.state_space_size()
        new_global_agent = DQNAgent(num_state_dimensions, num_actions, old_solver.num_original_actions,
                                    trained_options, seed=self.seed, name=old_solver.name,
                                    eps_start=old_solver.epsilon,
                                    tensor_log=old_solver.tensor_log,
                                    use_double_dqn=old_solver.use_ddqn,
                                    lr=old_solver.learning_rate)

        new_global_agent.policy_network.initialize_with_smaller_network(old_solver.policy_network)
        new_global_agent.target_network.initialize_with_smaller_network(old_solver.target_network)

        self.global_solver = new_global_agent
        return new_global_agent

    def create_goal_option(self, path_to_policy_net, path_to_target_net, init_classifier_pickle):
        name = "overall_goal_policy"
        goal_agent = DQNAgent(self.mdp.init_state.state_space_size(), len(self.mdp.actions), len(self.mdp.actions),
							   trained_options=[], seed=self.seed, name=name, use_double_dqn=self.global_solver.use_ddqn,
							   lr=self.global_solver.learning_rate, tensor_log=self.global_solver.tensor_log)
        goal_agent.policy_network.load_state_dict(torch.load(path_to_policy_net))
        goal_agent.target_network.load_state_dict(torch.load(path_to_target_net))
        goal_predicate = self.mdp.default_goal_predicate()
        with open(init_classifier_pickle, "rb") as _f:
            initiation_classifier = pickle.load(_f)
        init_predicate = Predicate(func=lambda s: initiation_classifier.predict([s.features()])[0] == 1,
                                   name=name + '_init_predicate')
        goal_option = Option(init_predicate=init_predicate, term_predicate=goal_predicate, overall_mdp=self.mdp,
                             init_state=self.mdp.init_state, actions=self.mdp.actions, policy={},
                             name=name, term_prob=0., global_solver=self.global_solver, buffer_length=self.buffer_length,
                             pretrained=True, num_subgoal_hits_required=self.num_subgoal_hits_required,
                             subgoal_reward=self.subgoal_reward, seed=self.seed, max_steps=self.max_steps, parent=None,
                             classifier_type="ocsvm")
        goal_option.initiation_classifier = initiation_classifier
        goal_option.solver = goal_agent
        goal_option.solver.epsilon = 0.01
        return goal_option

    # TODO: Something wrong with child option initiation set classifiers (not getting executed)
    def create_subgoal_option(self, name, previous_option, path_to_policy_net, path_to_target_net, init_classifier_pickle):
        agent = DQNAgent(self.mdp.init_state.state_space_size(), len(self.mdp.actions), len(self.mdp.actions),
                       trained_options=[], seed=self.seed, name=name, use_double_dqn=self.global_solver.use_ddqn,
                       lr=self.global_solver.learning_rate, tensor_log=self.global_solver.tensor_log)
        agent.policy_network.load_state_dict(torch.load(path_to_policy_net))
        agent.target_network.load_state_dict(torch.load(path_to_target_net))
        agent.epsilon = 0.01
        with open(init_classifier_pickle, "rb") as _f:
            initiation_classifier = pickle.load(_f)
        init_predicate = Predicate(func=lambda s: initiation_classifier.predict([s.features()])[0] == 1,
                                   name=name + '_init_predicate')
        option = previous_option.create_child_option(init_state=deepcopy(self.mdp.init_state),
													actions=self.mdp.actions,
													new_option_name=name,
													global_solver=self.global_solver,
													buffer_length=self.buffer_length,
													num_subgoal_hits=self.num_subgoal_hits_required,
													subgoal_reward=self.subgoal_reward,
                                                    pretrained=True)
        option.init_predicate = init_predicate
        option.initiation_classifier = initiation_classifier
        option.solver = agent
        return option

    def get_pretrained_options(self):
        data_dir = os.getcwd()

        path_to_goal_policy_network = os.path.join(data_dir, "overall_goal_policy_policy_dqn.pth")
        path_to_goal_target_network = os.path.join(data_dir, "overall_goal_policy_target_dqn.pth")
        path_to_o1_policy_network = os.path.join(data_dir, "option_1_policy_dqn.pth")
        path_to_o1_target_network = os.path.join(data_dir, "option_1_target_dqn.pth")
        path_to_o2_policy_network = os.path.join(data_dir, "option_2_policy_dqn.pth")
        path_to_o2_target_network = os.path.join(data_dir, "option_2_target_dqn.pth")

        path_to_og_init_clf_pkl = os.path.join(data_dir, "overall_goal_policy_svm.pkl")
        path_to_o1_init_clf_pkl = os.path.join(data_dir, "option_1_svm.pkl")
        path_to_o2_init_clf_pkl = os.path.join(data_dir, "option_2_svm.pkl")

        goal_option = self.create_goal_option(path_to_goal_policy_network, path_to_goal_target_network, path_to_og_init_clf_pkl)
        option_1 = self.create_subgoal_option("option_1", goal_option, path_to_o1_policy_network, path_to_o1_target_network, path_to_o1_init_clf_pkl)
        option_2 = self.create_subgoal_option("option_2", option_1, path_to_o2_policy_network, path_to_o2_target_network, path_to_o2_init_clf_pkl)

        return [goal_option, option_1, option_2]


