# Python imports.
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pdb

# Other imports.
from simple_rl.mdp.MDPClass import MDP
from simple_rl.abstraction.action_abs.PredicateClass import Predicate
from simple_rl.abstraction.action_abs.OptionClass import Option
from simple_rl.agents.AgentClass import Agent
from simple_rl.run_experiments import run_agent_on_mdp
from simple_rl.mdp.StateClass import State
from simple_rl.agents import LinearQAgent
from simple_rl.tasks import GymMDP
from sklearn import svm

class SkillChaining(object):
    def __init__(self, mdp, rl_agent=None):
        """
        Args:
            mdp (MDP): Underlying domain we have to solve
            rl_agent (Agent): RL agent used to determine the policy for each option
        """
        self.mdp = mdp
        self.agent = rl_agent if rl_agent is not None else LinearQAgent(mdp.get_actions(), mdp.get_num_state_feats())
        self.subgoal_reward = 0.

    def learn_option(self, goal_predicate, plot_initiation_set=False):
        """
        Given a trigger function, learn the policy and initiation set for the
        option corresponding to that trigger function.
        Args:
            goal_predicate (Predicate): Maps state to {0, 1} indicating whether or not state is a goal state

        Returns:
            learned_option (Option)
        """
        self.mdp.subgoal_predicate = goal_predicate
        policy, reached_goal, examples = self._unleash_rl_agent()
        initiation_classifier = self._learn_initiation_classifier(examples[0], examples[1])
        init_predicate = Predicate(func=lambda s: initiation_classifier.predict([s])[0])
        learned_option = Option(init_predicate, goal_predicate, policy)

        # While the policy is defined over the full state (for pendulum this is [cos(x), sin(x), xdot]), we are only
        # visualizing x and the predicted labels
        if plot_initiation_set:
            thetas = map(lambda s: np.arccos(s.data[0]), policy.keys())
            predicted_labels = initiation_classifier.predict(self._construct_feature_matrix(policy.keys()))
            true_labels = np.array( ([1] * len(examples[0])) + ([0] * (len(examples[1])-1)) )

            plt.figure()
            plt.scatter(np.rad2deg(thetas), predicted_labels, c=true_labels, alpha=0.5)
            plt.xlabel('Theta')
            plt.ylabel('Predicted label')
            plt.title('Initiation Set classifier training set')
            plt.legend()
            plt.show()

        return learned_option

    @staticmethod
    def _learn_initiation_classifier(positive_states, negative_states):
        """
        Construct a binary classifier for the current option.
        Args:
            positive_states (list): Example states in the initiation set of the option
            negative_states (list): Example states Not in the initiation set of the option

        Returns:
            classifier (svm.SVC): F: s --> {0, 1} indicating whether or not the state (s) is in the initiation set
        """
        all_states = positive_states + negative_states
        X = SkillChaining._construct_feature_matrix(states=all_states)
        Y = np.array( ([1] * len(positive_states)) + ([0] * len(negative_states)) )

        # Train a Support Vector Classifier
        classifier = svm.SVC()
        classifier.fit(X, Y)

        return classifier

    @staticmethod
    def _construct_feature_matrix(states):
        n_samples = len(states)
        n_features = states[0].get_num_feats()
        X = np.zeros((n_samples, n_features))
        for row in range(X.shape[0]):
            X[row, :] = states[row].features()
        return X

    def _unleash_rl_agent(self):
        """
        Given our current state and the goal trigger function, act in the environment until you encounter the goal.

        Returns:
            policy (function): pi(s) --> a
            reached_goal (bool): Did the RL Agent encounter the goal state
            examples (list): LoL of [[positive_examples], [negative_examples]] for constructing the initiation set
        """
        return run_agent_on_mdp(self.agent, self.mdp)


def construct_pendulum_domain(predicate):
    # Pendulum domain parameters
    max_torque = 1.5

    # Construct GymMDP wrapping around Gym's Pendulum MDP
    discretized_actions = np.arange(-max_torque, max_torque + 0.1, 0.1)
    return GymMDP(discretized_actions, subgoal_predicate=predicate, env_name='Pendulum-v0', render=True)

if __name__ == '__main__':
    target_cos_theta, cos_theta_tol = 1., 0.001
    goal_pred = Predicate(lambda s: (abs(s[0] - target_cos_theta) < cos_theta_tol))
    overall_mdp = construct_pendulum_domain(goal_pred)
    chainer = SkillChaining(overall_mdp)
    lo = chainer.learn_option(goal_pred, plot_initiation_set=True)