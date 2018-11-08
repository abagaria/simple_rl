# Python imports.
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()

# Other imports.
from simple_rl.skill_chaining.SkillChainingClass import SkillChaining, construct_lunar_lander_mdp
from simple_rl.agents.func_approx.TorchDQNAgentClass import DQNAgent, main

class SkillChainingExperiments(object):
    def __init__(self, mdp, num_episodes=1200, num_instances=3, random_seed=0):
        self.mdp = mdp
        self.num_episodes = num_episodes
        self.num_instances = num_instances
        self.random_seed = random_seed

    def compare_agents(self):
        print("=" * 80)
        print("Training skill chaining agent..")
        print("=" * 80)
        list_scores, episode_numbers = [], []
        for i in range(self.num_instances):
            print("\nInstance {} of {}".format(i+1, self.num_instances))
            solver = DQNAgent(self.mdp.env.observation_space.shape[0], self.mdp.env.action_space.n, 0)
            skill_chaining_agent = SkillChaining(self.mdp, self.mdp.goal_predicate, rl_agent=solver)
            episodic_scores = skill_chaining_agent.skill_chaining(num_episodes=self.num_episodes)
            episode_numbers += list(range(self.num_episodes))
            list_scores += episodic_scores

        print("\n")
        print("=" * 80)
        print("Training baseline DQN agent..")
        print("=" * 80)

        dqn_list_scores = []
        for i in range(self.num_instances):
            print("\nInstance {} of {}".format(i+1, self.num_instances))
            dqn_scores = main(num_training_episodes=self.num_episodes)
            dqn_list_scores += dqn_scores

        scores = list_scores + dqn_list_scores
        episodes = episode_numbers + episode_numbers
        algorithm = ["skill_chaining"] * len(episode_numbers) + ["dqn"] * len(episode_numbers)
        scores_dataframe = pd.DataFrame(np.array(scores), columns=["reward"])
        scores_dataframe["episode"] = np.array(episodes)

        plt.figure()
        scores_dataframe["method"] = np.array(algorithm)
        sns.lineplot(x="episode", y="reward", hue="method", data=scores_dataframe)
        plt.savefig("skill_chaining_comparison.png")

    def experiment_initializing_option_policy(self):
        print("=" * 80)
        print("Training skill chaining agent w/o any copying..")
        print("=" * 80)
        list_scores, episode_numbers = [], []
        for i in range(self.num_instances):
            print("\nInstance {} of {}".format(i + 1, self.num_instances))
            solver = DQNAgent(self.mdp.env.observation_space.shape[0], self.mdp.env.action_space.n, 0)
            skill_chaining_agent = SkillChaining(self.mdp, self.mdp.goal_predicate, rl_agent=solver)
            episodic_scores = skill_chaining_agent.skill_chaining(num_episodes=self.num_episodes)
            episode_numbers += list(range(self.num_episodes))
            list_scores += episodic_scores

        print("")
        print("=" * 80)
        print("Training skill chaining agent w/ copying the policy network..")
        print("=" * 80)
        copy_policy_net_scores = []
        for i in range(self.num_instances):
            print("\nInstance {} of {}".format(i + 1, self.num_instances))
            solver = DQNAgent(self.mdp.env.observation_space.shape[0], self.mdp.env.action_space.n, 0)
            skill_chaining_agent = SkillChaining(self.mdp, self.mdp.goal_predicate, rl_agent=solver, to_copy_policy_net=True)
            policy_scores = skill_chaining_agent.skill_chaining(num_episodes=self.num_episodes)
            copy_policy_net_scores += policy_scores

        print("")
        print("=" * 80)
        print("Training skill chaining agent w/ copying the target network..")
        print("=" * 80)
        copy_target_net_scores = []
        for i in range(self.num_instances):
            print("\nInstance {} of {}".format(i + 1, self.num_instances))
            solver = DQNAgent(self.mdp.env.observation_space.shape[0], self.mdp.env.action_space.n, 0)
            skill_chaining_agent = SkillChaining(self.mdp, self.mdp.goal_predicate, rl_agent=solver,
                                                 to_copy_policy_net=True, to_copy_target_net=True)
            target_scores = skill_chaining_agent.skill_chaining(num_episodes=self.num_episodes)
            copy_target_net_scores += target_scores

        scores = list_scores + copy_policy_net_scores + copy_target_net_scores
        episodes = episode_numbers + episode_numbers + episode_numbers
        algorithm = ["no_copying"] * len(episode_numbers) + ["copy_policy_net"] * len(episode_numbers) + \
                    ["copy_target_net"] * len(episode_numbers)
        scores_dataframe = pd.DataFrame(np.array(scores), columns=["reward"])
        scores_dataframe["episode"] = np.array(episodes)

        plt.figure()
        scores_dataframe["method"] = np.array(algorithm)
        sns.lineplot(x="episode", y="reward", hue="method", data=scores_dataframe)
        plt.savefig("policy_initialization_comparison.png")


if __name__ == '__main__':
    overall_mdp = construct_lunar_lander_mdp()
    experiments = SkillChainingExperiments(overall_mdp)
    # experiments.compare_agents()
    experiments.experiment_initializing_option_policy()
