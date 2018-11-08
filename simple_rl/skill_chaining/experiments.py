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
    def __init__(self, mdp, num_episodes=10, num_instances=3, random_seed=0):
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

    def compare_hyperparameter_subgoal_reward(self):
        print("=" * 80)
        print("Training skill chaining agent..")
        print("=" * 80)
        list_scores = []
        for n in [0., 20.]:
            for i in range(self.num_instances):
                print("\nInstance {} of {}".format(i + 1, self.num_instances))
                print("-" * 80)
                solver = DQNAgent(self.mdp.env.observation_space.shape[0], self.mdp.env.action_space.n, 0)
                skill_chaining_agent = SkillChaining(self.mdp, self.mdp.goal_predicate, subgoal_reward=n, rl_agent=solver)
                episodic_scores = skill_chaining_agent.skill_chaining(num_episodes=self.num_episodes)
                list_scores.append((n, np.mean(episodic_scores), np.std(episodic_scores)))

        return list_scores

    def to_step_or_not_to_step(self):
        print("=" * 80)
        print("To Step")
        print("=" * 80)
        to_step_scores, episode_numbers = [], []
        for i in range(self.num_instances):
            print("\nInstance {} of {}".format(i + 1, self.num_instances))
            solver = DQNAgent(self.mdp.env.observation_space.shape[0], self.mdp.env.action_space.n, 0)
            skill_chaining_agent = SkillChaining(self.mdp, self.mdp.goal_predicate, rl_agent=solver, step_before=True)
            episodic_scores = skill_chaining_agent.skill_chaining(num_episodes=self.num_episodes)
            episode_numbers += list(range(self.num_episodes))
            to_step_scores += episodic_scores

        print("")
        print("=" * 80)
        print("Not To Step")
        print("=" * 80)
        not_to_step_scores, episode_numbers = [], []
        for i in range(self.num_instances):
            print("\nInstance {} of {}".format(i + 1, self.num_instances))
            solver = DQNAgent(self.mdp.env.observation_space.shape[0], self.mdp.env.action_space.n, 0)
            skill_chaining_agent = SkillChaining(self.mdp, self.mdp.goal_predicate, rl_agent=solver, step_before=False)
            episodic_scores = skill_chaining_agent.skill_chaining(num_episodes=self.num_episodes)
            episode_numbers += list(range(self.num_episodes))
            not_to_step_scores += episodic_scores

        scores = to_step_scores + not_to_step_scores
        episodes = episode_numbers + episode_numbers
        algorithm = ["step_before"] * len(episode_numbers) + ["dont_step_before"] * len(episode_numbers)
        scores_dataframe = pd.DataFrame(np.array(scores), columns=["reward"])
        scores_dataframe["episode"] = np.array(episodes)

        plt.figure()
        scores_dataframe["method"] = np.array(algorithm)
        sns.lineplot(x="episode", y="reward", hue="method", data=scores_dataframe)
        plt.savefig("stepping_before_comparison.png")

        return scores_dataframe

if __name__ == '__main__':
    overall_mdp = construct_lunar_lander_mdp()
    experiments = SkillChainingExperiments(overall_mdp)
    # experiments.compare_agents()
    # subgoal_scores = experiments.compare_hyperparameter_subgoal_reward()
    scores_dataframe = experiments.to_step_or_not_to_step()
