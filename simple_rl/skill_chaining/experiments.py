# Python imports.
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
import pdb

# Other imports.
from simple_rl.skill_chaining.SkillChainingClass import SkillChaining
from simple_rl.agents.func_approx.TorchDQNAgentClass import DQNAgent, main
from simple_rl.skill_chaining.create_pre_trained_options import PretrainedOptionsLoader
from simple_rl.tasks.pinball.PinballMDPClass import PinballMDP

class SkillChainingExperiments(object):
    def __init__(self, mdp, num_episodes=100, num_instances=1, random_seed=0):
        self.mdp = mdp
        self.num_episodes = num_episodes
        self.num_instances = num_instances
        self.random_seed = random_seed

        self.data_frame = None

    def compare_agents(self):

        # Same buffer length used by SC-agent and transfer learning SC agent
        buffer_len = 50

        print("=" * 80)
        print("Training skill chaining agent..")
        print("=" * 80)
        list_scores, episode_numbers = [], []
        for i in range(self.num_instances):
            print("\nInstance {} of {}".format(i+1, self.num_instances))
            solver = DQNAgent(self.mdp.env.observation_space.shape[0], self.mdp.env.action_space.n, 0)
            skill_chaining_agent = SkillChaining(self.mdp, self.mdp.goal_predicate, rl_agent=solver, buffer_length=buffer_len)
            episodic_scores = skill_chaining_agent.skill_chaining(num_episodes=self.num_episodes)
            skill_chaining_agent.save_all_dqns()
            skill_chaining_agent.save_all_initiation_classifiers()
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

        print("\n")
        print("=" * 80)
        print("Training transfer skill chaining agent..")
        print("=" * 80)

        transfer_list_scores = []
        for i in range(self.num_instances):
            print("\nInstance {} of {}".format(i+1, self.num_instances))
            solver = DQNAgent(self.mdp.env.observation_space.shape[0], self.mdp.env.action_space.n, 0)
            loader = PretrainedOptionsLoader(overall_mdp, solver, buffer_length=buffer_len)
            pretrained_options = loader.get_pretrained_options()
            print("Running skill chaining with pretrained options: {}".format(pretrained_options))
            chainer = SkillChaining(overall_mdp, overall_mdp.goal_predicate, rl_agent=solver, buffer_length=buffer_len,
                                    pretrained_options=pretrained_options)
            transfer_scores = chainer.skill_chaining(num_episodes=self.num_episodes)
            transfer_list_scores += transfer_scores

        scores = list_scores + dqn_list_scores + transfer_list_scores
        episodes = episode_numbers + episode_numbers + episode_numbers
        algorithm = ["skill_chaining"] * len(episode_numbers) + ["dqn"] * len(episode_numbers) + ["given_options"] * len(episode_numbers)
        scores_dataframe = pd.DataFrame(np.array(scores), columns=["reward"])
        scores_dataframe["episode"] = np.array(episodes)

        # scores = list_scores + transfer_list_scores
        # episodes = episode_numbers + episode_numbers
        # algorithm = ["skill_chaining"] * len(episode_numbers) + ["given_options"] * len(episode_numbers)
        # scores_dataframe = pd.DataFrame(np.array(scores), columns=["reward"])
        # scores_dataframe["episode"] = np.array(episodes)

        plt.figure()
        scores_dataframe["method"] = np.array(algorithm)
        sns.lineplot(x="episode", y="reward", hue="method", data=scores_dataframe)
        plt.title("Learning curve with 1-class svm")
        plt.savefig("skill_chaining_comparison.png")

    def run_skill_chaining_with_different_seeds(self):
        # Same buffer length used by SC-agent and transfer learning SC agent
        buffer_len = 15

        random_seeds = [0]#, 24, 123, 4351]
        reward_scales = [10., 10000.]
        scores = []
        episodes = []
        algorithms = []

        for reward_scale in reward_scales:
            for random_seed in random_seeds:
                print()
                print("=" * 80)
                print("Training skill chaining agent (seed={}, Rmax={})".format(random_seed, reward_scale))
                print("=" * 80)
                list_scores, episode_numbers = [], []
                for i in range(self.num_instances):
                    print("\nInstance {} of {}".format(i + 1, self.num_instances))

                    self.mdp = PinballMDP(reward_scale=reward_scale, noise=0., episode_length=25000, render=False)
                    self.state_size = self.mdp.init_state.state_space_size()
                    self.num_actions = len(self.mdp.actions)

                    solver = DQNAgent(self.state_size, self.num_actions, self.num_actions, [], seed=random_seed, name="GlobalDQN")
                    skill_chaining_agent = SkillChaining(self.mdp, self.mdp.goal_predicate, rl_agent=solver,
                                                         buffer_length=buffer_len, max_num_options=0, subgoal_reward=reward_scale/2.)
                    episodic_scores = skill_chaining_agent.skill_chaining(num_episodes=self.num_episodes, num_steps=20000)
                    skill_chaining_agent.save_all_dqns()
                    skill_chaining_agent.save_all_initiation_classifiers()
                    episode_numbers += list(range(self.num_episodes))
                    list_scores += episodic_scores
                scores += list_scores
                episodes += episode_numbers
                algorithms += ["rScale={}".format(reward_scale)] * len(episode_numbers)
        # pdb.set_trace()
        scores_dataframe = pd.DataFrame(np.array(scores), columns=["reward"])
        scores_dataframe["episode"] = np.array(episodes)

        plt.figure()
        scores_dataframe["method"] = np.array(algorithms)
        sns.lineplot(x="episode", y="reward", hue="method", data=scores_dataframe)
        plt.title("Skill Chaining Learning Curves")
        plt.savefig("dqn_reward_scales.png")
        plt.show()

        self.data_frame = scores_dataframe
        scores_dataframe.to_pickle("successive_options_scores_df.pkl")


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

if __name__ == '__main__':
    experiments = SkillChainingExperiments(None)
    # experiments.compare_agents()
    # subgoal_scores = experiments.compare_hyperparameter_subgoal_reward()
    experiments.run_skill_chaining_with_different_seeds()
