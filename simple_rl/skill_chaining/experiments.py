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
    def __init__(self, mdp, num_episodes=120, num_instances=1, random_seed=0):
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
        subgoal_reward = 1.0
        learning_rate = 5e-4  # 0.1 of the one we usually use
        random_seeds = [0] #, 20, 123, 4351] Because I have only tested the init sets for seed=0
        max_num_options = [5, 0]
        scores = []
        episodes = []
        algorithms = []

        for num_options in max_num_options:
            for random_seed in random_seeds:
                print()
                print("=" * 80)
                print("Training skill chaining agent (seed={}, n_options={})".format(random_seed, num_options))
                print("=" * 80)
                list_scores, episode_numbers = [], []
                for i in range(self.num_instances):
                    print("\nInstance {} of {}".format(i + 1, self.num_instances))

                    self.mdp = PinballMDP(noise=0., episode_length=20000, render=True)
                    self.state_size = self.mdp.init_state.state_space_size()
                    self.num_actions = len(self.mdp.actions)

                    solver = DQNAgent(self.state_size, self.num_actions, self.num_actions, [], seed=random_seed,
                                      name="GlobalDDQN{}".format(num_options), tensor_log=False, use_double_dqn=True,
                                      lr=learning_rate)
                    skill_chaining_agent = SkillChaining(self.mdp, self.mdp.goal_predicate, rl_agent=solver,
                                                         subgoal_reward=subgoal_reward, buffer_length=buffer_len,
                                                         max_num_options=num_options, seed=random_seed, lr_decay=True)
                    episodic_scores, episodic_durations = skill_chaining_agent.skill_chaining(num_episodes=self.num_episodes, num_steps=20000)
                    skill_chaining_agent.perform_experiments()
                    # skill_chaining_agent.save_all_dqns()
                    # skill_chaining_agent.save_all_initiation_classifiers()
                    episode_numbers += list(range(self.num_episodes))
                    list_scores += episodic_scores
                scores += list_scores
                episodes += episode_numbers
                algorithms += ["nOptions={}".format(num_options)] * len(episode_numbers)
        scores_dataframe = pd.DataFrame(np.array(scores), columns=["reward"])
        scores_dataframe["episode"] = np.array(episodes)

        plt.figure()
        scores_dataframe["method"] = np.array(algorithms)
        sns.lineplot(x="episode", y="reward", hue="method", data=scores_dataframe, estimator=np.median)
        plt.title("Skill Chaining Learning Curves")
        plt.savefig("sc_v_dqn.png")
        plt.show()

        self.data_frame = scores_dataframe
        scores_dataframe.to_pickle("sc_v_dqn.pkl")

        return scores_dataframe

    def ddqn_hyper_params(self):
        from simple_rl.agents.func_approx.TorchDQNAgentClass import train
        random_seeds = [0, 20, 123] #, 4351]
        learning_rates = [5e-6, 5e-5, 5e-4, 5e-3]
        scores = []
        episodes = []
        algorithms = []

        for lr in learning_rates:
            for random_seed in random_seeds:
                print()
                print("=" * 80)
                print("Training DDQN agent (seed={}, lr={})".format(random_seed, lr))
                print("=" * 80)
                list_scores, episode_numbers = [], []
                for i in range(self.num_instances):
                    print("\nInstance {} of {}".format(i + 1, self.num_instances))

                    self.mdp = PinballMDP(noise=0., episode_length=20000, render=False)
                    self.state_size = self.mdp.init_state.state_space_size()
                    self.num_actions = len(self.mdp.actions)

                    solver = DQNAgent(self.state_size, self.num_actions, self.num_actions, [], seed=random_seed,
                                      name="GlobalDDQN", tensor_log=False, lr=lr, use_double_dqn=True)

                    episodic_scores = train(solver, self.mdp, self.num_episodes, 20000)
                    episode_numbers += list(range(self.num_episodes))
                    list_scores += episodic_scores
                scores += list_scores
                episodes += episode_numbers
                algorithms += ["lr={}".format(lr)] * len(episode_numbers)
        # pdb.set_trace()
        scores_dataframe = pd.DataFrame(np.array(scores), columns=["reward"])
        scores_dataframe["episode"] = np.array(episodes)

        plt.figure()
        scores_dataframe["method"] = np.array(algorithms)
        sns.lineplot(x="episode", y="reward", hue="method", data=scores_dataframe, estimator=np.median)
        plt.title("Double DQN Learning Curves")
        plt.savefig("ddqn_lr_tuning.png")
        plt.show()

        self.data_frame = scores_dataframe
        scores_dataframe.to_pickle("ddqn_lrs.pkl")

        return scores_dataframe

    def dqn_vs_ddqn(self):
        from simple_rl.agents.func_approx.TorchDQNAgentClass import train
        random_seeds = [0, 20, 123] #, 4351]
        use_ddqn = [True, False]
        learning_rate = 5e-5 # 0.1 of the one we usually use
        scores = []
        episodes = []
        algorithms = []

        for use in use_ddqn:
            for random_seed in random_seeds:
                print()
                print("=" * 80)
                print("Training DDQN agent (seed={}, DDQN={})".format(random_seed, use))
                print("=" * 80)
                list_scores, episode_numbers = [], []
                for i in range(self.num_instances):
                    print("\nInstance {} of {}".format(i + 1, self.num_instances))

                    self.mdp = PinballMDP(noise=0., episode_length=20000, render=False)
                    self.state_size = self.mdp.init_state.state_space_size()
                    self.num_actions = len(self.mdp.actions)

                    solver = DQNAgent(self.state_size, self.num_actions, self.num_actions, [], seed=random_seed,
                                      name="GlobalDQN", tensor_log=False, lr=learning_rate, use_double_dqn=use)

                    episodic_scores = train(solver, self.mdp, self.num_episodes, 20000)
                    episode_numbers += list(range(self.num_episodes))
                    list_scores += episodic_scores
                scores += list_scores
                episodes += episode_numbers
                algorithms += ["DDQN={}".format(use)] * len(episode_numbers)
        # pdb.set_trace()
        scores_dataframe = pd.DataFrame(np.array(scores), columns=["reward"])
        scores_dataframe["episode"] = np.array(episodes)

        plt.figure()
        scores_dataframe["method"] = np.array(algorithms)
        sns.lineplot(x="episode", y="reward", hue="method", data=scores_dataframe, estimator=np.median)
        plt.title("Double DQN vs DQN Learning Curves")
        plt.savefig("dqn_vs_ddqn.png")
        plt.show()

        self.data_frame = scores_dataframe
        scores_dataframe.to_pickle("dqn_vs_ddqn.pkl")

        return scores_dataframe

if __name__ == '__main__':
    experiments = SkillChainingExperiments(None)
    # experiments.compare_agents()
    # subgoal_scores = experiments.compare_hyperparameter_subgoal_reward()
    # experiments.run_skill_chaining_with_different_seeds()
    scdf = experiments.run_skill_chaining_with_different_seeds()
    # lrdf = experiments.ddqn_hyper_params()
    # dddf = experiments.dqn_vs_ddqn()
