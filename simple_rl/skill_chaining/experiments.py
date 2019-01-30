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
    def __init__(self, mdp, num_episodes=251, num_instances=1, random_seed=0):
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
        buffer_len = 20
        subgoal_reward = 1.0
        learning_rate = 1e-4  # 0.1 of the one we usually use
        random_seeds = [0] #, 20, 123, 4351, 77] # Because I have only tested the init sets for seed=0
        max_num_options = [1, 2]
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
                                                         max_num_options=num_options, seed=random_seed, lr_decay=False)
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
        plt.savefig("sc_v_dqn_neg_reward.png")
        plt.show()

        self.data_frame = scores_dataframe
        scores_dataframe.to_pickle("sc_v_dqn_neg_reward.pkl")

        return scores_dataframe

    def run_sc_and_pretrained(self):
        buffer_len = 20
        subgoal_reward = 1.0
        learning_rate = 1e-4  # 0.1 of the one we usually use
        random_seeds = [0, 20, 123, 4351, 77] # Because I have only tested the init sets for seed=0

        scores = []
        episodes = []
        algorithms = []

        for random_seed in random_seeds:
            print()
            print("=" * 80)
            print("Training skill chaining agent (seed={}, pretrained=False)".format(random_seed))
            print("=" * 80)
            list_scores, episode_numbers = [], []

            self.mdp = PinballMDP(noise=0., episode_length=20000, render=True)
            self.state_size = self.mdp.init_state.state_space_size()
            self.num_actions = len(self.mdp.actions)

            solver = DQNAgent(self.state_size, self.num_actions, self.num_actions, [], seed=random_seed,
                              name="GlobalDDQN", tensor_log=False, use_double_dqn=True,
                              lr=learning_rate)
            skill_chaining_agent = SkillChaining(self.mdp, self.mdp.goal_predicate, rl_agent=solver,
                                                 subgoal_reward=subgoal_reward, buffer_length=buffer_len,
                                                 max_num_options=3, seed=random_seed, lr_decay=False)
            episodic_scores, episodic_durations = skill_chaining_agent.skill_chaining(num_episodes=self.num_episodes, num_steps=20000)
            skill_chaining_agent.perform_experiments()
            skill_chaining_agent.save_all_scores(pretrained=False, episodic_scores=episodic_scores, episodic_durations=episodic_durations)


            episode_numbers += list(range(self.num_episodes))
            list_scores += episodic_scores
            scores += list_scores
            episodes += episode_numbers
            algorithms += ["pretrained=False"] * len(episode_numbers)

            print()
            print("=" * 80)
            print("Training skill chaining agent (seed={}, pretrained=True)".format(random_seed))
            print("=" * 80)
            list_scores, episode_numbers = [], []

            self.mdp = PinballMDP(noise=0., episode_length=20000, render=True)
            self.state_size = self.mdp.init_state.state_space_size()
            self.num_actions = len(self.mdp.actions)

            solver = DQNAgent(self.state_size, self.num_actions, self.num_actions, [], seed=random_seed,
                              name="GlobalDDQN", tensor_log=False, use_double_dqn=True,
                              lr=learning_rate)
            loader = PretrainedOptionsLoader(self.mdp, solver, buffer_len, num_subgoal_hits_required=3,
                                             subgoal_reward=subgoal_reward, max_steps=20000, seed=random_seed)
            pretrained_options = loader.get_pretrained_options()
            for pretrained_option in pretrained_options:
                solver = loader.augment_global_agent_with_option(solver, pretrained_option)

            skill_chaining_agent = SkillChaining(self.mdp, self.mdp.goal_predicate, rl_agent=solver, buffer_length=buffer_len,
                                seed=random_seed, subgoal_reward=subgoal_reward, max_num_options=0,
                                lr_decay=False, pretrained_options=pretrained_options)

            episodic_scores, episodic_durations = skill_chaining_agent.skill_chaining(num_episodes=self.num_episodes, num_steps=20000)
            skill_chaining_agent.save_all_scores(pretrained=True, episodic_scores=episodic_scores, episodic_durations=episodic_durations)

            episode_numbers += list(range(self.num_episodes))
            list_scores += episodic_scores
            scores += list_scores
            episodes += episode_numbers
            algorithms += ["pretrained=True"] * len(episode_numbers)

        scores_dataframe = pd.DataFrame(np.array(scores), columns=["reward"])
        scores_dataframe["episode"] = np.array(episodes)

        plt.figure()
        scores_dataframe["method"] = np.array(algorithms)
        sns.lineplot(x="episode", y="reward", hue="method", data=scores_dataframe, estimator=np.median)
        plt.title("Skill Chaining Learning Curves")
        plt.savefig("sc_with_pretrained_options.png")
        plt.show()

        self.data_frame = scores_dataframe
        scores_dataframe.to_pickle("sc_with_pretrained_options.pkl")

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

    def tune_skill_chaining_hyper_params(self, experiment_name):
        buffer_len = 20
        sub_reward = 1.
        max_num_options = 3
        negative_reward_ratios = [0., 0.5, 0.75, 1.0]
        learning_rate = 1e-4  # 0.1 of the one we usually use
        random_seeds = [0, 20, 123, 4351] # Because I have only tested the init sets for seed=0
        scores = []
        episodes = []
        algorithms = []

        for negative_reward_ratio in negative_reward_ratios:
            for random_seed in random_seeds:
                print()
                print("=" * 80)
                print("Training skill chaining agent (seed={}, neg_subgoal_reward_r={})".format(random_seed, negative_reward_ratio))
                print("=" * 80)
                list_scores, episode_numbers = [], []
                for i in range(self.num_instances):
                    print("\nInstance {} of {}".format(i + 1, self.num_instances))

                    self.mdp = PinballMDP(noise=0., episode_length=20000, render=False)
                    self.state_size = self.mdp.init_state.state_space_size()
                    self.num_actions = len(self.mdp.actions)

                    solver = DQNAgent(self.state_size, self.num_actions, self.num_actions, [], seed=random_seed,
                                      name="GlobalDDQN", tensor_log=False, use_double_dqn=True, lr=learning_rate)
                    skill_chaining_agent = SkillChaining(self.mdp, self.mdp.goal_predicate, rl_agent=solver,
                                                         subgoal_reward=sub_reward, buffer_length=buffer_len,
                                                         max_num_options=max_num_options, seed=random_seed, lr_decay=False,
                                                         option_subgoal_reward_ratio=negative_reward_ratio)
                    episodic_scores, episodic_durations = skill_chaining_agent.skill_chaining(
                        num_episodes=self.num_episodes, num_steps=20000)
                    skill_chaining_agent.save_all_scores(False, episodic_scores, episodic_durations)

                    episode_numbers += list(range(self.num_episodes))
                    list_scores += episodic_scores
                scores += list_scores
                episodes += episode_numbers
                algorithms += ["negativeRatio={}".format(negative_reward_ratio)] * len(episode_numbers)
        scores_dataframe = pd.DataFrame(np.array(scores), columns=["reward"])
        scores_dataframe["episode"] = np.array(episodes)

        plt.figure()
        scores_dataframe["method"] = np.array(algorithms)
        sns.lineplot(x="episode", y="reward", hue="method", data=scores_dataframe, estimator=np.median)
        plt.title(experiment_name)
        plt.savefig("{}.png".format(experiment_name))
        # plt.show()

        self.data_frame = scores_dataframe
        scores_dataframe.to_pickle("{}.pkl".format(experiment_name))

        return scores_dataframe


if __name__ == '__main__':
    experiments = SkillChainingExperiments(None)
    # experiments.compare_agents()
    # subgoal_scores = experiments.compare_hyperparameter_subgoal_reward()
    # experiments.run_skill_chaining_with_different_seeds()
    # scdf = experiments.run_skill_chaining_with_different_seeds()
    # lrdf = experiments.ddqn_hyper_params()
    # dddf = experiments.dqn_vs_ddqn()
    scdf = experiments.tune_skill_chaining_hyper_params(experiment_name="sc_negative_sub_reward_ratio_tuning_huber")
    # scdf = experiments.run_sc_and_pretrained()
