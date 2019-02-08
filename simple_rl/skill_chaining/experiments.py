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
        self.val_data_frame = None

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
        random_seeds = [0, 20, 123, 4351, 77] # Because I have only tested the init sets for seed=0
        max_number_of_options = 3

        scores = []
        val_scores = []
        episodes = []
        val_episodes = []

        for random_seed in random_seeds:
            print()
            print("=" * 80)
            print("Training skill chaining agent (seed={}, n_options={})".format(random_seed, max_number_of_options))
            print("=" * 80)

            self.mdp = PinballMDP(noise=0.0, episode_length=20000, reward_scale=1000., render=True)
            self.state_size = self.mdp.init_state.state_space_size()
            self.num_actions = len(self.mdp.actions)

            solver = DQNAgent(self.state_size, len(self.mdp.actions), len(self.mdp.actions), [],
                              seed=random_seed, lr=learning_rate,
                              name="GlobalDDQN_{}".format(random_seed), eps_start=1.0, tensor_log=False,
                              use_double_dqn=True)

            skill_chaining_agent = SkillChaining(self.mdp, rl_agent=solver, buffer_length=buffer_len,
                            seed=random_seed, subgoal_reward=subgoal_reward, max_num_options=max_number_of_options,
                            lr_decay=False)

            episodic_scores, episodic_durations = skill_chaining_agent.skill_chaining(num_episodes=self.num_episodes, num_steps=20000)
            validation_scores = skill_chaining_agent.validation_scores

            skill_chaining_agent.perform_experiments()

            scores += episodic_scores
            val_scores += np.cumsum(validation_scores).tolist()
            episodes += list(range(len(episodic_scores)))
            val_episodes += list(range(len(validation_scores)))

        training_scores_dataframe = pd.DataFrame(np.array(scores), columns=["reward"])
        training_scores_dataframe["episode"] = np.array(episodes)

        validation_scores_dataframe = pd.DataFrame(np.array(val_scores), columns=["cumulative_reward"])
        validation_scores_dataframe["episode"] = np.array(val_episodes)

        plt.figure()
        sns.lineplot(x="episode", y="reward", data=training_scores_dataframe, estimator=np.median)
        plt.title("Skill Chaining Scores (Training)")
        plt.savefig("sc_episodic_rewards.png")
        plt.show()

        plt.figure()
        sns.lineplot(x="episode", y="cumulative_reward", data=validation_scores_dataframe, estimator=np.median)
        plt.title("Skill Chaining Cumulative Rewards (Validation)")
        plt.savefig("sc_cum_validation_scores.png")
        plt.show()

        self.data_frame = training_scores_dataframe
        training_scores_dataframe.to_pickle("sc_training.pkl")
        validation_scores_dataframe.to_pickle("sc_val.pkl")

        return training_scores_dataframe, validation_scores_dataframe

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
        reward_scale = 1. # rewards in [-1., 10,000.]
        grad_clip_vals = [5., 10., 100., 1000.]
        learning_rate = 1e-4  # 0.1 of the one we usually use
        random_seeds = [0, 20, 123, 4351] # Because I have only tested the init sets for seed=0
        training_scores = []
        validation_scores = []
        episodes = []
        algorithms = []

        for grad_clip in grad_clip_vals:
            for random_seed in random_seeds:
                print()
                print("=" * 80)
                print("Training skill chaining agent (seed={}, clip_val={})".format(random_seed, grad_clip))
                print("=" * 80)
                list_scores, validation_list_scores, episode_numbers = [], [], []
                for i in range(self.num_instances):
                    print("\nInstance {} of {}".format(i + 1, self.num_instances))

                    self.mdp = PinballMDP(noise=0., episode_length=20000, reward_scale=reward_scale, render=False)
                    self.state_size = self.mdp.init_state.state_space_size()
                    self.num_actions = len(self.mdp.actions)

                    solver = DQNAgent(self.state_size, self.num_actions, self.num_actions, [], seed=random_seed,
                                      name="GlobalDDQN_Clip_{}".format(grad_clip), tensor_log=False, use_double_dqn=True,
                                      lr=learning_rate, loss_function="mse", gradient_clip=grad_clip)
                    skill_chaining_agent = SkillChaining(self.mdp, self.mdp.goal_predicate, rl_agent=solver,
                                                         subgoal_reward=sub_reward, buffer_length=buffer_len,
                                                         max_num_options=max_num_options, seed=random_seed, lr_decay=False,
                                                         option_subgoal_reward_ratio=0.5)
                    episodic_scores, episodic_durations = skill_chaining_agent.skill_chaining(
                        num_episodes=self.num_episodes, num_steps=20000)
                    skill_chaining_agent.save_all_scores(False, episodic_scores, episodic_durations)

                    episode_numbers += list(range(self.num_episodes))
                    list_scores += episodic_scores
                    validation_list_scores += skill_chaining_agent.validation_scores

                training_scores += list_scores
                validation_scores += validation_list_scores
                episodes += episode_numbers
                algorithms += ["gradClip={}".format(grad_clip)] * len(episode_numbers)

        scores_dataframe = pd.DataFrame(np.array(training_scores), columns=["reward"])
        val_scores_dataframe = pd.DataFrame(np.array(validation_scores), columns=["reward"])
        scores_dataframe["episode"] = np.array(episodes)
        val_scores_dataframe["episode"] = np.array(episodes)

        plt.figure()
        scores_dataframe["method"] = np.array(algorithms)
        sns.lineplot(x="episode", y="reward", hue="method", data=scores_dataframe, estimator=np.median)
        plt.title(experiment_name)
        plt.savefig("{}_training.png".format(experiment_name))
        # plt.show()

        self.data_frame = scores_dataframe
        scores_dataframe.to_pickle("{}_training.pkl".format(experiment_name))

        plt.figure()
        val_scores_dataframe["method"] = np.array(algorithms)
        sns.lineplot(x="episode", y="reward", hue="method", data=val_scores_dataframe, estimator=np.median)
        plt.title(experiment_name)
        plt.savefig("{}_validation.png".format(experiment_name))
        # plt.show()

        self.val_data_frame = val_scores_dataframe
        val_scores_dataframe.to_pickle("{}_validation.pkl".format(experiment_name))

        return scores_dataframe


if __name__ == '__main__':
    experiments = SkillChainingExperiments(None)
    # experiments.compare_agents()
    # subgoal_scores = experiments.compare_hyperparameter_subgoal_reward()
    tdf, vdf = experiments.run_skill_chaining_with_different_seeds()
    # scdf = experiments.run_skill_chaining_with_different_seeds()
    # lrdf = experiments.ddqn_hyper_params()
    # dddf = experiments.dqn_vs_ddqn()
    # scdf = experiments.tune_skill_chaining_hyper_params(experiment_name="sc_mse_grad_clipping_comparisons_sgr_0_5")
    # scdf = experiments.run_sc_and_pretrained()
