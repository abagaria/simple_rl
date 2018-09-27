#!/usr/bin/env python

# Python imports.
import pdb
import logging
import numpy as np

# Other imports.
from simple_rl.agents import RandomAgent, LinearQAgent
from simple_rl.tasks import GymMDP
from simple_rl.run_experiments import run_agent_on_mdp
from simple_rl.abstraction.action_abs.PredicateClass import Predicate

def main(open_plot=True):
    # simple_rl MDPs are set up for discrete actions, discretize the torque input space for the Pendulum environment
    max_torque = 2.
    discretized_actions = np.arange(-max_torque, max_torque+0.1, 0.1)

    # Gym MDP
    overall_target_cos_theta = 1.
    cos_theta_tolerance = 0.001
    predicate = Predicate(func=lambda s: (abs(s[0] - overall_target_cos_theta) < cos_theta_tolerance))
    gym_mdp = GymMDP(discretized_actions, subgoal_predicate=predicate, env_name='Pendulum-v0', render=True)
    num_feats = gym_mdp.get_num_state_feats()

    # Setup agents and run.
    rand_agent = RandomAgent(gym_mdp.get_actions())
    dqn_agent = LinearQAgent(gym_mdp.get_actions(), num_feats)
    # run_agents_on_mdp([dqn_agent, rand_agent], gym_mdp, instances=1, episodes=20, steps=200, open_plot=open_plot, verbose=True)
    policy, reached_goal, examples = run_agent_on_mdp(dqn_agent, gym_mdp)

if __name__ == "__main__":
    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(logging.ERROR)
    # main(open_plot=not sys.argv[-1] == "no_plot")
    main()
