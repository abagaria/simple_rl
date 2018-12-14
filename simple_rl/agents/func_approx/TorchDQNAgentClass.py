import numpy as np
import random
from collections import namedtuple, deque
import gym
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from copy import deepcopy
import pdb

import torch.optim as optim

import torch
import torch.nn as nn
import torch.nn.functional as F

from simple_rl.agents.AgentClass import Agent

## Hyperparameters
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 1  # how often to update the network
NUM_EPISODES = 50
NUM_STEPS = 100000

EPS_START = 1.0

RANDOM_SEED = 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("######### Using {} ############".format(device))

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128):
        """
        Set up the layers of the DQN
        Args:
            state_size (int): number of states in the state variable (can be continuous)
            action_size (int): number of actions in the discrete action domain
            seed (int): random seed
            fc1_units (int): size of the hidden layer
            fc2_units (int): size of the hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """
        DQN forward pass
        Args:
            state (torch.tensor): convert env.state into a tensor

        Returns:
            logits (torch.tensor): score for each possible action (1, num_actions)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def initialize_with_bigger_network(self, bigger_net):
        """
        Given a policy network or target network from the global DQN, initialize the corresponding option DQN
        Args:
            bigger_net (QNetwork): Outputs q values over larger number of actions
        """
        for local_param, global_param in zip(self.fc1.parameters(), bigger_net.fc1.parameters()):
            local_param.data.copy_(global_param)
        for local_param, global_param in zip(self.fc2.parameters(), bigger_net.fc2.parameters()):
            local_param.data.copy_(global_param)

        num_original_actions = 5 # TODO: Assuming that we are in pinball domain
        self.fc3.weight.data.copy_(bigger_net.fc3.weight[:num_original_actions, :])
        self.fc3.bias.data.copy_(bigger_net.fc3.bias[:num_original_actions])

        assert self.fc3.out_features == 5, "Expected Pinball with 5 actions, not {} ".format(self.fc3.out_features)

    def initialize_with_smaller_network(self, smaller_net):
        """
        Given a DQN over K actions, create a DQN over K + 1 actions. This is needed when we augment the
        MDP with a new action in the form of a learned option.
        Args:
            smaller_net (QNetwork)
        """
        for my_param, source_param in zip(self.fc1.parameters(), smaller_net.fc1.parameters()):
            my_param.data.copy_(source_param)
        for my_param, source_param in zip(self.fc2.parameters(), smaller_net.fc2.parameters()):
            my_param.data.copy_(source_param)

        smaller_num_labels = smaller_net.fc3.out_features
        self.fc3.weight[:smaller_num_labels, :].data.copy_(smaller_net.fc3.weight)
        self.fc3.bias[:smaller_num_labels].data.copy_(smaller_net.fc3.bias)

        new_action_idx = self.fc3.out_features - 1
        self.fc3.weight[new_action_idx].data.copy_(torch.max(smaller_net.fc3.weight, dim=0)[0])
        self.fc3.bias[new_action_idx].data.copy_(torch.max(smaller_net.fc3.bias, dim=0)[0])

class DQNAgent(Agent):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, num_original_actions, trained_options, eps_init, seed, tau=20., name="DQN-Agent"):
        self.state_size = state_size
        self.action_size = action_size
        self.num_original_actions = num_original_actions
        self.trained_options = trained_options
        self.seed = random.seed(seed)
        self.tau = tau

        # Q-Network
        self.policy_network = QNetwork(state_size, action_size, seed).to(device)
        self.target_network = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=LR)

        # Replay memory
        self.replay_buffer = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        # Epsilon strategy
        self.eps_init = eps_init
        self.epsilon = eps_init
        self.step_number = 0 # Number of times act() is called (used for eps-decay)

        # Debugging attributes
        self.num_updates = 0
        self.epsilon_history = []

        Agent.__init__(self, name, range(action_size), GAMMA)

    def act(self, state, eps=0.):
        """
        Interface to the DQN agent: state can be output of env.step() and returned action can be input into next step().
        Args:
            state (np.array): numpy array state from Gym env
            eps (float): epsilon value for action selection under epsilon-greedy program

        Returns:
            action (int): integer representing the action to take in the Gym env
        """
        self.step_number += 1

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.policy_network.eval()
        with torch.no_grad():
            action_values = self.policy_network(state)
        self.policy_network.train()

        # Argmax only over actions that can be implemented from the current state
        impossible_option_idx = [idx for idx, option in enumerate(self.trained_options) if (not option.is_init_true(state.cpu().data.numpy()[0]))
                                 or option.is_term_true(state.cpu().data.numpy()[0])]
        impossible_action_idx = map(lambda x: x + self.num_original_actions, impossible_option_idx)
        for impossible_idx in impossible_action_idx:
            action_values[0][impossible_idx] = torch.min(action_values, dim=1)[0] - 1.

        action_values = action_values.cpu().data.numpy()

        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            chosen_action = np.argmax(action_values)
        else:
            # Not allowing epsilon-greedy to select an option as a random action
            chosen_action = random.choice(np.arange(self.num_original_actions))

        self._update_epsilon()
        return chosen_action

    def get_value(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.policy_network.eval()
        with torch.no_grad():
            action_values = self.policy_network(state)
        self.policy_network.train()
        return np.max(action_values.cpu().data.numpy())

    def get_qvalue(self, state, action_idx):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.policy_network.eval()
        with torch.no_grad():
            action_values = self.policy_network(state)
        self.policy_network.train()
        return action_values[0][action_idx]

    def step(self, state, action, reward, next_state, done):
        """
        Interface method to perform 1 step of learning/optimization during training.
        Args:
            state (np.array): state of the underlying gym env
            action (int)
            reward (float)
            next_state (np.array)
            done (bool): is_terminal
        """
        # Save experience in replay memory
        self.replay_buffer.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.replay_buffer) > BATCH_SIZE:
                experiences = self.replay_buffer.sample()
                self._learn(experiences, GAMMA)
                self.num_updates += 1

    def _learn(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples.
        Args:
            experiences (tuple<torch.Tensor>): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.policy_network(states).gather(1, actions)

        # Compute loss
        # Akhil: Using Huber Loss here rather than the MSE loss from before
        loss = F.smooth_l1_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.policy_network, self.target_network, TAU)

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update of target network from policy network.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Args:
            local_model (nn.Module): weights will be copied from
            target_model (nn.Module): weights will be copied to
            tau (float): interpolation parameter - usually small eg 0.0001
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def _update_epsilon(self):
        self.epsilon_history.append(self.epsilon)
        tau = 10000 * self.tau
        t = self.step_number * (1. + self.episode_number)
        self.epsilon = self.eps_init / (1.0 + (t / tau))

    def end_of_episode(self):
        self._update_epsilon()

        if self.episode_number % 10 == 0:
            filename = "eps_history/{}_eps_history_{}_episodes.npy".format(self.name, self.episode_number)
            np.save(filename, self.epsilon_history)

        Agent.end_of_episode(self)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """
        Initialize a ReplayBuffer object.
        Args:
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """
        Add new experience to memory.
        Args:
            state (np.array): We add numpy arrays from gym env to the buffer, but sampling from buffer returns tensor
            action (int)
            reward (float_
            next_state (np.array)
            done (bool)
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

def train(agent, mdp, episodes, steps):
    per_episode_scores = []
    last_10_scores = deque(maxlen=10)

    for episode in range(episodes):
        mdp.reset()
        state = deepcopy(mdp.init_state)
        score = 0.
        for step in range(steps):
            action = agent.act(state.features(), agent.epsilon)
            reward, next_state = mdp.execute_agent_action(action)
            agent.step(state.features(), action, reward, next_state.features(), next_state.is_terminal())
            state = next_state
            score += reward
            if state.is_terminal():
                break
        last_10_scores.append(score)
        per_episode_scores.append(score)
        agent.end_of_episode()

        print('\rEpisode {}\tAverage Score: {:.2f}\tEps: {:.3f}'.format(episode, np.mean(last_10_scores), agent.epsilon), end="")
        if episode % 10 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tEps: {:.3f}'.format(episode, np.mean(last_10_scores), agent.epsilon))
    return per_episode_scores

def construct_pinball_mdp():
    from simple_rl.tasks.pinball.PinballMDPClass import PinballMDP
    mdp = PinballMDP(noise=0., episode_length=1000, render=True)
    return mdp

def test_forward_pass(dqn_agent, mdp):
    mdp.reset()
    state = deepcopy(mdp.init_state)
    overall_reward = 0.
    mdp.render = True

    while not state.is_terminal():
        action = dqn_agent.act(state.features(), eps=0.)
        reward, next_state = mdp.execute_agent_action(action)
        overall_reward += reward
        state = next_state

    mdp.render = False
    print("Score = {}".format(overall_reward))

    return overall_reward

def main(num_training_episodes=NUM_EPISODES, tau=20., to_plot=False):
    mdp = construct_pinball_mdp()
    state_space_size = mdp.init_state.state_space_size()
    action_space_size = len(mdp.actions)
    dqn_agent = DQNAgent(state_size=state_space_size, action_size=action_space_size, eps_init=EPS_START, tau=tau,
                         num_original_actions=action_space_size, trained_options=[], seed=RANDOM_SEED)
    episode_scores = train(dqn_agent, mdp, num_training_episodes, NUM_STEPS)

    if to_plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(episode_scores)), episode_scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.savefig('learning_curve.png')
        plt.close()

    return episode_scores, dqn_agent

if __name__ == '__main__':
    baseline_scores, flat_dqn_agent = main()
