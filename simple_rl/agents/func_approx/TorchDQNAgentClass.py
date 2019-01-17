import numpy as np
import random
from collections import namedtuple, deque
import gym
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import pdb
from copy import deepcopy
import shutil
import os

import torch.optim as optim

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from simple_rl.agents.AgentClass import Agent
from simple_rl.tasks.pinball.PinballMDPClass import PinballMDP

## Hyperparameters
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 1  # how often to update the network
NUM_EPISODES = 150
NUM_STEPS = 20000

class EpsilonSchedule:
    def __init__(self, eps_start, eps_end, eps_exp_decay, eps_linear_decay_length):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_exp_decay = eps_exp_decay
        self.eps_linear_decay_length = eps_linear_decay_length
        self.eps_linear_decay = (eps_start - eps_end) / eps_linear_decay_length

    def update_epsilon(self, current_epsilon, num_executions):
        pass

class GlobalEpsilonSchedule(EpsilonSchedule):
    def __init__(self, eps_start):
        EPS_END = 0.05
        EPS_EXPONENTIAL_DECAY = 0.995
        EPS_LINEAR_DECAY_LENGTH = 100000
        super(GlobalEpsilonSchedule, self).__init__(eps_start, EPS_END, EPS_EXPONENTIAL_DECAY, EPS_LINEAR_DECAY_LENGTH)

    def update_epsilon(self, current_epsilon, num_executions):
        if num_executions < self.eps_linear_decay_length:
            return current_epsilon - self.eps_linear_decay
        return max(self.eps_end, self.eps_exp_decay * current_epsilon)

class OptionEpsilonSchedule(EpsilonSchedule):
    def __init__(self, eps_start):
        EPS_END = 0.01
        EPS_EXPONENTIAL_DECAY = 0.998
        EPS_LINEAR_DECAY_LENGTH = 10000
        super(OptionEpsilonSchedule, self).__init__(eps_start, EPS_END, EPS_EXPONENTIAL_DECAY, EPS_LINEAR_DECAY_LENGTH)

    def update_epsilon(self, current_epsilon, num_executions):
        return max(self.eps_end, self.eps_exp_decay * current_epsilon)

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

    def __init__(self, state_size, action_size, num_original_actions, trained_options, seed, name="DQN-Agent",
                 eps_start=1., tensor_log=False, lr=LR):
        self.state_size = state_size
        self.action_size = action_size
        self.num_original_actions = num_original_actions
        self.trained_options = trained_options
        self.seed = random.seed(seed)
        self.tensor_log = tensor_log

        # Q-Network
        self.policy_network = QNetwork(state_size, action_size, seed).to(device)
        self.target_network = QNetwork(state_size, action_size, seed).to(device)

        # if "global" in name.lower():
        #     self.optimizer = optim.Adam(self.policy_network.parameters(), lr=LR)
        # else:
        #     self.optimizer = optim.Adam(self.policy_network.parameters(), lr=LR)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=LR)


        # Replay memory
        self.replay_buffer = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        # Epsilon strategy
        self.epsilon_schedule = GlobalEpsilonSchedule(eps_start) if "global" in name.lower() else OptionEpsilonSchedule(eps_start)
        self.epsilon = eps_start
        self.num_executions = 0 # Number of times act() is called (used for eps-decay)

        # Debugging attributes
        self.num_updates = 0
        self.num_epsilon_updates = 0

        if os.path.exists(name):
            print("Deleting folder: {}".format(name))
            shutil.rmtree(name)

        if self.tensor_log:
            self.writer = SummaryWriter(name)

        Agent.__init__(self, name, range(action_size), GAMMA)

    def initialize_optimizer_with_smaller_agent(self, smaller_agent):

        # Layers that dont need size changes
        old_ids = id(smaller_agent.policy_network.fc1.weight), id(smaller_agent.policy_network.fc1.bias), \
                  id(smaller_agent.policy_network.fc2.weight), id(smaller_agent.policy_network.fc2.bias)

        new_ids = id(self.policy_network.fc1.weight), id(self.policy_network.fc1.bias), \
                  id(self.policy_network.fc2.weight), id(self.policy_network.fc2.bias)


        opt_state_dict = smaller_agent.optimizer.state_dict()

        for old_id, new_id in zip(old_ids, new_ids):
            step = opt_state_dict['state'][old_id]['step']
            exp_avg = opt_state_dict['state'][old_id]['exp_avg']
            exp_avg_sq = opt_state_dict['state'][old_id]['exp_avg_sq']

            # new_exp_avg = torch.cat((), 1)
            pdb.set_trace()

        # Layers that do need size changes
        old_ids = id(smaller_agent.policy_network.fc3.weight), id(smaller_agent.policy_network.fc3.bias)
        new_ids = id(self.policy_network.fc3.weight), id(self.policy_network.fc3.bias)


    def act(self, state, eps=0.):
        """
        Interface to the DQN agent: state can be output of env.step() and returned action can be input into next step().
        Args:
            state (np.array): numpy array state from Gym env
            eps (float): epsilon value for action selection under epsilon-greedy program

        Returns:
            action (int): integer representing the action to take in the Gym env
        """
        self.num_executions += 1

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
            return np.argmax(action_values)

        # pdb.set_trace()
        original_actions = list(range(self.num_original_actions))
        all_option_idx = list(range(len(self.trained_options)))
        possible_option_idx = list(set(all_option_idx).difference(impossible_option_idx))
        all_possible_action_idx = original_actions + list(map(lambda x: x + self.num_original_actions, possible_option_idx))
        randomly_chosen_action = random.choice(all_possible_action_idx)

        # Not allowing epsilon-greedy to select an option as a random action
        return randomly_chosen_action

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
                if self.tensor_log:
                    self.writer.add_scalar("NumPositiveTransitions", self.replay_buffer.positive_transitions[-1], self.num_updates)
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

        # Gradient clipping: tried but the results looked worse -- needs more testing
        for param in self.policy_network.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        if self.tensor_log:
            self.writer.add_scalar("Loss", loss.item(), self.num_updates)
            self.writer.add_scalar("AverageTargetQvalue", Q_targets.mean().item(), self.num_updates)
            self.writer.add_scalar("AverageQValue", Q_expected.mean().item(), self.num_updates)
            parameters_nn = list(self.policy_network.parameters())
            means = []
            for param in parameters_nn:
                means.append(param.mean().item())
            self.writer.add_scalar("AverageWeights", np.mean(means), self.num_updates)

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

    def update_epsilon(self):
        self.num_epsilon_updates += 1
        self.epsilon = self.epsilon_schedule.update_epsilon(self.epsilon, self.num_epsilon_updates)

        # Log epsilon decay
        if self.tensor_log:
            self.writer.add_scalar("Epsilon", self.epsilon, self.num_epsilon_updates)

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

        self.positive_transitions = []

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

        # Log the number of times we see a non-negative reward (should be sparse)
        num_positive_transitions = sum([exp.reward >= 0 for exp in experiences])
        self.positive_transitions.append(num_positive_transitions)

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
    iteration_counter = 0

    for episode in range(episodes):
        mdp.reset()
        state = deepcopy(mdp.init_state)
        score = 0.
        for step in range(steps):
            iteration_counter += 1
            action = agent.act(state.features(), agent.epsilon)
            reward, next_state = mdp.execute_agent_action(action)
            agent.step(state.features(), action, reward, next_state.features(), next_state.is_terminal())
            agent.update_epsilon()
            state = next_state
            score += reward
            agent.writer.add_scalar("Score", score, iteration_counter)
            if state.is_terminal():
                break
        last_10_scores.append(score)
        per_episode_scores.append(score)

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(last_10_scores)), end="")
        if episode % 10 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(last_10_scores)))
    return per_episode_scores

def test_forward_pass(dqn_agent, mdp):
    # load the weights from file
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
    return overall_reward

def main(num_training_episodes=NUM_EPISODES, to_plot=False):
    mdp = PinballMDP(noise=0.0, episode_length=20000, render=False)

    # env.seed(RANDOM_SEED)

    dqn_agent = DQNAgent(state_size=mdp.init_state.state_space_size(), action_size=len(mdp.actions),
                         num_original_actions=len(mdp.actions), trained_options=[], seed=0, name="GlobalDQN")
    episode_scores = train(dqn_agent, mdp, num_training_episodes, NUM_STEPS)

    return episode_scores

if __name__ == '__main__':
    overall_mdp = PinballMDP(noise=0.0, episode_length=20000, render=True)
    dqn_agent = DQNAgent(state_size=overall_mdp.init_state.state_space_size(), action_size=len(overall_mdp.actions),
                         num_original_actions=len(overall_mdp.actions), trained_options=[], seed=0, name="GlobalDQN", tensor_log=True)
    episode_scores = train(dqn_agent, overall_mdp, NUM_EPISODES, NUM_STEPS)
