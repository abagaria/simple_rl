import numpy as np
import random
from collections import namedtuple, deque
import gym
import matplotlib.pyplot as plt
import seaborn as sns
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
NUM_EPISODES = 2000
NUM_STEPS = 1000

EPS_START = 1.0
EPS_END = 0.01
EPS_EXPONENTIAL_DECAY = 0.995
EPS_LINEAR_DECAY_LENGTH = 10000
EPS_LINEAR_DECAY = (EPS_START - EPS_END) / EPS_LINEAR_DECAY_LENGTH

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

    def __init__(self, state_size, action_size, num_original_actions, trained_options, seed, name="DQN-Agent"):
        self.state_size = state_size
        self.action_size = action_size
        self.num_original_actions = num_original_actions
        self.trained_options = trained_options
        self.seed = random.seed(seed)

        # Q-Network
        self.policy_network = QNetwork(state_size, action_size, seed).to(device)
        self.target_network = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=LR)

        # Replay memory
        self.replay_buffer = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        # Epsilon strategy
        self.epsilon = EPS_START
        self.num_executions = 0 # Number of times act() is called (used for eps-decay)

        # Debugging attributes
        self.num_updates = 0

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

        # Not allowing epsilon-greedy to select an option as a random action
        return random.choice(np.arange(self.num_original_actions))

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

    def update_epsilon(self):
        if self.num_executions < EPS_LINEAR_DECAY_LENGTH:
            self.epsilon -= EPS_LINEAR_DECAY
        else:
            self.epsilon = max(EPS_END, EPS_EXPONENTIAL_DECAY * self.epsilon)

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

def train(agent, env, episodes, steps):
    per_episode_scores = []
    last_100_scores = deque(maxlen=100)

    for episode in range(episodes):
        state = env.reset()
        score = 0.
        for step in range(steps):
            action = agent.act(state, agent.epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            agent.update_epsilon()
            state = next_state
            score += reward
            if done:
                break
        last_100_scores.append(score)
        per_episode_scores.append(score)

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(last_100_scores)), end="")
        if episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(last_100_scores)))

        if np.mean(last_100_scores) >= 200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode - 100,
                                                                                         np.mean(last_100_scores)))
            torch.save(agent.policy_network.state_dict(), 'checkpoint.pth')
            break
    return per_episode_scores

def test_forward_pass(dqn_agent, env):
    # load the weights from file
    dqn_agent.policy_network.load_state_dict(torch.load('checkpoint.pth'))

    for i in range(3):
        state = env.reset()
        for j in range(500):
            action = dqn_agent.act(state)
            env.render()
            state, reward, done, _ = env.step(action)
            if done:
                break

    env.close()

def main(num_training_episodes=NUM_EPISODES, to_plot=False):
    env = gym.make('LunarLander-v2')

    # env.seed(RANDOM_SEED)

    dqn_agent = DQNAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.n, seed=RANDOM_SEED)
    episode_scores = train(dqn_agent, env, num_training_episodes, NUM_STEPS)

    if to_plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(episode_scores)), episode_scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.savefig('learning_curve.png')
        plt.close()

    return episode_scores

if __name__ == '__main__':
    baseline_scores = main()
