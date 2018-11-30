import numpy as np
import random
from collections import namedtuple, deque
import gym
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from copy import deepcopy

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
NUM_EPISODES = 200
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

class DQNAgent(Agent):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, name="DQN-Agent"):
        self.state_size = state_size
        self.action_size = action_size
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

        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        return random.choice(np.arange(self.action_size))

    def get_value(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.policy_network.eval()
        with torch.no_grad():
            action_values = self.policy_network(state)
        self.policy_network.train()
        return np.max(action_values.cpu().data.numpy())

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
        self.negative_memory = deque(maxlen=buffer_size)
        self.positive_memory = deque(maxlen=10000)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "priority"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """
        Add new experience to memory.
        Args:
            state (np.array): We add numpy arrays from gym env to the buffer, but sampling from buffer returns tensor
            action (int)
            reward (float)
            next_state (np.array)
            done (bool)
        """
        priority = reward > 0
        e = self.experience(state, action, reward, next_state, done, priority)
        if priority:
            self.positive_memory.append(e)
        else:
            self.negative_memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        num_positive_transitions = max(self.batch_size // 10, len(self.positive_memory))
        num_negative_transitions = self.batch_size - num_positive_transitions

        # Instead of uniform random sampling from the replay buffer, in prioritized sampling we will
        # always sample a fraction of positive transitions from the buffer
        positive_experiences = random.sample(self.positive_memory, k=num_positive_transitions)
        negative_experiences = random.sample(self.negative_memory, k=num_negative_transitions)
        assert type(positive_experiences) == type(negative_experiences) == list, "Expected lists, got {}".format(type(positive_experiences), type(negative_experiences))
        experiences = positive_experiences + negative_experiences

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.positive_memory) + len(self.negative_memory)

def train(agent, mdp, episodes, steps):
    per_episode_scores = []
    last_10_scores = deque(maxlen=10)

    for episode in range(episodes):
        mdp.reset()
        state = deepcopy(mdp.init_state)
        score = 0.
        while not state.is_terminal():
            action = agent.act(state.features(), agent.epsilon)
            reward, next_state = mdp.execute_agent_action(action)
            agent.step(state.features(), action, reward, next_state.features(), next_state.is_terminal())
            agent.update_epsilon()
            state = next_state
            score += reward
        last_10_scores.append(score)
        per_episode_scores.append(score)

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(last_10_scores)), end="")
        if episode % 10 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(last_10_scores)))

        # if np.mean(last_100_scores) >= 200.0:
        #     print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode - 100,
        #                                                                                  np.mean(last_100_scores)))
        #     torch.save(agent.policy_network.state_dict(), 'checkpoint.pth')
        #     break
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

from simple_rl.tasks.pinball.PinballMDPClass import PinballMDP

if __name__ == '__main__':
    pinball_mdp = PinballMDP(noise=0., episode_length=1000, render=False)
    state_space_size = pinball_mdp.init_state.state_space_size()
    dqn_agent = DQNAgent(state_size=state_space_size, action_size=len(pinball_mdp.actions), seed=RANDOM_SEED)
    episode_scores = train(dqn_agent, env, NUM_EPISODES, NUM_STEPS)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(episode_scores)), episode_scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('learning_curve.png')
    plt.close()
