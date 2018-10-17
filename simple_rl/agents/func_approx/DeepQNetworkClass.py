# Python imports.
import random
import math
from collections import namedtuple
from itertools import count
import argparse
import os
import numpy as np

# PyTorch imports.
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Other imports.
from simple_rl.tasks.gym.GymMDPClass import GymMDP

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class Hyperparameters(object):
    BATCH_SIZE = 128
    GAMMA = 0.999
    EPS_START = 0.99
    EPS_END = 0.33
    EPS_DECAY = 10000
    TARGET_UPDATE = 50
    NUM_EPISODES = 500

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.memory = []
        self.position = 0

    def push(self, *args):
        """
        Saves a transition and evicts an old transition if needed
        Args:
            state
            action
            next_state
            reward
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, device, num_input_channels=3, num_actions=2, kernel_size=5, stride=2):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(num_input_channels, 16, kernel_size=kernel_size, stride=stride)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=kernel_size, stride=stride)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=kernel_size, stride=stride)
        self.bn3 = nn.BatchNorm2d(32)
        # self.fc = nn.Linear(448, num_actions) # This was the FC layer when using the sliced input image.
        self.fc = nn.Linear(256, num_actions) # Dimensions of the FC layer when using the full input image from Gym.

        self.num_actions = num_actions
        self.device = device
        self.steps_done = 0

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        logits = self.fc(x.view(x.size(0), -1))
        return logits

    def policy(self, state):
        epsilon_threshold = Hyperparameters.EPS_END + (Hyperparameters.EPS_START - Hyperparameters.EPS_END) * \
                            math.exp(-1. * self.steps_done / Hyperparameters.EPS_DECAY)
        self.steps_done += 1
        if random.random() > epsilon_threshold:
            with torch.no_grad():
                return self.forward(state).max(1)[1].view(1, 1)
        action = random.randrange(self.num_actions)
        return torch.tensor([[action]], device=self.device, dtype=torch.long)

def optimize_model(policy_network, target_network, optimizer, replay_buffer, device):
    if len(replay_buffer) < Hyperparameters.BATCH_SIZE:
        print("Returning early from optimizer_step because replay buffer is not big enough to sample from.")
        return
    transitions = replay_buffer.sample(Hyperparameters.BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_network(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(Hyperparameters.BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_network(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * Hyperparameters.GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_network.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return loss.item()

def train(mdp, policy_network, target_network, optimizer, replay_buffer, device, num_episodes):
    loss_history, episode_durations = [], []
    for episode in range(num_episodes):
        print("----- Episode {} ------".format(episode))
        mdp.reset()
        last_screen = mdp.get_screen(device)
        current_screen = mdp.get_screen(device)
        state = current_screen - last_screen
        for step in count():
            action = policy_network.policy(state)
            _, reward, done, _ = mdp.env.step(action.item())
            reward = torch.tensor([reward], device=device)

            # Observe new state
            last_screen = current_screen
            current_screen = mdp.get_screen(device)
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Store the transition in memory
            replay_buffer.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            loss_value = optimize_model(policy_network, target_network, optimizer, replay_buffer, device)
            loss_history.append(loss_value)

            if step % 20 == 0:
                print("Episode: {}\tStep: {}\tLoss: {}".format(episode, step, loss_value))

            if done:
                print("~~~~~~~~~~~ Hit terminal state. Episode lasted {} steps ~~~~~~~~~~~".format(step+1))
                episode_durations.append(step+1)
                break

        # Update the target network
        if episode % Hyperparameters.TARGET_UPDATE == 0:
            print("Updating target network..")
            target_network.load_state_dict(policy_network.state_dict())

        ckpt = os.path.join(args.save, "model-episode-{}.ckpt".format(episode))
        torch.save(policy_network.state_dict(), ckpt)
        print("Checkpoint saved to {}".format(ckpt))

    mdp.env.render()
    mdp.env.close()

    return loss_history, episode_durations

def run_trained_network(mdp, policy_network, device, num_episodes):
    cumulative_reward, episode_durations = 0., []
    for episode in range(num_episodes):
        print("----- Episode {} ------".format(episode))
        mdp.reset()
        last_screen = mdp.get_screen(device)
        current_screen = mdp.get_screen(device)
        state = current_screen - last_screen
        for step in count():
            with torch.no_grad():
                action = policy_network.policy(state)
            _, reward, done, _ = mdp.env.step(action.item())
            cumulative_reward += reward

            # Observe new state
            last_screen = current_screen
            current_screen = mdp.get_screen(device)
            next_state = (current_screen - last_screen) if not done else None

            state = next_state

            if done:
                episode_durations.append(step+1)
                print("Trained forward pass: episode lasted {} steps".format(step+1))
                break
    mdp.env.render()
    mdp.env.close()
    return cumulative_reward, episode_durations

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("############### Using device {} ###############".format(device))
    policy_network = DQN(device).to(device)
    target_network = DQN(device).to(device)
    target_network.load_state_dict(policy_network.state_dict())
    target_network.eval()

    if args.restore:
        policy_network.load_state_dict(torch.load(args.restore))
        print("Model restored from disk.")
        return [], [], policy_network

    # If not restoring model from disk, start the training process
    optimizer = optim.Adam(policy_network.parameters())
    replay_buffer = ReplayBuffer(10e4)

    mdp = GymMDP(render=True)

    loss_history, episode_durations = train(mdp, policy_network, target_network, optimizer, replay_buffer, device,
                                            num_episodes=Hyperparameters.NUM_EPISODES)

    return loss_history, episode_durations, policy_network

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--save", help="path to save directory", default="saved_runs")
    group.add_argument("--restore", help="path to model checkpoint")
    args = parser.parse_args()

    if not os.path.exists(args.save):
        os.mkdir(args.save)

    np.set_printoptions(linewidth=150)

    l_history, e_durations, p_network = main()