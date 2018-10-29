# Python imports.
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Other imports.
from simple_rl.abstraction.action_abs.OptionClass import Experience

def plot_trajectory(trajectory, show=True, color='k'):
    """
    Given State objects, plot their x and y positions and shade them based on time
    Args:
        trajectory (list): list of State objects
    """
    for i, state in enumerate(trajectory):
        plt.scatter(state.x, state.y, c=color, alpha=float(i) / len(trajectory))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Trajectory (Bolder color implies more recent point in time)')
    if show: plt.show()

def plot_all_trajectories_in_initiation_data(initiation_data):
    """
    Plot all the state buffers of an option
    Args:
        initiation_data (list) of deque objects where each queue represents a new state buffer (trajectory)
    """
    assert initiation_data.shape[1] == 10, "Assuming that input is a list of 10 queues."
    plt.figure()
    possible_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    for i in range(initiation_data.shape[1]):
        trajectory = initiation_data[:, i]
        plot_trajectory(trajectory, show=False, color=possible_colors[i])
    plt.show()

def plot_initiation_set(option):
    X0, X1 = np.array(option.initiation_examples)[:, 0], np.array(option.initiation_examples)[:, 1]
    pass

def visualize_option_policy(option):
    colors = ("red", "green", "blue", "yellow")
    # option.experience_buffer is a matrix with 10 columns representing the 10 times the option's
    # goal was encountered. Reshape as a column vector of Experience objects
    experience_buffer = option.experience_buffer.reshape(-1)
    x_positions = [experience.state.x for experience in experience_buffer]
    y_positions = [experience.state.y for experience in experience_buffer]
    actions = [option.solver.act(experience.state.features()) for experience in experience_buffer]
    color_map = [colors[action] for action in actions]
    plt.scatter(x_positions, y_positions, c=color_map, alpha=0.7, edgecolors='none')
    plt.xlabel("x"); plt.ylabel("y"); plt.title("{} policy \nred: noop, green: left, blue: right, yellow: main".format(option.name))
    plt.show()
    return x_positions, y_positions, actions

