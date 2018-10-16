# Python imports.
import matplotlib.pyplot as plt

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
