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

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def get_init_data_and_labels(option):
    positive_examples, negative_examples = option._split_experience_into_pos_neg_examples()
    positive_feature_matrix = option._construct_feature_matrix(positive_examples)
    negative_feature_matrix = option._construct_feature_matrix(negative_examples)
    positive_labels = [1] * positive_feature_matrix.shape[0]
    negative_labels = [0] * negative_feature_matrix.shape[0]
    X = np.concatenate((positive_feature_matrix, negative_feature_matrix))
    Y = np.concatenate((positive_labels, negative_labels)); return X, Y

def plot_initiation_set(option):
    trained_classifier = option.initiation_classifier
    fig, sub = plt.subplots(1, 1)
    X, Y = get_init_data_and_labels(option)
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    plot_contours(sub, trained_classifier, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X0, X1, c=Y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    plt.show()

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

# Perform forward passes through the given DQN model
# so that we can visually see how it is performing
def render_dqn_policy(env, dqn_model):
    for i in range(3):
        state = env.reset()
        for j in range(750):
            action = dqn_model.act(state)
            env.render()
            state, reward, done, _ = env.step(action)
            if done: break
    env.close()
