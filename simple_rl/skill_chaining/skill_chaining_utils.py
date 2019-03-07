# Python imports.
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import scipy.interpolate
from scipy.misc import imread
import pdb
import random

# Other imports.
from simple_rl.tasks.lunar_lander.PositionalLunarLanderStateClass import PositionalLunarLanderState
from simple_rl.tasks.acrobot.AcrobotStateClass import AcrobotState


def plot_trajectory(trajectory, show=True, color='k', with_experiences=False, marker="o"):
    """
    Given State objects, plot their x and y positions and shade them based on time
    Args:
        trajectory (list): list of State objects
    """

    for i, state in enumerate(trajectory):
        if with_experiences:
            state = state.state
        theta_1 = AcrobotState.acrobot_acos(state.cos_theta_1)
        theta_2 = AcrobotState.acrobot_acos(state.cos_theta_2)
        plt.scatter(theta_1, theta_2, c=color, alpha=float(i) / len(trajectory), marker=marker)
    if show: plt.show()

def plot_all_trajectories_in_initiation_data(initiation_data, with_experiences=False, new_fig=False, show=False, option_name="", marker="o"):
    """
    Plot all the state buffers of an option
    Args:
        initiation_data (list) of deque objects where each queue represents a new state buffer (trajectory)
    """
    if new_fig: plt.figure()
    possible_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    for i, trajectory in enumerate(initiation_data):
        plot_trajectory(trajectory, show=show, color=possible_colors[i], with_experiences=with_experiences, marker=marker)
    if new_fig:
        plt.xlim((0., 1.))
        plt.ylim((0., 1.))
        plt.gca().invert_yaxis()
        plt.title('Option Policy Init Data')
        plt.savefig("{}_policy_init_data.png".format(option_name))

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, option, xx, yy, **params):
    Z = option.batched_is_init_true(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def get_init_data_and_labels(option):
    positive_feature_matrix = option._construct_feature_matrix(option.positive_examples)
    negative_feature_matrix = option._construct_feature_matrix(option.negative_examples)
    positive_labels = [1] * positive_feature_matrix.shape[0]
    negative_labels = [0] * negative_feature_matrix.shape[0]
    X = np.concatenate((positive_feature_matrix, negative_feature_matrix))
    Y = np.concatenate((positive_labels, negative_labels))
    return X, Y

def plot_initiation_set(option):
    trained_classifier = option.initiation_classifier
    fig, sub = plt.subplots(1, 1)
    X, Y = get_init_data_and_labels(option)
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    plot_contours(sub, trained_classifier, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X0, X1, c=Y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    plt.savefig("{}_initiation_set_{}.png".format(option.name, time.time()))
    plt.close()

def plot_binary_initiation_set(option):
    fig, sub = plt.subplots(1, 1)
    X, Y = get_init_data_and_labels(option)
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    plot_contours(sub, option, xx, yy, cmap=plt.cm.coolwarm, alpha=0.5)
    plt.scatter(X0, X1, c=Y, cmap=plt.cm.coolwarm, s=40, edgecolors='k')
    plt.xlim((0., 1.))
    plt.ylim((0., 1.))
    plt.gca().invert_yaxis()

    background_image = imread("pinball_domain.png")
    plt.imshow(background_image, zorder=0, alpha=0.5, extent=[0., 1., 1., 0.])

    plt.savefig("{}_initiation_set_{}.png".format(option.name, time.time()))
    plt.close()

def get_one_class_init_data(option):
    positive_feature_matrix = option._construct_feature_matrix(option.positive_examples)
    return positive_feature_matrix

def plot_one_class_initiation_classifier(option, is_pinball_domain=True):
    trained_classifier = option.initiation_classifier
    classifier_name = "OCSVM"
    legend = {}
    background_image = imread("pinball_domain.png")

    plt.figure(figsize=(8.0, 5.0))
    X = get_one_class_init_data(option)
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    Z1 = trained_classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z1 = Z1.reshape(xx.shape)
    legend[classifier_name] = plt.contour(xx, yy, Z1, levels=[0], linewidths=2, colors="m")

    # plt.plot(X0, X1, '.')

    # for row in range(X.shape[0]):
    #     plt.scatter(X0[row], X1[row], c='k', alpha=0.5)
    plot_all_trajectories_in_initiation_data(option.positive_examples)

    center_point = option.get_center_of_initiation_data(option.positive_examples)
    plt.scatter(center_point[0], center_point[1], s=50, marker="x", c="black", zorder=1)
    plt.imshow(background_image, zorder=0, alpha=0.5, extent=[0., 1., 1., 0.])

    if is_pinball_domain:
        plt.xlim((0., 1.))
        plt.ylim((0., 1.))
        plt.gca().invert_yaxis()
    else:
        plt.xlim((-1., 1.))
        plt.ylim((-0.25, 1.75))
    plt.xlabel("xpos")
    plt.ylabel("ypos")

    plt.savefig("{}_one_class_svm_{}.png".format(option.name, time.time()))
    plt.close()

def visualize_dqn_replay_buffer(solver, experiment_name=""):
    goal_transitions = list(filter(lambda e: e.reward >= 0 and e.done == 1, solver.replay_buffer.memory))
    cliff_transitions = list(filter(lambda e: e.reward < 0 and e.done == 1, solver.replay_buffer.memory))
    non_terminals = list(filter(lambda e: e.done == 0, solver.replay_buffer.memory))

    goal_x = [AcrobotState.acrobot_acos(e.next_state[0]) for e in goal_transitions]
    goal_y = [AcrobotState.acrobot_acos(e.next_state[2]) for e in goal_transitions]
    cliff_x = [AcrobotState.acrobot_acos(e.next_state[0]) for e in cliff_transitions]
    cliff_y = [AcrobotState.acrobot_acos(e.next_state[2]) for e in cliff_transitions]
    non_term_x = [AcrobotState.acrobot_acos(e.next_state[0]) for e in non_terminals]
    non_term_y = [AcrobotState.acrobot_acos(e.next_state[2]) for e in non_terminals]

    # background_image = imread("pinball_domain.png")

    plt.figure()
    plt.scatter(cliff_x, cliff_y, alpha=0.67, label="cliff")
    plt.scatter(non_term_x, non_term_y, alpha=0.2, label="non_terminal")
    plt.scatter(goal_x, goal_y, alpha=0.67, label="goal")
    # plt.imshow(background_image, zorder=0, alpha=0.5, extent=[0., 1., 1., 0.])

    plt.legend()
    plt.title("# transitions = {}".format(len(solver.replay_buffer)))
    plt.savefig("{}_replay_buffer_analysi_{}s.png".format(solver.name, experiment_name))
    plt.close()

def visualize_smdp_updates(global_solver, mdp, experiment_name=""):
     smdp_transitions = list(filter(lambda e: not mdp.is_primitive_action(e.action), global_solver.replay_buffer.memory))
     negative_transitions = list(filter(lambda e: e.done == 0, smdp_transitions))
     terminal_transitions = list(filter(lambda e: e.done == 1, smdp_transitions))

     positive_start_x = [AcrobotState.acrobot_acos(e.state[0]) for e in terminal_transitions]
     positive_start_y = [AcrobotState.acrobot_acos(e.state[2]) for e in terminal_transitions]
     positive_end_x = [AcrobotState.acrobot_acos(e.next_state[0]) for e in terminal_transitions]
     positive_end_y = [AcrobotState.acrobot_acos(e.next_state[2]) for e in terminal_transitions]

     negative_start_x = [AcrobotState.acrobot_acos(e.state[0]) for e in negative_transitions]
     negative_start_y = [AcrobotState.acrobot_acos(e.state[2]) for e in negative_transitions]
     negative_end_x = [AcrobotState.acrobot_acos(e.next_state[0]) for e in negative_transitions]
     negative_end_y = [AcrobotState.acrobot_acos(e.next_state[2]) for e in negative_transitions]

     plt.figure(figsize=(8, 5))
     plt.scatter(positive_start_x, positive_start_y, alpha=0.6, label="+s")
     plt.scatter(negative_start_x, negative_start_y, alpha=0.6, label="-s")
     plt.scatter(positive_end_x, positive_end_y, alpha=0.6, label="+s'")
     plt.scatter(negative_end_x, negative_end_y, alpha=0.6, label="-s'")

     plt.legend()
     plt.title("# updates = {}".format(len(smdp_transitions)))
     plt.savefig("DQN_SMDP_Updates_{}.png".format(experiment_name))
     plt.close()

def get_qvalue(agent, state, device):
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    with torch.no_grad():
        action_values = agent.policy_network(state)
    return action_values

def get_values(solver, device):
    values = []
    for theta_1 in np.arange(-np.pi, np.pi + np.pi/4, np.pi/4):
        for theta_2 in np.arange(-np.pi, np.pi + np.pi/4, np.pi/4):
            v = []
            for vx in np.arange(-4*np.pi, 5*np.pi, 2*np.pi):
                for vy in np.arange(-9*np.pi, 10*np.pi, 3*np.pi):
                    s = AcrobotState(np.cos(theta_1), np.sin(theta_1), np.cos(theta_2), np.sin(theta_2), vx, vy)
                    v.append(solver.get_value(s.features()))
            values.append(np.mean(v))
    return values

def get_grid_states():
    ss = []
    for theta_1 in np.arange(-np.pi, np.pi + np.pi/4, np.pi/4):
        for theta_2 in np.arange(-np.pi, np.pi + np.pi/4, np.pi/4):
            s = AcrobotState(np.cos(theta_1), np.sin(theta_1), np.cos(theta_2), np.sin(theta_2), 0, 0)
            ss.append(s)
    return ss

def render_value_function(solver, device, episode=None, show=False):
    states = get_grid_states()
    values = get_values(solver, device)
    x = np.array([AcrobotState.acrobot_acos(state.cos_theta_1) for state in states])
    y = np.array([AcrobotState.acrobot_acos(state.cos_theta_2) for state in states])
    xi, yi = np.linspace(x.min(), x.max(), 1000), np.linspace(y.min(), y.max(), 1000)
    xx, yy = np.meshgrid(xi, yi)
    try:
        rbf = scipy.interpolate.Rbf(x, y, values, function="linear")
    except:
        return
    zz = rbf(xx, yy)
    plt.imshow(zz, vmin=min(values), vmax=max(values), extent=[x.min(), x.max(), y.min(), y.max()])
    plt.colorbar()
    # plt.gca().invert_yaxis()
    if show: plt.show()
    name = solver.name if episode is None else solver.name + "_{}".format(episode)
    plt.savefig("{}_value_function.png".format(name))
    plt.close()

def render_sampled_value_function(solver, device, episode=None, show=False):

    # Array of experience tuples
    replay_buffer = solver.replay_buffer.memory

    # Extract all visited states in the replay buffer
    if len(replay_buffer) > 1000:
        states = random.sample([e.state for e in replay_buffer], 1000)
    else:
        states = [e.state for e in replay_buffer]

    states_array = np.vstack(states)
    states_tensor = torch.from_numpy(states_array).float().to(device)

    # Forward pass through our Q-function
    with torch.no_grad():
        q_values = solver.get_batched_qvalues(states_tensor)
        values = torch.max(q_values, dim=1)[0].cpu().data.numpy()

    x = np.arccos(states_array[:, 0])  # theta 1
    y = np.arccos(states_array[:, 2])  # theta 2

    xi, yi = np.linspace(x.min(), x.max(), 500), np.linspace(y.min(), y.max(), 500)
    xx, yy = np.meshgrid(xi, yi)

    try:
        rbf = scipy.interpolate.Rbf(x, y, values, function="linear")
    except:
        print("Could not render value function because of interpolation error")
        plt.close()
        return

    zz = rbf(xx, yy)

    plt.contourf(xx, yy, zz)
    plt.colorbar()
    plt.title("Value Function at episode {}".format(episode))

    name = solver.name if episode is None else solver.name + "_{}".format(episode)
    plt.savefig("{}_value_function.png".format(name))
    plt.close()

def render_sampled_initiation_classifier(option, global_solver):
    replay_buffer = global_solver.replay_buffer.memory

    if len(replay_buffer) > 1000:
        states = random.sample([e.state for e in replay_buffer], 1000)
    else:
        states = [e.state for e in replay_buffer]

    states_array = np.vstack(states)

    inits = option.batched_is_init_true(states_array)

    # pdb.set_trace()

    def cos_to_thetas(cosines):
        theta = np.zeros_like(cosines)
        for i, cos_theta in enumerate(cosines):
            theta[i] = AcrobotState.acrobot_acos(cos_theta)
        return theta

    x = cos_to_thetas(states_array[:, 0])  # theta 1
    y = cos_to_thetas(states_array[:, 2])  # theta 2

    xi, yi = np.linspace(x.min(), x.max(), 500), np.linspace(y.min(), y.max(), 500)
    xx, yy = np.meshgrid(xi, yi)

    try:
        rbf = scipy.interpolate.Rbf(x, y, inits, function="linear")
    except:
        print("Could not render initiation classifier because of interpolation error")
        plt.close()
        return

    zz = rbf(xx, yy)

    plt.contourf(xx, yy, zz, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.colorbar()

    plot_all_trajectories_in_initiation_data(option.positive_examples)

    plt.xlabel("theta1")
    plt.ylabel("theta2")

    plt.savefig("{}_svm_{}.png".format(option.name, time.time()))
    plt.close()

def sample_termination_set_classifier(option):
    plt.figure(figsize=(8., 5.))
    y = np.arange(1.5, -0.2, -0.1)
    x = np.zeros_like(y)
    for x0, y0 in zip(x, y):
        state = PositionalLunarLanderState(x0, y0)
        color = 'b' if option.is_term_true(state) == 1 else 'k'
        plt.scatter(x0, y0, c=color)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("{}: sampled termination set (blue: in term)".format(option.name))
    plt.savefig("{}_sampled_termination_set.png".format(option.name))
    plt.close()

def visualize_option_policy(option):
    colors = ("red", "green", "blue", "yellow", "cyan")
    # option.experience_buffer is a matrix with 10 columns representing the 10 times the option's
    # goal was encountered. Reshape as a column vector of Experience objects
    experience_buffer = option.experience_buffer.reshape(-1)
    x_positions = [experience.state.x for experience in experience_buffer]
    y_positions = [experience.state.y for experience in experience_buffer]
    actions = [option.solver.act(experience.state.features(), train_mode=False) for experience in experience_buffer]
    color_map = [colors[action] for action in actions]
    plt.scatter(x_positions, y_positions, c=color_map, alpha=0.7, edgecolors='none')
    plt.xlabel("x"); plt.ylabel("y"); plt.title("{} policy \nred: noop, green: left, blue: right, yellow: main".format(option.name))
    plt.savefig("{}_policy.png".format(option.name))
    plt.close()
    return x_positions, y_positions, actions

def visualize_option_starting_and_ending_points(option):
    start_x = [x.x for x in option.starting_points]
    start_y = [x.y for x in option.starting_points]
    end_x = [x.x for x in option.ending_points]
    end_y = [x.y for x in option.ending_points]

    plt.plot(start_x, start_y, '.', label='start points')
    plt.plot(end_x, end_y, '.', label='end points')

    plt.xlim((0., 1.))
    plt.ylim((0., 1.))
    plt.gca().invert_yaxis()

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('{}'.format(option.name))
    plt.legend()
    plt.savefig('starting_ending_{}.png'.format(option.name))
    plt.close()

def visualize_global_dqn_execution_points(states):
    plt.figure()
    x_execution_states = [state.x for state in states]
    y_execution_states = [state.y for state in states]
    plt.plot(x_execution_states, y_execution_states, '.')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Global DQN Execution States')
    plt.savefig("global_dqn_execution_states.png")
    plt.close()

def visualize_reason_for_option_termination(option):
    init_trues = [s for s in option.ending_points if not option.is_init_true(s)]
    term_trues = [s for s in option.ending_points if option.is_term_true(s)]
    is_terminals = [s for s in option.ending_points if s.is_terminal()]
    out_of_frames = [s for s in option.ending_points if s.is_out_of_frame()]

    plt.figure()
    plt.plot([s.x for s in init_trues], [s.y for s in init_trues], '.', label="init false")
    plt.plot([s.x for s in term_trues], [s.y for s in term_trues], '.', label="term true")
    plt.plot([s.x for s in is_terminals], [s.y for s in is_terminals], '.', label="is_terminal true")
    plt.plot([s.x for s in out_of_frames], [s.y for s in out_of_frames], '.', label="out of frame true")
    plt.legend()
    plt.title('{}: reason for termination'.format(option.name))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('{}_reason_for_termination.png'.format(option.name))
    plt.close()

def plot_epsilon_history(option):
    plt.figure()
    plt.plot(range(len(option.epsilon_history)), option.epsilon_history, '.')
    plt.xlabel('Iteration of option execution')
    plt.ylabel('epsilon')
    plt.title('{} Epsilon History'.format(option.name))
    plt.savefig('{}_epsilon_history.png'.format(option.name))
    plt.close()

def plot_replay_buffer_size(option):
    plt.figure()
    plt.plot(range(len(option.num_states_in_replay_buffer)), option.num_states_in_replay_buffer)
    plt.xlabel('Iteration of option execution')
    plt.ylabel('# states')
    plt.title("{}: replay buffer size over episodes".format(option.name))
    plt.savefig('{}_replay_buffer_size.png'.format(option.name))
    plt.close()

def visualize_replay_buffer(option):
    states = [experience[0] for experience in option.solver.replay_buffer.memory]
    rewards = [experience[2] for experience in option.solver.replay_buffer.memory]
    x_positions = [state[0] for state in states]
    y_positions = [state[1] for state in states]
    sizes = [6. if reward <= 0 else 24. for reward in rewards]
    plt.figure()
    plt.scatter(x_positions, y_positions, s=sizes)
    plt.xlabel("x")
    plt.ylabel("y")

    plt.xlim((0., 1.))
    plt.ylim((0., 1.))
    plt.gca().invert_yaxis()

    plt.title("{} replay buffer".format(option.name))
    plt.savefig("{}_replay_buffer".format(option.name))
    plt.close()


def plot_num_learning_updates(option):
    plt.figure()
    plt.plot(range(len(option.num_learning_updates_dqn)), option.num_learning_updates_dqn)
    plt.xlabel('Iteration of option execution')
    plt.ylabel('# updates')
    plt.title('{}: number of DQN learning updates'.format(option.name))
    plt.savefig('{}_num_dqn_updates.png'.format(option.name))
    plt.close()

def plot_policy_refinement_data(option):
    refinement_data = [item for sublist in option.policy_refinement_data for item in sublist]
    refinement_x = [state.x for state in refinement_data]
    refinement_y = [state.y for state in refinement_data]
    plt.figure()
    plt.scatter(refinement_x, refinement_y)
    plt.title('Policy refinement data for {}'.format(option.name))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('{}_policy_refinement_data.png'.format(option.name))
    plt.close()

# Perform forward passes through the given DQN model
# so that we can visually see how it is performing
def render_dqn_policy(env, dqn_model, show_value_plot=False):
    plt.figure()
    for i in range(4):
        values = []
        state = env.reset()
        episodic_score = 0.
        for j in range(1000):
            action = dqn_model.act(state)
            values.append(dqn_model.get_value(state))
            env.render()
            state, reward, done, _ = env.step(action)
            episodic_score += reward
            if done: break
        print("Episode {}\tScore={}".format(i, episodic_score))
        plt.subplot(2, 2, i+1)
        plt.plot(values)
        plt.xlabel("# Frame")
        plt.ylabel("V(s)")
    if show_value_plot:
        plt.show()
    plt.close()
    env.close()

def render_learned_policy(skill_chainer):
    for i in range(3):
        score = skill_chainer.trained_forward_pass()
        print("Episode {}: Score: {:.2f}".format(i, score))
