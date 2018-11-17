# Python imports.
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import numpy as np

# Other imports.
from simple_rl.tasks.lunar_lander.PositionalLunarLanderStateClass import PositionalLunarLanderState


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
    plt.savefig("{}_initiation_set.png".format(option.name))
    plt.close()

def get_one_class_init_data(option):
    positive_feature_matrix = option._construct_feature_matrix(option.initiation_data)
    return positive_feature_matrix

def plot_one_class_initiation_classifier(option):
    trained_classifier = option.initiation_classifier
    classifier_name = "OCSVM"
    legend = {}

    plt.figure(figsize=(8.0, 5.0))
    X = get_one_class_init_data(option)
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    Z1 = trained_classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z1 = Z1.reshape(xx.shape)
    legend[classifier_name] = plt.contour(xx, yy, Z1, levels=[0], linewidths=2, colors="m")

    plt.plot(X0, X1, '.')

    for row in range(X.shape[0]):
        buffer_length = option.buffer_length
        plt.scatter(X0[row], X1[row], c='k', alpha=(float(row) % buffer_length) / float(buffer_length))

    plt.xlim((-1., 1.))
    plt.ylim((-0.25, 1.75))
    plt.xlabel("xpos")
    plt.ylabel("ypos")

    plt.savefig("{}_one_class_svm.png".format(option.name))
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
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('{}'.format(option.name))
    plt.legend()
    plt.savefig('starting_ending_{}.png'.format(option.name))
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
