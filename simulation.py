import numpy as np
import matplotlib.pyplot as plt
from agent import Agent
from environment import Environment
import pickle


def main():
    run_experiments()


def run_experiments():
    # Set up trial:
    dog_day = 10000
    n_moves = 50000

    modality_weights_one = [1, 0, 0]
    modality_weights_two = [0.5, 0.5, 0]
    modality_weights_three = [0.334, 0.333, 0.333]

    env = Environment(dog_day)
    agent = Agent(env, modality_weights=modality_weights_three)

    # Run a trial of n moves:
    for i in range(n_moves):
        agent.act()

    # Collect results:
    moves, dd = env.report()

    n_splits = int(n_moves/100)
    X = np.arange(n_splits)
    print(n_splits)
    split_moves = np.split(np.array(moves), n_splits)
    split_move_counts = np.array([[len(group[group == i]) for i in range(3)] for group in split_moves]).T

    plt.bar(X, split_move_counts[0], width=0.5)
    plt.bar(X, split_move_counts[1], width=0.5, bottom=split_move_counts[0])
    plt.bar(X, split_move_counts[2], width=0.5, bottom=split_move_counts[1] + split_move_counts[0])

    plt.show()

    print(agent.pred_outcome_given_state_per_modality)
    print(agent.state_visits)


if __name__ == '__main__':
    main()
