import numpy as np
import matplotlib.pyplot as plt
from agent import Agent
from agent2 import Agent2
from environment import Environment
from environment_alt_2 import Environment2
import pickle


def main():
    flag = False

    if flag:
        run_env_1()
    else:
        run_env_2()


def run_env_2():
    # Set up trial:
    dog_day = 10000
    n_moves = 50000

    env = Environment2(dog_day)
    agent = Agent2(env)

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
