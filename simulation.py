import numpy as np
import matplotlib.pyplot as plt
from agent import Agent
from environment import Environment
import pickle
from tqdm import tqdm
from multiprocessing import Pool

N_PROCESSES = 10


def main():
    # run_experiments()
    # vary_dog_modalities()
    # vary_dog_day()
    vary_salience()
    # control()


def control():
    n_days = 50000
    dog_day = 50000
    n_runs = 100
    modality_weights = [0.3334, 0.3333, 0.3333]
    results = []
    saliences = np.array([1, 1, 1])
    p_bar = tqdm(total=n_runs)

    agents = []

    for i in range(n_runs):
        env = Environment(dog_day=dog_day, dog_modalities=3)
        agent = Agent(env=env, modality_weights=modality_weights, saliences=saliences, n_moves=n_days)
        agents.append(agent)

    with Pool(processes=N_PROCESSES) as pool:
        sim_results = pool.map(run_agent, agents)
        results.append(sim_results)
        p_bar.update(1)

    p_bar.close()

    pickle_this(results, "control")


def vary_salience():
    exponential = False

    d_salience = 50

    n_days = 10200
    dog_day = 10000

    n_experiments = 20
    runs_per_experiment = 20

    n_runs = n_experiments * runs_per_experiment

    modality_weights = [1, 0, 0]

    all_results = []

    p_bar = tqdm(total=n_runs)

    for i in range(n_experiments):  # loop over experiments
        # Setup all agents:
        agents = []
        for j in range(runs_per_experiment):
            p_bar.update(1)
            env = Environment(dog_day=dog_day, dog_modalities=1)
            saliences = np.array([1, 1, i*d_salience])
            if exponential:
                saliences = np.array([1, 1, 2**i])
            agent = Agent(env=env, modality_weights=modality_weights, saliences=saliences, n_moves=n_days)
            agents.append(agent)

        # Run agent and collect results:
        with Pool(processes=N_PROCESSES) as pool:
            sim_results = pool.map(run_agent, agents)
            all_results.append(sim_results)

    # close progress bar:
    p_bar.close()

    # Store results to file:
    if exponential:
        pickle_this(all_results, "vary_dog_salience_exponential")
    else:
        pickle_this(all_results, "vary_dog_salience_linear")


def vary_dog_day():
    n_days = 20000
    delta_dog_day = 2000

    n_experiments = int(n_days/delta_dog_day)
    runs_per_experiment = 10

    n_runs = n_experiments * runs_per_experiment

    modality_weights = [1, 0, 0]  # Only consider a single modality

    saliences = np.array([1, 1, 500])

    all_results = []

    p_bar = tqdm(total=n_runs)

    for i in range(n_experiments):  # loop over possible dog days
        # Setup all agents:
        agents = []
        for j in range(runs_per_experiment):
            p_bar.update(1)
            env = Environment(dog_day=i*delta_dog_day, dog_modalities=1)
            agent = Agent(env=env, modality_weights=modality_weights, saliences=saliences, n_moves=n_days)
            agents.append(agent)

        # Run agent and collect results:
        with Pool(processes=N_PROCESSES) as pool:
            sim_results = pool.map(run_agent, agents)
            all_results.append(sim_results)

    # close progress bar:
    p_bar.close()

    # Store results to file:
    pickle_this(all_results, "vary_dog_day")


def vary_dog_modalities():
    dog_day = 15000
    n_days = 50000
    n_runs = 10

    modality_weights = np.array(
        [
            [1.0, 0.0, 0.0],  # one modality
            [0.5, 0.5, 0.0],  # two modalities
            [0.3334, 0.3333, 0.3333]  # three modalities
        ]
    )

    # How salient is the sight of the dog (3rd value) compared to the other outcomes:
    saliences = np.array([1, 1, 500])

    # all results:
    all_results = []

    # Create progress bar:
    p_bar = tqdm(total=4*n_runs)

    for i in range(4):      # loop over possible count of dog modalities:
        # Setup all agents:
        agents = []
        for j in range(n_runs):
            p_bar.update(1)
            dog_modalities = i
            # Setup environment:
            env = Environment(dog_day=dog_day, dog_modalities=dog_modalities)

            # Setup agent:
            agent = Agent(env=env, modality_weights=modality_weights[2], saliences=saliences, n_moves=n_days)
            agents.append(agent)

        # Run agent and collect results:
        with Pool(processes=N_PROCESSES) as pool:
            sim_results = pool.map(run_agent, agents)
            all_results.append(sim_results)

    # close progress bar:
    p_bar.close()

    # Store results to file:
    pickle_this(all_results, "vary_dog_modalities")


def run_agent(agent):
    return agent.run()


def vary_modalities():
    dog_day = 5000
    n_days = 20000

    dog_modalities = 3

    saliences = np.array([1, 1, 500])

    modality_weights = np.array(
        [
            [1.0, 0.0, 0.0],                # one modality
            [0.5, 0.5, 0.0],                # two modalities
            [0.3334, 0.3333, 0.3333]        # three modalities
        ]
    )

    # Run experiments:
    for i in range(3):

        # Setup environment:
        env = Environment(dog_day=dog_day, dog_modalities=dog_modalities)

        # Setup agent:
        agent = Agent(env=env, modality_weights=modality_weights[i], saliences=saliences, n_moves=n_days)

        # Run agent:
        results = agent.run()


def run_experiments():
    # Set up trial:
    dog_day = 15000
    n_moves = 50000

    bin_size = 500

    # experiment parameters
    saliences = np.array([1, 1, 500])

    dog_modalities = 3

    modality_weights_one = [1, 0, 0]
    modality_weights_two = [0.5, 0.5, 0]
    modality_weights_three = [0.334, 0.333, 0.333]

    env = Environment(dog_day, dog_modalities=dog_modalities)
    agent = Agent(env=env, modality_weights=modality_weights_three, saliences=saliences, n_moves=n_moves, deterministic=False)

    # Run a trial of n moves:
    for i in range(n_moves):
        agent.act()

    epist_vals = np.array(agent.epistemic_log)
    plt.plot(epist_vals[:, 0])
    plt.plot(epist_vals[:, 1])
    plt.plot(epist_vals[:, 2])
    plt.show()

    # Collect results:
    report = env.report()
    move_log = report["log"]
    dd = report["dog_encounter"]
    show_results_running_avg(agent, move_log, dd, n_moves, bin_size)


def show_results_running_avg(agent, move_log, dd, n_moves, bin_size):
    running_avgs = [[], [], []]

    for i in range(len(move_log) - bin_size):
        slce = move_log[i:i+bin_size]
        for j in range(3):
            slce_count = 0
            for k in range(bin_size):
                if slce[k] == j:
                    slce_count += 1
            running_avgs[j].append(slce_count)
    for i in range(3):
        plt.plot(running_avgs[i])

    plt.legend(["action 0", "action 1", "action 2"])
    plt.show()


def show_results(agent, move_log, dd, n_moves, bin_size):
    n_splits = int(n_moves/bin_size)
    X = np.arange(n_splits)
    print(n_splits)
    split_moves = np.split(np.array(move_log), n_splits)
    split_move_counts = np.array([[len(group[group == i]) for i in range(3)] for group in split_moves]).T

    #plt.bar(X, split_move_counts[0], width=0.5)
    #plt.bar(X, split_move_counts[1], width=0.5, bottom=split_move_counts[0])
    #plt.bar(X, split_move_counts[2], width=0.5, bottom=split_move_counts[1] + split_move_counts[0])

    #plt.show()

    print(agent.pred_outcome_given_state_per_modality)

    plt.plot(split_move_counts[0])
    plt.plot(split_move_counts[1])
    plt.plot(split_move_counts[2])
    plt.legend(["action 0", "action 1", "action 2"])
    plt.show()


def pickle_this(data, filename):
    f = open(filename, "wb")
    pickle.dump(data, f)
    f.close()


def unpickle_this(filename):
    f = open(filename, "rb")
    data = pickle.load(f)
    f.close()
    return data


if __name__ == '__main__':
    main()
