import matplotlib.pyplot as plt
import pickle
import numpy as np


def main():
    plot_dog_modalities()
    # plot_dog_day()
    # plot_dog_salience()
    #plot_control()


def unpickle_this(filename):
    f = open(filename, "rb")
    data = pickle.load(f)
    f.close()
    return data

def plot_control():
    data = np.array(unpickle_this("control"))
    for i in range(100):
        print(data[i])



def plot_dog_salience():
    exponential = False
    data = None
    if exponential:
        data = np.array(unpickle_this("vary_dog_salience_exponential"))
    else:
        data = np.array(unpickle_this("vary_dog_salience_linear"))
    avgs_after_dog_day = []

    bin_size = 200
    n_experiments = 20
    runs_per_experiment = 20

    all_avgs_before = []

    all_avgs_after = []

    plot_colors = ["maroon", "red", "orange", "yellow", "greenyellow", "green", "aqua", "blue", "darkviolet",
                   "magenta", ]

    for i in range(n_experiments):
        partial_data = data[i]
        avgs_before = []
        avgs_after = []
        for j in range(runs_per_experiment):
            before = np.array(partial_data[j]["log"][9800:10000])
            after = np.array(partial_data[j]["log"][10000:10200])
            avgs_before.append(len(before[before == 1])/200)
            avgs_after.append(len(after[after == 1])/200)
        all_avgs_before.append(avgs_before)
        all_avgs_after.append(avgs_after)

    all_avgs_after = np.array(all_avgs_after)
    all_avgs_before = np.array(all_avgs_before)

    x = None
    if exponential:
        x = range(n_experiments)
    else:
        x = np.multiply(range(n_experiments), 50)

    for i in range(20):
        plt.scatter(x,  all_avgs_before[:,i], alpha=0.1, color="blue")
        plt.scatter(x, all_avgs_after[:,i], alpha=0.1, color="orange")
        if i == 0:
            plt.scatter(x, np.mean(all_avgs_before, axis=1), alpha=1,  color="blue", label="Before dog-day")
            plt.scatter(x, np.mean(all_avgs_after, axis=1), alpha=1,  color="orange", label="After dog-day")
        else:
            plt.scatter(x, np.mean(all_avgs_before, axis=1), alpha=1, color="blue")
            plt.scatter(x, np.mean(all_avgs_after, axis=1), alpha=1, color="orange")
    plt.legend()
    plt.xlabel("Salience of seeing dog")
    plt.ylabel("Percentage of visits to dog state")
    plt.title("Percentage of performing action 1,\n over 200 days before and 200 days after dog-day \n as a function of salience")
    plt.show()

    plt.scatter(np.multiply(range(n_experiments), 50), np.mean(all_avgs_before, axis=1), alpha=1)
    plt.scatter(np.multiply(range(n_experiments), 50), np.mean(all_avgs_after, axis=1), alpha=1)


    plt.legend(["Visits before dog encounter", "Visits after dog encounter"])
    plt.xlabel("Salience of seeing dog")
    plt.ylabel("Percentage of action 1")
    plt.title("Percentage of visits to the dog state out of all actions,\n over 200 days before and 200 days after dog encounter for different saliences")
    plt.show()


def plot_dog_day():
    data = np.array(unpickle_this("vary_dog_day"))
    bin_size = 100

    running_avgs = [
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        []
    ]

    mean_running_avgs = [
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        []
    ]

    plot_colors = ["maroon", "red", "orange", "yellow", "greenyellow", "green", "aqua", "blue", "darkviolet", "magenta", ]

    mean_qualities = np.array([[data[i][j]["q_log"] for j in range(10)] for i in range(10)])
    print(mean_qualities.shape)
    fig, ax = plt.subplots(1, 3)
    for i in range(3):
        for j in range(10):
            for k in range(10):
                ax[i].plot(mean_qualities[j, k, :, i], color=plot_colors[j], alpha=0.2)
            ax[i].plot(np.mean(np.array(mean_qualities)[j, :, :, i], axis=0), color=plot_colors[j], label="dd: "+str(j*2000))
    ax[2].legend()
    ax[0].set_ylabel("quality")
    for i in range(3):
        ax[i].set_title("action "+str(i))
    ax[1].set_xlabel("time (days)")
    plt.suptitle("Qualities per action over time with varying 'dog day' (dd)")
    plt.show()

    for i in range(10):
        partial_data = data[i]
        partial_running_avg = []
        for j in range(10):
            single_run = partial_data[j]
            single_run_running_avgs = get_running_avg(move_log=single_run["log"], bin_size=bin_size)
            partial_running_avg.append(single_run_running_avgs)
        running_avgs[i] = partial_running_avg
        mean_running_avgs[i] = np.mean(partial_running_avg, axis=0)

    mean_running_avgs = np.array(mean_running_avgs)
    running_avgs = np.array(running_avgs)
    # Plot the mean per dog day:
    for i in range(10):
        plt.plot(mean_running_avgs[i, 1].T, color=plot_colors[i], alpha=0.8)

    plt.legend(["Dog spotted at day " + str(i * 2000)for i in range(10)], loc="lower right")
    plt.xlabel("Time (days)")
    plt.ylabel("Percentage of dog state visits")
    plt.title("Percentage of visiting the 'dog state' by the agent over time (dog salience = 500, average values of 10 agents per condition")
    plt.show()

    # Plot all runs of 3 dog modalities:
    #plt.plot(running_avgs[3, :, 1].T, color="blue", alpha=0.1)
    #plt.plot(mean_running_avgs[3, 1].T, color="red")
    #plt.legend(["All runs", "Mean of all runs"])
    #plt.show()


def plot_dog_modalities():
    data = unpickle_this("vary_dog_modalities")
    bin_size = 100

    running_avgs = [
        [],
        [],
        [],
        []
    ]

    mean_running_avgs = [
        [],
        [],
        [],
        []
    ]

    plot_colors = ["red", "green", "blue", "orange"]

    mean_qualities = np.array([[data[i][j]["q_log"] for j in range(10)] for i in range(4)])
    print(mean_qualities.shape)
    fig, ax = plt.subplots(1, 3)
    for i in range(3):

        for j in range(4):
            if j ==0:
                for k in range(10):
                    ax[i].plot(mean_qualities[j, k, :, i], color=plot_colors[j], alpha=0.2)
                ax[i].plot(np.mean(np.array(mean_qualities)[j, :, :, i], axis=0), color=plot_colors[j],
                           label=str(j) + "/3 exposed modalities")
    #ax[2].legend()
    ax[0].set_ylabel("quality")
    for i in range(3):
        ax[i].set_title("action " + str(i))
    ax[1].set_xlabel("time (days)")
    #plt.suptitle("Qualities per action over time while varying the number of sensory modalities exposed to the dog encounter")
    plt.suptitle("Qualities per action over time without a dog encounter")
    plt.show()


    for i in range(4):
        partial_data = data[i]
        partial_running_avg = []
        for j in range(10):
            single_run = partial_data[j]
            single_run_running_avgs = get_running_avg(move_log=single_run["log"], bin_size=bin_size)
            partial_running_avg.append(single_run_running_avgs)
        running_avgs[i] = partial_running_avg
        mean_running_avgs[i] = np.mean(partial_running_avg, axis=0)

    mean_running_avgs = np.array(mean_running_avgs)
    running_avgs = np.array(running_avgs)
    # Plot the mean per dog modality count:
    for i in range(4):
        plt.plot(mean_running_avgs[i, 1].T, color=plot_colors[i], alpha=0.8)

    plt.legend(["0 dog modalities", "1 dog modalities", "2 dog modalities", "3 dog modalities"], loc="lower left")
    plt.show()

    # Plot all runs of 3 dog modalities:
    plt.plot(running_avgs[3, :, 1].T, color="blue", alpha=0.1)
    plt.plot(mean_running_avgs[3, 1].T, color="red")
    plt.legend(["All runs", "Mean of all runs"])
    plt.show()


def get_running_avg(move_log, bin_size):
    running_avgs = [[], [], []]

    for i in range(len(move_log) - bin_size):
        slce = move_log[i:i+bin_size]
        for j in range(3):
            slce_count = 0
            for k in range(bin_size):
                if slce[k] == j:
                    slce_count += 1
            running_avgs[j].append(slce_count)
    return running_avgs

def better_running_avg(move_log, bin_size):
    pass

if __name__ == '__main__':
    main()
