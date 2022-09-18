# for all the subdirectories of the current directory read the density.csv file and calculate the average density
# and the standard deviation of the density

# import the necessary packages
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def scatter_hist(x, y, ax, ax_histx, ax_histy):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y)

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')


# read the density.csv file from 10 subsdirectories of agents_performance_analysis
# and calculate the average density and the standard deviation of the density


def read_density_csv(path, subdirectories, quantities):

    nr_of_quantities = len(quantities)
    # initialize the list of outputs
    outputs = [[] for _ in range(nr_of_quantities)]
    # loop through all the subdirectories of the current directory
    for counter, file_extension in enumerate(quantities):
        for subdir in subdirectories:
            csv_file = os.path.join(path, str(subdir), file_extension + '.csv')
            data = np.genfromtxt(csv_file, delimiter=',')
            # if key is density remove the last element of the array
            if file_extension == 'density':
                data = data[:-1]
            outputs[counter].append(data)

    # create a dictionary with file extensions as keys and the list of outputs as values
    output_dict = dict(zip(quantities, outputs))
    return output_dict


def hist_plot(outputs, quantity, xlabel, bins=50, last=True):
    """

    """
    if last:
        id = -1
        init_final = 'Final'
    else:
        id = 0
        init_final = 'Initial'

    init_final_values = [value[id] for value in outputs[quantity]]
    # plot the histogram
    plt.hist(init_final_values, bins=bins)
    if last:
        plt.xlabel(init_final + ' ' + xlabel)
    else:
        plt.xlabel(init_final + ' ' + xlabel)

    plt.ylabel('Frequency')
    plt.title('Histogram of the ' + init_final + ' ' + quantity)
    # save the histogram as a png file
    plt.savefig('histogram_' + init_final + '_' + quantity + '.png')
    plt.close()


quantities = ['density', 'energy_cost', 'G2', 'accumulated_reward' ]
ylabels = ['Density', 'Energy cost', '|Grad(PF)| ^ 2', 'Reward']
nr_tests = 1000
print("numer of tests: ", nr_tests)
subdirectories = list(range(0, nr_tests))
outputs = read_density_csv('.',
                           subdirectories=subdirectories,
                           quantities=quantities)

# Plot some histograms
for quantity, ylabel in zip(quantities, ylabels):
    hist_plot(outputs, quantity, ylabel)


# create a dataframe with the initial and final density and marginal plot
initial_density = [value[0] for value in outputs['density']]
final_density = [value[-1] for value in outputs['density']]
df = pd.DataFrame({'Initial density': initial_density, 'Final density': final_density})
sns.jointplot(x='Initial density', y='Final density', data=df, kind='reg')
# plt.show()
plt.savefig('initial_density_vs_final_density_scatter.png')
plt.close()

""" # create a dataframe with the initial density and energy cost and marginal plot
energy_cost = [value[-1] for value in outputs['energy_cost']]
df = pd.DataFrame({'Initial density': initial_density, 'Energy cost': energy_cost})
sns.jointplot(x='Initial density', y='Energy cost', data=df, kind='reg')
# plt.show()
plt.savefig('initial_density_vs_energy_cost_scatter.png')
plt.close() """


# create a dataframe with the initial density and G2 and marginal plot
G2 = [value[-1] for value in outputs['G2']]
df = pd.DataFrame({'Initial density': initial_density, '|Grad(PF)| ^ 2': G2})
sns.jointplot(x='Initial density', y='|Grad(PF)| ^ 2', data=df, kind='reg')
# plt.show()
plt.savefig('initial_density_vs_G2_scatter.png')
plt.close()


# create a dataframe with the initial density and the length of the density and marginal plot
episode_length = [len(value) for value in outputs['density']]
df = pd.DataFrame({'Initial density': initial_density, 'Episode length': episode_length})
sns.jointplot(x='Initial density', y='Episode length', data=df, kind="reg")
# plt.show()
plt.savefig('initial_density_vs_episode_length_scatter.png')
plt.close()

# create a dataframe with the initial density and the final reward and marginal plot
reward_final = [value[-1] for value in outputs['accumulated_reward']]
df = pd.DataFrame({'Initial density': initial_density, 'Reward': reward_final})
sns.jointplot(x='Initial density', y='Reward', data=df, kind="reg")
# plt.show()
plt.savefig('initial_density_vs_reward_scatter.png')
plt.close()


# create a dataframe with the initial density and the final reward and marginal plot
reward_final_only_geometry = np.array(reward_final) # + np.array(energy_cost)
df = pd.DataFrame({'Initial density': initial_density, 'Reward (only geometry)': reward_final_only_geometry})
sns.jointplot(x='Initial density', y='Reward (only geometry)', data=df, kind="reg")
# plt.show()
plt.savefig('initial_density_vs_reward_only_geometry_scatter.png')
plt.close()


# End the program
print('Program finished')
