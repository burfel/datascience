from math import *
from graphics import *
from operator import *
from random import *
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import timeit


def save_datapoint(x, y, data_file):
    file = open(data_file, 'a')
    file.write(str(x) + '\t' + str(y) + '\n')
    file.close()
    return


def avg_dev(agents):
    poses = [np.array([a.getX(), a.getY()]) for a in agents]
    return np.linalg.norm(np.std(poses, axis=0))


def quality(agents, speeds, N, bias, window, old_cm):
    # ACCURACY (DIRECTIONS DISTRIBUTION)
    dev = 0.0
    for i in range(N):
        dev += angle(bias, speeds[i])
    dev_avg = dev / (N * 2 * (np.pi))

    # ELONGATION: SHAPE OF SWARM
    # can be smarter if we make agents become poses before,
    # more globally in  the code
    poses = [np.array([a.getX(), a.getY()]) for a in agents]

    # Center of Mass
    cm = np.mean(poses, axis=0)

    # Standard Deviation
    std = np.std(poses, axis=0)

    # Elongation >1 means
    elong = std[1] / std[0]

    # GROUP DIRECTION
    # Vector
    group_dir = np.array([cm[0] - old_cm[0], cm[1] - old_cm[1]])
    # Norm
    group_dir = np.linalg.norm(group_dir)
    return cm


def plot(file):
    data = np.genfromtxt(file, delimiter='\t', skip_header=7,
                         skip_footer=3, names=['x', 'y'])
    fig = plt.figure()

    ax1 = fig.add_subplot(111)

    ax1.set_title("Agents's dispersion")
    ax1.set_xlabel('time')
    ax1.set_ylabel('Dipersion')

    ax1.plot(data['x'], data['y'], c='r', label='the data')

    leg = ax1.legend()

    plt.show()


    fig.plot(data['x'], data['y'], color='r', label='the data')




