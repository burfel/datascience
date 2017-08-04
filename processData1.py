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
    line = ""
    for val in y:
        line += "\t" + str(val)
    file.write(str(x) + line + '\n')
    file.close()
    return

def save_parameters(parameters, values, par_file):
    file = open(par_file, 'r')
    lines = list(file)
    lines = [par.split(" = ") for par in lines]
    file.close()

    for i in range(len(parameters)):
        for line in lines:
            if line[0] == parameters[i]:
                line[1] = str(values[i]) + '\n'

    file = open(par_file, 'w')

    for line in lines:
        file.write(line[0] + ' = ' + str(line[1]))

    file.close()


def load_parameters(par_file):
    par_file = open(par_file, 'r')
    parameters = list(par_file)
    parameters = [float(par.split(" = ")[1]) for par in parameters]
    par_file.close()

    return parameters


def load_model(par_file):
    file = open(par_file, 'r')
    line = file.readline()
    file.close()

    return line.split(" = ")[1][:-1]


def normalized(vector):
    if vector[0] == vector[1] == 0:
        return vector
    return vector / np.linalg.norm(vector)


def angle(vec1, vec2):  # (Not so) stupid way of doing an angle
    return np.arccos(np.dot(normalized(vec1),normalized(vec2)))


def rot_matrix(theta, unit = "None"):
    if unit in ["None", "deg"]:
        c, s = np.cos(pi * theta / 180), np.sin(pi * theta / 180)
    elif unit == "rad":
        c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],[s, c]])


def avg_dev(agents):
    # reeives array of agents, returns cm and dispersion
    poses = [np.array([a.getX(), a.getY()]) for a in agents]
    return avg_vec(poses), std_dev(poses)

def avg_vec(vecs):
    # receives an array of vectors, computes average vector 
    return np.mean(vecs, axis = 0)

def std_dev(vecs):
    # receives an array of vectors, computes std deviation vector
    return np.std(vecs, axis=0)

def avg_dir(vecs):
    norm_vecs = [normalized(vec) for vec in vecs]
    return avg_vec(norm_vecs)

def dispersion(agents):
    cm, std =  avg_dev(agents)
    return np.linalg.norm(std)

def accuracy(speeds, bias):
    # ACCURACY (AVERAGE DEVIATION TO BIAS) 
    return angle(avg_dir(speeds), bias) / np.pi


def elongation(agents):
    # ELONGATION: SHAPE OF SWARM
    cm, std = avg_dev(agents)
    return std[0] / std[1]


#def load_data():

def plot(file_name):
    file = open(file_name, 'r')
    line = file.readline()[:-1]
    line = line.split("\t")
    data = np.genfromtxt(file_name, delimiter='\t', skip_header=0,
                         skip_footer=0, names=line)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    colors = ['r', 'g', 'b']

    x = data[line[0]]

    for i in range(1,len(line)):

        ax1.plot(x, data[line[i]], c=colors[i-1], marker="o", label=line[i])
    plt.legend(loc='upper left');
    plt.show()
    #fig.plot(data['x'], data['y'], color='r', label='the data')








