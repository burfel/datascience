from sim300 import *
import sys
import numpy as np


# to be called in the command line as:
# python menu.py <N> <Width> <Height>
#
# loading comand line arguments
# N - Number of particles
# Width and Height of window

[N, Width, Height] = map(int, sys.argv[1:])


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
    parameters = list(par_file)[1:]
    parameters = [float(par.split(" = ")[1]) for par in parameters]
    par_file.close()

    return parameters

def load_model(par_file):
    file = open(par_file, 'r')
    line = file.readline()
    file.close()

    return line.split(" = ")[1][:-1]


parameters_file = "parameters.txt"
model = 0


repeat = raw_input("Repeat simulation ([y]/n)?\n")
while repeat not in ['', 'y', 'n']:
    repeat = raw_input("Please enter valid input ([y]/n)?\n")

if repeat == 'n':
    if model not in ['None', 'smpl', 'czn', 'vsck', 'czn2']:
        print('Models available:\n'
              'Simple speed coupling............ smpl\n'
              'Couzin model.....................  czn\n'
              'Viscek model..................... vsck\n'
              'Couzin-2 model................... czn2')
        model = raw_input("Please choose a model.")
        while model not in ['smpl', 'czn', 'vsck', 'czn2']:
            model = raw_input("Please enter a valid model.")
        save_parameters(['model'], [model], parameters_file)

    rep_par = raw_input("Use saved parameters ([y]/n)?\n")
    while rep_par not in ['', 'y', 'n']:
        rep_par = raw_input("Please enter valid input ([y]/n)?\n")

    if rep_par in ['', 'y']:
         pass

    else:
        print('Previous parameters will be overwritten.')
        if model in ['None', 'smpl']:
            s = input("Speed: s = ")
            a = input("Coupling: a = ")
            save_parameters(['s', 'a'], [s, a], parameters_file)

        elif model == 'czn':
            s = input("Speed: s = ")
            noise = input("Noise: noise = ")
            dTheta = input("Max angle of turn: dTheta = ")
            rr = input("Repulsion radius: rr = ")
            ro = input("Orientation radius: ro = ")
            ra = input("Atraction radius: ra = ")
            sight_theta = input("Field of view: sight_theta = ")
            save_parameters(['s', 'noise', 'dTheta', 'rr', 'ro', 'ra', 'sight_theta'], [s, noise, dTheta, rr, ro, ra, sight_theta], parameters_file)

        elif model == 'czn2':
            s = input("Speed: s = ")
            noise = input("Noise: noise = ")
            dTheta = input("Max angle of turn: dTheta = ")
            rr = input("Repulsion radius: rr = ")
            roa = input("Orientation/atraction radius: roa = ")
            #orient = input("Orientation weight: ra = ")
            #atract = input("Atract weight: sight_theta = ")
            save_parameters(['s', 'noise', 'dTheta', 'rr', 'roa'], [s, noise, dTheta, rr, roa], parameters_file)


        elif model == 'vsck':
            s = input("Speed: s = ")
            noise = input("Noise: noise = ")
            r = input("Repulsion radius: r = ")
            save_parameters(['s', 'noise', 'r'], [s, noise, r], parameters_file)

else:
    model = load_model('parameters.txt')

[a, s, r, rr, ro, ra, roa, noise, prop, weight, biasx,
biasy, dev_bias, dTheta, sight_theta, atract, orient] = load_parameters(parameters_file)

if model in ['None', 'smpl']:
    def interaction(agents, speeds):
        return couple_speeds(agents, speeds, a, s, N)

elif model == 'czn':
    def interaction(agents, speeds):
        return couzin(agents, speeds, N, Width, Height, s, noise, dTheta, rr, ro, ra, sight_theta, 0, roa, atract, orient)

elif model == 'czn2':
    def interaction(agents, speeds):
        return couzin(agents, speeds, N, Width, Height, s, noise, dTheta, rr, ro, ra, sight_theta, 1, roa, atract, orient)

elif model == 'vsck':
    def interaction(agents, speeds):
        return vicsek(agents, speeds, N, s, noise, r)




def run(N_steps, dt):
    # N_steps, a, dt, N, width, height, s, rr, ro, ra, noise,
    # prop, weight, bias, dev_bias, dTheta, sight_theta):
    """
    Simulates motion of swarm. Recieves following parameters:

    N_steps  - number of steps to perform
    a -  coupling between neighbouring points
    dt - time step to be used
    N - number of points to be used
    width & heigth of window
    s - module of speed throughout agents

    """
    prop = 0.4
    weight = 0.8
    bias = np.array([biasx,biasy])
    dev_bias = 0.1
    rot_bias = 1


    agents, speeds = initialize_agents(s, N, Width, Height)
    window = initialize_window(agents, Width, Height)
    old_cm=np.array([0.0,0.0])

    for i in range(N_steps):

        next_step(agents, speeds, dt, N)
        start = timeit.default_timer()
        ## MODEL
        #couple_speeds(agents, speeds, a, s, N)
        #vicsek(agents, speeds, N, s, 0.1, r = 500)
        #couzin(agents, speeds, N, s, noise, dTheta, rr, ro, ra, sight_theta)
        #couzin2(agents, speeds, N, s, noise,20,200, 1.5,1) #(...)last 3 parameters: roa, orient, atract
        interaction(agents, speeds)


        biaser(speeds, N, s, i, prop, bias, dev_bias, weight)

        #INFORMATION TRANSFER: SHAPE & DIRECTION & ALIGNMENT QUALITY 
        #point_cm.undraw()
        old_cm = quality(agents, speeds, N, bias, window, old_cm)

        ## BOUNDARY CONDITIONS
        #rigid_boundary(width, height, agents, speeds, N)
        periodic_boundary(Width, Height, agents, speeds, N)

        #time.sleep(0.02)
        stop = timeit.default_timer()
        #print stop - start
    window.close()
    return


run(200, 1)

