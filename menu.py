from processData import *
from sim300 import *
import numpy as np
import sys


# to be called in the command line as:
# python menu.py <N> <Width> <Height>
#
# loading comand line arguments
# N - Number of particles
# Width and Height of window

[N, Width, Height] = map(int, sys.argv[1:])
steps = 1000
time_step = 1
parameters_file = 'mill_I'

parameters_file = 'parameters/' + parameters_file + ".txt"
data_file = 'data.csv'
model = 0


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


####################################### Ask for requirements ################################################


N = input("Number of agents: N = ")
Width = input("Window dimensions: Width = ")
Height = input("Window dimensions: Height = ")



repeat = raw_input("Repeat simulation ([y]/n)?\n")
while repeat not in ['', 'y', 'n']:
    repeat = raw_input("Please enter valid input ([y]/n)?\n")

if repeat == 'n':
    
    ############################ Model Selection ################################
    if model not in ['None', 'smpl', 'czn', 'vsck', 'czn2', 'mill']:
        print('Models available:\n'
              'Simple speed coupling............ 0\n'
              'Couzin model..................... 1\n'
              'Viscek model..................... 2\n'
              'Couzin-2 model................... 3\n'
              'Mill model....................... 4')

        model = input("Please choose a model.\n")
        while model not in range(5):
            model = input("Please enter a valid model.\n")
        save_parameters(['model'], [model], parameters_file)


    ############################## Use same Parameter? ####################################
    rep_par = raw_input("Use saved parameters ([y]/n)?\n")
    while rep_par not in ['', 'y', 'n']:
        rep_par = raw_input("Please enter valid input ([y]/n)?\n")

    if rep_par in ['', 'y']:
        pass

    else:
        print('Previous parameters will be overwritten.')
        if model in ['None', 0]:
            s = input("Speed: s = ")
            a = input("Coupling: a = ")
            save_parameters(['s', 'a'], [s, a], parameters_file)

        elif model == 1:
            s = input("Speed: s = ")
            noise = input("Noise: noise = ")
            dTheta = input("Max angle of turn: dTheta = ")
            rr = input("Repulsion radius: rr = ")
            ro = input("Orientation radius: ro = ")
            ra = input("Atraction radius: ra = ")
            sight_theta = input("Field of view: sight_theta = ")
            save_parameters(['s', 'noise', 'dTheta', 'rr',
                             'ro', 'ra', 'sight_theta'],
                            [s, noise, dTheta, rr, ro, ra, sight_theta],
                            parameters_file)

        elif model == 3:
            s = input("Speed: s = ")
            noise = input("Noise: noise = ")
            dTheta = input("Max angle of turn: dTheta = ")
            rr = input("Repulsion radius: rr = ")
            roa = input("Orientation/atraction radius: roa = ")
            orient = input("Orientation weight: orient = ")
            atract = input("Atract weight: atract = ")
            save_parameters(['s', 'noise', 'dTheta', 'rr',
                             'roa', 'atract', 'orient'],
                            [s, noise, dTheta, rr, roa, atract, orient],
                            parameters_file)

        elif model == 2:
            s = input("Speed: s = ")
            noise = input("Noise: noise = ")
            r = input("Repulsion radius: r = ")
            save_parameters(['s', 'noise', 'r'],
                            [s, noise, r], parameters_file)

        elif model == 4:
            cr = input("Repulsion coeficient: cr = ")
            ca = input("Atraction coeficient: ca = ")
            lr = input("Repulsion length: lr = ")
            la = input("Atraction length: la = ")
            alpha = input("Self propulsion: alpha = ")
            beta = input("Friction coeficient: beta = ")
            mass = input("Agents's mass: mass = ")
            save_parameters(['cr', 'ca', 'lr', 'la', 'alpha', 'beta', 'mass'],
                            [cr, ca, lr, la, alpha, beta, mass], parameters_file)

        #########################  Bias Selection ############################
        use_bias = raw_input("Use bias ([y]/n)?\n")
        while use_bias not in ['', 'y', 'n']:
            use_bias = raw_input("Please enter valid input ([y]/n)?\n")

        if use_bias in ['', 'y']:
            use_bias = 1
            prop = input("Fraction of leading agents: prop = ")
            weight = input("Intensity of bias: weight = ")
            dev_bias = input("Jitter of bias: dev_bias = ")
            n_lead = input("Number of biased groups: n_lead = ")
            save_parameters(['n_lead', 'prop', 'weight', 'dev_bias'],
                            [n_lead, prop, weight, dev_bias], parameters_file)
            for i in range(n_lead):
                bias_angle = input("Bias angle for group " + str(i + 1) + ": bias_angle_" + str(i + 1) + " = ")
                save_parameters(['bias_angle_' + str(i + 1)],
                                [bias_angle], parameters_file)
        else:
            use_bias = 0
        save_parameters(['use_bias'],
                        [use_bias], parameters_file)
        ######################################################################


        ####################  Periodic Boudary Conditions ####################
        use_pbc = raw_input("Use PBCs ([y]/n)?\n")
        while use_pbc not in ['', 'y', 'n']:
            use_pbc = raw_input("Please enter valid input ([y]/n)?\n")

        if use_pbc in ['', 'y']:
            use_pbc = 1
        else:
            use_pbc = 0
        save_parameters(['use_pbc'],
                        [use_pbc], parameters_file)
        ######################################################################
else:
    model = load_model('parameters.txt')


#####################################################################################################


[model, use_pbc, use_bias, N, a, s, r, rr, ro, ra, roa, noise, n_lead,
 prop, weight, bias_angle_1, bias_angle_2, bias_angle_3, dev_bias,
 dTheta, sight_theta, atract, orient,
 cr, ca, lr, la, alpha, beta, mass] = load_parameters(parameters_file)


########################################## Implement requirements ################################################

# define treatment of boundaries
if use_pbc:
    def treat_boundary(agents, speeds):
        return periodic_boundary(Width, Height, agents, speeds, N)

    def agents_to_interact(agents):
        return virtualizer(agents, Height, Width, N)

else:
    def treat_boundary(agents, speeds):
        return rigid_boundary(Width, Height, agents, speeds, N)

    def agents_to_interact(agents):
        return N * [agents]

# define bias

if use_bias:
    biases = [np.array([np.cos(np.pi * bias_angle_1 / 180), np.sin(np.pi * bias_angle_1 / 180)]),
            np.array([np.cos(np.pi * bias_angle_2 / 180), np.sin(np.pi * bias_angle_2 / 180)]),
            np.array([np.cos(np.pi * bias_angle_3 / 180), np.sin(np.pi * bias_angle_3 / 180)])]
    print(biases)

    def treat_biases(agents, speeds, leaders, win):
        for i in range(int(n_lead)):
            biases[i] = biaser(agents, leaders[i], speeds, N, s, prop, biases[i], dev_bias, weight, win)


else:
    def treat_biases(agents, leaders, speeds, win):
        return

# define particle interaction based on chosen model
if model in ['None', 0]:
    def interaction(agents, speeds, dt):
        other_agents = agents_to_interact(agents)
        return couple_speeds(agents, other_agents, speeds, a, s, N)

elif model == 1:
    def interaction(agents, speeds, dt):
        other_agents = agents_to_interact(agents)
        return couzin(agents, other_agents, speeds, N, Width, Height, s, noise, dTheta,
                      rr, ro, ra, sight_theta, 0, roa, atract, orient)

elif model == 3:
    def interaction(agents, speeds, dt):
        other_agents = agents_to_interact(agents)
        return couzin(agents, other_agents, speeds, N, Width, Height, s, noise, dTheta,
                      rr, ro, ra, sight_theta, 1, roa, atract, orient)

elif model == 2:
    def interaction(agents, speeds, dt):
        other_agents = agents_to_interact(agents)
        return vicsek(agents, other_agents, speeds, N, s, noise, r)

elif model == 4:
    def interaction(agents, speeds, dt):
        other_agents = agents_to_interact(agents)
        return mill(agents, other_agents, speeds, dt, N, Width, Height, cr, ca, lr, la, alpha, beta, mass)

# define introduction of biases



######################################################################################################################

def run(N_steps, dt):
    """
    Simulates motion of swarm. Recieves following parameters:
    N_steps  - number of steps to perform
    dt - time step to be used
    """

    agents, speeds = initialize_agents(s, N, Width, Height)
    cm = get_cm(agents, N)
    cmagent = Point(cm[0], cm[1])
    cmagent.setFill("green")

    window = initialize_window(agents + [cmagent], Width, Height)

    leaders = 0
    if use_bias:
        leaders = initialize_leaders(agents, prop, int(n_lead), N)

    for i in range(N_steps):

        next_step(agents, speeds, dt, N)

        # Intruduction of a bias in "prop" of the agents
        treat_biases(agents, speeds, leaders, window)

        # Model for agent interactions
        interaction(agents, speeds, dt)

        # INFORMATION TRANSFER: SHAPE & DIRECTION & ALIGNMENT QUALITY
        [dx, dy] = get_cm(agents, N) - cm
        cmagent.move(dx, dy)
        cm = cm + [dx, dy]

        # BOUNDARY CONDITIONS
        treat_boundary(agents, speeds)

        dev = avg_dev(agents)
        save_datapoint(i * dt, dev, data_file)
    window.close()

    #plot(data_file)
    return


run(5000, 0.1)
