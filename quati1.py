import numpy as np
from processData import *
from sim3001 import *
from menu1 import *
#import matplotlib
#matplotlib.use("TkAgg")
#from matplotlib import pyplot as plt
from timeit import default_timer


def def_and_run(parameters, N_steps, dt, graphics=True):
    [model, use_pbc, use_bias, N,
     s, a,
     r, rr, ro, ra, roa, noise, n_lead, prop,
     weight, bias_angle_1, bias_angle_2, bias_angle_3, dev_bias,
     dTheta, sight_theta, atract, orient,
     cr, ca, lr, la, alpha, beta, mass] = parameters

    N = int(N)
    n_lead = int(n_lead)   

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
        biases = [np.array([np.cos(np.pi * bias_angle_1 / 180),
                            np.sin(np.pi * bias_angle_1 / 180)]),
                  np.array([np.cos(np.pi * bias_angle_2 / 180),
                            np.sin(np.pi * bias_angle_2 / 180)]),
                  np.array([np.cos(np.pi * bias_angle_3 / 180),
                            np.sin(np.pi * bias_angle_3 / 180)])]

        def treat_biases(agents, speeds, leaders):
            for i in range(int(n_lead)):
                biaser(agents, leaders[i], speeds, N,
                        s, prop, biases[i], dev_bias, weight)

    else:
        def treat_biases(agents, leaders, speeds):
            return

    # define particle interaction based on chosen model
    if model in ['None', 0]:
        def interaction(agents, speeds, dt):
            other_agents = agents_to_interact(agents)
            return couple_speeds(agents, other_agents, speeds, a, s, N)

    elif model == 1:
        def interaction(agents, speeds, dt):
            other_agents = agents_to_interact(agents)
            return couzin(agents, other_agents, speeds, N, Width, Height, s, noise,
                          dTheta, rr, ro, ra, sight_theta, 0, roa, atract, orient)

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
        agents, speeds = initialize_agents(s, N, Width, Height, Height/2)
        leaders = window = closed = 1
        if use_bias:
            leaders = initialize_leaders(agents, prop, n_lead, N)
        if graphics:
            window = initialize_window(agents, Width, Height)
            closed = False

        start = default_timer()
        for i in range(N_steps):
            
            # Intruduction of a bias in "prop" of the agents
            if use_bias:
                treat_biases(agents, speeds, leaders)

            # Model for agent interactions
            interaction(agents, speeds, dt)

            # Let system evolve with simple Euler method
            next_step(agents, speeds, dt, N)

            # BOUNDARY CONDITIONS
            treat_boundary(agents, speeds)

            if graphics and not closed:
                key = window.checkKey()
                if key == "Escape":
                    window.close()
                    closed = True
                window.update()

        stop = default_timer()
        print stop - start
        if graphics:
            window.close()

        return agents, speeds
    
    return run(N_steps, dt)



def analysis(study_par, par_file):
    model_names = ['smpl', 'czn1', 'vsck', 'czn2', 'mill']

    saved_parameters = load_parameters(par_file)
    used_model = model_names[int(saved_parameters[0])]
    N = saved_parameters[3]

    par_names = ['model', 'use_pbc', 'use_bias', 'N',
                 's', 'a',
                 'r', 'rr', 'ro', 'ra', 'roa', 'noise', 'n_lead', 'prop',
                 'weight', 'bias_angle_1', 'bias_angle_2', 'bias_angle_3',
                 'dev_bias', 'dTheta', 'sight_theta', 'atract', 'orient',
                 'cr', 'ca', 'lr', 'la', 'alpha', 'beta', 'mass']


    ############ You should work HERE ##################
    def treat_data(agents, speeds, val, data_file):
        # This function should do whatever treatment your working on
        # and save a new data point. First indicate the data file in use:

        # then you should call the observable function(s) you are interested in:
        elong = elongation(agents)
        accur = accuracy(speeds, bias)
        disp = dispersion(agents)

        # then you should save the data point appropriately
        save_datapoint(val, [elong, accur, disp], data_file)
        return
    ####################################################


    par_index = par_names.index(study_par)
    bias_angle_1 = saved_parameters[15]
    bias = np.array([np.cos(np.pi * bias_angle_1 / 180),
                            np.sin(np.pi * bias_angle_1 / 180)])


    
    data_file = 'data/' + str(N) + "_" + study_par + ".csv"
    file = open(data_file, 'w')
    file.truncate()
    file.write(study_par + '\telong\taccur\tdisp\n' )
    file.close()
    par_range = np.arange(1.0/N, 1.0, 1.0/N)

    #agents, speeds = def_and_run(saved_parameters, 150, 1, True)

    for val in par_range:
        print(val)
        saved_parameters[par_index] = val
        elong = accur = disp = 0
        
        for j in range(15):
            agents, speeds = def_and_run(saved_parameters, 300, 1, False)
            elong += elongation(agents)
            accur += accuracy(speeds, bias)
            disp += dispersion(agents)
        elong /= 15
        accur /= 15
        disp /= 15
        save_datapoint(val, [elong, accur, disp], data_file)
        #treat_data(agents, speeds, val, data_file)

    #plot(data_file)

N_steps = int(sys.argv[1])
dt = float(sys.argv[2])
parameters_file = 'parameters/' + sys.argv[3] + '.txt'
print(N_steps)

Height = 500
Width = 500

parameters = menu(parameters_file)
def_and_run(parameters, N_steps, dt, graphics=True)





























#N = 10 * (i + 1)
#analysis('prop', 'parameters/parameters4.txt')
'''
par_names = ['model', 'use_pbc', 'use_bias', 'N',
                 's', 'a',
                 'r', 'rr', 'ro', 'ra', 'roa', 'noise', 'n_lead', 'prop',
                 'weight', 'bias_angle_1', 'bias_angle_2', 'bias_angle_3',
                 'dev_bias', 'dTheta', 'sight_theta', 'atract', 'orient',
                 'cr', 'ca', 'lr', 'la', 'alpha', 'beta', 'mass']

parameters = load_parameters('parameters/parameters3.txt')
par_index = par_names.index('prop')
bias_angle_1 = parameters[15]
bias = np.array([np.cos(np.pi * bias_angle_1 / 180),
                            np.sin(np.pi * bias_angle_1 / 180)])

for prop in range(1,10):
    parameters[par_index] = prop/10.0
    print(prop)
    def_and_run(parameters, 300, 1, True)

'''
