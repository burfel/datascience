import numpy as np
from processData import *
from sim300 import *
#from menu import *
#import matplotlib
#matplotlib.use("TkAgg")
#from matplotlib import pyplot as plt
#import timeit

Height = 500
Width = 500

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
        #biases = [np.array([np.cos(np.pi * bias_angle_1 / 180),
         #                   np.sin(np.pi * bias_angle_1 / 180)]),
          #        np.array([np.cos(np.pi * bias_angle_2 / 180),
           #                 np.sin(np.pi * bias_angle_2 / 180)]),
            #      np.array([np.cos(np.pi * bias_angle_3 / 180),
             #               np.sin(np.pi * bias_angle_3 / 180)])]
        biases = [[0, 0], [0, 0], [0, 0]]

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
        #biases = [[0, 0], [0, 0], [0, 0]]

        comands = {"Up": [0, -1], "Down": [0, 1], "Right": [1, 0], "Left": [-1, 0]}
        agents, speeds = initialize_agents(s, N, Width, Height)
        #print len(speeds), ' - ', len(agents)
        cm = get_cm(agents, N)
        cmagent = Point(cm[0], cm[1])
        cmagent.setFill("green")

        if graphics:
            window = initialize_window(agents + [cmagent], Width, Height)
            closed = False
        else:
            window = 0
            closed = True


        leaders = 0
        if use_bias:
            leaders = initialize_leaders(agents, prop, n_lead, N)


        for i in range(N_steps):

            #key = window.checkKey()
            #if key in ["Up","Down","Left","Right"]:
            #     biases = [np.array(comands[key]) for i in range(3)]
            #elif key == "Return":
            #    leaders = initialize_leaders(agents, prop, int(n_lead), N)
            #key = ""


            next_step(agents, speeds, dt, N)


            if graphics and not closed:
                key = window.checkKey()
                if key in ["Up","Down","Left","Right"]:
                    biases = [np.array(comands[key]) for i in range(3)]
                elif key == "Return":
                    leaders = initialize_leaders(agents, prop, int(n_lead), N)
                elif key == "Escape":
                    window.close()
                    closed = True
                window.update()
            # Model for agent interactions
            interaction(agents, speeds, dt)

            # Intruduction of a bias in "prop" of the agents
            treat_biases(agents, speeds, leaders)

            # INFORMATION TRANSFER: SHAPE & DIRECTION & ALIGNMENT QUALITY
            [dx, dy] = get_cm(agents, N) - cm
            cmagent.move(dx, dy)
            cm = cm + [dx, dy]

            # BOUNDARY CONDITIONS
            treat_boundary(agents, speeds)

            #save_datapoint(i * dt, dev, data_file)sp
        if graphics:
            window.close()

        #plot(data_file)
        return agents, speeds
    
    return run(N_steps, dt)


def analysis(study_par, par_range, par_file):
    '''
    one should choose the parameters in study a priori having numbers
    corresponding to parameters as such:
    model ...................1
    use_pbc ................ 2
    use_bias ............... 3
    a ...................... 4
    s ...................... 5
    r ...................... 6
    rr ..................... 7
    ro ..................... 8
    ra ..................... 9
    roa .................... 10
    noise .................. 11
    n_lead ................. 12
    prop ................... 13
    weight ................. 14
    bias_angle_1 ........... 15
    bias_angle_2 ........... 16
    bias_angle_3 ........... 17
    dev_bias ............... 18
    dTheta ................. 19
    sight_theta ............ 20
    atract ................. 21
    orient ................. 22
    cr ..................... 23
    ca ..................... 24
    lr ..................... 25
    la ..................... 26
    alpha .................. 27
    beta ................... 28
    mass ................... 29

    '''
    model_names = ['smpl', 'czn1', 'vsck', 'czn2', 'mill']

    saved_parameters = load_parameters(par_file)
    used_model = model_names[int(saved_parameters[0])]
    N = saved_parameters[3]

    par_names = ['model', 'use_pbc', 'use_bias', 'N'
                 's', 'a',
                 'r', 'rr', 'ro', 'ra', 'roa', 'noise', 'n_lead', 'prop',
                 'weight', 'bias_angle_1', 'bias_angle_2', 'bias_angle_3', 'dev_bias',
                 'dTheta', 'sight_theta', 'atract', 'orient',
                 'cr', 'ca', 'lr', 'la', 'alpha', 'beta', 'mass']

    
    par_index = par_names.index(study_par)
    bias_angle_1 = saved_parameters[14]
    bias = np.array([np.cos(np.pi * bias_angle_1 / 180),
                            np.sin(np.pi * bias_angle_1 / 180)])

    ############ You should work HERE #################3
    def treat_data(agents, speeds, val):
        # This function should do whatever treatment your working on
        # and save a new data point. First indicate the data file in use:
        data_file1 = 'data/' + str(N) + "_" + used_model + "_elongation_" + study_par + ".csv"
        data_file2 = 'data/' + str(N) + "_" + used_model + "_accuracy_" + study_par + ".csv"


        # then you should call the observable function(s) you are interested in:
        elong = elongation(agents)
        #print len(speeds), ' - ', len(agents)
        accur = accuracy(speeds, bias)
        
        # then you should save the data point appropriately
        save_datapoint(val, elong, data_file1)
        save_datapoint(val, accur, data_file2)


    for val in par_range:
        print(val)
        saved_parameters[par_index] = val
        agents, speeds = def_and_run(saved_parameters, 500, 1, True)
        treat_data(agents, speeds, val)


pr = np.arange(0.0, 1.0, 0.1)
analysis('prop', pr, 'parameters/parameters.txt')


        
plot("data/" + str(N) + 'czn2_elongation_prop.csv')
plot("data/" + str(N) + 'czn2_accuracy_prop.csv')

















