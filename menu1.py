from processData import *
from sim3001 import *
import numpy as np
import sys


# to be called in the command line as:
# python menu.py <N> <Width> <Height>
#
# loading comand line arguments
# N - Number of particles
# Width and Height of window

#parameters_file = 'parameters'

#parameters_file = 'parameters/' + parameters_file + ".txt"
#data_file = 'data.csv'




def menu(parameters_file):
    ####################################### Ask for requirements ################################################

    repeat = raw_input("Repeat simulation ([y]/n)?\n")
    while repeat not in ['', 'y', 'n']:
        repeat = raw_input("Please enter valid input ([y]/n)?\n")

    repeat = 'y'
    if repeat == 'n':

        N = input("Number of agents: N = ")
        save_parameters(['N'], [N], parameters_file)

        ############################ Model Selection ################################
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


    return load_parameters(parameters_file)
########################################## Implement requirements ################################################



