
# coding: utf-8

# In[ ]:

import matplotlib.pyplot as plt
from graphics import *
from operator import *
from random import *
#from visual import *
import numpy as np
import timeit
from math import *


# In[ ]:

def initialize(speed = 1, N = 100, width = 1000, height = 1000):
    """
    Initializes our agent set with randomly directed speeds, draws the window and the agents
    """
    agents = [Point(width*random(), height*random()) for i in range(N)]
    speeds = [np.array([0.0, 0.0]) for i in range(N)]
    for i in range(N):
        theta = 2 * np.pi * random()
        speeds[i][0] = speed * np.cos(theta)
        speeds[i][1] = speed * np.sin(theta)

    return agents, speeds, draw_agents(agents, width, height)


def draw_agents(agents, width, height):
    win = GraphWin("Swarm", width, height) # size of box
    for agent in agents:
        agent.draw(win)
    win.getMouse()
    return win


def couple_speeds(agents, speeds, a, s, N):
    """
    Simplest model: for each agent, it will give its nearest neighbour a fraction of its speed and re-normalize it
    
    """
    nearest_neighbours = [nearest_neighbour(agent, agents, N) for agent in agents]
    for i in range(N):
            weightedSpeed = a * speeds[nearest_neighbours[i]]
            speeds[i] = speeds[i] + weightedSpeed
            speeds[i] = s * normalized(speeds[i]) ### PROBLEMA?-------------------------------------------------


def get_distances(agent, agents):
    """
    Given one angent and the set of all agents, 
    computes the distances from the first to all of the others
    """
    
    dists = [(a.getX() - agent.getX())**2 + (a.getY() - agent.getY())**2 for a in agents]
    for i in range(len(dists)):
        if dists[i] == 0:
            dists[i] = 0.1
    return dists


def nearest_neighbour(agent, agents, N):
    """
    Returns the index for the agent with smallest Eucledian distance to agent in question
    """
    distances = get_distances(agent, agents)
    j = next(i for i in range(N) if agents[i] is agent)
    distances[j] = distances[j-1] + 1

    return distances.index(min(distances))


def treat_boundary(x_bound, y_bound, agents, speeds, N):
    for i in range(N):
        [dx, dy] = [0, 0]
        [x, y] = [agents[i].getX(), agents[i].getY()]
        if x > x_bound:
            speeds[i][0] = -speeds[i][0]
            dx = x_bound - x         
        
        elif x < 0:
            speeds[i][0] = -speeds[i][0]
            dx = -x
        
        if y > y_bound:
            speeds[i][1] = -speeds[i][1]
            dy = y_bound - y
        
        elif y < 0:
            speeds[i][1] = -speeds[i][1]
            dy = -y

        agents[i].move(dx, dy)
        
        
def periodic_boundary(x_bound, y_bound, agents, speeds, N):  #Changed from rigid boundaries do periodic boundary condition
    [dx, dy] = [0, 0]
    for i in range(N):
        [x, y] = [agents[i].getX(), agents[i].getY()]
        if x > x_bound:
            dx = -x_bound         
        
        elif x < 0:
            dx = x_bound
        
        if y > y_bound:
            dy = -y_bound
        
        elif y < 0:
            dy = y_bound

        agents[i].move(dx, dy)

def next_step(agents, speeds, dt, N):
    dxvec = [dt * speeds[i][0] for i in range(N)]
    dyvec = [dt * speeds[i][1] for i in range(N)]
    for i in range(N):
        agents[i].move(dxvec[i], dyvec[i])
        

def normalized(vector):
    if vector[0] == vector[1] == 0:
        return vector
    return vector / np.linalg.norm(vector)


#def get_v_avg(speeds, N):
#    return
#   
#def get_potencials(distances, N):
#    potentials = np.zeros(N, N)
#    for i in range(N):
#        for j in range(i+1,N):
#            potentials[i][j] = potentials[j][i] = morse_pot(distances[i][j])
#    return potentials  
#    
#def morse_pot(r, ca = 0.5, cr = 1, la = 3, lr = 0.5):
#    return cr*np.exp(-r/lr) - ca * np.exp(-r)
#
#def weird_paper(agents, N, dt):
#    distances = [get_distances(agent, agents) for agent in agents]
#    potencials = get_potencials(distances, N)
#    for i in range(N):
#            newSpeed = speed[i] + dt * potentials
#            speeds[i] = speeds[i] + weightedSpeed
#            speeds[i] = s * normalized(speeds[i]) ### PROBLEMA?-------------------------------------------------
#
#    return


# In[66]:

def couzin(agents, speeds, N, s, noise, rr=1, ro=2, ra=3):
    # watch only particles inrepuls
    for i in range(N): ### FIX - Eliminate, in some way the i-i interaction
        curr_agent = agents[i]
        distances = get_distances(curr_agent, agents)
        r_dir = np.array([0.0, 0.0])
        o_dir = np.array([0.0, 0.0])
        a_dir = np.array([0.0, 0.0])
        repulsion_flag = False    
        
        for j in range(N):
            if i == j:
                continue

            if distances[j] < rr:
                temp_vec = np.array([agents[j].getX() - curr_agent.getX(), agents[j].getY() - curr_agent.getY()])
                temp_vec = normalized(temp_vec)
                r_dir = r_dir + temp_vec
                repulsion_flag = True
            
            elif not repulsion_flag:

                if distances[j] < ro:
                    o_dir = o_dir + speeds[j]
                
                elif distances[j] < ra:
                    temp_vec = np.array([agents[j].getX() - curr_agent.getX(), agents[j].getY() - curr_agent.getY()])
                    temp_vec = normalized(temp_vec)
                    a_dir = a_dir + temp_vec
            
        #Out of for (j), we treat now the resulting direction vector
        if repulsion_flag:
            tot_dir =  normalized(-r_dir)
        else:
            o_dir = normalized(o_dir)
            a_dir = normalized(a_dir)
            tot_dir = o_dir + a_dir
            tot_dir = normalized(tot_dir)
            
        tot_dir = normalized(normalized(tot_dir) + noise * np.array([(-1)+2*random(),(-1)+2*random()]))
        
        #avoid pts stoping when not interacting
        if np.linalg.norm(tot_dir) != 0:
            speeds[i] = s*tot_dir 

def gbias(speeds, N, s, prop, bias, weight):
    #bias = np.array([0.0,1.0])
    #Ns has to be integer 
    Ns = int(N * prop)
    for i in range(Ns):
        tot_dir = normalized(normalized(speeds[i]) + weight * bias)
        
        if np.linalg.norm(tot_dir) != 0:
            speeds[i] = s*tot_dir

def accuracy(speeds, N, bias):
    dev = 0.0
    for i in range(N):
        dev = dev + np.arccos(np.dot(bias,normalized(speeds[i])))
    dev_avg = dev/(N*2*(np.pi))
    print (dev_avg)
    
def elongation(agents, N):
    cn = np.array([0.0, 0.0])
    for i in range(N):
        cn = agents
    
def vicsek(agents, speeds, N, s, noise, r): # s=speed, noise= letter csi temperature factor, r=radius of interaction
    # consider only particles within 'r' from pt_i, align pt_i with v_avg
    for i in range(N):
        distances = get_distances(agents[i], agents)
        tot_dir = np.array([0.0, 0.0])
        
        for j in range(N):
            if distances[j] < r:
                tot_dir = tot_dir + speeds[j]

        tot_dir = s * normalized(normalized(tot_dir) + noise * np.array([(-1)+2*random(),(-1)+2*random()]))
        if np.linalg.norm(tot_dir) != 0:
            speeds[i] = tot_dir

def gueron():
    return


# In[70]:

def simulate(N_steps, a, dt, N, width, height, s, noise, prop, weight, bias):
    """
    Simulates motion of swarm. Recieves following parameters:

    N_steps  - number of steps to perform
    a -  coupling between neighbouring points
    dt - time step to be used 
    N - number of points to be used
    width & heigth of window
    s - module of speed throughout agents

    """
    agents, speeds, window = initialize(s, N, width, height)
    for i in range(N_steps):
        next_step(agents, speeds, dt, N)
        import timeit

        start = timeit.default_timer()
        ## MODEL
        #couple_speeds(agents, speeds, a, s, N)
        #vicsek(agents, speeds, N, s, 0.1, r = 500)
        couzin(agents, speeds, N, s, noise,100,1900,2000)
        gbias(speeds, N, s, prop, bias, weight)
        accuracy(speeds, N, bias)
        
        ## BOUNDARY CONDITIONS
        #treat_boundary(width, height, agents, speeds, N)
        periodic_boundary(width, height, agents, speeds, N)
        #print(v_avg)
        #v_avg = get_v_avg(speeds)
        #time.sleep(0.02)
        stop = timeit.default_timer()
        #print stop - start
    window.close()
    

#simulate(N_steps, a, dt, N, width, height, s, noise, prop, weight, bias)
simulate(2000, 0.1, 1, 50, 500, 500, 2, 0.02, 0.4,0.5, np.array([0.0,-1.0]))


# In[ ]:




# In[ ]:




# In[ ]:



