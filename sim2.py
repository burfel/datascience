
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
from graphics import *
from operator import *
from random import *
import numpy as np


# In[2]:

def initialize(randspeeds = True, N = 100, width = 1000, height = 1000):

    agents = [Point(width*random(), height*random()) for i in range(N)]

    if randspeeds:
        speeds = [[(random() - 0.5), (random() - 0.5)] for i in range(N)]
    else:
        speeds = N * [[0, 0]]

    return agents, speeds, plot(agents, width, height)


def couple_speeds(agents, speeds, a, N):
    nearest_neighbours = [nearest_neighbour(agent, agents, N) for agent in agents]
    for i in range(N):
            weightedSpeed = map(lambda x: a * x, speeds[nearest_neighbours[i]])
            #print(agents[i],'<-',weightedAgent)
            speeds[i] = map(add, speeds[i], weightedSpeed)
            speeds[i] = map(lambda x: 10 * x/np.sqrt(speeds[i][0]**2+speeds[i][1]**2), speeds[i]) 
    return

def get_distances(agent, agents):
    dists = [(a.getX() - agent.getX())**2 + (a.getY() - agent.getY())**2 for a in agents]
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
        [x, y] = [agents[i].getX(), agents[i].getY()]
        if x > x_bound or x < 0:
            speeds[i][0] = -speeds[i][0]
        if y > y_bound or y < 0:
            speeds[i][1] = -speeds[i][1]


def next_step(agents, speeds, dt, N):
    dxvec = [dt * speeds[i][0] for i in range(N)]
    dyvec = [dt * speeds[i][1] for i in range(N)]
    
    for i in range(N):
        agents[i].move(dxvec[i], dyvec[i])
        

def plot(agents, width, height):
    win = GraphWin("Swarm", width, height) # size of box
    for agent in agents:
        agent.draw(win)
    win.getMouse()
    return win
    #win.close()


# In[3]:

def couzin(agents, speeds, N, rr=1, ro=2, ra=3):
    # watch only particles inrepuls
    distances = [get_distances(agent, agents) for agent in agents]
    for i in range(N): ### FIX - Eliminate, in some way the i-i interaction
        r_agents = []
        o_agents = []
        a_agents = []
        repulsion_flag = False    
        for j in range(N):
            if i == j:
                continue
            if distances[i][j] < rr:
                temp_vec = [agents[j].getX()-agents[i].getX(), agents[j].getY()-agents[i].getY()]
                norm_vec = map(lambda x: x/distances[i][j], temp_vec)
                r_agents.append(norm_vec)
                repulsion_flag = True
            elif not repulsion_flag and distances[i][j] < ro:
                
                
                pass
        
        if repulsion_flag:
            rep_direction = np.sum(r_agents, axis=0)
            print(rep_direction)
            rep_direction = map(lambda x: x/np.sqrt(rep_direction[0]**2+rep_direction[1]**2), rep_direction)

            speeds[i] = map(add, speeds[i], rep_direction) # bias in the direction
            speeds[i] = map(lambda x: x*np.sqrt(speeds[i][0]**2+speeds[i][1]**2), rep_direction) # re-normalization 
        
        


def vicsek():
    return

def gueron():
    return


# In[4]:

def simulate(N_steps = 10, a = 0.1, dt = 0.01, N = 100, width = 1000, height = 1000, randspeeds = True):
    """
    Simulates motion of swarm. Recieves following parameters:

    N_steps  - number of steps to perform
    a -  coupling between neighbouring points
    dt - time step to be used 
    N - number of points to be used
    width & heigth of window
    randspeed - indicates wether we should use random initial speeds

    """

    agents, speeds, window = initialize(randspeeds, N)
    for i in range(N_steps):
        next_step(agents, speeds, dt, N)
        couzin(agents, speeds, N, 10,20,30)
        #couple_speeds(agents, speeds, a, N)
        treat_boundary(width, height, agents, speeds, N)
    window.close()

simulate(N_steps = 100, a =0.1, dt = 10)

