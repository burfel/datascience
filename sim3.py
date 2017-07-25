
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
from graphics import *
from operator import *
from random import *
import numpy as np


# In[13]:

def initialize(speed = 1, N = 100, width = 1000, height = 1000):

    agents = [Point(width*random(), height*random()) for i in range(N)]
    speeds = N * [0]
    for i in range(N):
        theta = 2 * np.pi * random()
        speeds[i] = [speed * np.cos(theta), speed * np.sin(theta)]

    return agents, speeds, plot(agents, width, height)


def couple_speeds(agents, speeds, a, N):
    nearest_neighbours = [nearest_neighbour(agent, agents, N) for agent in agents]
    for i in range(N):
            weightedSpeed = map(lambda x: a * x, speeds[nearest_neighbours[i]])
            #print(agents[i],'<-',weightedAgent)
            speeds[i] = map(add, speeds[i], weightedSpeed)
            speeds[i] = normalized(speeds[i])
    return

def get_distances(agent, agents):
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

def normalized(vector):
    return map(lambda x: x/np.sqrt(vector[0]**2 + vector[1]**2), vector)
    


# In[14]:

def couzin(agents, speeds, N, rr=1, ro=2, ra=3):
    # watch only particles inrepuls
    distances = [get_distances(agent, agents) for agent in agents]
    for i in range(N): ### FIX - Eliminate, in some way the i-i interaction
        r_dir = [0, 0]
        o_dir = [0, 0]
        a_dir = [0, 0]
        repulsion_flag = False    
        
        for j in range(N):
            if i == j:
                continue

            if distances[i][j] < rr:
                temp_vec = [agents[j].getX() - agents[i].getX(), agents[j].getY()-agents[i].getY()]
                temp_vec = normalized(temp_vec)
                r_dir = map(add, r_dir, temp_vec)
                repulsion_flag = True
            
            elif not repulsion_flag:

                if distances[i][j] < ro:
                    o_dir = map(add, o_dir, speed[j])
                elif distances[i][j] < ra:
                    temp_vec = [agents[i].getX()-agents[j].getX(), agents[i].getY()-agents[j].getY()]
                    temp_vec = normalized(temp_vec)
                    a_dir = map(add, a_dir, temp_vec)
            
                    
        if repulsion_flag:
            tot_dir = - normalized(r_dir)
        
        else:
            o_dir = normalized(o_dir)
            a_dir = normalized(a_dir)
            tot_dir = map(add, o_dir, a_dir)
            tot_dir = normalized(tot_dir)
        
        speeds[i] = map(lambda x: x*np.sqrt(speeds[i][0]**2+speeds[i][1]**2), tot_dir) # re-normalization 
        


def vicsek():
    return

def gueron():
    return


# In[16]:

def simulate(N_steps = 10, a = 0.1, dt = 0.01, N = 100, width = 1000, height = 1000, s = 10):
    """
    Simulates motion of swarm. Recieves following parameters:

    N_steps  - number of steps to perform
    a -  coupling between neighbouring points
    dt - time step to be used 
    N - number of points to be used
    width & heigth of window
    s - module of speed throughout agents

    """

    agents, speeds, window = initialize(s, N)
    for f in speeds:
        print(f)
    for i in range(N_steps):
        print('pixota1 ' + str(i))
        next_step(agents, speeds, dt, N)
        print('pixota2 ' + str(i))
        #couzin(agents, speeds, N, 10,20,30)
        couple_speeds(agents, speeds, a, N)
        print('pixota3 ' + str(i))
        treat_boundary(width, height, agents, speeds, N)
        print('pixota4 ' + str(i))
    window.close()

simulate(N_steps = 100)


# In[5]:




# In[ ]:



