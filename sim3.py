
# coding: utf-8

# In[112]:

import matplotlib.pyplot as plt
from graphics import *
from operator import *
from random import *
import numpy as np
import time


# In[113]:

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

    return agents, speeds, plot(agents, width, height)


def plot(agents, width, height):
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
        
def get_v_avg(speeds, N):
    return

def next_step(agents, speeds, dt, N):
    dxvec = [dt * speeds[i][0] for i in range(N)]
    dyvec = [dt * speeds[i][1] for i in range(N)]
    for i in range(N):
        agents[i].move(dxvec[i], dyvec[i])


def normalized(vector):
    if vector[0] == vector[1] == 0:
        return vector
    return vector / np.linalg.norm(vector)


# In[114]:

def couzin(agents, speeds, N, s, rr=1, ro=2, ra=3):
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
            
                    
        if repulsion_flag:
            tot_dir =  normalized(-r_dir)
        else:
            o_dir = normalized(o_dir)
            a_dir = normalized(a_dir)
            tot_dir = o_dir + a_dir
            tot_dir = normalized(tot_dir)

        if np.linalg.norm(tot_dir) != 0:
            speeds[i] = s*tot_dir 


def vicsek(agents, speeds, N, s, noise, r): # s=speed, noise= letter csi temperature factor, r=radius of interaction
    # consider only particles within 'r' from pt_i, align pt_i with v_avg
    for i in range(N):
        distances = get_distances(agents[i], agents)
        o_dir = np.array([0.0, 0.0])
        
        for j in range(N):
            if distances[j] < r:
                o_dir = o_dir + speeds[j]

        o_dir = s * normalized(normalized(o_dir) + noise * np.array([(-1)+2*random(),(-1)+2*random()]))
        if np.linalg.norm(o_dir) != 0:
            speeds[i] = o_dir

def gueron():
    return


# In[119]:

def simulate(N_steps, a, dt, N, width, height, s):
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
        
        ## MODEL
        vicsek(agents, speeds, N, s, noise = 0.9, r = 500)
        #couzin(agents, speeds, N, s,10,1900,2000)
        #couple_speeds(agents, speeds, a, s, N
        
        ## BOUNDARY CONDITIONS
        #treat_boundary(width, height, agents, speeds, N)
        periodic_boundary(width, height, agents, speeds, N)
        #print(v_avg)
        #v_avg = get_v_avg(speeds)
        #time.sleep(0.02)
    window.close()

simulate(2000, 0.1, 1, 100, 500, 500, 2)


# In[ ]:




# In[ ]:



