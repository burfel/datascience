
# coding: utf-8

# In[37]:

import matplotlib.pyplot as plt
from graphics import *
from operator import *
from random import *
import numpy as np
import time


# In[38]:

def initialize(speed = 1, N = 100, width = 1000, height = 1000):

    agents = [Point(width*random(), height*random()) for i in range(N)]
    speeds = N * [0]
    for i in range(N):
        theta = 2 * np.pi * random()
        speeds[i] = [speed * np.cos(theta), speed * np.sin(theta)]

    return agents, speeds, plot(agents, width, height)


def couple_speeds(agents, speeds, a, s, N):
    nearest_neighbours = [nearest_neighbour(agent, agents, N) for agent in agents]
    for i in range(N):
            weightedSpeed = map(lambda x: a * x, speeds[nearest_neighbours[i]])
            #print(agents[i],'<-',weightedAgent)
            speeds[i] = map(add, speeds[i], weightedSpeed)
            speeds[i] = map(lambda x: s * x, normalized(speeds[i]))
            print(speeds[i])

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
    [dx, dy] = [0, 0]
    for i in range(N):
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
        

def plot(agents, width, height):
    win = GraphWin("Swarm", width, height) # size of box
    for agent in agents:
        agent.draw(win)
    win.getMouse()
    return win
    #win.close()

def normalized(vector):
    if vector == [0, 0]:
        return vector
    return map(lambda x: x/np.sqrt(vector[0]**2 + vector[1]**2), vector)
    


# In[41]:

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
                    o_dir = map(add, o_dir, speeds[j])
                elif distances[i][j] < ra:
                    temp_vec = [agents[j].getX()-agents[i].getX(), agents[j].getY()-agents[i].getY()]
                    temp_vec = normalized(temp_vec)
                    a_dir = map(add, a_dir, temp_vec)
            
                    
        if repulsion_flag:
            tot_dir = map(lambda x: -x, normalized(r_dir))
        else:
            o_dir = normalized(o_dir)
            a_dir = normalized(a_dir)
            tot_dir = map(add, o_dir, a_dir)
            tot_dir = normalized(tot_dir)

        if tot_dir != [0,0]:
            speeds[i] = map(lambda x: x*np.sqrt(speeds[i][0]**2+speeds[i][1]**2), tot_dir) # re-normalization 


def vicsek(agents, speeds, N, s, noise, r): # s=speed, noise= letter csi temperature factor, r=radius of interaction
    # consider only particles within 'r' from pt_i, align pt_i with v_avg
    distances = [get_distances(agent, agents) for agent in agents]
    for i in range(N):
        angle_avg = angle_noise = angle_dir = 0
        v_sum = [0 , 0]
        v_dir = [0 , 0]
        for j in range(N):
            if distances[i][j] < r:
                v_sum = map(add, v_sum, speeds[j])
#       [cos_avg, sin_avg] = map(lambda x: x/(s*nr), v_sum)
        angle_avg = np.arctan( v_sum[1] / v_sum[0] )
        angle_noise = 2 * np.pi * (-noise/2 + noise * random())
        angle_dir = angle_avg + angle_noise
        speeds[i] = [s*np.cos(angle_dir), s*np.sin(angle_dir)]

    return

def gueron():
    return


# In[46]:

def simulate(N_steps = 10, a = 1, dt = 1, N = 200, width = 500, height = 500, s = 10):
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
        print(i)
        vicsek(agents, speeds, N, s, 0.1, 5000)
        #couzin(agents, speeds, N, 10,1900,2000)
        #couple_speeds(agents, speeds, a, s, N)
        #treat_boundary(width, height, agents, speeds, N)
        periodic_boundary(width, height, agents, speeds, N)
        #time.sleep(0.01)
        next_step(agents, speeds, dt, N)
    window.close()

simulate(N_steps = 2000, N = 100, dt = 1, width = 600, height = 600, s = 5)


# In[ ]:




# In[ ]:




# In[ ]:



