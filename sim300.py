
# coding: utf-8

# In[13]:

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from graphics import *
from operator import *
from random import *
import numpy as np
import timeit
from math import *

"""
a = 3
s = 3
r = 3
rr = 20
ro = 70
ra = 300
noise = 0.01
prop = 0.4
weight = 3
bias = np.array([0.0,-1.0])
dev_bias = 3
dTheta = 120
sight_theta = 180
"""


# In[10]:

def initialize_agents(speed, N, width, height):
    """
    Initializes our agent set with randomly directed speeds, draws the window and the agents
    """
    agents = [Point(width*random(), height*random()) for i in range(N)]
    speeds = [np.array([0.0, 0.0]) for i in range(N)]
    
    for i in range(N):
        theta = np.pi * random()
        speeds[i][0] = speed * np.cos(theta)
        speeds[i][1] = speed * np.sin(theta)

    return agents, speeds


def initialize_window(agents, width, height):
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
        speeds[i] = s * normalized(speeds[i])


def angle(vec1, vec2):  # (Not so) stupid way of doing an angle
    return np.arccos(np.dot(normalized(vec1),normalized(vec2)))
            
def get_distances(agent, agents, N):
    """
    Given one angent and the set of all agents, 
    computes the distances from the first to all of the others
    """
    dists = N * [0.0]
    for i in range(N):
        a, dists[i] = relative_pos(agent, agents[i])
        if dists[i] == 0:
            dists[i] = 0.1
    
    return dists

def relative_pos(agent1, agent2):
    dx = agent2.getX() - agent1.getX()
    dy = agent2.getY() - agent1.getY()

    return np.array([dx, dy]), np.linalg.norm([dx, dy])

def get_cm(agents, N):
    poses = [np.array([a.getX(), a.getY()]) for a in agents]
    return np.mean(poses,axis = 0)

def nearest_neighbour(agent, agents, N):
    """
    Returns the index for the agent with smallest Eucledian distance to agent in question
    """
    distances = get_distances(agent, agents, N)
    j = next(i for i in range(N) if agents[i] is agent)
    distances[j] = distances[j-1] + 1

    return distances.index(min(distances))


def softened_angle(speed, newspeed, s, maxTheta):
    theta = angle(speed, newspeed)
    if maxTheta**2 > theta**2:
        return s*newspeed
    else:
        return np.dot(rot_matrix(maxTheta), speed)    

def in_sight_range(rel_pos, speed1, angle_range):
    return 360*angle(speed1, rel_pos)/np.pi < angle_range
    

def noisy_vector(noise):
    return noise * np.array([2 * random() - 1, 2 * random() - 1])
    
def biaser(speeds, N, s, i, prop, bias, dev_bias, weight):
    #bias = np.array([0.0,1.0])
    #Ns has to be integer 
    Ns = int(N * prop)
    gbias = np.random.normal(bias, dev_bias, )
    for i in range(Ns):
        tot_dir = normalized(normalized(speeds[i]) + weight * gbias)
        
        if np.linalg.norm(tot_dir) != 0:
            speeds[i] = s*tot_dir
    #bias = np.dot(tot_dir,np.array([[np.cos(rot_bias*i), 0],[0, np.sin(rot_bias*i)]]))
    #return bias

def rigid_boundary(x_bound, y_bound, agents, speeds, N):
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


def rot_matrix(theta, unit = "None"):
    if unit in ["None", "deg"]:
        c, s = np.cos(pi * theta / 180), np.sin(pi * theta / 180)
    elif unit == "rad":
        c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],[s, c]])

def warn_me_args(N_steps, a, dt, N, width, height, s, rr, ro, ra, noise, prop, weight, bias, dTheta):
    if dt*s < rr:
        print("Warning - step length bigger the repultion radius.")


def virtualizer (current, agents, h, w, N):
    lower_limit = current.getY() - h / 2
    upper_limit = current.getY() + h / 2
    left_limit = current.getX() - w / 2
    right_limit = current.getX() + w / 2
    
    vagents=[np.array([agent.getX(),agent.getY()]) for agent in agents]
    virtuals = N * [0.0, 0.0]
    vvirtuals = N * [Point(0.0, 0.0)]

    for j in range(N):
        #make more compact!!!
        virtuals[j]=vagents[j]
        
        #newcopy = agents[i].clone()
        candidates = [vagents[j],
                      vagents[j]+[w,0],vagents[j]+[-w,0],vagents[j]+[w,h],vagents[j]+[w,-h],
                      vagents[j]+[-w,h],vagents[j]+[-w,-h],vagents[j]+[0,h],vagents[j]+[0,-h]]
        #print candidates
        
        #virtuals[j] = next((cand for cand in candidates if lower_limit < cand[1] < upper_limit and left_limit < cand[0] < right_limit),False
        for i in range(9):
            if lower_limit < candidates[i][1] < upper_limit and left_limit < candidates[i][0] < right_limit:
                virtuals[j]=candidates[i]
            
        ##ATTENTION HERE!!
#        if not virtuals[j]:
 #           virtuals[j] = agents[j]
        vvirtuals[j] = Point(virtuals[j][0],virtuals[j][1])
        
    return vvirtuals

        
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


# In[ ]:




# In[ ]:




# In[15]:

#COUZIN MODEL IMPLEMENTED WITH REPULSE, ATRACT AND ORIENT ZONES SEPARATED (1ST PAPER)
def couzin(agents, speeds, N, width, height, s, noise, dTheta, rr, ro, ra, sight_range, model2, roa, atract, orient, pbc):
    
    if not model2:
        atract = orient = 1
    # watch only particles in repulsion zone
    for i in range(N):
        r_dir = np.array([0.0, 0.0])
        o_dir = np.array([0.0, 0.0])
        a_dir = np.array([0.0, 0.0])
        repulsion_flag = False    
        
        if pbc:
            virtuals = virtualizer(agents[i], agents, height, width, N)
        else:
            virtuals = agents
        
        for j in range(N):     
            if i == j:
                #Eliminate the i-i interaction
                continue

            rel_pos, distance = relative_pos(agents[i], virtuals[j])
            
            if in_sight_range(rel_pos, speeds[i], sight_range):
                if distance < rr:
                    rel_pos = normalized(rel_pos)
                    r_dir = r_dir + rel_pos
                    repulsion_flag = True

                elif not repulsion_flag:
                    
                    if model2: # Couzin 2
                        if distance < roa:
                            o_dir = o_dir + speeds[j]
                            rel_pos = normalized(rel_pos)
                            a_dir = a_dir + rel_pos

                    else: # Couzin 1
                        if distance < ro:
                            o_dir = o_dir + speeds[j]

                        elif distance < ra:
                            rel_pos = normalized(rel_pos)
                            a_dir = a_dir + rel_pos

        #Out of for (j), we treat now the resulting direction vector
        if repulsion_flag:
            tot_dir =  normalized(- r_dir)
        else:
            tot_dir = orient * normalized(o_dir) + atract * normalized(a_dir)
            tot_dir = normalized(tot_dir)
            
        tot_dir = normalized(normalized(tot_dir) + noisy_vector(noise))
        
        #avoid pts stoping when not interacting
        if np.linalg.norm(tot_dir) != 0:
            speeds[i] = softened_angle(speeds[i], tot_dir, s, dTheta)
    return

    
def vicsek(agents, speeds, N, s, noise, r): # s=speed, noise= letter csi temperature factor, r=radius of interaction
    # consider only particles within 'r' from pt_i, align pt_i with v_avg
    for i in range(N):
        tot_dir = np.array([0.0, 0.0])
        
        for j in range(N):
            rel_pos, distance = relative_pos(agents[i], agents[j])
            if distance < r:
                tot_dir = tot_dir + speeds[j]

        tot_dir = s * normalized(normalized(tot_dir) + noise * np.array([(-1)+2*random(),(-1)+2*random()]))
        
        if np.linalg.norm(tot_dir) != 0:
            speeds[i] = tot_dir
    return


# MILL MODEL
def mill (agents, speeds, dt, N, width, height, cr, ca, lr, la, alpha, beta):
    clr = cr / lr
    cla = ca / la
    for i in range(N):
        u_dir = np.array([0.0, 0.0])
        propulsion = np.array([0.0, 0.0])
        friction = np.array([0.0, 0.0])
        grad_U = np.array([0.0, 0.0])
        
        for j in range(N): # Duality interactions, by the Morse potential  
            if i == j:
                #Eliminate the i-i interaction
                continue

            rel_pos, distance = relative_pos(agents[i], agents[j])
            u_dir = normalized(rel_pos)
            grad_U = grad_U + u_dir * (clr*np.exp(-distance / lr) - cla *np.exp(- distance / la))
            
        norm = np.linalg.norm(speeds[i])
        
        propulsion = alpha * speeds[i] # self-propulsion propto alpha
        friction =  beta * ((np.linalg.norm(speeds[i])) ** 2) * speeds[i] #friction force propto beta
        #friction =  beta * speeds[i] #friction force propto beta

        d_speed = propulsion - friction - grad_U
        speeds[i] = speeds[i] + dt * d_speed

    return


# In[16]:

#run(300, 5, 10, 1, 500, 500)
# Field of view is around 340º for pigeons as saind in the source:
# https://www.quora.com/What-does-a-pigeons-field-of-view-look-like


# In[ ]:



