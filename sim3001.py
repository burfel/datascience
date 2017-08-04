
# coding: utf-8

# In[11]:

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from processData import *
from graphics import *
from operator import *
from random import *
import numpy as np
import timeit


# In[12]:

# initialises our agent set with randomly directed speeds
def initialize_agents(speed, N, width, height, radius):
    """
    Initializes our agent set with randomly directed speeds, draws the window and the agents
    """
    seed()

    agents = N * [0]
    #agents = [Point(uniform(0, width), uniform(0, height)) for i in range(N)]
    speeds = N * [[0,0]] 
    
    for i in range(N):
        theta = uniform(0, 2 * np.pi)
        speeds[i]= speed * np.array([np.cos(theta), np.sin(theta)])
        
        theta = uniform(0, 2 * np.pi)
        r = uniform(0, radius)
        agents[i] = Point(width / 2 + r * np.cos(theta),
                          height / 2 + r * np.sin(theta))

    return agents, speeds

def initialize_leaders(agents, prop, N_groups, N):
    Ns = int(N * prop)
    colors = ['red', 'blue', 'yellow']
    leader_groups = []
    for j in range(N_groups):
        leader_groups.append([])
        for i in range(Ns):
            agent_index = i + j * Ns
            leader_groups[j].append(agent_index)
            agents[agent_index].setFill(colors[j])
    return leader_groups

# draws the window and the agents
def initialize_window(agents, width, height):
    #win = GraphWin("Swarm", width, height) # size of box
    win = GraphWin("My Swarm", width, height, autoflush=False)
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
    proposal = normalized(newspeed + normalized(speed))
    theta = angle(speed, proposal)
    if maxTheta > theta: # changed to non-square ---> TEST!
        return s * proposal
    else:
        return np.dot(rot_matrix(maxTheta), speed)    

def in_sight_range(rel_pos, speed1, angle_range):
    return 360 * angle(speed1, rel_pos) / np.pi < angle_range
    

def noisy_vector(noise):
    return noise * np.array([2 * random() - 1, 2 * random() - 1])
    
def biaser(agents, leaders, speeds, N, s, prop, bias, dev_bias, weight):
    #bias = np.array([0.0,1.0])
    #Ns has to be integer

    gbias = np.random.normal(bias, dev_bias, )
    for i in leaders:
        tot_dir = normalized(normalized(speeds[i]) + weight * gbias)
        if np.linalg.norm(tot_dir) != 0:
            speeds[i] = s * tot_dir
            
    #bias = np.dot(tot_dir,np.array([[np.cos(rot_bias*i), 0],[0, np.sin(rot_bias*i)]]))
    return

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
        
        
def periodic_boundary(x_bound, y_bound, agents, speeds, N):  #Changed from rigid boundaries to periodic boundary condition
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


def virtualizer(agents, h, w, N):
    vagents = [np.array([agent.getX(),agent.getY()])
               for agent in agents]
    virtuals = N * [N * [0.0, 0.0]]
    virtual_points = N * [N * [0]]
    
    for i in range(N):
        d_limit = agents[i].getY() - h / 2
        u_limit = agents[i].getY() + h / 2
        l_limit = agents[i].getX() - w / 2
        r_limit = agents[i].getX() + w / 2
        for j in range(N):
            virtuals[i][j] = vagents[j]
            candidates = [vagents[j],
                          vagents[j] + [w, 0], vagents[j] + [-w, 0],
                          vagents[j] + [w, h], vagents[j] + [w, -h],
                          vagents[j] + [-w, h], vagents[j] + [-w, -h],
                          vagents[j] + [0, h], vagents[j] + [0, -h]]

            #virtuals[j] = next((cand for cand in candidates if lower_limit < cand[1] < upper_limit and left_limit < cand[0] < right_limit),False
            for candidate in candidates:
                if d_limit < candidate[1] < u_limit and l_limit < candidate[0] < r_limit:
                    virtuals[i][j] = candidate

            virtual_points[i][j] = Point(virtuals[i][j][0],virtuals[i][j][1])
        
    return virtual_points


def get_cm_std(agents, N):
    poses = [np.array([a.getX(), a.getY()]) for a in agents]
    return np.mean(poses,axis = 0), np.std(poses,axis = 0)

def mill_observables (N, agents, speeds):
    cm, std = get_cm_std(agents, N)
    mean_R = np.linalg.norm(std)
    point_cm = Point(cm[0],cm[1])
    norm_R = N*[0.0]
    vector_R = N*[0,0]
    angles = N*[0.0]
    for i in range(N):
        vector_R[i],norm_R[i] = relative_pos(point_cm, agents[i])
        angles[i] = np.pi() - angle(vector_R[i],speeds[i])
    min_R = min(norm_R)
    max_R = max(norm_R)
    
    return mean_R, min_R, max_R, angles


# In[13]:

#COUZIN MODEL IMPLEMENTED WITH REPULSION, ATTRACTION AND ORIENTATION ZONE SEPARATED (1ST PAPER)
def couzin(agents, other_agents, speeds, N, width, height, s, noise, dTheta, rr, ro, ra, sight_range, model2, roa, atract, orient):
    
    if not model2:
        atract = orient = 1
    # watch only particles in repulsion zone
    for i in range(N):
        r_dir = np.array([0.0, 0.0])
        o_dir = np.array([0.0, 0.0])
        a_dir = np.array([0.0, 0.0])
        repulsion_flag = False    

        for j in range(N):     
            if i == j:
                #Eliminate the i-i interaction
                continue

            rel_pos, distance = relative_pos(agents[i], other_agents[i][j])
            
            if in_sight_range(rel_pos, speeds[i], sight_range):
                if distance < rr:
                    rel_pos = normalized(rel_pos)
                    r_dir = r_dir + rel_pos
                    repulsion_flag = True

                elif not repulsion_flag:
                    
                    # Couzin 2
                    if model2:
                        if distance < roa:
                            o_dir = o_dir + speeds[j]
                            rel_pos = normalized(rel_pos)
                            a_dir = a_dir + rel_pos
                    # Couzin 1
                    else:
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

    
def vicsek(agents, other_agents, speeds, N, s, noise, r): # s=speed, noise= letter csi temperature factor, r=radius of interaction
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


##MILL MODEL
def mill(agents, other_agents, speeds, dt, N, width, height, cr, ca, lr, la, alpha, beta, mass):
    
    clr = cr / lr
    cla = ca / la
    
    for i in range(N):
        u_dir = np.array([0.0, 0.0])
        propulsion = np.array([0.0, 0.0])
        friction = np.array([0.0, 0.0])
        grad_U = np.array([0.0, 0.0])
        
        #virtuals = virtualizer(agents[i], agents, height, width, N)
        
        for j in range(N):   #Duality interactions, by the Morse potential  
            if i == j:
                #Eliminate the i-i interaction
                continue

            rel_pos, distance = relative_pos(agents[i], other_agents[i][j])
            u_dir = normalized(rel_pos)
            grad_U = grad_U + u_dir * (clr*np.exp(- distance / lr) - cla * np.exp(- distance / la))
            
        
        propulsion = alpha * speeds[i] # self-propulsion propto alpha
        norm = (np.linalg.norm(speeds[i]))
        friction =  beta * (norm ** 2) * speeds[i] #friction force prop to beta
        
        d_speed = (propulsion - friction - grad_U) / mass
        speeds[i]= speeds[i] + dt * d_speed
    return

#mill (dt, agents, speeds, N, width, height, cr=100, ca=150, lr=80, la=200.0, alpha=1.0, beta=0.01, mass=1) # we're there! N=30, s=5, dt=0.1, Radius=height/4  
        #mill (dt, agents, speeds, N, width, height, cr=100, ca=150, lr=80, la=200.0, alpha=1.2, beta=0.01, mass=3) # we're there! N=30, s=5, dt=0.1, Radius=height/4  
        #mill (dt, agents, speeds, N, width, height, cr=100, ca=150, lr=80, la=200.0, alpha=1.0, beta=0.01, mass=1) # we're there aswell! N=20, s=5, dt=0.1, Radius=height/4  
        #mill (dt, agents, speeds, N, width, height, cr=50, ca=100, lr=50, la=100.0, alpha=1.0, beta=0.01, mass=1) # The winner at 12:28 (1/8/2017)! N=60, s=5, dt=0.1, Radius=height/2  
        #mill (dt, agents, speeds, N, width, height, cr=50, ca=100, lr=50, la=100.0, alpha=1.0, beta=0.5, mass=1)
        
        ##OBSERVABLES FOR MILL MODEL
        #mean_R, min_R, max_R, angles = mill_observables(N, agents, speeds)
        #if (i > 500):
        #    print mean_R, min_R, max_R
        #    print angles[0],angles[N/2], angles[N-1]
        
        # WE WANT TO PLOT:
        #Rmin, Rmax, Rmean OVER THE TIME OF THE SIMULATION, SEE IT CONVERGES
        #angles[i] AT ONE STABLE MOMENT OF THE SIMULATION (END) AS A DISTRIBUTION OF ANGLES, SEE TWO PEAKS AROUND 90º
        
        


# In[ ]:



