from graphics import *
import time
import random
import math
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------------------
# help functions

# Distance function betwen points xi, xii and yj,yii
def distance(xi,xii,yi,yii):
    sq1 = (xi-xii)*(xi-xii)
    sq2 = (yi-yii)*(yi-yii)
    return math.sqrt(sq1 + sq2)

#abs of a vector
def absvec(a, b):
    m = math.sqrt(a*a + b*b)
    if m == 0: m = 0.001
    return m

# angle between vectors x= (x1,y1) and y= (x2,y2), in degrees
def calc_angle(x1, y1, x2, y2):
        skalar = x1*x2 + y1*y2
        abs1 = absvec(x1, y1)
        abs2 = absvec(x2, y2)

        erg = skalar/(abs1* abs2)
        if erg > 1:
            #print erg
            erg=1

        elif erg < -1:
            #print erg
            erg=-1
        return math.degrees(math.acos(erg))

# -------------------------------------------------------------------
# simplest simulation
def nearest_neighbour(a,aas):

    minDis = float('inf')
    nn = None

    for b in aas:
        if (a == b):
            True
        elif (nn == None):
            nn = b
        else:
            dis = distance(a[0].getX(), b[0].getX(), a[0].getY(), b[0].getY())
            if(dis < minDis):
                minDis = dis
                nn = b
    return b

# updateVelociy
def updateV(agent, nn, maxV):

    vx = agent[1] + 0.1*nn[1] + random.uniform(-3, 3)
    vy = agent[2] + 0.1*nn[2] + random.uniform(-3, 3)

    if(abs(vx) < maxV) :
        agent[1] = vx
    elif (vx <= -maxV):
        agent[1] = -maxV
    else :
        agent[1] = maxV

    if(abs(vy) < maxV ):
        agent[2] = vy
    elif (vy <= -maxV):
        agent[2] = -maxV
    else :
        agent[2] = maxV
    return agent

# -------------------------------------------------------------------
# couzin


# returns three lists, one for each zone,
# contaning all other agent in the zone.
# ignores al egents ind the angle behind the current agent defined by blind.
def neigbour_in_zones(a, aas, zor_r, zoo_r, zoa_r, blind):
    zor = []
    zoo = []
    zoa = []


    for agent in aas:
        disVecX = agent[0].getX() - a[0].getX()
        disVecY = agent[0].getY() - a[0].getY()
        alpha = calc_angle(a[1],a[2], disVecX, disVecY) 
        
        if (a == agent):
            True
        elif alpha < 180 - blind and alpha > 180 + blind:
            True
        else:
            dis = absvec(agent[0].getX() - a[0].getX() , agent[0].getY() - a[0].getY() )
            if dis <= zor_r:
                zor.append(agent)
            elif dis <= zoo_r:
                zoo.append(agent)
            elif dis <= zoa_r:
                zoa.append(agent)

    #print len(zoo)+len(zor)+len(zoa)
    return [zor, zoo, zoa]

#update Velocity a la couzin
def updateV_couzin(a, matrix, maxV):
    dx=0
    dy=0

    #zor
    if matrix[0] != []:
        for agent in matrix[0]:
            disX = agent[0].getX() - a[0].getX()
            disY = agent[0].getY() - a[0].getY()

            rX = disX/absvec(disX, disY)
            rY = disY/absvec(disX, disY)

            dx += rX / absvec(rX, rY)
            dy += rY / absvec(rX, rY)

        dx = -dx
        dy = -dy

    # zoo ; zoa leer
    elif matrix[1] != []  and matrix[2] == []:
        for agent in matrix[1]:
            dx += agent[1] / absvec(agent[1], agent[2])
            dy += agent[2] / absvec(agent[1], agent[2])
        dx += a[1] / absvec(a[1], a[2])
        dy += a[2] / absvec(a[1], a[2])

    # zoo leer ; zoa
    elif matrix[1] == []  and matrix[2] != []:
        for agent in matrix[2]:
            disX = agent[0].getX() - a[0].getX()
            disY = agent[0].getY() - a[0].getY()

            rX = disX/absvec(disX, disY)
            rY = disY/absvec(disX, disY)

            dx += rX / absvec(rX, rY)
            dy += rY / absvec(rX, rY)

    # zoo ; zoa
    elif matrix[1] != []  and matrix[2] != []:
        for agent in matrix[1]:
            dx += agent[1] / absvec(agent[1], agent[2])
            dy += agent[2] / absvec(agent[1], agent[2])
        dx += a[1] / absvec(a[1], a[2])
        dy += a[2] / absvec(a[1], a[2])

        for agent in matrix[2]:
            disX = agent[0].getX() - a[0].getX()
            disY = agent[0].getY() - a[0].getY()

            rX = disX/absvec(disX, disY)
            rY = disY/absvec(disX, disY)

            dx += rX / absvec(rX, rY)
            dy += rY / absvec(rX, rY)

        dx = 0.5*dx
        dy = 0.5*dy

	# all zones empty
    else:
        dx = a[1]
        dy = a[2]

	# randomness factor / error
    dx += random.uniform(-1, 1)
    dy += random.uniform(-1, 1)

    return [dx, dy]

# check for window boundaries and move agents
def checkBoundary(agent, winWidth, winHeight):
    point = agent[0]
    point.move(agent[1],agent[2])

    x = point.getX()
    y = point.getY()

    if x > 0 and y < winHeight and x < winWidth and y > 0:
        agent[0] = point

    elif x <= 0 or x >= winWidth:
        agent[1] = agent[1] * (-1)
        agent[0].move(agent[1],agent[2])
    
    elif y <= 0 or y >= winHeight:
        agent[2] = agent[2] * (-1)
        agent[0].move(agent[1],agent[2])
    return agent


def avgdistancetoall(agent,agents):
    i= 0
    totaldistance = 0
    for i in range (len (agents)):
        if agents[i] == agent:
            True
        else:
            totaldistance += distance (agent[0].getX(),agent[0].getY(),agents[i][0].getX(),agents[i][0].getY())
    avgdistance = totaldistance/i
    return avgdistance
            
    
def totalavgdistance(agents,numplist):
    for agent in agents:
        numplist = np.append(numplist,avgdistancetoall(agent,agents))
    return numplist
    

def main():
    winWidth = 500
    winHeight = 500
    window = GraphWin("Window", winWidth, winHeight)
    maxTime = 300
    distancedata = np.array([])
    distancestd = np.array([])
    numpycount = np.array([])
	# radii of zones
    zor_r = 15
    zoo_r = 50
    zoa_r = 200

    maxV = 8 			# maxVelocity
    speed = 8			# constant speed

    blind = 30			# angle of blindness

    maxTurn = 50
    radTurn = math.radians(maxTurn)
    negRadTurn = math.radians(-maxTurn)

    agentNum = 50

    #generate agent
    # 0 Point
    # 1 XVelocity
    # 2 YVelocity
    # 3 Line
    # 4 temp. VelocityPoint
    agents = [[0 for x in range(5)] for y in range(agentNum)]
    for agent in agents:
        agent[0] = Point(random.uniform(0,winWidth), random.uniform(0,winHeight))

        agent[1] = random.uniform(-2,2)
        agent[2] = random.uniform(-2,2)

        agent[0].draw(window)
        agent[3] = Line(agent[0], Point(agent[0].getX() + agent[1], agent[0].getY() + agent[2]))
        agent[3].setArrow("last")
        agent[3].draw(window)

    """for testing
	agentA = [Point(200, 200) , 0, 0,None,[0,0]]
    agentB = [Point(205, 200) , 0, 0,None,[0,0]]
    agentC = [Point(210, 200) , 0, 0,None,[0,0]]
    agentA[0].draw(window)
    agentB[0].draw(window)
    agentC[0].draw(window)
    agents = [agentA, agentB, agentC]"""

    #main loop
    for i in range(maxTime):
        rawdata = totalavgdistance(agents,distancedata)
        #print ("rawdata: "+ str(rawdata))
        distancedata = np.append(distancedata,np.mean(rawdata))
        distancestd = np.append(distancestd,np.std(rawdata))
        numpycount = np.append(numpycount,i)
        # Velocity update
        for agent  in agents:
            neigh_matrix = neigbour_in_zones(agent, agents, zor_r, zoo_r, zoa_r, blind)
            agent[4] = updateV_couzin(agent, neigh_matrix, maxV)

            #print str(i) + " zor: " + str(len(neigh_matrix[0]))
            #print str(i) + " zoo: " + str(len(neigh_matrix[1]))
            #print str(i) + " zoa: " + str(len(neigh_matrix[2]))

        # move, draw
        for agent in agents:

            alpha = calc_angle(agent[1], agent[2],agent[4][0],agent[4][1])
			# test if in ragne of maxturn, if not rotate angle by maxTurn in
			# direction of new direction
            if alpha < maxTurn or alpha > 360-maxTurn:
                agent[1] = agent[4][0]
                agent[2] = agent[4][1]
            elif alpha < 180:
                agent[1] =  agent[1] * math.cos(radTurn) - agent[2]  * math.sin(radTurn)
                agent[2] =  agent[1] * math.sin(radTurn) + agent[2]  * math.cos(radTurn)
            else:
                agent[1] =  agent[1] * math.cos(negRadTurn) - agent[2]  * math.sin(negRadTurn)
                agent[2] =  agent[1] * math.sin(negRadTurn) + agent[2]  * math.cos(negRadTurn)

			# normalise diection vector to 1, and multiply by constant speed
	    
            agent[1] = 1/absvec(agent[1], agent[2]) * agent[1] * speed
            agent[2] = 1/absvec(agent[1], agent[2]) * agent[2] * speed

            agent = checkBoundary(agent, winWidth, winHeight)

			# draw arrow
            agent[3].undraw()
            agent[3] = Line(agent[0], Point(agent[0].getX() + agent[1], agent[0].getY() + agent[2]))
            agent[3].setArrow("last")
            agent[3].draw(window)
        time.sleep(0.01)
    plt.errorbar(numpycount,distancedata,yerr=distancestd)
    plt.show()
    window.getMouse()
    window.close()

main()
