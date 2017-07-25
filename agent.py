
from graphics import *
import time
import random
import math

def nearest_neighbour(a,aas):

    minDis = float('inf')
    nn = None

    for b in aas:
        if (a == b):
            True
        elif (nn == None):
            nn = b
        else:
            dis = distance(a[0].getX(), a[0].getY(),b[0].getX(), b[0].getY())
            if(dis < minDis):
                minDis = dis
                nn = b
    return b

# Distance function
def distance(xi,xii,yi,yii):
    sq1 = (xi-xii)*(xi-xii)
    sq2 = (yi-yii)*(yi-yii)
    return math.sqrt(sq1 + sq2)

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

# check for window boundaries
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

def main():
    winWidth = 1000
    winHeight = 700

    window = GraphWin("Window", winWidth, winHeight)

    maxTime = 4000
    maxV = 8
    agentNum = 50
    agents = [[0 for x in range(3)] for y in range(agentNum)]

    #generate point
    for agent in agents:
        agent[0] = Point(random.uniform(0,winWidth), random.uniform(0,winHeight))

        agent[1] = random.uniform(-2,2)
        agent[2] = random.uniform(-2,2)
        agent[0].draw(window)

    #update points
    for i in range(maxTime):
        for agent in agents:
            nn = nearest_neighbour(agent, agents)

            agent = updateV(agent, nn, maxV)
            agent = checkBoundary(agent, winWidth, winHeight)

        time.sleep(0.01)
        
    window.getMouse()
    window.close()

main()
