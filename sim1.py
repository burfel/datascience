import matplotlib.pyplot as plt
from operator import *
import random


def nearest_neighbour(agent, agents):
	"""
	Returns the agent that has smallest Eucledian distance to agent in question
	"""
	distances = [(a[0] - agent[0])**2 + (a[1] - agent[1])**2 for a in agents]
	i = next(i for i in range(len(agents)) if agents[i] == agent)
	distances[i] = distances[-i] + 1

	return distances.index(min(distances))

	##^^^^^^^FIX^^^^^^^^^##

def initialize(rand = True):
	if rand:
		return [[i,j,2*random.random()-1,2*random.random()-1] for i in range(10) for j in range(10)]
	else:
		return [[i,j, 0, 0] for i in range(10) for j in range(10)]


def simulate(steps = 10, a = 1, dt = 0.01, randomspeeds = True):
	agents = initialize(randomspeeds)
	
	for i in range(steps):
		for agent in agents:
			step = map(lambda x: dt * x, agent[2:4])
			#print(agent)
			agent[0:2] = map(add, agent[0:2], step)

		nearest_neighbours = [agents[nearest_neighbour(agent, agents)][:] for agent in agents]
		#print(nearest_neighbours)

		for i in range(len(agents)):
			weightedAgent = map(lambda x: a * x, nearest_neighbours[i])
			#print(agents[i],'<-',weightedAgent)
			agents[i][2:4] = map(add, agents[i][2:4], weightedAgent[2:4])
			treat_boundary(150, 150, agents[i])
			#print(agent)

		plotPoints(agents)


def treat_boundary(x_bound, y_bound, agent):
	[x, y] = agent[0:2]
	if x > x_bound or x < 0:
		agent[2] = -agent[2]

	if y > y_bound or y < 0:
		agent[3] = -agent[3]


def plotPoints(agents):
	plt.plot([x[0] for x in agents], [x[1] for x in agents], 'ro')
	plt.show()
