import gym
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import random

class DQN(nn.Module):
	def __init__(self, state_dim, hidden_dim, action_dim):
		super(DQN , self).__init__()
		self.model = torch.nn.Sequential(
					torch.nn.Linear(state_dim, hidden_dim),
					torch.nn.LeakyReLU(),
					torch.nn.Linear(hidden_dim, hidden_dim*2),
					torch.nn.LeakyReLU(),
					torch.nn.Linear(hidden_dim*2, action_dim)
					)
	def forward(self, state):
		return self.model(state)

#Initializing models and hyper parameters
env = gym.envs.make("CartPole-v1")
env.reset()

#Initializing policy parameters
numStates = 4
numActions = 2
hiddenDim = 50
lr = 0.0005
model = DQN(numStates, hiddenDim, numActions)
optim = torch.optim.Adam(list(model.parameters()), lr)

#Initialzing parameters
criterion = nn.L1Loss()
numEpisodes = 200
gamma=0.9 
epsilon=0.3
eps_decay=0.99

def getState(state):
	return torch.tensor(state)

def performAction(state):
	if random.random() < epsilon:
		action = env.action_space.sample()
	else:
		q_values = model.forward(state)
		action = torch.argmax(q_values).item()
	return action

def optimize():
	
	for episode in range(numEpisodes):
		done = False
		allStates = []
		totalReward = 0
		# Get initial state
		state = getState(env.reset()).unsqueeze(0).float()

		while (done != True):
			action = performAction(state) #Perform action according to epsilon greedy strategy
			qValues = model.forward(state) #Get Q values
			nextState, reward, done, _ = env.step(action) #Perform action in env and get next state, reward, status
			
			if not done:   # If terminal state is not acheived
				qValuesNext = model(getState(nextState).unsqueeze(0).float()) #Get Q values for future state
			else:
				qValuesNext = torch.tensor(0) #For future state of terminal state, the Q value is zero

			qValuesTarget = torch.tensor(reward) + gamma * torch.max(qValuesNext) #Get expected Q value according to bellman equation Q(s, a) <- R + gamma*Q(s`, a`) 
			loss = criterion(qValues[0][action], qValuesTarget)
			loss.backward()
			optim.step()

			rgbState = env.render("rgb_array")
			allStates.append(rgbState)
			totalReward+= reward

			state = getState(nextState).unsqueeze(0).float()

		global epsilon
		epsilon = max(epsilon * eps_decay, 0.01)  #Update epsilon
		out = cv2.VideoWriter("cartpole-Q/"+str(episode)+'.avi',0,1, (400, 600))
		print("Total reward for this episode", totalReward)

		for i in range(0, len(allStates)):
			out.write(allStates[i])
			out.release()

# Training
optimize()

#Saving the model 



