import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import pygame
import imageio

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, 2)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc1(x.view(-1)))
        x = F.relu(self.fc2(x))
        return x


resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


def get_image(screen):
	pil_string_image = pygame.image.tostring(screen, "RGBA",False)
	pil_image = Image.frombytes("RGBA",(400,300),pil_string_image)
	pil_image = remove_transparency(pil_image)
	np_image = np.array(pil_image)
	return np_image

def get_next_state(curr_screen, action, x_coord, block_break, x1, y1, angle):
	screen = pygame.display.set_mode((400,300))
	screen = curr_screen
	image = np.zeros((300,400,4))
	reward = 0.0
	for i in range(0, 4):
		if action == 0 : x_coord -= 30
		if action == 1 : x_coord += 30
		if(x_coord<=0):
			x_coord = 0
		if(x_coord>=370):
			x_coord = 370
		life_end = False
		screen.fill((0, 0, 0))
		reward = 0.0
		for bl in range(0, len(block_break)):
			pygame.draw.rect(screen, (255,255,255), block_break[bl])
		block = pygame.draw.rect(screen, (0,255,120), pygame.Rect(x_coord, 290, 30,10))
		if(x1>=0 and x1<400 and y1>=0 and y1<300):
			x1 += math.sin(angle)*15.0
			y1 += math.cos(angle)*15.0
		if(x1>=400):
			x1 = 395
			angle = -angle 
		elif(x1<0):
			x1 = 0
			angle = -angle 
		elif(y1>=300):
			life_end = True
			reward += -1.0
			done = True
		elif(y1<0):
			y1 = 0
			angle = 3.14-angle 
		ball = pygame.draw.circle(screen, (0, 255,255), (int(x1), int(y1)), 10)
		if ball.colliderect(block):
			reward += 0.5
			angle = 3.14-angle
		new_blocks = []
		for l in range(0, len(block_break)):
			if ball.colliderect(block_break[l]):
				angle = 3.14-angle
				reward += 1.0
			else:
				new_blocks.append(block_break[l])
		block_break = new_blocks
		image += get_image(screen)
	return image , screen, reward, x_coord, block_break, x1, y1, angle,life_end



def remove_transparency(im, bg_colour=(255, 255, 255)):
	if im.mode in ('RGBA') or (im.mode == 'P' and 'transparency' in im.info):
		alpha = im.convert('RGBA').split()[-1]
		bg = Image.new("RGBA", im.size, bg_colour + (255,))
		bg.paste(im, mask=alpha)
		return bg
	else:
		return im

def get_initial_state():
	initial_history = 4
	block_break=[]
	screen = pygame.display.set_mode((400,300))
	image = np.zeros((300,400,4))
	x1 = 200
	y1 = 150
	start_y = 100
	start_x = 0
	y_coord = 290
	x_coord = 200
	angle = random.randint(-120, 120)
	angle = angle*3.14/180
	life_end = False
	for i in range(0, 2):
		start_x = 0
		for b in range(0, 10):
			block_break.append(pygame.Rect(start_x, start_y, 40,10))
			start_x +=40
		start_y +=10
	for bl in range(0, len(block_break)):
		pygame.draw.rect(screen, (225,0,0) , block_break[bl])
	block = pygame.draw.rect(screen, (0,255,120), pygame.Rect(x_coord, y_coord, 30,10))
	ball = pygame.draw.circle(screen, (0, 255,255), (int(x1), int(y1)), 10)
	for i in range(0, initial_history):
		screen.fill((0, 0, 0))
		for bl in range(0, len(block_break)):
			pygame.draw.rect(screen, (255,255,255), block_break[bl])
		block = pygame.draw.rect(screen, (0,255,120), pygame.Rect(x_coord, y_coord, 30,10))
		ball = pygame.draw.circle(screen, (0, 255,255), (int(x1), int(y1)), 10)
		if(x1>=0 and x1<400 and y1>=0 and y1<300):
			x1 += math.sin(angle)*15.0
			y1 += math.cos(angle)*15.0
		if(x1>=400):
			x1 = 395
			angle = -angle 
		elif(x1<0):
			x1 = 0
			angle = -angle 
		elif(y1>=300):
			life_end = True
			done = True
		elif(y1<0):
			y1 = 0
			angle = 3.14-angle 

		if ball.colliderect(block):
			angle = 3.14-angle
		
		new_blocks = []
		for l in range(0, len(block_break)):
			if ball.colliderect(block_break[l]):
				angle = 3.14-angle
			else:
				new_blocks.append(block_break[l])
		block_break = new_blocks
		image += get_image(screen)
	return image, screen, x_coord, block_break, x1, y1, angle


BATCH_SIZE = 1
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


screen_height = 300
screen_width = 400
n_actions = 2
policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return np.argmax(policy_net(state))
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    state_action_values = policy_net(state_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = max(target_net(non_final_next_states))
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss

num_episodes = 5000000000000000000
for i_episode in range(num_episodes):
    # Initialize the environment and state
	scenes=[]
	done = False
	init_screen_image , game_screen, x_coord, block_break, x1, y1, angle = get_initial_state()
	scenes.append(init_screen_image)
	loss = 0.0
	frame_count = 0
	for t in range(0, 100):
		frame_count+=1
		# Select and perform an action
		init_screen_image = cv2.resize(init_screen_image, (84,84))
		init_screen_image_ = torch.tensor(init_screen_image).float()
		init_screen_image_ = torch.reshape(init_screen_image_, [1, 4, 84, 84])
		action = select_action(init_screen_image_)
		action_ = torch.tensor(action).long()
		action_ = torch.reshape(action_, [1,1])
		next_state, screen , reward,x_coord, block_break, x1, y1, angle, done = get_next_state(game_screen, action, x_coord, block_break, x1, y1, angle)					
		reward = torch.tensor([reward], device=device)

		# Store the transition in memory
		next_state_arr = cv2.resize(next_state, (84,84))
		next_state_ = torch.tensor(next_state_arr).float()
		next_state_ = torch.reshape(next_state_, [1, 4, 84, 84])
		memory.push(init_screen_image_, action_, next_state_, reward)

		# Move to the next state
		init_screen_image = next_state
		scenes.append(next_state)
		# Perform one step of the optimization (on the target network)
		loss += optimize_model()
		if done:
		    episode_durations.append(t + 1)
		    plot_durations()
		    break
		# Update the target network, copying all weights and biases in DQN
		if i_episode % TARGET_UPDATE == 0:
			target_net.load_state_dict(policy_net.state_dict())

		# print('Complete')
	print("loss", loss/frame_count)
	vid_name = 'vid' +str(i_episode)+'.gif'
	imageio.mimsave(vid_name ,  scenes)
	print("Completed one episode")
plt.ioff()
plt.show()
