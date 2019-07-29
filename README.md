# atari-dqn
Deep reinforcement learning for self made atari game environment


# Environement

The environment has been made using Pygame instead of gym. The number of actions are only 2, left and right. The game involves one bar which prevents the ball from touching the bottom of the window. The reward is maximised if the ball hits the blocks above and is prevented from hitting the bottom


# Code

Some classes in the code like Replay Memory has been adopted from https://github.com/pytorch/tutorials/blob/master/intermediate_source/reinforcement_q_learning.py


# Results till now

After 10 mins:


![vid8](https://user-images.githubusercontent.com/32021556/62021861-1b97b680-b1e7-11e9-9dc4-cd7be0d93e44.gif)




After 60 mins:



![vid714](https://user-images.githubusercontent.com/32021556/62021862-1d617a00-b1e7-11e9-8af8-cd0f059a8d04.gif)




The model needs to be trained for more episodes. 
