import numpy as np
import pandas as pd
from time import time,sleep
from random import randint as r
import random
import pickle
import pygame

# Number of row and colmns
n = 10
xpyplt = n*100
ypyplt = n*100
background = (51,51,51)
# Visualize
screen = pygame.display.set_mode((xpyplt,ypyplt))
colors = [(51,51,51),(51,51,51),(51,51,51),(51,51,51),(51,51,51),(255,0,0),(255,0,0),(51,51,51),(51,51,51),(51,51,51),(0,255,0),(51,51,51),(51,51,51),(51,51,51),(51,51,51),(51,51,51)]
# Create a play ground
reward = np.array([[-2, -100, -2, -2, -2, -2, -2, -2, -2, -2],
                   [-2, -2, -2, 50 , -2, -100, -2, -2, -2, -2],
                   [50 , -2, -2, -2, -2, -100, 50, -2, -2, -2],
                   [-100, -100, -2, -100, -100, -2, -100, -2, -2, -2],
                   [-2, 50 , -100, -2, -100, -2, 50 , -100, -100, -2],
                   [-2, -2, -100, -2, -100, -2, -2, -2, -2, -2],
                   [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
                   [-2, 50 , -2, -2, -2, -2, -100, -100, -100, -100],
                   [-2, -100, -100, -100, -100, -100, -2, -2, -2, -2],
                   [-2, -2, -2, -2, -2, -2, -2, -100, 50, 100]])
# Create walls
colors = [(51,51,51) for i in range(n**2)]
reward = np.zeros((n,n))
term = []
walls = [[0,1], [1,5], [2,5], [3,0], [3,1], [3,3], [3,4], [3,6], [4,2], [4,4], [4,6], [4,7], [4,8], [5,2], [5,4], [8,1], [8,2], [8,3], [8,4], [8,5], [7,6], [7,7], [7,8], [7,9]  ]
for _ in walls:
    i = _[0]
    j = _[1]
    reward[i,j] = -1
    colors[n*i+j] = (255,0,0)
    term.append(n*i+j)
reward[n-1,n-1] = 1
colors[n**2 - 1] = (0,255,0)
term.append(n**2 - 1)
# Create flags
flags = [[2,0], [1,3], [2,6], [4,1], [4,5], [7,1], [9, 8] ]
for _ in flags:
    i = _[0]
    j = _[1]
    colors[n*i+j] = (128, 0, 128)
# Create Q table
Q = np.zeros((n**2,4))
# Create actions
actions = {"up": 0,"down" : 1,"left" : 2,"right" : 3}
states = {}
k = 0
for i in range(n):
    for j in range(n):
        states[(i,j)] = k
        k+=1

# Q learning factors
alpha = 0.01
gamma = 0.9
# Start from first
current_pos = [0,0]
epsilon = 0.25
def layout():
    c = 0
    for i in range(0,xpyplt,100):
        for j in range(0,ypyplt,100):
            pygame.draw.rect(screen,(255,255,255),(j,i,j+100,i+100),0)
            pygame.draw.rect(screen,colors[c],(j+3,i+3,j+95,i+95),0)
            c+=1

def choose_actions(current_state):
    global current_pos,epsilon, reward
    possible_actions = []
    if np.random.uniform() <= epsilon:
        if current_pos[1] != 0:
            possible_actions.append("left")
        if current_pos[1] != n-1:
            possible_actions.append("right")
        if current_pos[0] != 0:
            possible_actions.append("up")
        if current_pos[0] != n-1:
            possible_actions.append("down")
        action = actions[possible_actions[r(0,len(possible_actions) - 1)]]
    else:
        m = np.min(Q[current_state])
        if current_pos[0] != 0:
            possible_actions.append(Q[current_state,0])
        else:
            possible_actions.append(m - 100)
        if current_pos[0] != n-1:
            possible_actions.append(Q[current_state,1])
        else:
            possible_actions.append(m - 100)
        if current_pos[1] != 0:
            possible_actions.append(Q[current_state,2])
        else:
            possible_actions.append(m - 100)
        if current_pos[1] != n-1:
            possible_actions.append(Q[current_state,3])
        else:
            possible_actions.append(m - 100)
        action = random.choice([i for i,a in enumerate(possible_actions) if a == max(possible_actions)])
        return action

def pass_episods():
    global current_pos,epsilon
    current_state = states[(current_pos[0],current_pos[1])]
    action = choose_actions(current_state)
    if action == 0:
        current_pos[0] -= 1
    elif action == 1:
        current_pos[0] += 1
    elif action == 2:
        current_pos[1] -= 1
    elif action == 3:
        current_pos[1] += 1
    new_state = states[(current_pos[0],current_pos[1])]
    if new_state not in term:
        Q[current_state,action] += alpha*(reward[current_pos[0],current_pos[1]] + gamma*(np.max(Q[new_state])) - Q[current_state,action])
    else:
        Q[current_state,action] += alpha*(reward[current_pos[0],current_pos[1]] - Q[current_state,action])
        current_pos = [0,0]
        epsilon -= 1e-3

# Run the game
run = True
for i in range(000):
    pass_episods()
current_pos = [0,0]
while run:
    screen.fill(background)
    layout()
    pygame.draw.circle(screen,(25,129,230),(current_pos[1]*100 + 50,current_pos[0]*100 + 50),30,0)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
    pygame.display.flip()
    pass_episods()

# Quit if needed
pygame.quit()
print (epsilon)
# Print Q table
print(Q)
print(Q.shape)
# Save Q table
dict = {'up, down, left, right': Q}  
df = pd.DataFrame(dict) 
df.to_csv('Q_table.csv') 