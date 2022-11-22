# Import routines

import numpy as np
import math
import random
from itertools import permutations, product

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        # total of 20 permuation of (5 obj and 2 space = 5!/3! = 5*4 = 20 + 1([0,0]: initial state))
        self.action_space = [(0,0)] + list(permutations([i for i in range(m)], 2)) 
        # m*t*d = 5*24*7 = 840
        self.state_space = [pro for pro in product(range(m),range(t),range(d))]
        self.state_init = random.choice(self.state_space)

        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. 
        This method converts a given state into a vector format. 
        Hint: The vector is of size m + t + d."""

        state_encod = np.zeros((m + t + d)).astype(int).tolist() # zeros of 5+24+7=36

        state_encod[state[0]] = 1 # understandable
        state_encod[m+state[1]] = 1
        state_encod[m+t+state[2]] = 1
        return state_encod



    # Use this function if you are using architecture-2 
    # def state_encod_arch2(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""

        
    #     return state_encod


    ## Getting number of requests




    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)
        elif location == 1:
            requests = np.random.poisson(12)
        elif location == 2:
            requests = np.random.poisson(4)
        elif location == 3:
            requests = np.random.poisson(7)
        elif location == 4:
            requests = np.random.poisson(8)

        if requests > 15:
            requests = 15

        # if request is 15 then it should choose random 15 numbers from 1-21(20 actions in total)
        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]       # only returning those action which are selected in possible action idx     
        actions.append((0,0))
        possible_actions_index.append(0)                                       # appending index for (0,0) action
        return possible_actions_index, actions   


    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        curr_loc, curr_time, curr_day = state
        pickup_loc, drop_loc = action
        # print("t1, curr_loc, pickup_loc, curr_time, curr_day\n", curr_loc, pickup_loc, curr_time, curr_day)
        
        if action == (0,0):
            reward = -1 * C

        else:
            # time from curr_loc to reach pickup_loc
            t1 = int(Time_matrix[curr_loc][pickup_loc][curr_time][curr_day]) # Time_matrix.shape: (5, 5, 24, 7)
            # print("t1, curr_loc, pickup_loc, curr_time, curr_day\n", t1, curr_loc, pickup_loc, curr_time, curr_day)
            curr_time, curr_day = self.time_day_update_func(curr_time, curr_day, t1)
            # print("curr_time, curr_day", curr_time, curr_day)
            # time from pickup_loc to reach drop_loc
            t2 = int(Time_matrix[pickup_loc][drop_loc][curr_time][curr_day])
            
            # reward is total ride time (R*t2) minus fule cost(C) from t1 to t2
            reward = (R * t2) - (C * (t1 + t2))

        return reward


    def time_day_update_func(self, curr_time, curr_day, ride_duration):
        """ Takes current time of the day, the current day of the week and the ride_duration. 
            returns the current time and day post ride."""
        curr_day = (curr_day + ((curr_time + ride_duration) // t)) % d      # t=24, d=7
        curr_time = (curr_time + ride_duration) % t
        return curr_time, curr_day



    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""

        curr_loc, curr_time, curr_day = state
        pickup_loc, drop_loc = action
        
        rewards = self.reward_func(state, action, Time_matrix)
        total_time = 0
        
        if action == (0,0):
            # update time by 1 hour
            curr_time, curr_day = self.time_day_update_func(curr_time, curr_day, 1)
            next_state = (curr_loc, curr_time, curr_day)
            total_time = 1
        else:
            # time from curr_loc to reach pickup_loc
            t1 = int(Time_matrix[curr_loc][pickup_loc][curr_time][curr_day])
            curr_time, curr_day = self.time_day_update_func(curr_time, curr_day, t1)
            
            # time from pickup_loc to reach drop_loc
            t2 = int(Time_matrix[pickup_loc][drop_loc][curr_time][curr_day])
            curr_time, curr_day = self.time_day_update_func(curr_time, curr_day, t2)
               
            total_time = t1 + t2
            next_state = (drop_loc, curr_time, curr_day)
        
        return next_state, rewards, total_time



    def reset(self):
        return self.action_space, self.state_space, self.state_init
