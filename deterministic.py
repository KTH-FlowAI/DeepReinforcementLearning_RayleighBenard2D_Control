#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''

deterministic.py: Deterministinc runner for trained agent evaluation

Created: 2/5/2023

Author: Pol Suarez, adapted for shenfun by Colin Vignon

'''

import os, time, sys
import subprocess
import copy as cp
from tensorforce.agents import Agent
from tensorforce.execution import Runner

# Change to your case name here 
case_name = 'RB_2D_multiAgent_6_4pi'
general_path = os.getcwd()

# Set case path 
case_path = general_path+'/shenfun_files/cases/'+case_name
sys.path.append(case_path)

from MARL_env import Environment2D
from parameters import nb_actuations, num_episodes, num_servers, simu_name, nb_inv_envs, n_seg

# Set to Fasle for SARL 
MARL = True
if MARL:
    nb_envs = nb_inv_envs
else:
    nb_envs = num_servers

# Begin timer
initial_time = time.time()


# Read the list of nodes
fp = open('nodelist','r')
nodelist = [h.strip() for h in fp.readlines()]
fp.close()

def split(environment, np):  
    
    
    ''' input: one of the parallel environments (np); output: a list of nb_inv_envs invariant 
    environments identical to np. 
    
    Their ID card: (np, ni)
    
    np:= number of the parallel environment. e.g. between [1,4] for 4 parallel environments
    ni:= env_ID[1]:= number of the 'pseudo-parallel' invariant environment. e.g. between [1, 10] for 10 invariant envs
    nb_inv_envs:= total number of 'pseudo-parallel' invariant environments. e.g. 10

    '''

    list_inv_envs = []
    for i in range(nb_inv_envs):
        env = cp.copy(environment)
        env.ENV_ID = [np, i+1]
        env.host="environment{}".format((np-1)*nb_inv_envs + (i+1))
        list_inv_envs.append(env)
    return list_inv_envs

environment_base = Environment2D(simu_name = simu_name, path=general_path, do_baseline=False, ENV_ID=[1,0], deterministic=True, host="environment1", node=nodelist)  # Baseline
environments_MARL = [split(environment_base, 1)[j] for j in range(nb_inv_envs)]


# Load TensorForce Agent from saved agent during training 
agent = Agent.load(directory=os.path.join(case_path, 'saver_data'), format='checkpoint', environment=environment_base)

# Initialise TensorForce Runner
runner = Runner(agent=agent, environments=environments_MARL, remote='multiprocessing', max_episode_timesteps=5000)

# Run the runner
runner.run(num_episodes=nb_envs, evaluation=True)

# Close agent, runner and end timer 
runner.close()
agent.close()
end_time = time.time()

print("Deterministic Runner time :\nStart at : {}.\nEnd at {}\nDone in : {}".format(initial_time,end_time,end_time-initial_time))
