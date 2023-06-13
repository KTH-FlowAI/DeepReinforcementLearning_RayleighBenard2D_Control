#!/bin/env python
#
# DEEP REINFORCEMENT LEARNING FOR RAYLEIGH-BENARD CONVECTION
#
# Multi-Agent Reinforcement Learning launcher
#
# train_marl.py: main launcher for the MARL framework. 
#
# Pol Suarez, Francisco Alcantara, Colin Vignon & Joel Vasanth
#
# FLOW, KTH Stockholm | 09/04/2023

from __future__ import print_function, division
import os, sys
import copy as cp
import time
from tensorforce.agents import Agent
from tensorforce.execution import Runner
from env_utils import run_subprocess, generate_node_list, read_node_list


#### Set up which case to run:
# this is the name of the parameters file without the 'parameters_' prefix
training_case = "RB_2D_MARL"  

simu_name = training_case
general_path = os.getcwd()
case_path = general_path+'/data/'+simu_name
sys.path.append(case_path)

try:
    os.system('rm -r '+case_path)
except:
    print('No existing case with this name')
os.mkdir(case_path)
os.system('cp ./parameters/parameters_{}.py '.format(training_case)+case_path+'/parameters.py')

from marl_env import Environment2D 
from parameters import nb_actuations, num_episodes, num_servers, simu_name, nb_inv_envs


#### Run
initial_time = time.time()

# Generate the list of nodes
generate_node_list(num_servers=num_servers) 
nodelist = read_node_list()

print("\n\nDRL for 2D Rayleigh-Benard convection\n")
print("---------------------------------------\n")
print('Case: '+simu_name+' (Multi-Agent RL)\n')
environment_base = Environment2D(simu_name=simu_name, path=general_path, node=nodelist[0])
network = [dict(type='dense', size=512), dict(type='dense', size=512)]

# Define tensorforce agent
agent = Agent.create(
    # Agent + Environment
    agent='ppo', environment=environment_base, max_episode_timesteps=nb_actuations,
    # Network
    network=network,
    # Optimization
    batch_size=20, learning_rate=1e-3, subsampling_fraction=0.2, multi_step=25,
    # Reward estimation
    likelihood_ratio_clipping=0.2, predict_terminal_values=True,
    # Critic
    baseline=network,
    baseline_optimizer=dict(
        type='multi_step', num_steps=5,
        optimizer=dict(type='adam', learning_rate=1e-3)
    ),
    # Regularization
    entropy_regularization=0.01,
    parallel_interactions=num_servers*nb_inv_envs,
    saver=dict(directory=os.path.join(os.getcwd(), 'saver_data'), frequency=1, max_checkpoints=1),
)


def split(environment, np): 
    ''' input: one of the parallel environments (np); 
        output: a list of nb_inv_envs invariant environments identical to np. 
        Their ID card: (np, ni)
    '''
    list_inv_envs = []
    for i in range(nb_inv_envs):
        env = cp.copy(environment)
        env.ENV_ID = [np, i+1]
        env.host="environment{}".format((np-1)*nb_inv_envs + (i+1))
        list_inv_envs.append(env)
    return list_inv_envs

parallel_environments = [Environment2D(simu_name=simu_name, path=general_path, do_baseline=False, \
                                       ENV_ID=[i+1,0], host="environment{}".format(i+1), node=nodelist[i+1]) \
                                        for i in range(num_servers)]

environments = [split(parallel_environments[i], i+1)[j] for i in range(num_servers) for j in range(nb_inv_envs)]


runner = Runner(agent=agent, environments=environments, remote='multiprocessing')    
runner.run(num_episodes=num_episodes, sync_episodes=False)
runner.close()

agent.save(directory=os.path.join(os.getcwd(),'model-numpy'), format='numpy', append='episodes')

agent.close()

for env in environments:
    env.close()
end_time = time.time()

print("DRL simulation :\nStart at : {}.\nEnd at {}\nDone in : {}".format(initial_time,end_time,end_time-initial_time))







