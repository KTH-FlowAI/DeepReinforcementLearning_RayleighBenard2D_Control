#!/bin/env python
#
# DEEP REINFORCEMENT LEARNING FOR RAYLEIGH-BENARD CONVECTION
#
# Single-Agent Reinforcement Learning launcher
#
# train_sarl.py: main launcher for the SARL framework. 
#
# Pol Suarez, Francisco Alcantara, Colin Vignon & Joel Vasanth
#
# FLOW, KTH Stockholm | 09/04/2023

from __future__ import print_function, division
import os, sys, time
from tensorforce.agents import Agent
from tensorforce.execution import Runner
from env_utils     import run_subprocess, generate_node_list, read_node_list


#### Set up which case to run
training_case = "RB_2D_SARL"  
simu_name = training_case

general_path = os.getcwd()
case_path = general_path+'/data/'+simu_name
sys.path.append(case_path)

os.system('rm -r '+case_path)
os.mkdir(case_path)

os.system('cp ./parameters/parameters_{}.py '.format(training_case)+case_path+'/parameters.py')

from sarl_env import Environment2D
from parameters import nb_actuations, num_episodes, num_servers, simu_name


#### Run
initial_time = time.time()

# Generate the list of nodes
generate_node_list(num_servers=num_servers) 

# Read the list of nodes
nodelist = read_node_list()

print("\n\nDRL for 2D Rayleigh-Benard convection\n")
print("---------------------------------------\n")
print('Case: '+simu_name+' (Single-Agent RL)\n')
environment_base = Environment2D(simu_name=simu_name, path=general_path, node=nodelist[0]) # Baseline  #+simu_name

network = [dict(type='dense', size=512), dict(type='dense', size=512)]

agent = Agent.create(
    # Agent + Environment
    agent='ppo', environment=environment_base, max_episode_timesteps=nb_actuations,
    # Network
    network=network,
    # Optimization
    batch_size=20, learning_rate=1e-3, subsampling_fraction=0.2, multi_step=25,
    # Reward estimation
    likelihood_ratio_clipping=0.2, predict_terminal_values=True,
    baseline=network,
    baseline_optimizer=dict(
        type='multi_step', num_steps=5,
        optimizer=dict(type='adam', learning_rate=1e-3)
    ),
    # Regularization
    entropy_regularization=0.01,
    parallel_interactions=num_servers,
    saver=dict(directory=os.path.join(os.getcwd(), 'saver_data'), frequency=1, max_checkpoints=1),#parallel_interactions=number_servers,
)

environments = [Environment2D(simu_name=simu_name, path=general_path, do_baseline=False, ENV_ID=i, host="environment{}".format(i+1), node=nodelist[i+1]) for i in range(num_servers)]


#start all environments at the same time
runner = Runner(agent=agent, environments=environments, remote='multiprocessing')

#now start the episodes and sync_episodes is very useful to update the DANN efficiently
runner.run(num_episodes=num_episodes, sync_episodes=False)
runner.close()

#saving all the model data in model-numpy format 
agent.save(directory=os.path.join(os.getcwd(),'model-numpy'), format='numpy', append='episodes')

agent.close()

end_time = time.time()

print("DRL simulation :\nStart at : {}.\nEnd at {}\nDone in : {}".format(initial_time,end_time,end_time-initial_time))

