#!/bin/env python
#
# DEEP REINFORCEMENT LEARNING FOR RAYLEIGH-BENARD CONVECTION
#
# parameters file for the SARL framework
#
#
# FLOW, KTH Stockholm | 09/04/2023

from __future__ import print_function, division
import numpy as np
import math


# case name - should be the same name as this file, without the prefix parameters_
case            = 'RB_2D_SARL'
simu_name       = case
dimension       = '2D'
reward_function = 'Nusselt' 


# Number of calculation processors
nb_proc     = 1 

# number of environment in parallel
num_servers = 1 

# Number of segments (actuators) on the lower boundary  
n_seg = 10   

# Number of invariant parallel environments ('multi-agents' - set to one for single agent)
nb_inv_envs = 1  

# Duration of baseline simulation (in nondimensional simulation time)
simulation_duration   = 10.0   
simulation_time_start = 0.0

# Duration of each actuation (in nondimensional simulation time)
delta_t_smooth   = 1.5      
delta_t_converge = 0.0       
smooth_func      = 'linear' 


# post options
post_process_steps = 200 

# Total number of episodes
num_episodes = 1 

# Number of actuations per episode 
nb_actuations = 2 
nb_actuations_deterministic = nb_actuations*4

# Probes
probes_location      = 'cartesian_grid'
number_of_probes     = (8,32)


# Simulation parameters
simulation_params = {
	'simulation_duration':  simulation_duration,  
	'simulation_timeframe': [simulation_time_start,simulation_time_start+simulation_duration],
	'delta_t_smooth':       delta_t_smooth,
	'delta_t_converge':     delta_t_converge,
	'smooth_func':          smooth_func,
	'post_process_steps' :  post_process_steps
}

# Variational input  
variational_input = {
	'filename':        'RB_2D', 
	'porous':          False, 
	"d":               0, 
	"time":            -0.25, 
	"initial_time":    None, 
}


output_params = {
	'nb_probes':  number_of_probes,
	'probe_type': 'u_T'
}

# Optimization
optimization_params = {
	"min_ampl_temp":             -1.,
	"max_ampl_temp":             1.,
	#"norm_Temp":                 0.4,  
	"norm_reward":               1.,  
	#"norm_press":                    2,  
	"offset_reward":                 2.6788,  
}

inspection_params = {
	"plot":                False,  
	"step":                50,
	"dump":                100,
	"show_all_at_reset":   True,
	"single_run":          False
}
