#!/bin/env python
#
# DEEP REINFORCEMENT LEARNING FOR RAYLEIGH-BENARD CONVECTION
#
# parameters file for the MARL framework
#
#
# FLOW, KTH Stockholm | 09/04/2023

from __future__ import print_function, division
from Tfunc import Tfunc, domain
import numpy as np
import math
import sympy


# case name - should be the same name as this file, without the prefix parameters_
case            = 'RB_2D_MARL'
simu_name       = case
dimension       = '2D'
reward_function = 'Nusselt' 

# Number of calculation processors
nb_proc     = 1  

# Number of environments in parallNel
num_servers = 1  

# Number of segments (actuators) on the lower boundary 
n_seg = 10  

# Number of invariant parallel environments ('multi-agents') : 
nb_inv_envs = n_seg  
# always take nb_inv_envs = n_seg for MARL


# Duration of baseline simulation (in nondimensional simulation time)
simulation_duration   = 10.0   
simulation_time_start = 0.0

# Duration of each actuation (in nondimensional simulation time)
delta_t_smooth   = 1.5      
delta_t_converge = 0.0       
smooth_func      = 'linear' 


# Total number of episodes : 
# CHOOSE A MULTIPLE OF nb_inv_envs*num_servers and add +1 
num_episodes = 11 

# Number of actuations per episode 
nb_actuations = 2 
nb_actuations_deterministic = nb_actuations*4

# Probes
probes_location      = 'cartesian_grid'
number_of_probes     = (8,32)
N = (64,96)  # simulation mesh grid

# misc
post_process_steps = 200  
alphaRestart = 0.9 
x, y, tt = sympy.symbols('x,y,t', real=True)



CFD_params = {
            'number_of_actuators': n_seg, 
            'hor_inv_probes':number_of_probes[1]//nb_inv_envs,  
            'dico_d':        {'N': N,
                              'domain': domain,
                              'Ra': 10000.,
     	                      'Pr': 0.7,
     	                      'dt': 0.05,
     	                      'filename': f'RB_{N[0]}_{N[1]}',
     	                      'conv': 0,
     	                      'modplot': 10,
     	                      'obsGrid': number_of_probes,  
    	                      'moderror': 10000,
    	                      'modsave': 10000,
     	                      'bcT': (Tfunc(nb_seg=n_seg, dicTemp={'T'+str(i):1. for i in range(n_seg)}).apply_T(y), 1), 
     	                      'family': 'C',
     	                      'checkpoint': 10,
     	                      #'padding_factor': 1,
     	                      'timestepper': 'IMEXRK3'
     	                     }
}
     	    

simulation_params = {
	'simulation_duration':  simulation_duration,  
	'simulation_timeframe': [simulation_time_start,simulation_time_start+simulation_duration],
	'delta_t_smooth':       delta_t_smooth,
	'delta_t_converge':     delta_t_converge,
	'smooth_func':          smooth_func,
	'post_process_steps' :  post_process_steps
}



# Optimization
optimization_params = {
	"min_ampl_temp":             -1.,
	"max_ampl_temp":             1.,
	"norm_reward":               1.,
	"offset_reward":             2.6726  
}

inspection_params = {
	"plot":                False,  
	"step":                50,
	"dump":                100,
	"show_all_at_reset":   True,
	"single_run":          False
}




