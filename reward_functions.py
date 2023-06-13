#!/bin/env python
#
# DEEP REINFORCEMENT LEARNING FOR RAYLEIGH-BENARD CONVECTION
#
# reward_functions.py: computes reward, Nusselt number and kinetic energy.
#
# Colin Vignon
#
# FLOW, KTH Stockholm | 09/04/2023

import copy as cp
import numpy as np
from parameters import CFD_params, nb_inv_envs, optimization_params


def compute_reward(probes_values, reward_function):
    out = cp.copy(probes_values)
    obsGrid = CFD_params.get('dico_d').get('obsGrid')
    out2 = np.array(out).reshape(3, obsGrid[0], obsGrid[1])

    #un-normalization of the data
    out2[0] *= 1/1.5  # horizontal speed (ux) un-normalization
    out2[1] *= 1/1.5  # vertical speed (uy)  un-normalization
    out2[2] *= 1/2  # temperature un-normalization
    out2[2] += 0.8

    hor_inv_probes = CFD_params.get('hor_inv_probes')
    out_red = np.zeros((3, obsGrid[0], hor_inv_probes))  
    out_red = out2[:, :, (nb_inv_envs//2)*hor_inv_probes:((nb_inv_envs//2)+1)*hor_inv_probes]  

    kappa = 1./np.sqrt(CFD_params.get('dico_d').get('Pr')*CFD_params.get('dico_d').get('Ra'))
    T_up = CFD_params['dico_d'].get('bcT')[1]
    div = kappa*(2.-T_up)/2  # H = 2, Tb = 2.

    uyT_ = np.mean(np.mean(np.multiply(out2[1], out2[2]), axis=1), axis = 0)
    T_ = np.mean(np.gradient(np.mean(out2[2], axis=1), axis=0))
    gen_Nus = (uyT_ - kappa*T_)/div

    uyT_loc = np.mean(np.mean(np.multiply(out_red[1], out_red[2]), axis=1), axis = 0)
    T_loc = np.mean(np.gradient(np.mean(out_red[2], axis=1), axis=0))
    loc_Nus = (uyT_loc - kappa*T_loc)/div

    gen_kinEn = np.sum(out2[1]*out2[1] + out2[0]*out2[0])
    loc_kinEn = np.sum(out_red[1]*out_red[1] + out_red[0]*out_red[0])

    Reward_Nus   = 0.9985*gen_Nus + 0.0015*loc_Nus
    Reward_kinEn = 0.4*gen_kinEn + 0.6*loc_kinEn
    if reward_function == 'Nusselt':
        reward = optimization_params["norm_reward"]*(-Reward_Nus + optimization_params["offset_reward"])
    elif reward_function == 'kinEn':
        reward = optimization_params["norm_reward"]*(-Reward_kinEn + optimization_params["offset_reward"])
    elif reward_function == 'meanFlow':
        reward = None
        print("ERROR: 'meanFlow' not encoded yet")
    else:
        print("ERROR: Choose 'Nusselt' or 'kinEn' or 'meanFlow' for the reward function") 

    return reward, gen_Nus, gen_kinEn