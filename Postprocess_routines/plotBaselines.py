import numpy as np
import os, sys, time
import matplotlib.pyplot as plt
import json


'''
This script plots the baseline Nu and KE over the time of the baseline simulation.

Author: Joel Vasanth

Date: 3/20/2023

'''

# Set case
simu_name = 'RB_3D_MARL_01'  
general_path = os.getcwd()
path = general_path+'/../shenfun_files/cases/'+simu_name
os.chdir(path)


# Set parameters
baseline_t = 400  # duration baseline simulation
ampl = 0.75  # Amplitude of variation of T
nb_parr_envs = 1  # number of parallel environments
nb_seg = 10 # number of segments


MARL = True
if MARL:
    nb_envs = nb_parr_envs*nb_seg
else:
    nb_envs = nb_parr_envs

reward_function = 'Nusselt'  # choose 'Nusselt' or 'kinEn'


os.chdir(path+'/baseline/')    
with open('evo_Nusselt_baseline.json', 'r') as f:
    Nusselt_baseline = json.load(f)
with open('evo_kinEn_baseline.json', 'r') as f2:
    kinEn_baseline = json.load(f2)

# Time of simulation. 0.5 here is the dt, can be changed.
timerange = np.arange(0.5,baseline_t,0.5)

# Perform plotting 
fig, ax = plt.subplots()
p1 = ax.plot(Nusselt_baseline, 'b-', label='Nusselt Number')
ax2 = ax.twinx()
p2 = ax2.plot(kinEn_baseline, 'r-', label='Kinetic Energy')
allp = p1+p2
labs = [p.get_label() for p in allp]
ax.legend(allp, labs, loc = 4)
ax.set_xlabel('Time')
ax.set_ylabel('Nusselt Number', color='b')
ax2.set_ylabel('Kinetic Energy', color='r')
# plt.grid()
savepath = general_path+'/../baseline3D.png'
print(savepath)
plt.savefig(savepath)
# plt.show()
