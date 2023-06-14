import numpy as np
import os, sys, time
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator)
import json
from scipy.signal import savgol_filter

'''

The following plots can be generated with this script, one at a time.
 
1. Plots learning curve - instantaneous reward vs episode.
2. Plots learning curve - instantaneous Nusselt number vs episode.
3. Plots learning curve (Nu) from a single training episode (ep_no), along with the Nu from the baseine. Horizontal axis is time.
4. Plots many learning curves (Nu) from diff episodes
5. Plot of learning curves from SARL and MARL for comparison 
6. Plots learning curve (Nu) from a single episode (ep_no), along with the baseine. x-axis is time, along with the removing control 
7. Plots Nu after control is removed   

You can plot either one, by setting the 'case' variable below to any index number above.

Author: Joel Vasanth | FLOW, KTH Stockholm

Date: 3/25/2023

'''

# Set which case to plot. See options above 
cas = 7

# Set simulation name and path 
simu_name = 'RB_2D_MARL' 
general_path = os.getcwd()
path = general_path+'/../shenfun_files/cases/'+simu_name
os.chdir(path)
sys.path.append(path)
from parameters import simulation_params, reward_function, optimization_params, nb_proc, nb_actuations, nb_actuations_deterministic, n_seg


# Number of actions/actuations per episode
nb_actions = 100  

# CFD time duration between two actuations
dt = 1.5  

# duration of the baseline simulation
baseline_t = 1000  

# Amplitude of variation of T
ampl = 0.75  

# Number of segments 
nb_seg = 10

# Number of pseudo-environments 
nb_envs = 10


# Evolution of the reward obtained at the end of an episode, for all episodes, for all environments (list(R1, R2, R3,...))
total_rewards = [] 

rewards_interEps = {}  

# Evolution of the reward obtained at the end of an episode, for all episodes, for all environments (dictionary {env 1: , env2: ...})
total_rewards_averaged = []
rewards_interEps_averaged = {}

# Evolution of the normalized actions during all the episodes, for all episodes, for all environment for all segment (actuator) (between [-1,+1])
Actions_interEnvs = {}  

# Evolution of the actions during all the episodes, for all episodes, for all environment for all segment (actuator) (between [2-0.75,2+0.75])
Actions_unN_interEnvs = {}  

# Evolution of the rewards during an episode, for all episodes, for all environment ({environment i: array([[R1_ep1, R2_ep1, ...], [R1_ep2, R2_ep2, ...]])})
reward_in_ep = {}  
Nusselt_in_ep = {}  

# the episodes we consider (num_episode, num_environment)
episodes = [(204,9), (234,6), (224, 2)]  

# Episodes considered in the last figure (where we plot the evolution of the actions)
episodes_trace_action = [(1,1)]  
colors = ['red', 'blue', 'green', 'black']

max_ep = 1e8
max_ep_arr = []



if (cas == 1):

    ### PLOTS INSTANTANEOUS REWARD VS EPISODES
    ## the inst. reward is the reward at the end of each episode, and averaged over all environments.
    ## ------------------------------------

    num_eps = 451 # number of episodes you want to plot

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.tick_params(direction="in")
    ax.yaxis.set_ticks_position('both')
    for num_env in range (1,nb_envs+1):
        os.chdir(path+'/final_rewards/environment'+str(num_env))
        eps, rewards = [], []
        
        with open('output_final_rewards.csv', 'r') as f:
            c = 0
            for line in f:
                if c == 0:
                    c +=1
                else:
                    episode, reward = line.split(';')
                    eps.append(int(episode))
                    rewards.append(float(reward.strip('\n')))
                    c += 1 
        
        rewards_interEps.update({'env'+str(num_env):rewards})

    lmm = np.zeros((num_eps,),dtype=np.float64)
    print("No. of episodes printed: ", num_eps)

    # averaging over no fo envs
    for envno in range(1,nb_envs+1):
        ar1 = np.array(rewards_interEps['env'+str(envno)])
        lmm = lmm + ar1

    lmm = lmm/float(nb_envs)
    # plt.rcParams["figure.figsize"] = (15,6)
    plt.plot(lmm)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    # plt.xlim(0,450)
    # plt.ylim(-0.2,1)
    #plt.show()
    os.chdir(general_path+'/../shenfun_files/cases/')
    plt.savefig('FR_1_LearningCurve_reward.png')

elif (cas == 2):
    ## PLOTS INSTANTANEOUS NUSSELT VS EPISODE
    # the Nusselt is the value at the end of each episode, and averaged over all environments.
    # ------------------------------------

    numeps = 446
    nusselts = []
    for num_epis in range (1,numeps+1):
        totalEnvNusselt = 0.0
        for num_env in range (1,nb_envs+1):
            os.chdir(path+'/rewards/environment'+str(num_env)+'/ep_'+str(num_epis))
            eps, rewards = [], []
            with open('output_Nusselts.csv', 'r') as f:
                for line in f:
                    pass
                last_line = line
            actionNumber, envNusselt = last_line.split(';')
            # nusselts.append(float(envNusselt.strip('\n')))
            totalEnvNusselt += float(envNusselt.strip('\n'))
        nusselts.append(totalEnvNusselt/float(num_env))


    os.chdir(path+'/baseline/')    
    with open('evo_Nusselt_baseline.json', 'r') as f:
        Nusselt_baseline = json.load(f)
    print("No. of episodes printed: ", num_epis)

    baselineNusselt = np.array(Nusselt_baseline)
    # xax_base = np.linspace(0,400,799)-400
    xax_epis = np.arange(0,300, 1.5)
    # print(len(nusselts))

    # plt.rcParams["figure.figsize"] = (15,6)
    plt.plot(nusselts, lw = 1)
    # plt.plot(xax_base, baselineNusselt)
    plt.xlabel('Episode')
    #plt.xlim(0,numeps)
    # plt.grid()
    plt.ylabel('$Nu$')
    # plt.xlim(0,365)
    # plt.ylim(1.8,3.0)
    #plt.show()
    os.chdir(general_path+'/../')
    print(general_path)
    plt.savefig('FR_2_LearningCurve.png')

elif (cas == 3):
    # PLOTS learning curve (Nu) from a single episode (ep_no), along with the baseine. x-axis is time.
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.tick_params(direction="in")
    ax.yaxis.set_ticks_position('both')
    ep_no = 130
    total_nusselts = np.zeros((nb_actions,))
    for num_env in range(1,nb_envs+1):
        print(num_env)
        os.chdir(path+'/rewards/environment'+str(num_env)+'/ep_'+str(ep_no))
        nusselts = []
        c = 0
        with open('output_Nusselts.csv', 'r') as f:
            for line in f:
                if (c > 0):
                    _ , envNusselt = line.split(';')
                    nusselts.append(float(envNusselt.strip('\n')))
                c += 1
        if (len(nusselts) != nb_actions):
            print('Length of episode wrong')
        total_nusselts += np.array(nusselts)

    os.chdir(path+'/baseline/')    
    with open('evo_Nusselt_baseline.json', 'r') as f:
        Nusselt_baseline = json.load(f)

    # print(len(Nusselt_baseline))
    
    x_bs = np.arange(1, baseline_t)-baseline_t
    x_ep = np.arange(0,100)
    print(np.size(x_ep))
    print(len(total_nusselts))
    x_ep_vid = np.arange(0,600,3)
    x_join = [0.0, 1.5]
    join_line = [Nusselt_baseline[-1], total_nusselts[0]]
    plt.plot(x_bs,Nusselt_baseline, color='black', linewidth=1)
    # plt.plot(x_join, join_line, lw = 1, color='black')
    plt.plot(x_ep,total_nusselts/float(nb_envs), lw = 1, color= 'blue')
    # nno = np.mean(total_nusselts[99:199])
    # nno_line = [nno, nno, nno]
    # rem_con = [2.0546285693099984,2.0546285693099984,2.0546285693099984]
    # plt.plot([-100,0,300],nno_line,linewidth=0.8,linestyle='--', color='black', label='Actively controlled')
    # plt.plot([-100,0,300],rem_con,linewidth=0.8,linestyle='--', color='red', label='Control Removed')
    # x_pts = [0, 50, 100, 114, 116, 118, 135, 200]
    # pts = [2.6785, 2.675, 2.587, 2.325, 2.2855, 2.3408, 2.5139, 1.9636]
    # plt.scatter(x_pts,pts, lw = 1, color= 'red', marker='o')
    # baseline_array = [2.675,2.675,2.675]
    # plt.plot([-100,0,300],baseline_array,linewidth=0.8,linestyle='--', color='blue', label='Baseline')
    # plt.axvspan(148.54, 300, color='0.8', alpha = 0.5)
    # print(nno)
    plt.xlabel('Time')
    # plt.legend()
    # plt.grid()
    plt.xlim(-800,100)
    # plt.ylim(1.78,3)
    # plt.ylabel('$Nu$')
    plt.show()
    # os.chdir(general_path+'/../')
    # plt.savefig('FR_8_mainEpisode.png')
    


elif (cas ==4):
    # Plots many curves (Nu) from diff episodes
    eps_nos = [45, 275, 357]
    eps_nos_paper = [45,275,350]
    colors = ['green', 'red', 'blue']
    

    def moving_averge(a, n):
        mean_array = np.zeros((a.size-n+1,))
        for i in range(a.size-n+1):
            mean_array[i] = np.mean(a[i:i+n-1])
        return mean_array

    os.chdir(path+'/baseline/')    
    with open('evo_Nusselt_baseline.json', 'r') as f:
        Nusselt_baseline = json.load(f)
    x_bs = np.arange(0, baseline_t-0.5,0.5) - 400
    x_join = [-1,0]

    f = plt.figure(figsize=(10,5))
    plt.rcParams.update({'font.size': 15})
    ax = f.add_subplot(111)
    ax.tick_params(direction="in")
    ax.yaxis.set_ticks_position('both')
    
    n=0
    for ep in eps_nos:
        total_nusselts = np.zeros((nb_actions,))
        for num_env in range(1,nb_envs+1):
            os.chdir(path+'/rewards/environment'+str(num_env)+'/ep_'+str(ep))
            nusselts = []
            c = 0
            with open('output_Nusselts.csv', 'r') as f:
                for line in f:
                    if (c > 0):
                        _ , envNusselt = line.split(';')
                        nusselts.append(float(envNusselt.strip('\n')))
                    c += 1
            if (len(nusselts) != nb_actions):
                print('Length of episode wrong')
            total_nusselts += np.array(nusselts)
        
        x_ep = np.arange(0,300,1.5)
        # x_ep_vid = np.arange(0,600,3)
        # plt.plot(x_bs,Nusselt_baseline)
        plt.plot(x_ep,total_nusselts, lw = 0.5, color=colors[n],)
        nusavg = moving_averge(total_nusselts,20)
        x_avg = moving_averge(x_ep,20)
        plt.plot(x_avg,nusavg,color=colors[n], lw = 1.4,  label='Episode '+str(eps_nos_paper[n]))
        join_line = [Nusselt_baseline[-1], total_nusselts[0]]
        plt.plot(x_join, join_line, lw = 0.3, color=colors[n])
        n += 1
    
    
    plt.plot(x_bs, Nusselt_baseline, color = 'black', lw = 1,label='Baseline')
    baseline_array = [2.678,2.678,2.678]
    plt.plot([0,150,350],baseline_array,linewidth=1.2,linestyle='--', color='black')
    nno_line = [2.0546, 2.0546, 2.0546]
    nno_line2 = [2.0496, 2.0496, 2.0496]
    plt.plot([-50,0,400],nno_line,linewidth=1.2,linestyle='-.', color='black', label='Control Removed')
    plt.plot([-50,0,400],nno_line2,linewidth=1.2,linestyle='--', color='blue', label='Actively Controlled')
    plt.xlabel('Time')
    plt.legend(fontsize=12, ncol=2)
    # plt.grid()
    plt.xlim(-50,300)
    plt.ylim(1.75,3)
    plt.ylabel('$Nu$')
    # plt.show()
    os.chdir(general_path+'/../')
    print(general_path)
    plt.savefig('FR_3_multipleEpisodeLearningCurve')

elif (cas == 5): 
    # Plot of SARL vs MARL
    simu_names = ['RB_2D_MARL_2pi_drl_3', 'RB_2D_SARL']  # RB_2D_MARL_2pi_drl_2 RB_2D_SARL
    colors = ['0', '0.45']
    labels = ['MARL', 'SARL']
    ep_nos = [365, 500] # num of total episodes in MARL (first entry) and SARL (second entry)
    si = 0

    def moving_averge(a, n):
        mean_array = np.zeros((a.size-n+1,))
        for i in range(a.size-n+1):
            mean_array[i] = np.mean(a[i:i+n-1])
        return mean_array

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.tick_params(direction="in")
    ax.yaxis.set_ticks_position('both')
    for simu_name in simu_names:
        path = general_path+'/../shenfun_files/cases/'+simu_name
        os.chdir(path)
        sys.path.append(path)

        numeps = ep_nos[0]
        nusselts = []
        
        for num_epis in range (1,numeps+1):
            totalEnvNusselt = 0.0
            for num_env in range (1,nb_envs+1):
                os.chdir(path+'/rewards/environment'+str(num_env)+'/ep_'+str(num_epis))
                eps, rewards = [], []
                with open('output_Nusselts.csv', 'r') as f:
                    for line in f:
                        pass
                    last_line = line
                actionNumber, envNusselt = last_line.split(';')
                # nusselts.append(float(envNusselt.strip('\n')))
                totalEnvNusselt += float(envNusselt.strip('\n'))
            nusselts.append(totalEnvNusselt/float(num_env))

        # plt.rcParams["figure.figsize"] = (15,6)
        nusarr = np.array(nusselts)
        nusavg = moving_averge(nusarr,25)
        x = np.arange(1,365+1)
        x_avg = moving_averge(x,25)
        plt.plot(x, nusselts, lw = 0.5, linestyle='--', color = colors[si])
        plt.plot(x_avg, nusavg, linewidth=1.5, color = colors[si], label=labels[si])
        si += 1
        # plt.plot(xax_base, baselineNusselt)
    plt.xlabel('Episode')
    baseline_array = [2.675,2.675,2.675]
    nuc = 2.0546247073036668 # this is Nu of the controlled single RB cell, after the control has been removed
    controlled_array = [nuc, nuc, nuc]
    plt.plot([0,150,350],baseline_array,linewidth=2,linestyle='--', color='blue', label='Baseline')
    plt.plot([0,150,350],controlled_array,linewidth=2,linestyle='--', color='black', label='Controlled')
    #plt.xlim(0,numeps)
    # plt.grid()
    plt.ylabel('$Nu$')
    plt.legend()
    plt.xlim(0,350)
    plt.ylim(1.8,3.0)
    plt.show()
    # os.chdir(general_path+'/../')
    # plt.savefig('FR_4_MARLvsSARL.png')

elif (cas == 6):
    # Plots learning curve (Nu) from a single episode (ep_no), along with the baseine. x-axis is time, along with the removing control 
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.tick_params(direction="in")
    ax.yaxis.set_ticks_position('both')
    # ax.rcParams["figure.figsize"] = (10,6)
    plt.figure(figsize=(10,5))
    plt.rcParams.update({'font.size': 15})

    ep_no = 357 # in drl2
    total_nusselts = np.zeros((nb_actions,))
    for num_env in range(1,nb_envs+1):
        os.chdir(path+'/rewards/environment'+str(num_env)+'/ep_'+str(ep_no))
        nusselts = []
        c = 0
        with open('output_Nusselts.csv', 'r') as f:
            for line in f:
                if (c > 0):
                    _ , envNusselt = line.split(';')
                    nusselts.append(float(envNusselt.strip('\n')))
                c += 1
        if (len(nusselts) != nb_actions):
            print('Length of episode wrong')
        total_nusselts += np.array(nusselts)

    os.chdir(path+'/baseline/')    
    with open('evo_Nusselt_baseline.json', 'r') as f:
        Nusselt_baseline = json.load(f)

    os.chdir(general_path+'/../Postprocess_routines/drl3_ep357/removeControl/baseline/')    
    with open('evo_Nusselt_baseline.json', 'r') as f:   
        Nusselt_removeControl = json.load(f)

    x_bs = np.arange(0, baseline_t-0.5,0.5)-baseline_t
    x_ep = np.arange(1.5,300+1.5,1.5)
    x_rc = np.arange(300+0.5,500,0.5)
    print("Length 1:", len(total_nusselts), len(x_rc))
    x_ep_vid = np.arange(0,600,3)
    x_join_bs = [0.0, 1.5]
    join_line_bs = [Nusselt_baseline[-1], total_nusselts[0]]
    x_join_rc = [300.0, 300.5]
    join_line_rc = [total_nusselts[-1], Nusselt_removeControl[0]]
    
    plt.plot(x_bs,Nusselt_baseline, color='black', linewidth=1)
    plt.plot(x_rc,Nusselt_removeControl, color='black', linewidth=1)
    plt.plot(x_join_bs, join_line_bs, lw = 1, color='black')
    plt.plot(x_join_rc, join_line_rc, lw = 1, color='black')
    plt.plot(x_ep,total_nusselts, lw = 1, color= 'black')
    nno = np.mean(total_nusselts[119:199])
    nno_line = [nno, nno, nno]
    print(nno)
    print(Nusselt_removeControl[-1])
    rem_con = [Nusselt_removeControl[-1],Nusselt_removeControl[-1],Nusselt_removeControl[-1]]
    plt.plot([-50,0,400],nno_line,linewidth=0.8,linestyle='--', color='black', label='Actively controlled')
    plt.plot([-50,0,400],rem_con,linewidth=0.8,linestyle='--', color='red', label='Control Removed')
    x_pts = [51, 102, 114, 120, 123, 126, 141, 252]
    pts = [total_nusselts[33], total_nusselts[67],total_nusselts[75],total_nusselts[79],total_nusselts[81], \
           total_nusselts[83],total_nusselts[93],total_nusselts[167]]
    
    baseline_array = [2.675,2.675,2.675]
    plt.plot([-50,0,400],baseline_array,linewidth=0.8,linestyle='--', color='blue', label='Baseline')
    plt.axvspan(0, 112, color='purple', alpha = 0.5)
    plt.axvspan(112, 130, color='yellow', alpha = 0.5)
    plt.axvspan(130, 300, color='green', alpha = 0.5)
    plt.axvspan(300, 400, color='grey', alpha = 0.5)
    plt.scatter(x_pts,pts, lw = 1, color= 'red', marker='o')
    # print(nno)
    plt.xlabel('Time')
    plt.legend()
    # plt.grid()
    plt.xlim(-50,400)
    plt.ylim(1.78,3)
    plt.ylabel('$Nu$')
    # plt.show()
    # os.chdir(general_path+'/../')
    # print(general_path)
    # plt.savefig('FR_8_mainEpisode.png')


elif (cas == 7):
    # Plots Nu after control is removed

    plt.rcParams.update({'font.size': 15})
    f = plt.figure(figsize=(7,3))
    ax = f.add_subplot(111)
    ax.tick_params(direction="in")
    ax.yaxis.set_ticks_position('both')

    ep_no = 357 # in drl2

    os.chdir(general_path+'/../Postprocess_routines/drl3_ep357/removeControl/baseline/')    
    with open('evo_Nusselt_baseline.json', 'r') as f:   
        Nusselt_removeControl = json.load(f)

    x_rc = np.arange(300+0.5,500,0.5)
    plt.plot(x_rc,Nusselt_removeControl, color='black', linewidth=1)    
    plt.axvspan(300, 500, color='grey', alpha = 0.5)
    # print(nno)
    plt.tight_layout()
    plt.xlabel('Time')
    #plt.ylim(2,2.1)
    plt.ylabel('$Nu$', labelpad=-1)
    #plt.show()
    os.chdir(general_path+'/../')
    print(general_path)
    plt.savefig('controlRemoved.png')