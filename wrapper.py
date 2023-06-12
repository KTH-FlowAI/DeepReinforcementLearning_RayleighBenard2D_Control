#!/bin/env python
#
# DEEP REINFORCEMENT LEARNING FOR RAYLEIGH-BENARD CONVECTION
#
# wrapper.py: Wrapper file that serves as a link between the DRL (tensorforce environments) and the CFD (shenfun).
#
# Colin Vignon
#
# FLOW, KTH Stockholm | 09/04/2023

import matplotlib.pyplot as plt
import os, csv, numpy as np
import shutil
import time
import json
import copy as cp
import sympy
import random
from shenfun import *
from Tfunc import Tfunc 
from rb_marl import RayleighBenard
from channelflow2d import KMM  
from parameters import nb_inv_envs, n_seg, simu_name, num_servers, \
    num_episodes, CFD_params, simulation_params, alphaRestart

x, y, tt = sympy.symbols('x,y,t', real=True)



class Wrapper():


    def __init__(self, episode, general_path):
        
        self.ep = episode
        self.local_path = general_path+'/data/'+simu_name


    def run_CFD(self, simu, env_ID, act, simulation_timeframe, which, evolve, End_episode, t_end_ini, tstep_end_ini):
        ''' for all the invariant environments (of one parallel env),
            run_CFD() runs a unique simulation and gives the results to all of them 
        '''
        np = env_ID[0]
        ep_path = self.local_path+'/CFD_n'+str(np)+'/EP_'+str(self.ep)+'/'
        os.chdir(ep_path)
        
        self.d = CFD_params['dico_d']
        
        if env_ID[1]==1:  # the environment(s) with ID (., 1) launches the CFD simulation(s)
            if act==1:
                alpha = random.random()
                if alpha > alphaRestart and self.ep > 1:
                    os.system('cp '+self.local_path+'/CFD_n'+str(np)+'/EP_'+str(self.ep-1)+'/*.h5 '+ep_path)
                else:
                     os.system('cp '+self.local_path+'/baseline/*.h5 '+ep_path)
                    
                if self.ep > 1:  
                    os.system('rm -r '+self.local_path+'/CFD_n'+str(np)+'/EP_'+str(self.ep-1))
                simu = RayleighBenard(**self.d)
                evolve = False

            # prepare the lower boundary
            actions = self.pull_actions(act, env_ID)
            dicTemp = {'T'+str(i):actions[i] for i in range(n_seg)}
            self.d.update({'bcT':(Tfunc(nb_seg=n_seg, dicTemp=dicTemp).apply_T(y), 1)})

            # Launch simulation
            t_ini, tstep_ini = simu.define_timeframe(which, evolve, \
                                                     self.d.get('bcT'), t_end_ini, tstep_end_ini)
            if act == 1:
                simulation_timeframe = [t_ini, t_ini+simulation_params["delta_t_smooth"]]
            probes_values, t_end_ini, tstep_end_ini = simu.launch(simulation_timeframe, which, \
                                                                  t_ini, tstep_ini, evolve, End_episode)

            # low cost mode: clean useless files
            if act > 1:
                os.system('rm '+ep_path+'Results_ep'+str(self.ep)+'_env_'+\
                          str(np)+'_actuation_'+str(act-1)+'.json')
                os.system('rm '+ep_path+'is_finished'+'_Actuation'+str(act-1)+'.csv')

            # write results  
            file_w = ep_path+'Results_ep'+str(self.ep)+'_env_'+str(np)+'_actuation_'+str(act)+'.json'
            push_info = [t_end_ini, float(tstep_end_ini)]+actions+list(probes_values)
            with open(file_w, 'w') as f:
                json.dump(push_info, f)

            # tell the other environments that the results are ready-to-be-read
            open(ep_path+'is_finished'+'_Actuation'+str(act)+'.csv', 'w').close()  
                        
        else:
            while(not os.path.isfile(ep_path+'is_finished'+'_Actuation'+str(act)+'.csv')):  
                time.sleep(0.05)

            file_r = ep_path+'Results_ep'+str(self.ep)+'_env_'+str(np)+'_actuation_'+str(act)+'.json'
            with open(file_r, 'r') as f:
               pull_info = json.load(f)
            t_end_ini, tstep_end_ini, actions, probes_values = pull_info[0], pull_info[1],\
                  pull_info[2:2+n_seg], pull_info[2+n_seg:]
            
        return probes_values, actions, t_end_ini, tstep_end_ini, simu
        
        

    def run_baseline_CFD(self, base, do_baseline, simulation_timeframe, which):

        if do_baseline == True:
            dico = CFD_params['dico_d']
            dico.update({'moderror':10, 'modsave':10}) 
            base = RayleighBenard(**dico)
            t_ini, tstep_ini = base.define_timeframe(which)
            probes_values, t_end_ini, tstep_end_ini = base.launch(simulation_timeframe, \
                                                                  which, t_ini, tstep_ini)
            evo_Nusselt_baseline, evo_kinEn_baseline = base.instant_Nusselt, base.instant_kinEn

            with open('evo_Nusselt_baseline.json', 'w') as f:
                json.dump(evo_Nusselt_baseline, f)
            with open('evo_kinEn_baseline.json', 'w') as f2:
                json.dump(evo_kinEn_baseline, f2)

            info_baseline = [t_end_ini, tstep_end_ini]+list(probes_values)
            with open('data_baseline.json', 'w') as f3:
                json.dump(info_baseline, f3)
        else:
            with open('data_baseline.json', 'r') as f3:
                info_baseline = json.load(f3)
            t_end_ini, tstep_end_ini, probes_values = info_baseline[0], \
                info_baseline[1], info_baseline[2:]
        
        return probes_values, t_end_ini, tstep_end_ini, base
        
    
    def env1_give_answers(self, Port, env_ID):
        for par_env in range(num_servers):
            for inv_env in range(2,nb_inv_envs+1):
                c = 0
                while c<30:
                    try:
                        socket_client((par_env+1, inv_env), PORT=Port-\
                                      (inv_env+nb_inv_envs*par_env))
                        c+=50
                    except:
                        time.sleep(0.1)
                        c +=1


    def other_wait_env1(self, Port, env_ID):

        socket_server(PORT=Port-(env_ID[1]+nb_inv_envs*(env_ID[0]-1)))
             

        
    def merge_actions(self, action, actuation, env_ID):
        
        CFD_path = self.local_path+'/CFD_n'+str(env_ID[0])
        ep_path = CFD_path+'/EP_'+str(self.ep)
        
        if env_ID[1]==1:  
            if actuation==1:
                if self.ep==1:
                    os.mkdir(CFD_path)
                os.mkdir(ep_path)
            
            act_file = ep_path+'/Actions_ep'+str(self.ep)+'_env'+str(env_ID[0])+\
                '_actuation'+str(actuation)+'.csv'
            with open(act_file, 'a') as csv_file:
                spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                spam_writer.writerow([env_ID[1], action])
  
        else:
            while(not os.path.exists(ep_path)):
                time.sleep(0.1)
            act_file = ep_path+'/Actions_ep'+str(self.ep)+'_env'+str(env_ID[0])+\
                '_actuation'+str(actuation)+'.csv'
            with open(act_file, 'a') as csv_file:
                spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                spam_writer.writerow([env_ID[1], action])

            
            
    def pull_actions(self, actuation, env_ID):
        
        ep_path = self.local_path+'/CFD_n'+str(env_ID[0])+'/EP_'+str(self.ep)
        Flag = False
        while Flag == False:
            with open(ep_path+'/Actions_ep'+str(self.ep)+'_env'+str(env_ID[0])+\
                    '_actuation'+str(actuation)+'.csv', 'r') as f:
                Flag = (len(f.readlines())==n_seg)
               
        actions = {}        
        with open(ep_path+'/Actions_ep'+str(self.ep)+'_env'+str(env_ID[0])+\
                  '_actuation'+str(actuation)+'.csv', 'r') as fbis:        
            for line in fbis:
                segment, action = line.split(';')
                action = action.strip('\n')
                action = action.strip('[]')
                actions.update({segment:float(action)})
    
        Actions = []
        for i in range(n_seg): 
            Actions.append(actions.get(str(i+1)))
        
        return Actions
                    
                    
                    
                    
                    
                    
        
