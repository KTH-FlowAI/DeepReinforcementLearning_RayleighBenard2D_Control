#!/bin/env python
#
# DEEP REINFORCEMENT LEARNING FOR RAYLEIGH-BENARD CONVECTION
#
# marl_env.py: Defines the tensorforce environments and adapts them for use in the MARL framework.
#
# Colin Vignon & Joel Vasanth
#
# FLOW, KTH Stockholm | 09/04/2023

from shenfun import *
import matplotlib.pyplot as plt
import sympy
import os, csv, numpy as np
import shutil
import time
import json

# Environment
from tensorforce.environments import Environment
from mpi4py import MPI
from wrapper import Wrapper
from reward_functions import compute_reward
import copy as cp

from parameters import CFD_params, simulation_params, reward_function, optimization_params, \
    nb_proc, nb_actuations, nb_actuations_deterministic, simu_name, nb_inv_envs
from env_utils import run_subprocess


general_path = os.getcwd()
case_path = general_path+'/data/'+simu_name
os.chdir(case_path)

np.warnings.filterwarnings('ignore')

x, y, tt = sympy.symbols('x,y,t', real=True)
comm = MPI.COMM_SELF

class Environment2D(Environment):

    def __init__(self, simu_name, path, number_steps_execution=1, do_baseline=True, \
                 continue_training=False, deterministic=False, ENV_ID=[1,1], host='', node=None):
                 
        self.simu_name = simu_name
        self.general_path = path
        self.case_path = self.general_path+'/data/'+self.simu_name
        self.ENV_ID    = ENV_ID
        self.nb_inv_envs = nb_inv_envs
        self.host      = host
        self.node      = node
        
        self.number_steps_execution = number_steps_execution
        self.reward_function        = reward_function
        self.optimization_params  = optimization_params
        
        self.simulation_timeframe = simulation_params["simulation_timeframe"]
        self.last_time            = round(self.simulation_timeframe[1],3)
        self.delta_t_smooth       = simulation_params["delta_t_smooth"]


        #### CFD-related attributes
         # number of segments on the lower boundary
        self.n_acts = CFD_params['number_of_actuators'] 
        # (cartesian) grid of probes
        self.obsGrid = CFD_params['dico_d'].get('obsGrid')  
        # number of columns of probes per invariant environment
        self.hor_inv_probes = CFD_params.get('hor_inv_probes')  

        self.simu = None
        self.base = None

        # postprocess values
        self.history_parameters = {}
        self.history_parameters['Nusselt'] = []
        self.history_parameters['kinEn'] = []
        self.history_parameters["time"] = []
        self.history_parameters["episode_number"] = []
        
        name="output.csv"
        # if we start from other episode already done
        last_row = None
        if(os.path.exists("saved_models/"+name)):
            with open("saved_models/"+name, 'r') as f:
                for row in reversed(list(csv.reader(f, delimiter=";", lineterminator="\n"))):
                    last_row = row
                    break
        if(not last_row is None):
            self.episode_number = int(last_row[0])
            self.last_episode_number = int(last_row[0])
        else:
            self.last_episode_number = 0
            self.episode_number = 0
            
        self.episode_Nusselts = np.array([])
        self.episode_kinEns = np.array([])
        
        self.do_baseline = do_baseline
        self.continue_training = continue_training
        self.deterministic = deterministic
        
        self.start_class()
        self.do_baseline = True


        #start episodes        
        super().__init__()


    #### Start baseline_flow from scratch, creating cfd setup
    def start_class(self):

        
        self.clean(True)
        self.create_baseline()
        t0 = time.time()
        self.run(which = 'reset', ep = None)
        print("Done. Time elapsed : ", np.round(time.time() - t0, 3)," seconds\n")
        
        self.action_count=0
        if self.continue_training == True or self.deterministic == True:
            temp_id = '{}'.format(self.host)
        else:
            temp_id = ''
        
        self.check_id = True

    
    def clean(self,full):
        
        if full:
            if self.do_baseline == True:
                if os.path.exists("saved_models"):
                    run_subprocess('./','rm -r','saved_models')        
        else:
            self.action_count = 1
            
    def create_baseline(self):
        if self.do_baseline == True:
            
            os.mkdir(self.case_path+'/baseline')
            
          
    def run(self, which, ep, evolve=False): 
         # Run a simulation with shenfun
            
        if which == 'baseline':
            os.chdir(self.case_path+'/baseline')
            wrap = Wrapper(ep, self.general_path)
            print("Running baseline simulation ... ")
            self.probes_values, self.t_end_ini, self.tstep_end_ini, \
                self.base = wrap.run_baseline_CFD(self.base, self.do_baseline, \
                                                  self.simulation_timeframe, which)
            self.t_end_baseline, self.tstep_end_baseline = self.t_end_ini, self.tstep_end_ini
            os.chdir(self.case_path)
            
        elif which == 'reset':
            os.chdir(self.case_path+'/baseline')
            os.system('mkdir -p logs') # Create logs folder
            self.run('baseline', ep)

        elif which == 'execute':
            wrap = Wrapper(ep, self.general_path)
            end_episode = (self.action_count == nb_actuations)
            self.probes_values, self.actions, self.t_end_ini, self.tstep_end_ini, \
                self.simu = wrap.run_CFD(self.simu, self.ENV_ID, self.action_count, \
                                         self.simulation_timeframe, which, evolve, \
                                         end_episode, self.t_end_ini, self.tstep_end_ini)
            
            if self.action_count == 1:
                self.simulation_timeframe = [self.t_end_ini-self.delta_t_smooth, self.t_end_ini]            
            os.chdir(self.case_path)
            
            return self.probes_values



    def save_history_parameters(self, nb_actuations):
        # Save at the end of every episode

        self.episode_Nusselts = np.append(self.episode_Nusselts, self.history_parameters['Nusselt'])
        self.episode_kinEns = np.append(self.episode_kinEns, self.history_parameters['kinEn'])        
        
        if self.action_count == nb_actuations or self.episode_number == 0:
            self.last_episode_number = self.episode_number
            last_instant_Nusselt = self.history_parameters['Nusselt'][-1]
            last_instant_kinEn = self.history_parameters['kinEn'][-1]
                        
            if self.do_baseline == True:
                name = "output.csv"
                if(not os.path.exists("saved_models")):
                    try:
                        os.mkdir("saved_models")
                    except:
                        pass
                if(not os.path.exists("saved_models/"+name)):
                    with open("saved_models/"+name, "w") as csv_file:
                        spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                        if self.reward_function == 'Nusselt': 
                            spam_writer.writerow(["Episode", "instantNusselt"])
                            spam_writer.writerow([self.last_episode_number, last_instant_Nusselt])
                        else: 
                            spam_writer.writerow(["Episode", "instant_kinEn"])
                            spam_writer.writerow([self.last_episode_number, last_instant_kinEn])
                else:
                    with open("saved_models/"+name, "a") as csv_file:
                        spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                        if self.reward_function == 'Nusselt':
                            spam_writer.writerow([self.last_episode_number, last_instant_Nusselt])
                        else:
                            spam_writer.writerow([self.last_episode_number, last_instant_kinEn])
            self.episode_Nusselts = np.array([])
            self.episode_kinEns = np.array([])
            
            if self.do_baseline == True:
                pass
                        
            

    def save_this_action(self):
        
        if self.ENV_ID[1] == 1:
            name_a = "output_actions.csv"
            if(not os.path.exists("actions")):
                try:
                    os.mkdir("actions")    
                except:
                    pass
            if(not os.path.exists("actions/ep_{}/".format(self.episode_number))):
                os.mkdir("actions/ep_{}/".format(self.episode_number))

            path_a = "actions/ep_{}/".format(self.episode_number)
            action_line = "{}".format(self.action_count)
            for i in range(self.n_acts):
                action_line = action_line + "; {}".format(self.actions[i])

            if(not os.path.exists(path_a+name_a)):
                header_line = "Action"
                for i in range(self.n_acts):
                    header_line = header_line + "; Segment_{}".format(i+1)
                with open(path_a+name_a, "w") as csv_file:
                    spam_writer=csv.writer(csv_file, lineterminator="\n")
                    spam_writer.writerow([header_line])
                    spam_writer.writerow([action_line])
            else:
                with open(path_a+name_a, "a") as csv_file:
                    spam_writer=csv.writer(csv_file, lineterminator="\n")
                    spam_writer.writerow([action_line])        


    def save_reward(self,reward, Nusselt):
                
        name_a = "output_rewards.csv"
        name_N = "output_Nusselts.csv"  

        if(not os.path.exists("rewards")):
            try:
                os.mkdir("rewards")
            except:
                pass
        if(not os.path.exists("rewards/{}".format(self.host))):
            os.mkdir("rewards/{}".format(self.host))
            
        if(not os.path.exists("rewards/{}/ep_{}/".format(self.host, self.episode_number))):
            os.mkdir("rewards/{}/ep_{}/".format(self.host, self.episode_number))
            
        path_a = "rewards/{}/ep_{}/".format(self.host, self.episode_number)
        
        if(not os.path.exists(path_a+name_a)):
                with open(path_a+name_a, "w") as csv_file:
                    spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                    spam_writer.writerow(["Action", "Reward"])
                    spam_writer.writerow([self.action_count, reward])
                with open(path_a+name_N, "w") as csv_file:
                    spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                    spam_writer.writerow(["Action", "Nusselt"])
                    spam_writer.writerow([self.action_count, Nusselt])
        else:
                with open(path_a+name_a, "a") as csv_file:
                    spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                    spam_writer.writerow([self.action_count, reward])
                with open(path_a+name_N, "a") as csv_file:
                    spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                    spam_writer.writerow([self.action_count, Nusselt])
 
       
    def save_final_reward(self,reward): 
    
     
        name_a = "output_final_rewards.csv"
        
        if(not os.path.exists("final_rewards")):
            try:
                os.mkdir("final_rewards")
            except:
                pass
        if(not os.path.exists("final_rewards/{}".format(self.host))):
            os.mkdir("final_rewards/{}".format(self.host))
            
        path_a = "final_rewards/{}/".format(self.host)
        
        if(not os.path.exists(path_a+name_a)):
            with open(path_a+name_a, "w") as csv_file:
                spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                spam_writer.writerow(["EPISODE", "REWARD"])
                spam_writer.writerow([self.episode_number, reward])
        else:
            with open(path_a+name_a, "a") as csv_file:
                spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                spam_writer.writerow([self.episode_number, reward])
      
   
    def save_comms_probes(self): 
         
        name_a = "output_probes_comms.csv"
        
        if(not os.path.exists("probes_comms")):
            os.mkdir("probes_comms")
            
        if(not os.path.exists("probes_comms/ep_{}/".format(self.episode_number))):
            os.mkdir("probes_comms/ep_{}/".format(self.episode_number))
            
        path_a = "probes_comms/ep_{}/".format(self.episode_number)
        
        if(not os.path.exists(path_a+name_a)):
                with open(path_a+name_a, "w") as csv_file:
                    spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                    array_acts = np.linspace(1, 24, dtype=int) 
                    spam_writer.writerow(["Action", array_acts])
                    spam_writer.writerow([self.action_count, self.probes_values])
        else:
                with open(path_a+name_a, "a") as csv_file:
                    spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                    spam_writer.writerow([self.action_count, self.probes_values])
 

    def recover_start(self): 
        runpath = self.case_path
        self.actions = np.zeros(self.n_acts)
        self.action = 0 
        self.t_end_ini, self.tstep_end_ini = self.t_end_baseline, self.tstep_end_baseline

    def create_cpuID(self):
        os.chdir(self.general_path)
        runpath = self.case_path
        runbin  = 'mkdir'
        if self.deterministic == False:
            pass
        else:
            runargs = 'deterministic'
            run_subprocess(runpath,runbin,runargs,check_return=False)
               
          
    def close(self):
        super().close()       
       

    def states(self):

        return dict(type='float',
                    shape=(3*self.obsGrid[0]*self.obsGrid[1], )
                    ) 
    

    def actions(self):
        
        """ Actions now correspond to the temperature of one segment. 
            All the actions are then gathered thanks to Wrapper().merge_actions()
            Return: dict_values(['float', 1, -1, 1]) (e.g)
        """
        
        return dict(type='float',
                    shape=(1), 
                           min_value=self.optimization_params["min_ampl_temp"],
                           max_value=self.optimization_params["max_ampl_temp"]
                    )
    
        
                
    def execute(self, actions):

         
        self.action = actions 
        wrap = Wrapper(self.episode_number, self.general_path)
        wrap.merge_actions(self.action, self.action_count, self.ENV_ID)

        
        self.last_time = self.simulation_timeframe[1]
        t1 = round(self.last_time,5)
        t2 = t1 + self.delta_t_smooth
            
        self.simulation_timeframe = [t1,t2]
        
        # Start a run
        t0 = time.time()
       
        # Run + get the new Nusselt: self.run give the global data, 
        # self.recentre_obs recenters the data according to the invariant env
        self.probes_values = self.recentre_obs(self.run('execute', self.episode_number, evolve=True))

        # Compute the reward
        reward, gen_Nus, gen_kinEn = compute_reward(self.probes_values, self.reward_function)
        self.history_parameters['Nusselt'].extend([gen_Nus])
        self.history_parameters['kinEn'].extend([gen_kinEn])
        self.history_parameters["time"].extend([self.last_time])
        self.history_parameters["episode_number"].extend([self.episode_number])
        self.save_history_parameters(nb_actuations)
        
        # Write the action
        self.save_this_action()
        
        self.save_reward(reward, gen_Nus)
        
        self.action_count += 1
        
        if self.deterministic == False and self.action_count <= nb_actuations:
            terminal = False  
        elif self.deterministic == True and self.action_count <= nb_actuations_deterministic:
            terminal = False  
        else:
            terminal = True   
            
            # write the last rewards at each episode to see the improvement 
            self.save_final_reward(reward)
        
        return self.probes_values, terminal, reward
        
        
    def reset(self):
        """Reset state"""
        
        # Create a folder for each environment
        if self.check_id == True:
            self.create_cpuID()
            self.check_id = False
        
        # Clean
        self.clean(False)
        
        # Apply new time frame
        t1 = simulation_params["simulation_timeframe"][0]
        t2 = simulation_params["simulation_timeframe"][1]
        self.simulation_timeframe = [t1,t2]
        
        # Advance in episode
        self.episode_number += 1
        if self.deterministic == True:
            self.host = 'deterministic'
        
        # Copy the baseline in the environment directory     
        if self.action_count == 1:
            self.recover_start()
        
        return self.probes_values
        

    def recentre_obs(self, probes_values):

        ''' This function is aimed at centering the data around the environment-segment 
        (1 env is attached to the behaviour of 1 segment)
        '''

        obs_array = np.array(probes_values).reshape(3, self.obsGrid[0], self.obsGrid[1])
        centered_array = np.zeros((3, self.obsGrid[0], self.obsGrid[1]))
        ux = obs_array[0]
        uy = obs_array[1]
        Temp = obs_array[2]
        
        ind = ((self.ENV_ID[1]-(nb_inv_envs-nb_inv_envs//2))%nb_inv_envs)*self.hor_inv_probes

        centered_array[0] = np.array(list(ux.T)[ind:]+list(ux.T)[:ind]).T
        centered_array[1] = np.array(list(uy.T)[ind:]+list(uy.T)[:ind]).T        
        centered_array[2] = np.array(list(Temp.T)[ind:]+list(Temp.T)[:ind]).T       
        centered_list = list(centered_array.reshape(3*self.obsGrid[0]*self.obsGrid[1],))

        return centered_list