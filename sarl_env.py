from shenfun import *
import matplotlib.pyplot as plt
import sympy
import os, csv, numpy as np
import shutil
import time
import json
from tensorforce.environments import Environment
from mpi4py import MPI
from Tfunc import Tfunc
from rayleighbenard2d import RayleighBenard
from channelflow2d import KMM
from parameters import case, simulation_params, reward_function, optimization_params, output_params, nb_proc, nb_actuations, nb_actuations_deterministic, n_seg, simu_name
from env_utils import run_subprocess


general_path = os.getcwd()
case_path = general_path+'/data/'+simu_name
os.chdir(case_path)



np.warnings.filterwarnings('ignore')

# pylint: disable=attribute-defined-outside-init

x, y, tt = sympy.symbols('x,y,t', real=True)

comm = MPI.COMM_SELF

class Environment2D(Environment):

    def __init__(self, simu_name, path, number_steps_execution=1, do_baseline=True, continue_training=False, deterministic=False, ENV_ID=-1, host='', node=None, inv_ID = 1, nb_inv_envs = 1):
                 
        #cr_start('ENV.init',0)
        self.simu_name = simu_name
        self.general_path = path
        self.case_path = self.general_path+'/data/'+self.simu_name
        self.case      = case
        self.ENV_ID    = ENV_ID
        self.host      = host
        self.node      = node
        
        self.number_steps_execution = number_steps_execution
        self.reward_function        = reward_function
        self.output_params          = output_params
        self.optimization_params  = optimization_params
        
        self.simulation_timeframe = simulation_params["simulation_timeframe"]
        self.last_time            = round(self.simulation_timeframe[1],3)
        self.delta_t_smooth       = simulation_params["delta_t_smooth"]
        #self.smooth_func          = simulation_params["smooth_func"]


        # Others
        self.n_seg = n_seg
        N = (64, 96)  # (cartesian) meshgrid
        self.obsGrid = output_params['nb_probes']  # (cartesian) grid of observation probes
        self.dicTemp = {'T0':1., 'T1':1., 'T2':1., 'T3':1., 'T4':1., 'T5':1., 'T6':1., 'T7':1., 'T8':1., 'T9':1.}  # starting temperatures
        self.d = {
     	    'N': N,
            'Ra': 10000.,
     	    'Pr': 0.7,
     	    'dt': 0.05,
     	    'filename': f'RB_{N[0]}_{N[1]}',
     	    'conv': 0,
     	    'modplot': 1000,
     	    'obsGrid':self.obsGrid,
    	    'moderror': 10000,
    	    'modsave': 10000,
     	    'bcT': (Tfunc(nb_seg=n_seg, dicTemp=self.dicTemp).apply_T(y), 1),  # Tfunc: Temperature function of the lower boundary (enforce ten different temperatures on ten segments)
     	    'family': 'C',
     	    'checkpoint': 10,
     	    #'padding_factor': 1,
     	    'timestepper': 'IMEXRK3'
     	    }
        self.simu = None
        
        #postprocess values
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
        #cr_stop('ENV.init',0)


    ## Start baseline_flow from scratch, creating cfd setup
    
    def start_class(self):

        t0 = time.time()
        
        self.clean(True)
        self.create_baseline()
        
        #----------------------------------------------------------------------
        # Run the first case
        t0 = time.time()
        
        #shenfun run in baseline
        self.run(which = 'reset')
        print("Done. time elapsed : ", time.time() - t0)
        
        # Get the new avg drag and lift and SAVE 
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
                
                ## Best model at the end of each episode
                if os.path.exists("best_model"):
                    run_subprocess('./','rm -r','best_model')
            
            
        else:
            self.action_count = 1
            
     
    def create_baseline(self):
        if self.do_baseline == True:
            os.mkdir(self.case_path+'/baseline')
           
          
    def run(self, which, evolve = False): 
                
        if which == 'baseline':
            os.chdir(self.case_path+'/baseline')

            self.base = RayleighBenard(**self.d)
            if self.do_baseline == True:
                data_reward, self.probes_values, self.t_end_ini, self.tstep_end_ini = self.base.launch(self.simulation_timeframe, which)
                self.t_end_baseline, self.tstep_end_baseline = self.t_end_ini, self.tstep_end_ini
                evo_Nusselt_baseline = self.base.instant_Nusselt
                evo_kinEn_baseline = self.base.instant_kinEn
                with open('evo_Nusselt_baseline.json', 'w') as f:
                    json.dump(evo_Nusselt_baseline, f)
                with open('evo_kinEn_baseline.json', 'w') as f2:
                    json.dump(evo_kinEn_baseline, f2)
                    
                info_baseline = [data_reward[0], data_reward[1], data_reward[2], self.t_end_ini, self.tstep_end_ini]+list(self.probes_values)
                with open('data_baseline.json', 'w') as f3:
                    json.dump(info_baseline, f3)
            else:
                with open('data_baseline.json', 'r') as f3:
                    info_baseline = json.load(f3)
                Nusselt, kinEn, meanFlow, self.t_end_ini, self.tstep_end_ini, self.probes_values = info_baseline[0], info_baseline[1], info_baseline[2], info_baseline[3], info_baseline[4], info_baseline[5:]
                data_reward = [Nusselt, kinEn, meanFlow]  
                self.t_end_baseline, self.tstep_end_baseline = self.t_end_ini, self.tstep_end_ini
            print('baseline run finished')
            os.chdir(self.case_path)          
            
        elif which == 'reset':
            os.chdir(self.case_path+'/baseline')
            os.system('mkdir -p logs') 
            self.run('baseline')

        elif which == 'execute':
            casepath = os.path.join(self.case_path,'%s'%self.host,'EP_%d'%self.episode_number)
            logsrun  = os.path.join('logs','log_last_execute_run.log')
            logssets = os.path.join('logs','log_sets.log')
            run_subprocess(casepath,'mkdir -p','logs') 
            os.chdir(casepath)
           
            if self.action_count ==1:
                self.d.update({'moderror':10000, 'modsave':10000})
                self.simu = RayleighBenard(**self.d)
                evolve = False
                
            end_episode = (self.action_count == nb_actuations)
            data_reward, self.probes_values, self.t_end_ini, self.tstep_end_ini = self.simu.launch(self.simulation_timeframe, which, evolve, end_episode, self.d.get('bcT'), self.t_end_ini, self.tstep_end_ini)
            os.chdir(self.case_path)
            return data_reward, self.probes_values



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
                    os.mkdir("saved_models")
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
                if(os.path.exists("saved_models/output.csv")):
                    if(not os.path.exists("best_model")):
                        shutil.copytree("saved_models", "best_model")
                    else:
                        with open("saved_models/output.csv", 'r') as csvfile:
                            data = csv.reader(csvfile, delimiter = ';')
                            for row in data:
                                lastrow = row
                            last_iter = lastrow[1]
                        with open("best_model/output.csv", 'r') as csvfile:
                            data = csv.reader(csvfile, delimiter = ';')
                            for row in data:
                                lastrow = row
                            best_iter = lastrow[1]
                        if float(best_iter) < float(last_iter):
                            if(os.path.exists("best_model")):
                                shutil.rmtree("best_model")
                            shutil.copytree("saved_models", "best_model")
                        
            
        
         
    def save_this_action(self):
        
        name_a = "output_actions.csv"
        if(not os.path.exists("actions")):
            os.mkdir("actions")    
        if(not os.path.exists("actions/{}".format(self.host))):
            os.mkdir("actions/{}".format(self.host))
        if(not os.path.exists("actions/{}/ep_{}/".format(self.host, self.episode_number))):
            os.mkdir("actions/{}/ep_{}/".format(self.host, self.episode_number))
        
        path_a = "actions/{}/ep_{}/".format(self.host, self.episode_number)
        action_line = "{}".format(self.action_count)
        for i in range(self.n_seg):
            action_line = action_line + "; {}".format(self.action[i])
        
        if(not os.path.exists(path_a+name_a)):
            header_line = "Action"
            for i in range(self.n_seg):
                header_line = header_line + "; Segment_{}".format(i+1)  # TODO: "Jet" -> modify
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
            os.mkdir("rewards")
            
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
 
        
    def save_final_reward(self,reward):  # TODO: nothing to change ?
    
        name_a = "output_final_rewards.csv"
        
        if(not os.path.exists("final_rewards")):
            os.mkdir("final_rewards")
        time.sleep(0.5)
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
        runbin  = 'cp -r'
        runargs = self.case_path+'/baseline %s'%os.path.join('%s'%self.host,'EP_%d'%self.episode_number)
        logs    = os.path.join(self.case_path+'/baseline','logs','log_restore_last_episode.log')
        run_subprocess(runpath,runbin,runargs,log=logs)
        
        self.action = np.zeros(self.n_seg)
        self.t_end_ini, self.tstep_end_ini = self.t_end_baseline, self.tstep_end_baseline
         
        if(self.episode_number>1):
            runbin  = 'rm -r'
            runargs = os.path.join('%s'%self.host,'EP_%d'%(self.episode_number-1))
            if self.deterministic == False:
               run_subprocess(runpath,runbin,runargs)
    
    def create_cpuID(self):
        os.chdir(self.general_path)
        runpath = self.case_path
        runbin  = 'mkdir'
        if self.deterministic == False:
            runargs = self.host
            run_subprocess(runpath,runbin,runargs)
            
            name = "nodes"
            if(not os.path.exists(self.case_path+"/{}/".format(self.host)+name)):
                with open(self.case_path+"/{}/".format(self.host)+name, "w") as csv_file:
                    spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                    spam_writer.writerow(["Nodes in this learning"])
                    spam_writer.writerow([self.node])
            else:
                with open(self.case_path+"/{}/".format(self.host)+name, "a") as csv_file:
                    spam_writer=csv.writer(csv_file, delimiter=";", lineterminator="\n")
                    spam_writer.writerow(["Nodes in this learning"])
                    spam_writer.writerow([self.node])
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
        
        """Action is a list of n_seg capped values of Temp"""
        return dict(type='float',
                    shape=(self.n_seg), 
                           min_value=self.optimization_params["min_ampl_temp"],
                           max_value=self.optimization_params["max_ampl_temp"]
                    )
    
        
                
    def execute(self, actions):
        
        self.action = actions        
        self.last_time = self.simulation_timeframe[1]
        t1 = round(self.last_time,3)
        t2 = t1 + self.delta_t_smooth
            
        self.simulation_timeframe = [t1,t2]
        
        simu_path = os.path.join(self.case_path,'%s'%self.host,'EP_%d'%self.episode_number)
        if case == 'RB_2D':
            for i in range(self.n_seg):

                self.dicTemp.update({'T'+str(i):self.action[i]})  
                
                
        self.d.update({'bcT':(Tfunc(nb_seg=n_seg, dicTemp=self.dicTemp).apply_T(y), 1)})
        t0 = time.time()
        data_reward, self.probes_values = self.run('execute', evolve=True)

        self.history_parameters['Nusselt'].extend([data_reward[0]])
        self.history_parameters['kinEn'].extend([data_reward[1]])
        self.history_parameters["time"].extend([self.last_time])
        self.history_parameters["episode_number"].extend([self.episode_number])
        self.save_history_parameters(nb_actuations)
        # Write the action
        self.save_this_action()
        # Compute the reward
        reward = self.compute_reward(data_reward)

        self.save_reward(reward, data_reward[0])
       
        self.action_count += 1
        
        if self.deterministic == False and self.action_count <= nb_actuations:
            terminal = False  
        elif self.deterministic == True and self.action_count <= nb_actuations_deterministic:
            terminal = False  
        else:
            terminal = True   
            
            self.save_final_reward(reward)
            time.sleep(0.1)
           
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
        self.simulation_timeframe = simulation_params["simulation_timeframe"]
        t1 = self.simulation_timeframe[0]
        t2 = self.simulation_timeframe[1]
        self.simulation_timeframe = [t1,t2]
        
        # Advance in episode
        self.episode_number += 1
        if self.deterministic == True:
            self.host = 'deterministic'
        
        # Copy the baseline in the environment directory     
        if self.action_count == 1:
            self.recover_start()
        
        NWIT_TO_READ=1 

        filename     = os.path.join('shenfun_files','%s'%self.host,'EP_%d'%self.episode_number,'%s.nsi.wit'%self.case)

        probes_value = self.probes_values
        return probes_value      


    def compute_reward(self, data):
        if self.reward_function == 'Nusselt':  
            reward = self.optimization_params["norm_reward"]*(-data[0] + self.optimization_params["offset_reward"])
        elif self.reward_function == 'kinEn':  
            reward = self.optimization_params["norm_reward"]*(-data[1] + self.optimization_params["offset_reward"])
        elif self.reward_function == 'meanFlow':  
            reward = self.optimization_params["norm_reward"]*(-data[2] + self.optimization_params["offset_reward"])
        else:
            print("ERROR: Choose 'Nusselt' or 'kinEn' or 'meanFlow' for the reward function") 
        return reward         