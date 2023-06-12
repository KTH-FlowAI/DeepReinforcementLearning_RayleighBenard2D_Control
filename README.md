# DeepReinforcementLearning_RayleighBenard2D_Control

Control of 2D Rayleigh Benard Convection using Deep Reinforcement Learning with Tensorforce and Shenfun.

Welcome to the repository DeepReinforcementLearning_RayleighBenard2D_Control. Here, you will find the scripts used for the control of two-dimensional Rayleigh Benard convection (RBC) using deep reinforcement learning (DRL). The computational fluid dynamics (CFD) solver employed is the spectral Galerkin code-suite 'shenfun' ([Documentation](https://shenfun.readthedocs.io/en/latest/)), which solves the flow field for the RBC. The DRL is solved using learning libraries from 'Tensorforce' ([Documentation](https://tensorforce.readthedocs.io/en/latest/)); specifically, the Proximal Policy Optimisation (PPO) algorithm is used. You will need shenfun and tensorforce installed on your system to run the code.

DRL is applied here using two different frameworks:
- Single agent reinforcement learning (SARL), and
- Multi-agent reinforcement learning (MARL).

Comparative studies for the performance of both frameworks may be performed if desired. Separate scripts that apply each framework are provided in the repository and are labelled with differently. These labels are further described below. To run either framework, you would have to run its corresponding script.

Installation instructions, a short tutorial, and links to pre-computed results for validation are given below.

## Installation

Use conda to install Tensorforce and shenfun in a virtual environment. If you dont have conda installed, you will have to do that first. You can find instructions for that [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). Once installed, follow the steps below in your terminal (applies to Linux users):

Create and activate virtual environment called 'drlrbc2d'. You can name the environment whatever you like:
```bash
conda create --name drlrbc2d -c conda-forge python=3.8.10
conda activate
```

Install tensorforce:
```bash
pip3 install tensorforce
```

Clone shenfun from its [repository](https://github.com/spectralDNS/shenfun.git), and then install it.
```bash
git clone https://github.com/spectralDNS/shenfun.git
cd shenfun
pip3 install -e .
```

Install a few more necessary packages, and set the python path variable as the shenfun folder.
```bash
conda install -c conda-forge mpich cython quadpy
export PYTHONPATH=path/to/shenfun/folder/:$PYTHONPATH
```

Install keras and protobuf:
```bash
pip3 install keras==2.6 
pip3 install protobuf==3.20
```

It is important to note that the requirement for the successful installation of the above is a working Linux operating system from any major distribution (Debian, CentOS , Redhat or Ubuntu)

## Repository description

The basic structure of the code is a main launcher script that launches the baseline and training episodes together, which call functions from some other scripts creating and defining the tensorforce (TF) environments, which in-turn call functions from other scripts that run the CFD. The remaining scripts not related to the main running of the program are meant for post-processing to visualise the results.

Some scripts are common to the SARL and MARL frameworks, the others are specific to each. These are described below. 

### Scripts specific to either SARL or MARL

#### Launcher scripts
**train_marl.py**: This is the main launcher script for the MARL framework
**train_sarl.py**: This is the main launcher script for the SARL framework
Both scripts above instantiate the TF environments calling functions from scripts in the next subsection below. Following this, they create a TF agent, create a TF runner, instantiate parallel environments and executes the runner. 

#### Scripts to instantiate TF environments
**marl_env.py**: For MARL - Defines several TF environments, which in the MARL framework are called pseudo-environments, and correspond to each individual agent. 
**sarl_env.py**: For SARL - defines a single TF environment used in the SARL framework.
All functionality in either of the scripts above is encapsulated in a class called `Environment2D()', which contains functions such as `init()', `clean()', `run()', `execute()', etc, which are typical TF functions, modified for use in RBC for the MARL or SARL.

#### Wrapper script
**wrapper.py**: This applies to the MARL framework only. The wrapper acts as a link between shenfun (the CFD) and TF. There are three main functions here:
- run_baseline_CFD(): runs a baseline simulation
- run_CFD(): runs training episodes, for the duration of a single action. 
- merge_actions(): merges actions from each control segment.
Each of these are called in a controlled manner from `marl_env.py'.

Note that all of the functionality done by the wrapper applies to the MARL only. For SARL, the wrapper's functions are included in `sarl_env.py' itself and does not use an exclusive wrapper script.

### Scripts common to SARL and MARL
**channelflow2d.py**: solver script from shenfun that solves the channel flow problem.
**rayleighbenard2d.py**: solver script from shenfun that solves the RB problem. The main class of this file inherits from the class of **channelflow2d.py**, since the problem at hand involves RB convection in a channel.
**Tfunc.py**: script that defines and modifies temperature profiles on bottom wall at each actuation step of the training episodes.
**reward_function.py**: computes the cumulative reward from local rewards from each pseudo-environment. Also computes the Nusselt number and kinetic energy.
**env_utils.py**: miscellaneous utility functions.

### Folders
**parameters**: this contains files that store parameters for the simulations. Each parameter file has the prefix `parameter_' to its name. What follows this prefix is the name of the specific case that is run. Multiple parameter files may be created for several different runs.
**data**: This is the folder where all of the output files are stored during and after each run. 

## Tutorial

### Modifying parameters
To execute the code, first, you must modify the parameters file to contain parameters of your choice. The main parameters that you can change are given below along with default values which were used in the paper (LINKTOPAPER) in parentheses - 

- name of the case to be run (RB_2D_MARL for MARL and RB_2D_SARL for SARL)
- the simulation time for the baseline run (400) 
- the number of episodes for the training run (3500 for MARL, 350 for SARL) 
- the duration of each action in any episode (1.5)
- the number of actions per episode (200)
- the number of environments that run in parallel (1)
- the number of observation probes (8,32)
- the size of the simulation mesh grid (64,96)

The parameters files in the `parameters/' folder are captioned and variables that correspond to the values above are clearly named, so you can find them easily.

### Execution
To execute the DRL using the SARL framework use the command :
```bash
python train_sarl.py
```

or alternatively to execute using MARL code, use the command - 
```bash
python train_marl.py
```

You should see outputs on your terminal screen indicating the run of the baseline simulation followed by the training phase.

### Output
The data written while the run proceeds is stored in the `data/' folder. First, a folder in the `data/' folder named with the case name is created, and all of the data corresponding to that case is written there.

#### Basline results
The `baseline/' folder contains these results. The Nusselt number as a function of time is stored in `evo_Nusselt_baseline.json'. The flow field in terms of the temperature and u and v velocities is stored in files with the `hdf5' format. `xdmf' format files are automatically generated for visualisation on paraview.

### Results from the training
These are in the folders actions, rewards and final_rewards that store a list of values for each environment and each episode.
