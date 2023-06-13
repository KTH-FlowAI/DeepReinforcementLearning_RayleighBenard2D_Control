# DeepReinforcementLearning_RayleighBenard2D_Control

Control of 2D Rayleigh Benard Convection using Deep Reinforcement Learning with Tensorforce and Shenfun.

If using this code, please cite our paper:

```
Effective control of two-dimensional Rayleigh-Benard convection: invariant multi-agent reinforcement learning is all you need
Colin Vignon, Jean Rabault, Joel Vasanth, Francisco Alcántara-Ávila, Mikael Mortensen, Ricardo Vinuesa
Physics of Fluids, 2023
```

Link to:

- the preprint: https://arxiv.org/abs/2304.02370
- the paper: TODO_UPDATE

## Introduction

Welcome to the repository DeepReinforcementLearning_RayleighBenard2D_Control. Here, you will find the scripts used for the control of two-dimensional Rayleigh Benard convection (RBC) using deep reinforcement learning (DRL). The computational fluid dynamics (CFD) solver employed is the spectral Galerkin code-suite 'shenfun' ([Documentation](https://shenfun.readthedocs.io/en/latest/)), which solves the flow field for the RBC. The DRL is solved using learning libraries from 'Tensorforce' ([Documentation](https://tensorforce.readthedocs.io/en/latest/)); specifically, the Proximal Policy Optimisation (PPO) algorithm is used. You will need shenfun and tensorforce installed on your system to run the code.

DRL is applied here using two different frameworks:
- Single agent reinforcement learning (SARL), and
- Multi-agent reinforcement learning (MARL).

Comparative studies for the performance of both frameworks may be performed if desired. Separate scripts that apply each framework are provided in the repository and are labelled with differently. These labels are further described below. To run either framework, you would have to run its corresponding script.

Installation instructions, a short tutorial, and links to pre-computed results for validation are given below.

## Installation

### Recommended: using our singularity sandbox

Installing the packages can be a bit challenging due to conflicting requirements in tensorforce and shenfun. Therefore, we recommend that you use our singularity sandbox that contains all the packages needed and is ready to use out of the box. This will require having singularity installed on your machine (check the singularity documentation for more information, see https://sylabs.io/docs/ ).

The singularity sandbox was created following the instructions / steps described at: https://github.com/jerabaul29/guidelines_workflow_project .

The sandbox segments are available at: https://drive.google.com/drive/folders/1CMarYhkRqhBhingRpbcxU8ORKzKsykWO?usp=sharing .

To use:

- download the segments and the checksums file
- check the integrity of the segments: ```sha256sum drlrbc2d.tar.gz*``` and compare with the content of the checksums file to check the integrity
- append the segments into the image: ```cat drlrbc2d.tar.gz__part.?? > drlrbc2d.tar.gz```
- check the integrity of the reconstructed image: compare ```sha256sum drlrbc2d.tar.gz``` with the content of the checksums file
- untar: ```tar xfp drlrbc2d.tar.gz```, this results into a folder: this is actually the singularity sandbox
- launch a terminal inside the sandbox: ```singularity shell --writable --fakeroot --no-home drlrbc2d```
- at this time, your terminal is inside the singularity sandbox, and all the software is available there. To run the code, you can simply do:

```
Singularity> cd /home/2D_DeepReinforcementLearning_RayleighBenard_Control_Shenfun_Tensorforce-master
Singularity> eval "$(/root/anaconda3/bin/conda shell.bash hook)"
(base) Singularity> conda activate shenfun
(shenfun) Singularity> python3 train_marl.py 
[... the code will start running then ...]
```

I.e.: move to the folder containing the code inside the sandbox, activate conda inside the sandbox, activate the correct conda environment containing all the packages in the right versions, and run any script.

Note that the sandbox is just a folder on your machine, so all the files inside the sandbox (such as data files, results of simulations and DRL trainings, etc) are available at the corresponding location if you need to use them from outside the sandbox at a later point.

Also note that the code within the sandbox may not be in the latest version (we do not update the sandbox each time we push updates here), but you can simply ```git clone``` or ```cp``` the version of the code you want to the sandbox - the aim of the sandbox is mostly to distribute an environment with the necessary packages in a cross compatible set of versions, and you can run any code version you want from within it.

### (not recommended): manual installation with ad hoc conda environment

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

It is important to note that the requirement for the successful installation of the above is a working Linux operating system from any major distribution (Debian, CentOS , Redhat or Ubuntu). The software stack is challenging to install as different packages have conflicting requirements: you may need to upgrade / downgrade packages by hand!

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
