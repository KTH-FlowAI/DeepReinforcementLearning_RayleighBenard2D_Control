#!/bin/env python
#
# DEEP REINFORCEMENT LEARNING FOR RAYLEIGH-BENARD CONVECTION
#
# Xavier Garcia, Pol Suarez, Arnau Miro, Francisco Alcantara
#
# 08/11/2022


from __future__ import print_function, division

import os, subprocess
from configuration import NODELIST


def run_subprocess(runpath,runbin,runargs,nprocs=1,host=None,log=None,srun=False,check_return=True):
	'''
	Use python to call a terminal command
	'''
	# Build command to run
	if srun:
		# Sometimes we will need to use srun...
		cmd = 'cd %s && srun -n %d %s %s'%(runpath,nprocs,runbin,runargs) if log is None else 'cd %s && srun -n %d %s %s > %s 2>&1'%(runpath,nprocs,runbin,runargs,log)
	else:
		if nprocs == 1:
			# Run a serial command
			cmd = 'cd %s && %s %s'%(runpath,runbin,runargs) if log is None else 'cd %s && %s %s > %s 2>&1'%(runpath,runbin,runargs,log)
		else:
			# Run a parallel command
			if host is None:
				cmd = 'cd %s && mpirun -np %d %s %s'%(runpath,nprocs,runbin,runargs) if log is None else 'cd %s && mpirun -np %d %s %s > %s 2>&1'%(runpath,nprocs,runbin,runargs,log)
			else:
				cmd = 'cd %s && mpirun -np %d -host %s %s %s'%(runpath,nprocs,host,runbin,runargs) if log is None else 'cd %s && mpirun -np %d -host {3} %s %s > %s 2>&1'%(runpath,nprocs,host,runbin,runargs,log)
	# Execute run
	retval = subprocess.call(cmd,shell=True)
	# Check return
	if check_return and retval != 0: raise ValueError('Error running command <%s>!'%cmd)
	# Return value
	return retval


def detect_system():
	'''
	Test if we are in a cluster or on a local machine
	'''
	# Start assuming we are on the local machine
	out = 'LOCAL' 
	# 1. Test for SRUN, if so we are in a SLURM machine
	# and hence we should use SLURM to check the available nodes
#	if (run_subprocess('./','which','srun',check_return=False) == 0): out = 'SLURM'
	# Return system value
	return out


def _slurm_generate_node_list(outfile,num_servers = os.getenv('SLURM_NNODES'),num_cores_server = os.getenv('SLURM_NTASKS_PER_CORE')):
	'''
	Generate the list of nodes using slurm
	'''
	# Recover some information from SLURM environmental variables
	hostlist = os.getenv('SLURM_JOB_NODELIST') # List of nodes used by the job
	# Perform some sanity checks
	if not len(hostlist) == num_servers: raise ValueError('Inconsistent number of nodes in SLURM or configuration!')
	run_subprocess('./','echo','"%s"'%hostlist,log=outfile)
#	run_subprocess('./','scontrol','show hostnames',log=outfile) 

def _localhost_generate_node_list(outfile,num_servers):
	'''
	Generate the list of nodes for a local run
	'''
	hostlist = 'localhost'
	for iserver in range(num_servers): hostlist += '\nlocalhost'
	# Basically write localhost as the list of nodes
	# Add n+1 nodes as required per the nodelist
	run_subprocess('./','echo','"%s"'%hostlist,log=outfile)

def generate_node_list(outfile=NODELIST,num_servers=1):
	'''
	Detect the system and generate the node list
	'''
	system  = detect_system()
	if system == 'LOCAL': _localhost_generate_node_list(outfile,num_servers)
	if system == 'SLURM': _slurm_generate_node_list(outfile,num_servers)

def read_node_list(file=NODELIST):
	'''
	Read the list of nodes
	'''
	fp = open(file,'r')
	nodelist = [h.strip() for h in fp.readlines()]
	fp.close()
	return nodelist
