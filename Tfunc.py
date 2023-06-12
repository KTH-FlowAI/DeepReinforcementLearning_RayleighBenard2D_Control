#!/bin/env python
#
# DEEP REINFORCEMENT LEARNING FOR RAYLEIGH-BENARD CONVECTION
#
# Tfunc.py: defines temperature profiles on bottom wall that are fed to agent for actuation 
#
# Colin Vignon
#
# FLOW, KTH Stockholm | 09/04/2023

from shenfun import *
import sympy
from sympy.parsing.sympy_parser import parse_expr

domain = ((-1, 1), (0, 2*sympy.pi))

class Tfunc():

	def __init__(self, nb_seg = None, dicTemp = None):
		''' N = number of actuators/segments on the hot boundary layer
		dicTemp = temperature variations of the segments: Ti' = Tnormal + Ti, Tnormal = 0.6 here''' 
	
		self.nb_seg = nb_seg
		self.dicTemp = dicTemp

		# Amplitude of variation of T
		self.ampl = 0.75  

		# half-length of the interval on which we do the smoothing
		self.dx = 0.03  
		
		
	def apply_T(self, x):
		values = self.ampl*np.array(list(self.dicTemp.values()))
		Mean = values.mean()
		K2 = max(1, np.abs(values-np.array([Mean]*self.nb_seg)).max()/self.ampl)
		
		# Position:
		xmax = domain[1][1]
		ind = sympy.floor(self.nb_seg*x//xmax)

		seq=[]
		count = 0
		while count<self.nb_seg-1:  # Temperatures will vary between: 2 +- 0.75
			
			x0 = count*xmax/self.nb_seg
			x1 = (count+1)*xmax/self.nb_seg
			
			T1 = 2+(self.ampl*self.dicTemp.get("T"+str(count))-Mean)/K2
			T2 = 2+(self.ampl*self.dicTemp.get("T"+str(count+1))-Mean)/K2
			if count == 0:
				T0 = 2+(self.ampl*self.dicTemp.get("T"+str(self.nb_seg-1))-Mean)/K2
			else:
				T0 = 2+(self.ampl*self.dicTemp.get("T"+str(count-1))-Mean)/K2
				
			seq.append((T0+((T0-T1)/(4*self.dx**3))*(x-x0-2*self.dx)*(x-x0+self.dx)**2, x<x0+self.dx))  # cubic smoothing		
			seq.append((T1, x<x1-self.dx))
			seq.append((T1+((T1-T2)/(4*self.dx**3))*(x-x1-2*self.dx)*(x-x1+self.dx)**2, x<x1))  # cubic smoothing

			count += 1
			
			if count == self.nb_seg-1:
				x0 = count*xmax/self.nb_seg
				x1 = (count+1)*xmax/self.nb_seg
				T0 = 2+(self.ampl*self.dicTemp.get("T"+str(count-1))-Mean)/K2
				T1 = 2+(self.ampl*self.dicTemp.get("T"+str(count))-Mean)/K2
				T2 = 2+(self.ampl*self.dicTemp.get("T0")-Mean)/K2
				
				seq.append((T0+((T0-T1)/(4*self.dx**3))*(x-x0-2*self.dx)*(x-x0+self.dx)**2, x<x0+self.dx))
				seq.append((T1, x<x1-self.dx))
				seq.append((T1+((T1-T2)/(4*self.dx**3))*(x-x1-2*self.dx)*(x-x1+self.dx)**2, True))
				
		return sympy.Piecewise(*seq)
		