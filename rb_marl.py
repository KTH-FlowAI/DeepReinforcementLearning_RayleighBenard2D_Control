#!/bin/env python
#
# DEEP REINFORCEMENT LEARNING FOR RAYLEIGH-BENARD CONVECTION
#
# rb_marl.py: Rayleigh Benard CFD environment
#
# Colin Vignon & Joel Vasanth
#
# FLOW, KTH Stockholm | 09/04/2023

from shenfun import *
import matplotlib.pyplot as plt
import sympy
import os, csv, numpy as np
import shutil
import copy
from time import time
from mpi4py import MPI
from tensorforce.environments import Environment
from mpi4py_fft import generate_xdmf
from channelflow2d import KMM
from parameters import nb_actuations, num_episodes, num_servers, \
    simu_name, nb_inv_envs, simulation_duration


np.warnings.filterwarnings('ignore')

x, y, tt = sympy.symbols('x,y,t', real=True)

comm = MPI.COMM_SELF



class RayleighBenard(KMM):

    def __init__(self, N=(64, 96), domain=((-1, 1), (0, 2*sympy.pi)), Ra=10000., Pr=0.7, dt=0.05, bcT=(2, 1), \
         conv=0, filename='RB_2D', family='C', padding_factor=(1, 1.5), modplot=10, obsGrid=(8,32), modsave=10,  \
            moderror=10, checkpoint=10, timestepper='IMEXRK3'):
        
        plt.close('all')
        KMM.__init__(self, N=N, domain=domain, nu=np.sqrt(Pr/Ra), dt=dt, conv=conv,
                     filename=filename, family=family, padding_factor=padding_factor,
                     modplot=modplot, modsave=modsave, moderror=moderror,
                     checkpoint=checkpoint, timestepper=timestepper, dpdy=0)
        self.kappa = 1./np.sqrt(Pr*Ra)
        self.bcT = bcT
        plt.close('all')
        
        # Additional spaces and functions for Temperature equation
        self.T0 = FunctionSpace(N[0], family, bc=bcT, domain=domain[0])
        self.TT = TensorProductSpace(comm, (self.T0, self.F1), modify_spaces_inplace=True) # Temperature
        self.uT_ = Function(self.BD)     # Velocity vector times T
        self.T_ = Function(self.TT)      # Temperature solution
        self.Tb = Array(self.TT)
        
        self.file_T = ShenfunFile('_'.join((filename, 'T')), self.TT, backend='hdf5', mode='w', mesh='uniform')

        # Modify checkpoint file
        self.checkpoint.data['0']['T'] = [self.T_]

        dt = self.dt
        kappa = self.kappa
        self.N = N

        sol2 = chebyshev.la.Helmholtz if self.B0.family() == 'chebyshev' else la.SolverGeneric1ND

        # Addition to u equation.
        self.pdes['u'].N = [self.pdes['u'].N, Dx(self.T_, 1, 2)]
        self.pdes['u'].latex += r'\frac{\partial^2 T}{\partial y^2}'

        # Remove constant pressure gradient from v0 equation
        self.pdes1d['v0'].N = self.pdes1d['v0'].N[0]

        # Add T equation
        q = TestFunction(self.TT)
        self.pdes['T'] = self.PDE(q,
                                  self.T_,
                                  lambda f: kappa*div(grad(f)),
                                  -div(self.uT_),
                                  dt=self.dt,
                                  solver=sol2,
                                  latex=r"\frac{\partial T}{\partial t} = \kappa \nabla^2 T - \nabla \cdot \vec{u}T")

        self.im1 = None
        self.im2 = None

        # Observation outputs
        self.out = []
        self.out2 = []
        self.instant_Nusselt = []
        self.instant_kinEn = []
        
        # Others
        self.obsGrid = obsGrid
        self.Nstep = (N[0]//self.obsGrid[0], N[1]//self.obsGrid[1])
        
      
    def update_bc(self, t):
        # Update time-dependent bcs.
        self.T0.bc.update(t)
        self.T_.get_dealiased_space(self.padding_factor).bases[0].bc.update(t)

    def prepare_step(self, rk):
        self.convection()
        Tp = self.T_.backward(padding_factor=self.padding_factor)
        self.uT_ = self.up.function_space().forward(self.up*Tp, self.uT_)

    def tofile(self, tstep):
        self.file_u.write(tstep, {'u': [self.u_.backward(mesh='uniform')]}, as_scalar=True)
        self.file_T.write(tstep, {'T': [self.T_.backward(mesh='uniform')]})

    def init_from_checkpoint(self):
        self.checkpoint.read(self.u_, 'U', step=0)
        self.checkpoint.read(self.T_, 'T', step=0)
        self.checkpoint.open()
        tstep = self.checkpoint.f.attrs['tstep']
        t = self.checkpoint.f.attrs['t']
        self.checkpoint.close()
        return t, tstep

    def print_energy_and_divergence(self, t, tstep):
        if tstep % self.moderror == 0 and self.moderror > 0:
            ub = self.u_.backward(self.ub)
            Tb = self.T_.backward(self.Tb)
            e0 = inner(1, ub[0]*ub[0])
            e1 = inner(1, ub[1]*ub[1])
            d0 = inner(1, Tb*Tb)
            divu = self.divu().backward()
            e3 = np.sqrt(inner(1, divu*divu))
            if comm.Get_rank() == 0:
                if tstep % (10*self.moderror) == 0 or tstep == 0:
                    pass
                   

    def initialize(self, rand=0.001, from_checkpoint=False):
        if from_checkpoint:
            self.checkpoint.read(self.u_, 'U', step=0)
            self.checkpoint.read(self.T_, 'T', step=0)
            self.checkpoint.open()
            tstep = self.checkpoint.f.attrs['tstep']
            t = self.checkpoint.f.attrs['t']
            self.checkpoint.close()
            self.update_bc(t)
            return t, tstep
        X = self.X
               
        fun = self.bcT[0]
        self.Tb[:] = 0.5*(1 + 0.5*self.bcT[1]-X[0]/(1+self.bcT[1])+ 0.125*(2-self.bcT[1])\
                          *np.sin(np.pi*X[0]))*fun + rand*np.random.randn(*self.Tb.shape)*(1-X[0])*(1+X[0])
        self.T_ = self.Tb.forward(self.T_)
        self.T_.mask_nyquist(self.mask)
        return 0, 0


    def outputs(self, tstep, count):
        if (tstep == 0 or count == 0):  # Make sure to reinitialize the outputs when a brand new simulation is launched
            self.out = []
            self.out2 = []
            self.out_red = []
            self.out_red2 = []
        else:
            ub = self.u_.backward(self.ub) 
            Tb = self.T_.backward(self.Tb)
            new_out = np.zeros((3, self.obsGrid[0], self.obsGrid[1]))
            new_out[0] = ub[1, ::self.Nstep[0], ::self.Nstep[1]]  # horizontal (x) axis
            new_out[1] = ub[0, ::self.Nstep[0], ::self.Nstep[1]]  # vertical (y) axis
            new_out[2] = Tb[::self.Nstep[0], ::self.Nstep[1]]
            self.out.append(new_out)
            
            # Normalization of the data
            new_out2 = copy.copy(new_out)
            new_out2[0]*= 1.5
            new_out2[1]*= 1.5
            new_out2[2] = 2*(new_out2[2] - 0.8)
            new_out2 = new_out2.reshape(3*self.obsGrid[0]*self.obsGrid[1],)
            self.out2.append(new_out2)
  

    def DRL_inputs(self):  # inputs for the DRL algo
        return np.mean(np.array([self.out2[-1], self.out2[-2], self.out2[-3], self.out2[-4]]), axis=0)

    def compute_Nusselt(self):
        '''Used just to have the evolution of Nu during the baseline simulation'''
        div = self.kappa*(2.-self.bcT[1])/2  # H = 2, Tb = 2.

        gen_Nus = []
        for i in range(1,5):
            uyT_ = np.mean(np.mean(np.multiply(self.out[-i][1], self.out[-i][2]), axis=1), axis = 0)
            T_ = np.mean(np.gradient(np.mean(self.out[-i][2], axis=1), axis=0))
            gen_Nus.append((uyT_ - self.kappa*T_)/div)
        
        return np.mean(np.array(gen_Nus))
                
        
    def compute_kinEn(self):
        '''Used just to have the evolution of kinEn during the baseline simulation'''
        
        u2_xy = self.out[-1][1]*self.out[-1][1] + self.out[-1][0]*self.out[-1][0]
        return np.sum(u2_xy)
        
        
    def evolve(self, new_bcT, t_ini=None, tstep_ini = None):
    
        self.bcT = new_bcT    
        new_t, new_tstep = t_ini, tstep_ini
        self.T0.bc.bc['left']['D'] = self.bcT[0]
        self.T0.bc.update()
        self.T0.bc.set_tensor_bcs(self.T0, self.T0.tensorproductspace)
        TP0 = self.T_.get_dealiased_space(self.padding_factor).bases[0]
        TP0.bc.bc['left']['D'] = self.bcT[0]
        TP0.bc.update()
        TP0.bc.set_tensor_bcs(TP0, TP0.tensorproductspace)
        
        
    def solve(self, which, t=0, tstep=0, end_time=1000, end_episode = False):
        c = self.pdes['u'].stages()[2]
        self.assemble()
        count = 0
        while t < end_time-1e-8:
            for rk in range(self.PDE.steps()):
                self.prepare_step(rk)
                for eq in ['u', 'T']:
                    self.pdes[eq].compute_rhs(rk)
                for eq in ['u']:
                    self.pdes[eq].solve_step(rk)
                self.compute_v(rk)
                self.update_bc(t+self.dt*c[rk+1])
                self.pdes['T'].solve_step(rk)
            self.outputs(tstep, count)
            count += 1
            if tstep >= 4 and which == 'baseline' and tstep%10 == 0:
                self.instant_Nusselt.append(self.compute_Nusselt())
                self.instant_kinEn.append(self.compute_kinEn())
            t += self.dt
            tstep += 1
            self.update(t, tstep)
            self.checkpoint.update(t, tstep)
            
            if tstep % self.modsave == 0:
                self.tofile(tstep)
        if end_episode:
            self.tofile(tstep-1)  
            self.TT.destroy()  
            self.TB.destroy()
            self.TD.destroy()
            self.TC.destroy()
            self.TDp.destroy()
            self.BD.destroy()
            self.CD.destroy()
        return t, tstep         
        
        
    def define_timeframe(self, which, Evolve = False, new_bcT = None, t_ini = None, tstep_ini = None):

        if which == 'baseline':
            t, tstep = self.initialize(rand=0.001, from_checkpoint=False)
            #self.evolve(new_bcT, t, tstep)
        else:
            if Evolve:
                t, tstep = t_ini, tstep_ini
            else:
                t, tstep = self.initialize(rand=0.001, from_checkpoint=True)
            self.evolve(new_bcT, t, tstep)
        return t, tstep


    def launch(self, timeframe, which, t_ini = None, tstep_ini = None, Evolve = False, End_episode = False):

        
        t_end, tstep_end = self.solve(which, t=t_ini, tstep=tstep_ini, end_time=timeframe[1], end_episode = End_episode)
        if which == 'baseline':
            generate_xdmf('RB_'+str(self.N[0])+'_'+str(self.N[1])+'_T.h5')
            generate_xdmf('RB_'+str(self.N[0])+'_'+str(self.N[1])+'_U.h5')
        return self.DRL_inputs(), t_end, tstep_end
    
    

