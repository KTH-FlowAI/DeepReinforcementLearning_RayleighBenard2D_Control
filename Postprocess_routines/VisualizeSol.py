from shenfun import *
from ChannelFlow2D import KMM
import matplotlib.pyplot as plt
import sympy
from Tfunc import Tfunc
from mpi4py_fft import generate_xdmf
import numpy as np

'''
VisualizeSol.py: Script aimed at visualizing the control sequence of a given episode

Author: Colin Vignon | FLOW, KTH Stockholm

Date: 3/25/2023

Set parameters in the __main__ function, as desired.

'''

np.warnings.filterwarnings('ignore')
x, y, tt = sympy.symbols('x,y,t', real=True)
comm = MPI.COMM_WORLD

class RayleighBenard(KMM):

    def __init__(self,
                 N=(64, 96),
                 domain=((-1, 1), (0, 2*sympy.pi)),
                 Ra=10000.,
                 Pr=0.7,
                 dt=0.05,
                 bcT=(2, 1),
                 conv=0,
                 filename='RB',
                 family='C',
                 padding_factor=(1, 1.5),
                 modplot=10,
                 modsave=1e8,
                 moderror=10,
                 checkpoint=10,
                 timestepper='IMEXRK3'):
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

        # Chebyshev matrices are not sparse, so need a tailored solver. Legendre has simply 5 nonzero diagonals
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
        
        # Others
        self.obsGrid = (8,32)
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
                    print(f"{'Time':^11}{'uu':^11}{'vv':^11}{'T*T':^11}{'div':^11}")
                print(f"{t:2.4e} {e0:2.4e} {e1:2.4e} {d0:2.4e} {e3:2.4e}")

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
        self.Tb[:] = 0.5*(1 + 0.5*self.bcT[1]-X[0]/(1+self.bcT[1])+ 0.125*(2-self.bcT[1])*np.sin(np.pi*X[0]))*fun + rand*np.random.randn(*self.Tb.shape)*(1-X[0])*(1+X[0])

        self.T_ = self.Tb.forward(self.T_)
        self.T_.mask_nyquist(self.mask)
        return 0, 0
                
                
    def outputs(self, tstep):
    	if tstep == 0:  
            self.out = []
            self.out2 = []
    	else:  
            new_out = np.zeros((3, self.obsGrid[0], self.obsGrid[1]))
            ub = self.u_.backward(self.ub)
            Tb = self.T_.backward(self.Tb)
            new_out[0] = ub[1, ::self.Nstep[0], ::self.Nstep[1]]  # horizontal speed
            new_out[1] = ub[0, ::self.Nstep[0], ::self.Nstep[1]]  # vertical speed
            new_out[2] = Tb[::self.Nstep[0], ::self.Nstep[1]]
            self.out.append(new_out)
            new_out2 = new_out.reshape(3*self.obsGrid[0]*self.obsGrid[1],)
            self.out2.append(new_out2)

    def DRL_inputs(self, D = 4):  # inputs for the DRL algo

        return self.out2[-1]

    def compute_Nusselt(self):

        div = self.kappa*(1/2)
        uyT_ = np.mean(np.mean(np.multiply(self.out[-1][1], self.out[-1][2]), axis=1), axis = 0)
        T_ = np.mean(np.gradient(np.mean(self.out[-1][2], axis=1), axis=0))
        return (uyT_ - self.kappa*T_)/div

    
    
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
        
        
    def solve(self, t=0, tstep=0, end_time=10000):
        c = self.pdes['u'].stages()[2]
        self.assemble()
        while t < end_time-1e-8:
            for rk in range(self.PDE.steps()):
                self.prepare_step(rk)
                for eq in ['u', 'T']:
                    self.pdes[eq].compute_rhs(rk)
                for eq in ['u']:
                    self.pdes[eq].solve_step(rk)
                self.compute_v(rk)
                self.update_bc(t+self.dt*c[rk+1]) # modify time-dep boundary condition
                self.pdes['T'].solve_step(rk)
            self.outputs(tstep)
            if tstep >= 1:
                self.instant_Nusselt.append(self.compute_Nusselt())
            t += self.dt
            tstep += 1
            self.update(t, tstep)
            self.checkpoint.update(t, tstep)
            if tstep % self.modsave == 0:
                self.tofile(tstep)
        return t, tstep

if __name__ == '__main__':
    from time import time
    import csv
    import os
    
    # Set parameters here, as desired
    nb_actions = 200  
    duration_baseline = 400.0
    duration_action = 1.5
    simu_name = 'RB_2D_SARL'
    num_ep = 20 # Which episode to visualise
    nb_segs = 10
    name = 'output_actions.csv'
    
    # Set paths here
    general_path = '/scratch/jvasanth/2022_Rayleigh_Benard_Control_DRL_Shenfun_2D_3D/shenfun_files/cases/'+simu_name
    specific_path = general_path+'/actions/environment1/ep_'+str(num_ep)+'/'
    move_path = general_path+'/CFD_n1/EP_'+str(num_ep)+'/'
    
    dicTemps = {}  # dictionary with the nb_actions successive boundary conditions
    with open(specific_path+name, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        line = next(reader)
        for i in range(nb_actions):
            line = next(reader)
            dicTi = {}

            dicTi.update({'T'+str(k):float(line[k+1]) for k in range(nb_segs)})
            dicTemps.update({'Action_'+str(i):dicTi})

    os.system('cp '+general_path+'/baseline/*.h5 /scratch/jvasanth/2022_Rayleigh_Benard_Control_DRL_Shenfun_2D_3D/Postprocess_routines/')
    os.chdir('/scratch/jvasanth/2022_Rayleigh_Benard_Control_DRL_Shenfun_2D_3D/Postprocess_routines/')

    for i in range(nb_actions):	
        N = (64, 96)
        d = {
            'N': N,
            'Ra': 10000.,
            'Pr': 0.7,
            'dt': 0.05,
            'filename': f'RB_{N[0]}_{N[1]}',
            'conv': 0,
            'modplot': 10,
            'moderror': 10,
            'modsave': 10,
            'bcT': (Tfunc(nb_seg=nb_segs, dicTemp=dicTemps.get('Action_'+str(i))).apply_T(y), 1),
            'family': 'C',
            'checkpoint': 10,
            'timestepper': 'IMEXRK3'
            }
        if i == 0:
            c = RayleighBenard(**d)
            t, tstep = c.initialize(rand=0.001, from_checkpoint=True)
            t0 = time()
            new_t, new_tstep = c.solve(t=t, tstep=tstep, end_time=duration_baseline + (i+1)*duration_action)
            print('Computing time %2.4f'%(time()-t0))
        else:
            c.evolve(d.get('bcT'), t_ini=new_t, tstep_ini = new_tstep)
            new_t, new_tstep = c.solve(t=new_t, tstep=new_tstep, end_time=duration_baseline + (i+1)*duration_action)

        plt.close('all')

    # Produces the visualisation of Temperature and velocity. 
    generate_xdmf('RB_'+str(N[0])+'_'+str(N[1])+'_T.h5')
    generate_xdmf('RB_'+str(N[0])+'_'+str(N[1])+'_U.h5')


