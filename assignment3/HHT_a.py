#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 11:06:53 2021

@author: kadde
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from assimulo.explicit_ode import Explicit_ODE
from assimulo.ode import *
import numpy as np
import matplotlib.pyplot as mpl
import scipy.linalg as SL
# from assimulo.solvers import CVode

class HHT_a(Explicit_ODE):
    """
    HHT_a
    """
    tol=1.e-8     
    maxit=10000     
    maxsteps=500000
    
    def __init__(self, problem, alpha=0):
        Explicit_ODE.__init__(self, problem) #Calls the base class
        self.alpha = alpha;
        self.f = problem.rhs
        self.gamma = 1/2 - alpha
        self.beta = ((1-alpha)/2)**2
        #Solver options
        self.options["h"] = 0.01
        if problem.C == None:
            problem.C = np.zeros(np.shape(problem.M))
        #Statistics
        self.statistics["nsteps"] = 0
        self.statistics["nfcns"] = 0
    
    def _set_h(self,h):
            self.options["h"] = float(h)

    def _get_h(self):
        return self.options["h"]
        
    h=property(_get_h,_set_h)
        
    def integrate(self, t, y, tf, opts):
        """
        _integrates (t,y) values until t > tf
        """
        h = self.options["h"]
        h = min(h, abs(tf-t))
        
        #Lists for storing the result
        tres = [t]
        yres = []
        y = self.init_HHT(t,y)
        curr_y = y
        yres.append(y.copy())
        for i in range(self.maxsteps):
            if t >= tf:
                break
            self.statistics["nsteps"] += 1
          
            
            t, y = self.HHT_step(t,y,h)
            yres.append(y.copy())
            tres.append(t)
            
        
            h=min(self.h,np.abs(tf-t))
        else:
            raise Explicit_ODE_Exception('Final time not reached within maximum number of steps')
        
        return ID_PY_OK, tres, yres
    
    def init_HHT(self,t,y):
            u = y[0:int(len(y)/3)]
            up = y[int(len(y)/3):int(2*len(y)/3)]
            rhs = self.f(t,u) - self.problem.C@up - self.problem.K(t,u)@u     
            upp = np.linalg.solve(self.problem.M,rhs)
            y[int(2*len(y)/3):] = upp
            return y
        
    def HHT_step(self,t,y,h):
            u = y[0:int(len(y)/3)]
            up = y[int(len(y)/3):int(2*len(y)/3)]
            upp = y[int(2*len(y)/3):]
            # PLACEHOLDER f !!!
            # Is the fucntion fixed step? 
            
            # eq 8''
            rhs1 =  self.problem.M@(u/(self.beta*h**2) + up/(self.beta*h) + (1/(2*self.beta) - 1)*upp)
            rhs2 =  self.problem.C@( (self.gamma/(self.beta*h))*u - (1- self.gamma/self.beta)*up -(1 - (self.gamma/(2*self.beta)))*h*upp)
            rhs3 = self.alpha*self.problem.K(t,u)@u
            rhs = self.f(t,u) + rhs1 + rhs2 + rhs3
            
            lhs = ( self.problem.M/(self.beta*h**2) + (self.gamma*self.problem.C)/(self.beta*h) + (1 + self.alpha)*self.problem.K(t,u))
            utp1 = np.linalg.solve(lhs,rhs)
                        
            # eq 6'
            upptp1 = (utp1 - u)/(self.beta*h**2) - up/(self.beta*h) - (1/(2*self.beta) - 1)*upp
            
            # eq 7'
            uptp1 = (self.gamma/self.beta)*(utp1-u)/h + (1 - self.gamma/self.beta)*up + (1 - (self.gamma/(2*self.beta)))*h*upp
            
            ytp1 = np.append(utp1,uptp1,0)
            ytp1 = np.append(ytp1,upptp1,0)
            return t+h,ytp1
            
    def print_statistics(self, verbose=NORMAL):
        self.log_message('Final Run Statistics            : {name} \n'.format(name=self.problem.name),        verbose)
        self.log_message(' Step-length                    : {stepsize} '.format(stepsize=self.options["h"]), verbose)
        self.log_message(' Number of Steps                : '+str(self.statistics["nsteps"]),          verbose)               
        self.log_message(' Number of Function Evaluations : '+str(self.statistics["nfcns"]),         verbose)
            
        self.log_message('\nSolver options:\n',                                    verbose)
        self.log_message(' Solver            : BDF4',                     verbose)
        self.log_message(' Solver type       : Fixed step\n',                      verbose)
            
        
'''
#Define the rhs
def f(t,y):
    ydot = -y[0]
    return np.array([ydot])
    
#Define an Assimulo problem
exp_mod = Explicit_Problem(f, 4)
exp_mod.name = 'Simple BDF-4 Example'

#Define another Assimulo problem
def pend(t,y):
    #g=9.81    l=0.7134354980239037
    gl=13.7503671
    return np.array([y[1],-gl*np.sin(y[0])])
    
pend_mod=Explicit_Problem(pend, y0=np.array([2.*np.pi,1.]))
pend_mod.name='Nonlinear Pendulum'

#Define an explicit solver
exp_sim = BDF_4(pend_mod) #Create a BDF solver
t, y = exp_sim.simulate(4)
exp_sim.plot()
mpl.show()
'''