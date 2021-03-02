#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 20:28:50 2021

@author: erik
"""
from assimulo.explicit_ode import Explicit_ODE
from assimulo.ode import *
import numpy as np
import scipy as sp
import matplotlib.pyplot as mpl
import scipy.linalg as SL
# from assimulo.solvers import CVode

class BDF_3(Explicit_ODE):
    """
    BDF-3   (Example of how to set-up own integrators for Assimulo)
    """
    tol=1.e-5     
    maxit=100000     
    maxsteps=5000
    
    def __init__(self, problem):
        Explicit_ODE.__init__(self, problem) #Calls the base class
        
        #Solver options
        self.options["h"] = 0.01
        
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
        tres = []
        yres = []
        
        for i in range(self.maxsteps):
            if t >= tf:
                break
            self.statistics["nsteps"] += 1
            
            if i==0:  # initial step
                t_np1,y_np1 = self.step_EE(t,y, h)
                t_np2,y_np2 = self.step_BDF2([t_np1, t], [y_np1, y], h)
                
                t, t_nm1, t_nm2 = t_np2, t_np1, t
                y, y_nm1, y_nm2 = y_np2, y_np1, y
                i = i + 1
            else:   
                t_np1, y_np1 = self.step_BDF3([t,t_nm1,t_nm2], [y,y_nm1,y_nm2], h)
                t, t_nm1, t_nm2 = t_np1, t, t_nm1
                y, y_nm1, y_nm2 = y_np1, y, y_nm1
            
            
            tres.append(t)
            yres.append(y.copy())
            'print(i)'
            h=min(self.h,np.abs(tf-t))
        else:
            raise Explicit_ODE_Exception('Final time not reached within maximum number of steps')
        
        return ID_PY_OK, tres, yres
    
    def step_EE(self, t, y, h):
        """
        This calculates the next step in the integration with explicit Euler.
        """
        self.statistics["nfcns"] += 1
        
        f = self.problem.rhs
        return t + h, y + h*f(t, y) 
        
    def step_BDF3(self,T,Y, h):
        """
        BDF-3 with Fixed Point Iteration and Zero order predictor
        
        alpha_0*y_np1+alpha_1*y_n+alpha_2*y_nm1=h f(t_np1,y_np1)
        alpha=[3/2,-2,1/2]
        """
        alpha=[11./6.,-3.,3./2,-1./3]
        f=self.problem.rhs
        
        t_n,t_nm1,t_nm2=T
        y_n,y_nm1,y_nm2=Y
        # predictor
        t_np1=t_n+h
        y_np1_i=y_n   # zero order predictor
        # corrector with fixed point iteration
        for i in range(self.maxit):
            self.statistics["nfcns"] += 1
            
            y_np1_ip1=(-(alpha[1]*y_n+alpha[2]*y_nm1+alpha[3]*y_nm2)+h*f(t_np1,y_np1_i))/alpha[0]
            if SL.norm(y_np1_ip1-y_np1_i) < self.tol:
                return t_np1,y_np1_ip1
            y_np1_i=y_np1_ip1
        else:
            raise Explicit_ODE_Exception('Corrector could not converge within % iterations'%i)
            
    def step_BDF2(self,T,Y, h):
        """
        BDF-2 with Fixed Point Iteration and Zero order predictor
        
        alpha_0*y_np1+alpha_1*y_n+alpha_2*y_nm1=h f(t_np1,y_np1)
        alpha=[3/2,-2,1/2]
        """
        alpha=[3./2.,-2.,1./2]
        f=self.problem.rhs
        
        t_n,t_nm1=T
        y_n,y_nm1=Y
        # predictor
        t_np1=t_n+h
        y_np1_i=y_n   # zero order predictor
        # corrector with fixed point iteration
        for i in range(self.maxit):
            self.statistics["nfcns"] += 1
            
            y_np1_ip1=(-(alpha[1]*y_n+alpha[2]*y_nm1)+h*f(t_np1,y_np1_i))/alpha[0]
            if SL.norm(y_np1_ip1-y_np1_i) < self.tol:
                return t_np1,y_np1_ip1
            y_np1_i=y_np1_ip1
        else:
            raise Explicit_ODE_Exception('Corrector could not converge within % iterations'%i)
            
    def print_statistics(self, verbose=NORMAL):
        self.log_message('Final Run Statistics            : {name} \n'.format(name=self.problem.name),        verbose)
        self.log_message(' Step-length                    : {stepsize} '.format(stepsize=self.options["h"]), verbose)
        self.log_message(' Number of Steps                : '+str(self.statistics["nsteps"]),          verbose)               
        self.log_message(' Number of Function Evaluations : '+str(self.statistics["nfcns"]),         verbose)
            
        self.log_message('\nSolver options:\n',                                    verbose)
        self.log_message(' Solver            : BDF3',                     verbose)
        self.log_message(' Solver type       : Fixed step\n',                      verbose)
 
'''           
#Define the rhs
def f(t,y):
    ydot = -y[0]
    return np.array([ydot])
    
#Define an Assimulo problem
exp_mod = Explicit_Problem(f, 4)
exp_mod.name = 'Simple BDF-3 Example'

#Define another Assimulo problem
def pend(t,y):
    #g=9.81    l=0.7134354980239037
    gl=13.7503671
    return np.array([y[1],-gl*np.sin(y[0])])
    
pend_mod=Explicit_Problem(pend, y0=np.array([2.*np.pi,1.]))
pend_mod.name='Nonlinear Pendulum'

#Define an explicit solver
exp_sim = BDF_3(pend_mod) #Create a BDF solver
t, y = exp_sim.simulate(4)
exp_sim.plot()
mpl.show()
'''