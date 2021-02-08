#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 21:15:08 2021

@author: erik
"""

from assimulo.explicit_ode import Explicit_ODE
from assimulo.ode import *
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as mpl
import scipy.linalg as SL

class trapezoidal_Rule(Explicit_ODE):

    tol=1.e-8     
    maxit=10000     
    maxsteps=5000000
    
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

        h = self.options["h"]
        h = min(h, abs(tf-t))
        
        #Lists for storing the result
        tres = []
        yres = []
        
        for i in range(self.maxsteps):
            if t >= tf:
                break
            self.statistics["nsteps"] += 1
            
            t_np1, y_np1 = self.trapezoidalStep(t,y,h)
            t, y = t_np1, y_np1
            
            tres.append(t)
            yres.append(y.copy())
        
            h=min(self.h,np.abs(tf-t))
        else:
            raise Explicit_ODE_Exception('Final time not reached within maximum number of steps')
        
        return ID_PY_OK, tres, yres
    
    def trapezoidalStep(self, T, Y, h):
        
        f = self.problem.rhs
        t_n = T
        y_n = Y
        t_np1 = t_n+h
        res  = lambda y : y_n - y +0.5*h*(f(t_n,y_n)+f(t_np1,y))
        y_np1 = fsolve(res, y_n)
        return t_np1, y_np1
    
        
            
    def print_statistics(self, verbose=NORMAL):
        self.log_message('Final Run Statistics            : {name} \n'.format(name=self.problem.name),        verbose)
        self.log_message(' Step-length                    : {stepsize} '.format(stepsize=self.options["h"]), verbose)
        self.log_message(' Number of Steps                : '+str(self.statistics["nsteps"]),          verbose)               
        self.log_message(' Number of Function Evaluations : '+str(self.statistics["nfcns"]),         verbose)
            
        self.log_message('\nSolver options:\n',                                    verbose)
        self.log_message(' Solver            : Trapezoidal Rule',                     verbose)
        self.log_message(' Solver type       : Fixed step\n',                      verbose)
            
    