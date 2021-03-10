#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 14:01:03 2021

@author: erik
"""

import numpy as np
import Explicit_Problem_2t1 as c2t1
from assimulo.ode import *
from scipy.linalg import inv
#from assimulo.solvers import CVode
from assimulo.explicit_ode import Explicit_ODE
#import matplotlib.pyplot as mpl
import Explicit_Problem_2nd
import scipy.sparse.linalg as ssl

class Explicit_Problem_MCK_to_1wrap(Explicit_Problem_2nd.Explicit_Problem_2nd):
    
    def __init__(self, M, K, rhs, y0, t0, C=None):
        Explicit_Problem_2nd.Explicit_Problem_2nd.__init__(self, M, K, rhs, y0, t0, C=C)
        self.M = M
        self.x0 = y0[0]
        self.x0d = y0[1]
        self.F = rhs
        
        
    #converts problem from MCK form to second order 
    def convRHS(self, t, x, xd):
        Kt = self.K(t,x)
        if not self.C==None:
            xdd = ssl.spsolve(self.M,(self.rhs(t,x)-self.C(self.M,Kt)@np.transpose(xd)-Kt@np.transpose(x)))
        else:
            xdd = ssl,spsolve(self.M,(self.rhs(t,x)-Kt@np.transpose(x)))          
        return xdd
    
    def makeModel(self):
        converter = c2t1.Explicit_Problem_2_to_1wrap(self.convRHS,self.t0,self.x0,self.x0d)
        model = converter.makeModel()
        return model
    
    

'''
Kt = np.eye(2)
Ct = np.array([[0.1,0 ],[0, 0.1]])
Mt = np.eye(2)
x0 = np.array([1.5,-1.5])
x0d = np.array([0,0])

def F(t,x):
    return np.array([0,0])

def Kf(t,x,xd):
    #return np.eye(2)
    return np.array([[1/np.sqrt(abs(x[0])), 0.7],[-0.1, 1/np.sqrt(abs(x[1]))]])

converter = Explicit_Problem_MCK_to_1wrap(Mt, Ct, Kf, F, 0, x0, x0d)
print(converter.convRHS(0,x0,x0d))
model = converter.makeModel()



sim = CVode(model)
t,y = sim.simulate(10)
mpl.plot(t,y[:,0])
mpl.plot(t,y[:,1])
mpl.grid()
'''