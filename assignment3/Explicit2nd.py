#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 14:07:52 2021

@author: erik
"""

import numpy as np
from assimulo.ode import *
#from assimulo.solvers import CVode
from assimulo.explicit_ode import Explicit_ODE
#import matplotlib.pyplot as mpl

class Explicit_Problem2nd():
    
    def __init__(self, rhs, y0, t0):
        self.rhs2nd = rhs
        self.y0 = y0
        self.t0 = t0
        self.dim = int(len(y0)/2)
        #print(self.dim)
        
    
    #translates the right-hand-side function to first order
    def buffRHS(self, t, x):
        xpos = x[0:self.dim]
        v = x[self.dim:2*self.dim]
        vdot = self.rhs2nd(t,xpos,v)
        ret = np.concatenate([v,vdot])
        return ret
      

    #makes the Explicit_Problem model to be passed to a solver
    def makeModel(self):
        
        model = Explicit_Problem(self.buffRHS,self.y0,self.t0)
        return model
        

    
'''

def rhs1(t, y, v):
    
    k = 100
    lmd = k*(np.sqrt(y[0]*y[0] + y[1]*y[1])-1)/np.sqrt(y[0]*y[0] + y[1]*y[1])
    y2d = -y[1]*lmd-1
    y1d = -y[0]*lmd
    xdot = np.array([y1d,y2d])
    return xdot

def rhs2(t,y):
    k = 100
    lmd = k*(np.sqrt(y[0]*y[0] + y[1]*y[1])-1)/np.sqrt(y[0]*y[0] + y[1]*y[1])
    y4d = -y[1]*lmd-1
    y3d = -y[0]*lmd
    y2d = y[3]
    y1d = y[2]
    
    yd = np.array([y1d, y2d, y3d, y4d])
    return yd



s2 = 1/np.sqrt(2)
x0 = np.array([-s2+0.05,-s2+0.05,0,0])

testprob = Explicit_Problem2nd(rhs1,x0,0)
model2= Explicit_Problem(rhs2,x0,0)

#print(testprob.dim)
#print(testprob.buffRHS(0,x0))

model = testprob.makeModel()


sim = CVode(model)
sim2 = CVode(model2)
print(model.rhs(0,[1,0]))

t,y = sim.simulate(10)
t2, y2 = sim2.simulate(10)
mpl.plot(t2,y2[:,0])
mpl.plot(t2,y2[:,1])
'''



    
    
    
    
    
    
    
    