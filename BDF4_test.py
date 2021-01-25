#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 21:31:54 2021

@author: erik
"""

from assimulo.explicit_ode import Explicit_ODE
from assimulo.ode import *
import numpy as np
import matplotlib.pyplot as mpl
import BDF4 as bdf4


'''

#Define another Assimulo problem
def pend(t,y):
    #g=9.81    l=0.7134354980239037
    gl=13.7503671
    return np.array([y[1],-gl*np.sin(y[0])])
    
pend_mod=Explicit_Problem(pend, y0=np.array([2.*np.pi,1.]))
pend_mod.name='Nonlinear Pendulum'

#Define an explicit solver
exp_sim = bdf4.BDF_4(pend_mod) #Create a BDF solver
t, y = exp_sim.simulate(4)
exp_sim.plot()
mpl.show()

'''

def rhs(t,y):
    k = 2000
    lmd = k*(np.sqrt(y[0]*y[0] + y[1]*y[1])-1)/np.sqrt(y[0]*y[0] + y[1]*y[1])
    y4d = -y[1]*lmd-1
    y3d = -y[0]*lmd
    y2d = y[3]
    y1d = y[2]
    
    yd = np.array([y1d, y2d, y3d, y4d])
    
    return yd



xinit = np.sin(20*np.pi/180)+0.1
yinit = np.cos(20*np.pi/180)
sqrt2 = np.sqrt(0.5);
y0 = np.array([xinit, yinit, 0.0, 0])
t0 = 0.0

model = Explicit_Problem(rhs, y0, t0)
model.name = 'Linear Test ODE'

sim = bdf4.BDF_4(model)
tfinal = 20.0

t, y = sim.simulate(tfinal)
mpl.plot(t,y[:,0])
mpl.plot(t,y[:,1])
mpl.show()




