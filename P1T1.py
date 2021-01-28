#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 17:50:45 2021

@author: erik
"""
import numpy as np
import pylab as pl
from assimulo.problem import Explicit_Problem
from assimulo.solvers import CVode, ExplicitEuler


def rhs(t,y):
    k = 2000
    lmd = k*(np.sqrt(y[0]*y[0] + y[1]*y[1])-1)/np.sqrt(y[0]*y[0] + y[1]*y[1])
    y4d = -y[1]*lmd-1
    y3d = -y[0]*lmd
    y2d = y[3]
    y1d = y[2]
    
    yd = np.array([y1d, y2d, y3d, y4d])
    
    return yd



xinit = np.sin(20*np.pi/180)+0.05
yinit = np.cos(20*np.pi/180)+0.05
sqrt2 = np.sqrt(0.5);
y0 = np.array([xinit, yinit, 0.0, 0])
t0 = 0.0

model = Explicit_Problem(rhs, y0, t0)
model.name = 'Linear Test ODE'

sim = CVode(model)
tfinal = 20.0

t, y = sim.simulate(tfinal)
pl.plot(t,y[:,0])
pl.plot(t,y[:,1])
pl.show()