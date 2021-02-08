#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 17:50:45 2021

@author: erik
"""
import numpy as np
import pylab as pl
from assimulo.problem import Explicit_Problem
from assimulo.solvers import CVode, ExplicitEuler, ImplicitEuler
import trapezoidalRule as trap
import BDF4_fsolve as bdf4
import BDF3_fsolve as bdf3


k = 500

def rhs(t,y):

    lmd = k*(np.sqrt(y[0]*y[0] + y[1]*y[1])-1)/np.sqrt(y[0]*y[0] + y[1]*y[1])
    y4d = -y[1]*lmd-1
    y3d = -y[0]*lmd
    y2d = y[3]
    y1d = y[2]
    
    yd = np.array([y1d, y2d, y3d, y4d])
    
    return yd

def rhs2(t,y):
    c = 0.2
    lmd = k*(np.sqrt(y[0]*y[0] + y[1]*y[1])-1)/np.sqrt(y[0]*y[0] + y[1]*y[1])
    y4d = -y[1]*lmd-1 - c*np.sign(y[3])*np.sqrt(abs(y[3]))
    y3d = -y[0]*lmd - c*np.sign(y[2])*np.sqrt(abs(y[2]))
    y2d = y[3]
    y1d = y[2]
    
    yd = np.array([y1d, y2d, y3d, y4d])
    
    return yd

xinit = np.sin(20*np.pi/180)+0.2
yinit = np.cos(20*np.pi/180)+0.2
sqrt2 = np.sqrt(0.5);
y0 = np.array([xinit, yinit, 0.0, 0])
t0 = 0.0

model = Explicit_Problem(rhs2, y0, t0)
model.name = 'Linear Test ODE'

sim = CVode(model)
simShit = trap.trapezoidal_Rule(model)
simBDF4 = bdf4.BDF_4(model)
simBDF3 = bdf3.BDF_3(model)
tfinal = 20

t, y = sim.simulate(tfinal)
t2, y2 = simShit.simulate(tfinal)
t3, y3 = simBDF4.simulate(tfinal)
t4, y4 = simBDF3.simulate(tfinal)
pl.plot(t2,y2[:,1],label='Trapezoidal rule')
pl.plot(t,y[:,1],label='CVode')
#pl.plot(t3, y3[:,1],label='BDF-4')
#pl.plot(t4, y4[:,1],label='BDF-3')
pl.show()


pl.title('Trapezoidal rule, h = 0.01, k = {}' .format(k))
pl.ylabel('y2')
pl.xlabel('t')
pl.legend(loc='lower left')
#pl.ylim(-1.7,1.2)
#pl.xlim(350,400)
pl.show()
