
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
import BDF3_fsolve as bdf3



def rhs(t,y):
    k = 200
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

sim = bdf3.BDF_3(model)
tfinal = 20

t, y = sim.simulate(tfinal)
mpl.plot(t,y[:,0])
mpl.plot(t,y[:,1])
mpl.show()




