# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 11:52:13 2021

@author: Magnus
"""

from assimulo.explicit_ode import Explicit_ODE
from assimulo.ode import *
import Explicit_Problem_2nd
import numpy as np
import matplotlib.pyplot as mpl
import Second_Order as newmark




def rhs(t,y):
    k = 2000
    lmd = k*(np.sqrt(y[0]*y[0] + y[1]*y[1])-1)/np.sqrt(y[0]*y[0] + y[1]*y[1])
    y4d = -y[1]*lmd-1
    y3d = -y[0]*lmd
    y2d = y[3]
    y1d = y[2]
    
    yd = np.array([0, -1])
    
    return yd

def K(t,y):
    k = 2000
    lmd = k*(np.sqrt(y[0]*y[0] + y[1]*y[1])-1)/np.sqrt(y[0]*y[0] + y[1]*y[1])
    y4d = -y[1]*lmd
    y3d = -y[0]*lmd
    yd=np.array([y3d,y4d])
    return yd



xinit = np.sin(20*np.pi/180)+0.1
yinit = np.cos(20*np.pi/180)
xpinit=0
ypinit=0
sqrt2 = np.sqrt(0.5);
y0 = list((np.array([xinit,yinit]),np.array([xpinit,ypinit])))
t0 = 0.0
model = Explicit_Problem_2nd.Explicit_Problem_2nd(rhs, y0, t0,M=np.eye(2),K=K)
model.name = 'Linear Test ODE'

sim = newmark.Second_Order(model)
tfinal = 20.0

t, y = sim.simulate(tfinal)
mpl.plot(t,y[:,0])
mpl.plot(t,y[:,1])
mpl.show()