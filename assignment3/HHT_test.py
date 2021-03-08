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
import HHT_a as HHT




def rhs(t,y):    
    yd = np.array([0, -1])
    return yd

def K(t,y):
    k = 200
    lmd = k*(np.sqrt(y[0]*y[0] + y[1]*y[1])-1)/np.sqrt(y[0]*y[0] + y[1]*y[1])
    yd=np.array([[lmd,0],[0,lmd]])
    return yd
   



xinit = np.sin(20*np.pi/180)+0.1
yinit = np.cos(20*np.pi/180)
xpinit=0
ypinit=0
sqrt2 = np.sqrt(0.5);
y0 = list((np.array([xinit,yinit]),np.array([xpinit,ypinit]), np.array([0,0])))
t0 = 0.0
K(0,y0)
model = Explicit_Problem_2nd.Explicit_Problem_2nd(rhs, y0, t0,M=np.eye(2),K=K)
model.name = 'Linear Test ODE'

sim = HHT.HHT_a(model,-1/3)
tfinal = 20

t, y = sim.simulate(tfinal)
mpl.plot(t,y[:,0])
mpl.plot(t,y[:,1])
mpl.show()  