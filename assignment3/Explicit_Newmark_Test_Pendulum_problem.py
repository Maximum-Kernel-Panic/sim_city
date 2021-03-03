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
import Explicit_Problem_MCKt1_v2
import BDF3_fsolve as BDF




def rhs(t,y):    
    yd = np.array([0, -1])
    return yd

def K(t,y):
    k =1000
    lmd = k*(np.sqrt(y[0]*y[0] + y[1]*y[1])-1)/np.sqrt(y[0]*y[0] + y[1]*y[1])
    yd=np.array([[lmd,0],[0,lmd]])
    return yd

def C_Rayleigh(M,K):
    Cm=1
    Ck=1
    return Cm*M+Ck*K



xinit = np.sin(20*np.pi/180)+0.1
yinit = np.cos(20*np.pi/180)
xpinit=0
ypinit=0
sqrt2 = np.sqrt(0.5);
y0 = list((np.array([xinit,yinit]),np.array([xpinit,ypinit])))
t0 = 0.0
model = Explicit_Problem_2nd.Explicit_Problem_2nd(np.eye(2), K, rhs, y0, t0)
model_2_class=Explicit_Problem_MCKt1_v2.Explicit_Problem_MCK_to_1wrap(np.eye(2), K, rhs, y0, t0)
model_2=model_2_class.makeModel()
model.name = 'Newmark'
model_2.name = 'Linear Test ODE'

sim = newmark.Second_Order(model)
sim2 = BDF.BDF_3(model_2)
tfinal = 20

t1, y1 = sim.simulate(tfinal)
t2, y2 = sim2.simulate(tfinal)

mpl.plot(t1,y1[:,0])
mpl.plot(t1,y1[:,1])
mpl.figure()
mpl.plot(t2,y2[:,0])
mpl.plot(t2,y2[:,1])
mpl.show()