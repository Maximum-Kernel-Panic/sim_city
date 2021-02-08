#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 21:31:54 2021

@author: erik
"""
from assimulo.solvers import CVode
from assimulo.explicit_ode import Explicit_ODE
from assimulo.ode import *
import numpy as np
import matplotlib.pyplot as mpl
import BDF4_fsolve as bdf4
import BDF3_fsolve as bdf3
import BDF2_Assimulo as bdf2
import BDF5_fsolve as bdf5


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

k = 1000

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

def getEigs(x):
    
    imeigs = np.zeros(np.shape(x))
    reeigs = np.zeros(np.shape(x))
    g = lambda x:k*(np.sqrt(x[0]**2 + x[1]**2)-1)/np.sqrt(x[0]**2 + x[1]**2)
    lmbd = list(map(g, x))
    
    for i,lmd in enumerate(lmbd):
         K = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [-lmd, 0, 0, 0],[0, -lmd, 0, 0]])
         imeigs[i,:] = np.imag(np.linalg.eig(K)[0])
         reeigs[i,:] = np.real(np.linalg.eig(K)[0])
         
    return np.transpose(reeigs),np.transpose(imeigs)


xinit = np.sin(20*np.pi/180)+0.05
yinit = np.cos(20*np.pi/180)+0.05
sqrt2 = np.sqrt(0.5);
y0 = np.array([xinit, yinit, 0.0, 0])
t0 = 0.0

model = Explicit_Problem(rhs2, y0, t0)
model.name = 'Linear Test ODE'

sim = bdf4.BDF_4(model)
sim2 = CVode(model)
sim3 = bdf3.BDF_3(model)
sim4 = bdf2.BDF_2(model)
sim5 = bdf5.BDF_5(model)
tfinal = 20.0

t, y = sim.simulate(tfinal)
t2,y2 = sim2.simulate(tfinal)
t3, y3 = sim3.simulate(tfinal)
#t4, y4 = sim4.simulate(tfinal)
t5, y5 = sim5.simulate(tfinal)

reigs, imeigs = getEigs(y2)

'''
re, im = getEigs(y2)
re, im = re, im
choice = 50
mpl.figure()
mpl.scatter(re[1,choice],im[1,choice])

#mpl.scatter(t2,imeigs[1,:],linewidths=0.01)
'''

mpl.plot(t2,y2[:,1], label='CVode')

#mpl.plot(t4,y4[:,1], label='BDF-2')
#mpl.title('BDF-2, h = 0.01, k = {}' .format(k))

#mpl.plot(t3,y3[:,1], label='BDF-3')
#mpl.title('BDF-3, h = 0.01, k = {}' .format(k))

mpl.plot(t,y[:,1], label='BDF-4')
#mpl.title('BDF-4, h = 0.01, k = {}' .format(k))

mpl.plot(t5,y5[:,1], label='BDF-5')


mpl.ylabel('y2')
mpl.xlabel('t')
mpl.legend(loc='lower left')
mpl.ylim(-1.2,1.2)
mpl.show()





