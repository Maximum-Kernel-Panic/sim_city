# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 13:50:37 2021

@author: kadde
"""

#import squeezer 
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from assimulo.problem import Implicit_Problem
from assimulo.solvers import IDA
import squeezer


a = squeezer.Seven_bar_mechanism()
t0 = 0
y0, yp0 = a.init_squeezer()


model = Implicit_Problem(a.f, y0, yp0, t0) #Create an Assimulo problem
sim = IDA(model)
sim.atol[7:20] = 10000
sim.algvar[7:20] = 0

te = 0.03
t, y, yp =  sim.simulate(te)
sim.plot()
plt.plot(t,y[:,0:7])
plt.legend()

print("hello, world")