#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 16:51:51 2021

@author: erik
"""

import numpy as np
import pylab as pl
from assimulo.problem import Explicit_Problem
from assimulo.solvers import CVode

def rhs(t,y):
    A = np.array([[0,1],[-2,-1]])
    yd = np.dot(A,y)
    
    return yd

y0 = np.array([1.0, 1.0])
t0 = 0.0

model = Explicit_Problem(rhs, y0, t0)
model.name = 'Linear Test ODE'

sim = CVode(model)
tfinal = 10.0

t, y = sim.simulate(tfinal)
pl.plot(t,y)
pl.show()