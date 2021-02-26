#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 13:33:35 2021

@author: kadde
"""
from assimulo.ode import *


class Explicit_Problem_2nd(Explicit_Problem):
    
    def __init__(self, rhs, y0, t0, M=None, K=None, C=None):
        Explicit_Problem.__init__(self,rhs, y0, t0)
        self.M  = M;
        self.C = C;
        self.K = K;
        
    
    