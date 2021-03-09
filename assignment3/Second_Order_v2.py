# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:34:49 2021

@author: Magnus
"""
from assimulo.explicit_ode import Explicit_ODE
from assimulo.ode import *
import numpy as np
import matplotlib.pyplot as mpl
import scipy.linalg as SL
import scipy.sparse.linalg as ssl
import numpy.linalg as nplin

#Explicit Newmark Method
class Explicit_Newmark(Explicit_ODE):
    tol=1.e-5     
    maxit=100000     
    maxsteps=200000
    
    def __init__(self, problem):
        Explicit_ODE.__init__(self, problem) #Calls the base class
        
        #Solver options
        self.options["h"] = 0.005
        
        #Statistics
        self.statistics["nsteps"] = 0
        self.statistics["nfcns"] = 0
    
    def _set_h(self,h):
            self.options["h"] = float(h)

    def _get_h(self):
        return self.options["h"]
        
    h=property(_get_h,_set_h)
        
    def integrate(self, t, y, tf, opts):
        """
        _integrates (t,y) values until t > tf
        """
        h = self.options["h"]
        h = min(h, abs(tf-t))
        
        #Lists for storing the result
        tres = []
        yres = []
        ypres = []
        yppres = []
        
        
        for i in range(self.maxsteps):
            if t >= tf:
                break
            self.statistics["nsteps"] += 1
            
            if i==0:  # initial step
                yp=y[int(len(y)/2):]
                y=y[:int(len(y)/2)]
                t,ypp = self.step_Newmark_explicit_init(t,y,yp, h,opts)
                i = i + 1
            else:   
                t,y,yp,ypp = self.step_Newmark_explicit(t,y,yp,ypp, h,opts)
            
            
            tres.append(t)
            yres.append(np.concatenate((y.copy(),yp.copy())))
            ypres.append(yp.copy())
            yppres.append(ypp.copy())
            h=min(self.h,np.abs(tf-t))
        else:
            raise Explicit_ODE_Exception('Final time not reached within maximum number of steps')
        

        return ID_PY_OK, tres, yres
    
    def step_Newmark_explicit_init(self, t, y, yp, h,opts):
        """
        This calculates the next step in the integration with explicit Euler.
        """
        self.statistics["nfcns"] += 1
        
        f = self.problem.rhs
        M = self.problem.M
        C = self.problem.C
        K = self.problem.K
        if C==None:
            ypp=ssl.spsolve(M,f(t,np.concatenate((y.copy(),yp.copy())))[:int(len(y))]-K(t,y)@y)
        else:
            ypp=ssl.spsolve(M,f(t,np.concatenate((y.copy(),yp.copy())))[:int(len(y))]-C(M,K(t,y))@yp-K(t,y)@y)              
        return t + h, ypp
    
    def step_Newmark_explicit(self,t,y,yp,ypp,h,opts):
        f = self.problem.rhs
        M = self.problem.M
        C = self.problem.C
        K = self.problem.K        
        y=y+yp*h+ypp*h**2/2
        if C==None:
            ypp_new=ssl.spsolve(M,f(t,np.concatenate((y.copy(),yp.copy())))[:int(len(y))]-K(t,y)@y)
        else:
            ypp_new=ssl.spsolve(M,f(t,np.concatenate((y.copy(),yp.copy())))[:int(len(y))]-C(M,K(t,y))@yp-K(t,y)@y)            
        yp=yp+ypp*h/2+ypp_new*h/2
        return t+h,y,yp,ypp_new
            
    def print_statistics(self, verbose=NORMAL):
        self.log_message('Final Run Statistics            : {name} \n'.format(name=self.problem.name),        verbose)
        self.log_message(' Step-length                    : {stepsize} '.format(stepsize=self.options["h"]), verbose)
        self.log_message(' Number of Steps                : '+str(self.statistics["nsteps"]),          verbose)               
        self.log_message(' Number of Function Evaluations : '+str(self.statistics["nfcns"]),         verbose)
            
        self.log_message('\nSolver options:\n',                                    verbose)
        self.log_message(' Solver            : BDF3',                     verbose)
        self.log_message(' Solver type       : Fixed step\n',                      verbose)

# -*- coding: utf-8 -*-
# from assimulo.solvers import CVode

class HHT_a(Explicit_ODE):
    """
    HHT_a
    """
    tol=1.e-8     
    maxit=100000     
    maxsteps=20000
    
    def __init__(self, problem, alpha=0):
        Explicit_ODE.__init__(self, problem) #Calls the base class
        self.alpha = alpha;
        self.f = problem.rhs
        self.gamma = 1/2 - alpha
        self.beta = ((1-alpha)/2)**2
        #Solver options
        self.options["h"] = 0.01
        if problem.C == None:
            problem.C = np.zeros(np.shape(problem.M))
        #Statistics
        self.statistics["nsteps"] = 0
        self.statistics["nfcns"] = 0
    
    def _set_h(self,h):
            self.options["h"] = float(h)

    def _get_h(self):
        return self.options["h"]
        
    h=property(_get_h,_set_h)
        
    def integrate(self, t, y, tf, opts):
        """
        _integrates (t,y) values until t > tf
        """
        h = self.options["h"]
        h = min(h, abs(tf-t))
        
        #Lists for storing the result
        tres = []
        yres = []
        ypres = []
        
        yp = y[int(len(y)/2):]
        y = y[:int(len(y)/2)]
        for i in range(self.maxsteps):
            if t >= tf:
                break
            self.statistics["nsteps"] += 1
            if i == 0:
                ypp = self.init_HHT(t,y,yp)

            else:    
                t, y, yp, ypp = self.HHT_step(t,y,yp,ypp,h)
            
            
            tres.append(t)
            yres.append(np.concatenate((y.copy(),yp.copy())))
        
            h=min(self.h,np.abs(tf-t))
        else:
            raise Explicit_ODE_Exception('Final time not reached within maximum number of steps')
        
        return ID_PY_OK, tres, yres
    
    def init_HHT(self,t,y,yp):
            rhs = self.f(t,np.concatenate((y.copy(),yp.copy())))[:int(len(y))] - self.problem.C(0,0)@yp - self.problem.K(t,y)@y     
            ypp = ssl.spsolve(self.problem.M,rhs)
            return ypp
        
    def HHT_step(self,t,y,yp,ypp,h):
            # PLACEHOLDER f !!!
            # Is the fucntion fixed step? 
            
            # eq 8''
            rhs1 =  self.problem.M@(y/(self.beta*(h**2)) + yp/(self.beta*h) + (1/(2*self.beta) - 1)*ypp)
            rhs2 = self.problem.C(0,0)@( (self.gamma*y)/(self.beta*h) - (1- self.gamma/self.beta)*yp -(1 - (self.gamma/(2*self.beta)))*h*ypp)
            rhs3 = self.alpha*self.problem.K(t,y)@y
            rhs = self.f(t,np.concatenate((y.copy(),yp.copy())))[:int(len(y))] + rhs1 + rhs2 + rhs3
            
            lhs = ( self.problem.M/(self.beta*h**2) + (self.gamma*self.problem.C(0,0))/(self.beta*h) + (1 + self.alpha)*self.problem.K(t,y))
            ytp1 = ssl.spsolve(lhs,rhs)
                        
            # eq 6'
            ypptp1 = (ytp1 - y)/(self.beta*h**2) - yp/(self.beta*h) - (1/(2*self.beta) - 1)*ypp
            
            # eq 7'
            yptp1 = (self.gamma/self.beta)*(ytp1-y)/h + (1 - self.gamma/self.beta)*yp + (1 - (self.gamma/(2*self.beta)))*h*ypp
            

            return t+h,ytp1, yptp1,ypptp1
            
    def print_statistics(self, verbose=NORMAL):
        self.log_message('Final Run Statistics            : {name} \n'.format(name=self.problem.name),        verbose)
        self.log_message(' Step-length                    : {stepsize} '.format(stepsize=self.options["h"]), verbose)
        self.log_message(' Number of Steps                : '+str(self.statistics["nsteps"]),          verbose)               
        self.log_message(' Number of Function Evaluations : '+str(self.statistics["nfcns"]),         verbose)
            
        self.log_message('\nSolver options:\n',                                    verbose)
        self.log_message(' Solver            : BDF4',                     verbose)
        self.log_message(' Solver type       : Fixed step\n',                      verbose)
  


'''           
#Define the rhs
def f(t,y):
    ydot = -y[0]
    return np.array([ydot])
    
#Define an Assimulo problem
exp_mod = Explicit_Problem(f, 4)
exp_mod.name = 'Simple BDF-3 Example'

#Define another Assimulo problem
def pend(t,y):
    #g=9.81    l=0.7134354980239037
    gl=13.7503671
    return np.array([y[1],-gl*np.sin(y[0])])
    
pend_mod=Explicit_Problem(pend, y0=np.array([2.*np.pi,1.]))
pend_mod.name='Nonlinear Pendulum'

#Define an explicit solver
exp_sim = BDF_3(pend_mod) #Create a BDF solver
t, y = exp_sim.simulate(4)
exp_sim.plot()
mpl.show()
'''
        