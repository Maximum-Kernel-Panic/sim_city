#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 17:50:45 2021

@author: erik
"""
import numpy as np
import pylab as pl
from assimulo.problem import Explicit_Problem
from assimulo.solvers import CVode, ExplicitEuler


def rhs(t,y):
    k = 100
    lmd = k*(np.sqrt(y[0]*y[0] + y[1]*y[1])-1)/np.sqrt(y[0]*y[0] + y[1]*y[1])
    y4d = -y[1]*lmd-1
    y3d = -y[0]*lmd
    y2d = y[3]
    y1d = y[2]
    
    yd = np.array([y1d, y2d, y3d, y4d])
    
    return yd


global tfinal
tfinal=20.0

#Run either rtol or atol simulation for all tolerances given in Tols and plot the result for the given coordinate in xVals. Reset solver to default values afterwards.
#xVals -True => plot for y1, false =>
#rtol - True => vary rtol parameter, false => vary atol parameter
#sim - the CVODE object
#Tols - numpy array of desired tolerances to study
def runAtolRtolSim(xVals,rtol,sim,Tols):
    global tfinal
    for i,x in enumerate(Tols):
        if rtol:
            theLabel="RTOL = "
            sim.rtol=x
        else:
            theLabel="ATOL = "
            sim.atol=x*np.ones(4)
        t,y=sim.simulate(tfinal)
        if xVals:
            yVals=y[:,0]
        else:
            yVals=y[:,1]            
        pl.plot(t,yVals,label=theLabel+str(x))
        sim.reset()
    if rtol:
        sim.rtol=1e-6
    else:
        sim.atol=1e-6*np.ones(4)

#Check how the norm of the given coordinate of the solution depends on the ATOL/RTOL parameter. Reset solver to default values afterwards.
#xVals -True => plot for y1, false =>
#rtol - True => vary rtol parameter, false => vary atol parameter
#sim - the CVODE object
#xTol - numpy array of desired tolerances to study        
def runNormVsTolSim(xVals,rtol,xTol,sim):
    global tfinal
    for i,x in enumerate(xTol):
        if rtol:
            sim.rtol=x
        else:
            sim.atol=x*np.ones(4)
        t,y=sim.simulate(tfinal)
        if xVals:
            theVal=y[:,0]
        else:
            theVal=y[:,1]
        normValues[i]=np.linalg.norm(theVal)
        sim.reset()
    if Rtol:
        sim.rtol=1e-6
    else:
        sim.atol=1e-6*np.ones(4)

#Run simulation varying the MAXORD of the solver for a given coordinate xVals. Reset solver to default values afterwards.
#xVals -True => plot for y1, false => plot for y2
#sim - the CVODE object
#order - numpy array of desired orders to simulate
def runMaxOrdSim(xVals,sim,order):
    global tfinal
    for i,x in enumerate(order):
        #print("    ----   "+str(xVals)+"    -----"+str(sim.y0[0]))
        sim.maxord=x
        t,y=sim.simulate(tfinal)
        #sim.print_statistics()
        if xVals:
            yVals=y[:,0]
        else:
            yVals=y[:,1]            
        pl.plot(t,yVals,label="ORDER = "+str(x))
        sim.reset()
    sim.maxord=5
    
#Run simulation varying the DISCR parameter of the solver for a given coordinate xVals. Reset solver to default values afterwards.
#xVals -True => plot for y1, false => plot for y2
#sim - the CVODE object
def runDiscMethodSim(sim,xVals):
    global tfinal
    for Adams in [0,1]:
        if Adams:
            method="ADAMS"
            sim.discr="Adams"
        else:
            method="BDF"
        #print("    ----   "+str(xVals)+"    -----"+str(sim.y0[0]))
        t,y=sim.simulate(tfinal)
        #sim.print_statistics()
        if xVals:
            yVals=y[:,0]
        else:
            yVals=y[:,1]            
        pl.plot(t,yVals,label="METHOD = "+method)
        sim.reset()
        sim.discr="BDF"

#Plot AtolRtol Simulation and save the image to folder.
#x true => plot for y1, x false => plot for y2
#rtol true => plot for rtol simulation, rtol false => plot for atol simulation
#highOsc true => plot for high oscillating case, highOsc false => plot for low oscillating case
def LabelAtolRtolSim(x,rtol,highOsc):
    pl.legend(loc="center right")    
    pl.xlabel("Time[t]")
    pl.ylabel("Distance[m]")
    if rtol:
        titleText_2="RTOL"
        fileText_2="Rtol"
    else:
        titleText_2="ATOL"
        fileText_2="Atol"
    if highOsc:
        titleText_1="High Oscillation: "
        fileText_1="High_Osc_"
    else:
        titleText_1="Low Oscillation: "
        fileText_1="Low_Osc_"
    pl.title("plotted for different values of "+titleText_2)
    if x:
        titleText_3="Y1"
    else:
        titleText_3="Y2"
    pl.suptitle(titleText_1+titleText_3+"-coordinate as function of time")
    pl.savefig("./Project_1_Task_4_Plots/task_4_"+fileText_1+fileText_2+"VsTime_"+titleText_3+".png")

#Plot DISC Simulation and save the image to folder.
#xVals true => plot for y1, xVals false => plot for y2
#highOsc true => plot for high oscillating case, highOsc false => plot for low oscillating case
def labelDiscMethodSim(highOsc,xVals):
    pl.legend(loc="center right")    
    pl.xlabel("Time[t]")
    pl.ylabel("Distance[m]")
    pl.title("Plotted for different Discretization methods")
    if highOsc:
        FileName_1="High_Osc_"
        LabelName_1="High Oscillation: "
    else:
        FileName_1="Low_Osc_"
        LabelName_1="Low Oscillation: "
    if xVals:
        LabelFileName="Y1"
    else:
        LabelFileName="Y2"
    pl.suptitle(LabelName_1+LabelFileName+"-coordinate as function of time")
    pl.savefig("./Project_1_Task_4_Plots/task_4_"+FileName_1+"DiscretizationVsTime_"+LabelFileName+".png")

#Plot MAXORD Simulation and save the image to folder.
#xVals true => plot for y1, xVals false => plot for y2
#highOsc true => plot for high oscillating case, highOsc false => plot for low oscillating case
def labelMaxOrdSim(xVals,highOsc):
    pl.legend(loc="center right")    
    pl.xlabel("Time[t]")
    pl.ylabel("Distance[m]")
    pl.title("Plotted for different MAXORD values")
    if highOsc:
        FileName_1="High_Osc_"
        LabelName_1="High Oscillation: "
    else:
        FileName_1="Low_Osc_"
        LabelName_1="Low Oscillation: "
    if xVals:
        LabelFileName="Y1"
    else:
        LabelFileName="Y2"
    pl.suptitle(LabelName_1+LabelFileName+"-coordinate as function of time")
    pl.savefig("./Project_1_Task_4_Plots/task_4_"+FileName_1+"MAXORDERVsTime_"+LabelFileName+".png")

    

offset=0.05
xinit = np.sin(20*np.pi/180)
yinit = np.cos(20*np.pi/180)
sqrt2 = np.sqrt(0.5);
y0_low = np.array([xinit, yinit, 0.0, 0])
y0_high = np.array([xinit+offset, yinit+offset, 0.0, 0])
t0 = 0.0

#Low oscillation object
model_low_osc = Explicit_Problem(rhs, y0_low, t0)
model_low_osc.name = 'Linear Test ODE'

#high oscillation object
model_high_osc = Explicit_Problem(rhs, y0_high, t0)
model_high_osc.name = 'Linear Test ODE'



sim_low = CVode(model_low_osc)
sim_low.verbosity=50
sim_high = CVode(model_high_osc)
sim_high.verbosity=50
Sims=[sim_low,sim_high]


#xTol=np.logspace(-6,-1,num=50)
#normValues=np.zeros(xTol.shape)
#tfinal = 20.0
#sim.verbosity=50


#pl.plot(xTol,normValues)
#pl.xscale("log")

Tols=np.array([1e-6,1e-2,1e-1])

pl.figure()

#Runs both atol and rtol simulation for both coordinates and for both high- and low oscillating case. Plot the result and save the images.
for i,sim in enumerate(Sims):
    for xVal in [0,1]:
        for Rtol in [0,1]:
            runAtolRtolSim(xVal,Rtol,sim,Tols)
            LabelAtolRtolSim(xVal,Rtol,i)
            pl.figure()
            
#Runs MAXORD simulation for both coordinates and for both high- and low oscillating case. Plot the result and save the images.
Orders=np.array([2,3,4,5])
for i,sim in enumerate(Sims):
    for xVal in [0,1]:
        runMaxOrdSim(xVal,sim,Orders)
        labelMaxOrdSim(xVal,i)
        pl.figure()
        
#Runs DISCR simulation for both coordinates and for both high- and low oscillating case. Plot the result and save the images.
for i,sim in enumerate(Sims):
    for xVal in [0,1]:
        runDiscMethodSim(sim,xVal)
        labelDiscMethodSim(xVal,i)
        if not (xVal==1 and i==1):
            pl.figure()


