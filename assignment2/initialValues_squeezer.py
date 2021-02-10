# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 13:36:18 2021

@author: Magnus
"""
from scipy import *
from numpy import array
class initialValues_squeezer():
    
    def __init__(self,Values):
        self.Values=Values
    
    def getInitalAngles(self):
        m1,m2,m3,m4,m5,m6,m7=.04325,.00365,.02373,.00706,.07050,.00706,.05498
        i1,i2,i3,i4,i5,i6,i7=2.194e-6,4.410e-7,5.255e-6,5.667e-7,1.169e-5,5.667e-7,1.912e-5
		# Geometry
        xa,ya=-.06934,-.00227
        xb,yb=-0.03635,.03273
        xc,yc=.014,.072
        d,da,e,ea=28.e-3,115.e-4,2.e-2,1421.e-5
        rr,ra=7.e-3,92.e-5
        ss,sa,sb,sc,sd=35.e-3,1874.e-5,1043.e-5,18.e-3,2.e-2
        ta,tb=2308.e-5,916.e-5
        u,ua,ub=4.e-2,1228.e-5,449.e-5
        zf,zt=2.e-2,4.e-2
        fa=1421.e-5
        # Driving torque
        mom=0.033
        # Spring data
        c0=4530.
        lo=0.07785

        # Initial computations and assignments
        beta=self.Values
		#theta,gamma,phi,delta,omega,epsilon=y[0:7]        
        sibeth =lambda x: sin(beta+x[0]);cobeth =lambda x: cos(beta+x[0])
        siphde =lambda x: sin(x[2]+x[3]);cophde =lambda x: cos(x[2]+x[3])
        siomep =lambda x: sin(x[4]+x[5]);coomep =lambda x: cos(x[4]+x[5])
        g=zeros((6,))
        g_0 =lambda x: rr*cos(beta) - d*cobeth(x) - ss*sin(x[1]) - xb
        g_1 =lambda x: rr*sin(beta) - d*sibeth(x) + ss*cos(x[1]) - yb
        g_2 =lambda x: rr*cos(beta) - d*cobeth(x) - e*siphde(x) - zt*cos(x[3]) - xa
        g_3 =lambda x: rr*sin(beta) - d*sibeth(x) + e*cophde(x) - zt*sin(x[3]) - ya
        g_4 =lambda x: rr*cos(beta) - d*cobeth(x) - zf*coomep(x) - u*sin(x[5]) - xa
        g_5 =lambda x: rr*sin(beta) - d*sibeth(x) - zf*siomep(x) + u*cos(x[5]) - ya        
        
        f=lambda x: [g_0(x),g_1(x),g_2(x),g_3(x),g_4(x),g_5(x)]
        return f
        