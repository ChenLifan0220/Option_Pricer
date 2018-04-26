#Assignment 3 
#encoding:utf-8

'''
European call/put option
S - Stock price
K - Strike price
T - Time to maturity
t - Any time less than T
tau - Volatility
r - Risk free interest
q - repo rate
sigma_true - The true value of sigma
'''

import scipy.stats
from scipy.stats import norm
import scipy
import sympy as sympy
import numpy as np
import matplotlib.pyplot as plt
from math import *
import csv as csv
import pandas as pd
import time

def BlackScholes(CallPutFlag, S, K, t, T, tau, r, q):
    d1 = (log(float(S)/K)+(r-q)*(T-t))/(tau*sqrt(T-t))+(1/2)*tau*sqrt(T-t)
    d2 = d1-tau*sqrt(T-t)
    if CallPutFlag =='C':
        N1 = norm.cdf(np.float64(d1))
        N2 = norm.cdf(np.float64(d2))
        return S*exp(-q*(T-t))*N1-K*exp(-r*(T-t))*N2
    elif CallPutFlag == 'P':
        N3 = norm.cdf(np.float64(-d1))
        N4 = norm.cdf(np.float64(-d2))
        return K*exp(-r*(T-t))*N4-S*exp(-q*(T-t))*N3

def Vega(S,K,T,t,sigma,r,q):
	tau = sympy.Symbol('tau')
	y = S*sympy.exp(-q*(T-t))*sympy.sqrt(T-t)*1/sympy.sqrt(2*pi)*sympy.exp(-((sympy.log(S/K)+(r-q)*(T-t))/(tau*sympy.sqrt(T-t))+(1/2)*tau*sympy.sqrt(T-t))**2/2)
	yPrime = y.evalf(subs = {tau:sigma})
	return yPrime

def Implied_Volatility_Calculation(CallPutFlag,S,K,T,t,r,q,market_value):
    sigmahat = sqrt(2*abs((log(S/K)+(r-q)*(T-t))/(T-t)))
	#print('sigmahat',sigmahat)
    tol = 1e-5
    sigma = sigmahat
    if CallPutFlag == 'C':
        C_true = market_value
    elif CallPutFlag == 'P':
        P_true = market_value
    sigmadiff = 1
    n = 1
    nmax = 100
    if S-K*exp(-r*T)>0:
        CLower = S-K*exp(-r*T)
    else:
        CLower = 0

    if K*exp(-r*T)-S>0:
        PLower = K*exp(-r*T)-S
    else:
        PLower = 0

    while(sigmadiff >= tol and n < nmax):
        if CallPutFlag == 'C':
            C = BlackScholes('C',S,K,t,T,sigma,r,q)
            #print('C',C)
            if (C<=S*exp(-q*(T-t))-K*exp(-r*(T-t)) or C>=S*exp(-q*(T-t))):
                return 'NaN'
                break
            else:
                Cvega = Vega(S,K,T,t,sigma,r,q)
                #print('Cvega',Cvega)
                if round(Cvega,6) == 0.0:
                    return 'NaN'
                    break
                increment = (C - C_true)/Cvega
                #print('increment',increment)
        elif CallPutFlag == 'P':
            P = BlackScholes('P',S,K,t,T,sigma,r,q)
            #print('P',P)
            if (P>=K*exp(-r*(T-t)) or P<=(K-S*exp(-r*(T-t)))):
                return "NaN"
                break
            else:
                Pvega = Vega(S,K,T,t,sigma,r,q)
                #print('Pvega',Pvega)
                if round(Pvega,6) == 0.0:
                    return 'NaN'
                    break
                increment = (P - P_true)/Pvega
                #print('increment',increment)
        sigma = sigma - increment
        #print('sigma',sigma)
        n = n+1
        sigmadiff=abs(increment)
        #print('sigmadiff',sigmadiff)
    return sigma

    def sigma_basket(asset_no,sigma_respective,rau_matrix):
        intermidiate=0
        for x in range(1,asset_no+1):
            for y in range(1,asset_no+1):
                intermidiate=intermidiate+sigma_basket[i-1]*sigma_respective[j-1]*rau_matrix[i-1][j-1]
        sigma_basket=sqrt(intermidiate)/asset_no