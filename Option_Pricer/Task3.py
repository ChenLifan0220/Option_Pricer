#Assignment 3 
#Task 3
#Geometric Asia option & Geomeetric basket option
#Group 7
#encoding:utf-8

'''
Geometric Asia call/put option
S - Stock price
K - Strike price
T - Time to maturity
t - Any time less than T
r - Risk free interest
sigma - Volatility
'''

from scipy.stats import norm
import scipy
import sympy as sympy
import numpy as np
import matplotlib.pyplot as plt
from math import *
import pandas as pd

def Geometric_Asia_Option(CallPutFlag,S,K,sigma,r,T,t,n):
	sigmahat=sigma*sqrt((n+1)*(2*n+1)/(6*n*n))
	miuhat=(r-(1/2)*sigma*sigma)*((n+1)/(2*n))+(1/2)*sigmahat*sigmahat
	d1=(log(S/K)+(miuhat+(1/2)*sigmahat*sigmahat)*(T-t))/(sigmahat*sqrt(T-t))
	d2=d1-sigmahat*sqrt(T-t)
	if CallPutFlag=='C':
		N1=norm.cdf(np.float64(d1))
		N2=norm.cdf(np.float64(d2))
		return exp(-r*(T-t))*(S*exp(miuhat*(T-t))*N1-K*N2)
	elif CallPutFlag=='P':
		N3=norm.cdf(np.float64(-d2))
		N4=norm.cdf(np.float64(-d1))
		return exp(-r*(T-t))*(K*N3-S*exp(miuhat*(T-t))*N4)

def Geometric_Basket_Option(CallPutFlag,S1,S2,K,tau1,tau2,rau,r,T,t):
	sigmaBg=sqrt(tau1*tau1+tau1*tau2*rau+tau2*tau1*rau+tau2*tau2)/2
	miuBg=r-1/2*(tau1*tau1+tau2*tau2)/2+1/2*sigmaBg*sigmaBg
	Bg0=pow((S1*S2),1/2)
	d1=(log(Bg0/K)+(miuBg+1/2*sigmaBg*sigmaBg)*(T-t))/(sigmaBg*sqrt(T-t))
	d2=d1-sigmaBg*sqrt(T-t)
	if CallPutFlag=='C':
		N1=norm.cdf(np.float64(d1))
		N2=norm.cdf(np.float64(d2))
		return exp(-r*(T-t))*(Bg0*exp(miuBg*(T-t))*N1-K*N2)
	elif CallPutFlag=='P':
		N3=norm.cdf(np.float64(-d2))
		N4=norm.cdf(np.float64(-d1))
		return exp(-r*(T-t))*(K*N3-Bg0*exp(miuBg*(T-t))*N4)

def Geometric_Basket_Option2(CallPutFlag,sigmaBg,miuBg,Bg0,K,r,T,t):
	d1=(log(Bg0/K)+(miuBg+1/2*sigmaBg*sigmaBg)*(T-t))/(sigmaBg*sqrt(T-t))
	d2=d1-sigmaBg*sqrt(T-t)
	if CallPutFlag=='C':
		N1=norm.cdf(np.float64(d1))
		N2=norm.cdf(np.float64(d2))
		return exp(-r*(T-t))*(Bg0*exp(miuBg*(T-t))*N1-K*N2)
	elif CallPutFlag=='P':
		N3=norm.cdf(np.float64(-d2))
		N4=norm.cdf(np.float64(-d1))
		return exp(-r*(T-t))*(K*N3-Bg0*exp(miuBg*(T-t))*N4)

def sigma_basket(asset_no,sigma_respective,rau_matrix):
	intermidiate=0
	for x in range(1,asset_no+1):
		for y in range(1,asset_no+1):
			intermidiate=intermidiate+sigma_respective[x-1]*sigma_respective[y-1]*rau_matrix[x-1][y-1]
	return sqrt(intermidiate)/asset_no
'''
#Test Geometric_Asia_Option
print("Geometric_Asia_Option")
print(Geometric_Asia_Option('P',100,100,0.3,0.05,3,0,50))
print(Geometric_Asia_Option('P',100,100,0.3,0.05,3,0,100))
print(Geometric_Asia_Option('P',100,100,0.4,0.05,3,0,50))

print(Geometric_Asia_Option('C',100,100,0.3,0.05,3,0,50))
#print(Geometric_Asia_Option('C',100,100,0.3,0.05,3,0,100))
#print(Geometric_Asia_Option('C',100,100,0.4,0.05,3,0,50))
'''
'''
#Test Geometric_Basket_Option
print("Test Geometric_Basket_Option")
print(Geometric_Basket_Option('P',100,100,100,0.3,0.3,0.5,0.05,3,0))
print(Geometric_Basket_Option('P',100,100,100,0.3,0.3,0.9,0.05,3,0))
print(Geometric_Basket_Option('P',100,100,100,0.1,0.3,0.5,0.05,3,0))
print(Geometric_Basket_Option('P',100,100,80,0.3,0.3,0.5,0.05,3,0))
print(Geometric_Basket_Option('P',100,100,120,0.3,0.3,0.5,0.05,3,0))
print(Geometric_Basket_Option('P',100,100,100,0.5,0.5,0.5,0.05,3,0))
print(Geometric_Basket_Option('C',100,100,100,0.3,0.3,0.5,0.05,3,0))
print(Geometric_Basket_Option('C',100,100,100,0.3,0.3,0.9,0.05,3,0))
print(Geometric_Basket_Option('C',100,100,100,0.1,0.3,0.5,0.05,3,0))
print(Geometric_Basket_Option('C',100,100,80,0.3,0.3,0.5,0.05,3,0))
print(Geometric_Basket_Option('C',100,100,120,0.3,0.3,0.5,0.05,3,0))
print(Geometric_Basket_Option('C',100,100,100,0.5,0.5,0.5,0.05,3,0))
'''