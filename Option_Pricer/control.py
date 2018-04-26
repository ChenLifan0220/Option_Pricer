# -*- coding: utf-8 -*-

from flask import Flask, request, render_template,url_for,redirect,json,send_from_directory
import scipy.stats
from scipy.stats import norm
import scipy
import sympy as sympy
import numpy as np
import matplotlib.pyplot as plt
from math import *
import Task12
import Task3
from sobol import i4_sobol
import os

control = Flask(__name__)
root=os.path.join(os.path.dirname(os.path.abspath(__file__)),"templates","static")
@control.route('/<path:path>',methods=['GET'])
def static_proxy(path):
    return send_from_directory(root,path)

#Functions
#Arithmetic Asian Option
'''
def Quasi_Random_No(dimension,simu_no):
    sequencer=ghalton.Halton(dimension)
    points=sequencer.get(simu_no)
    result = norm.ppf(points)
    #sequencer.reset()
    return result
'''

def i4_sobol_generate(dim_num, n, skip):
    r = np.full((n, dim_num), np.nan)
    for j in range(n):
        seed = j + skip
        r[j, 0:dim_num], next_seed = i4_sobol(dim_num, seed)
    return r

def Quasi_Random_Sobol(dimension,simu_no,skip=1):
    sobols = i4_sobol_generate(dimension, simu_no, skip)
    #print(sobols)
    normals = norm.ppf(sobols)
    return normals    

def Arithmetic_Asia_Option(Path_No,is_Control_Variate,CallPutFlag,S,K,sigma,r,T,t,n):   
    #geo_expected=apply(Geometric_Asia_Option,(CallPutFlag,S,K,sigma,r,T,t,n))  just in python 2.
    geo_expected=Task3.Geometric_Asia_Option(CallPutFlag,S,K,sigma,r,T,t,n)
    drift=exp((r-0.5*sigma*sigma)*T/n)
    arithPayoff_seq=[]
    geoPayoff_seq=[]
    #generate std norm rnd matrix
    for i in range(1,Path_No+1):
        #ST_path=[]
        Spath=[] 
        growthRatio=drift*exp(sigma*sqrt((T-t)/n)*np.random.standard_normal())
        Spath.append(S*growthRatio)
        for j in range(2,n+1):
            growthRatio=drift*exp(sigma*sqrt((T-t)/n)*np.random.standard_normal())
            Spath.append(Spath[j-2]*growthRatio)
        # Arithmetic mean
        arithMean=np.mean(Spath)
        geoMean=exp((1.0/n)*sum(np.log(Spath))) #math log() limit to single value
        if(CallPutFlag=='C'):
            arithPayoff=exp(-r*(T-t))*max(0,arithMean-K)
            geoPayoff=exp(-r*(T-t))*max(0,geoMean-K)
            arithPayoff_seq.append(arithPayoff)
            geoPayoff_seq.append(geoPayoff)
        elif(CallPutFlag=='P'):
            arithPayoff=exp(-r*(T-t))*max(0,K-arithMean)
            geoPayoff=exp(-r*(T-t))*max(0,K-geoMean)
            arithPayoff_seq.append(arithPayoff)
            geoPayoff_seq.append(geoPayoff)
    geoMCmean=np.mean(geoPayoff_seq)
        #control variate version
    if(is_Control_Variate=="1"):
        theta=np.cov(arithPayoff_seq, geoPayoff_seq)/np.var(geoPayoff_seq)
        Target=[]
        for i in range(0,len(arithPayoff_seq)):
            T_value=arithPayoff_seq[i] + theta * (geo_expected - geoPayoff_seq[i]) 
            Target.append(T_value)
        Tmean = np.mean(Target)
        Tstd = np.std(Target)
        confcv_lower= Tmean-1.96*Tstd/sqrt(Path_No)
        confcv_upper=Tmean+1.96*Tstd/sqrt(Path_No)
        return Tmean,confcv_lower,confcv_upper
    elif(is_Control_Variate=="0"):
        Tmean = np.mean(arithPayoff_seq)
        Tstd = np.std(arithPayoff_seq)
        confcv_lower= Tmean-1.96*Tstd/sqrt(Path_No)
        confcv_upper=Tmean+1.96*Tstd/sqrt(Path_No) 
        return Tmean,confcv_lower,confcv_upper,geoMCmean

#Geometric Basket Option
def Geometric_Basket_Option2(CallPutFlag,sigmaBg,miuBg,Bg0,K,r,T,t):
	d1=(log(Bg0/K)+(miuBg+1.0/2*sigmaBg*sigmaBg)*(T-t))/(sigmaBg*sqrt(T-t))
	d2=d1-sigmaBg*sqrt(T-t)
	if CallPutFlag=='C':
		N1=norm.cdf(np.float64(d1))
		N2=norm.cdf(np.float64(d2))
		return exp(-r*(T-t))*(Bg0*exp(miuBg*(T-t))*N1-K*N2)
	elif CallPutFlag=='P':
		N3=norm.cdf(np.float64(-d2))
		N4=norm.cdf(np.float64(-d1))
		return exp(-r*(T-t))*(K*N3-Bg0*exp(miuBg*(T-t))*N4)

#Arithmetic Basket Option
def Arithmetic_Mean_Basket_Option_im(Path_No,is_Control_Variate,CallPutFlag,S,K,r,T,t,asset_no,sigma_respetive,sigma_bg,miu_bg,Bg0,rau_matrix): 
    #geo_expected=apply(Geometric_Asia_Option,(CallPutFlag,S,K,sigma,r,T,t,n))  just in python 2.
    #S0 is a list
    geo_expected=Geometric_Basket_Option2(CallPutFlag,sigma_bg,miu_bg,Bg0,K,r,T,t)
    for i in range(1,asset_no+1):
        for j in range(i+1,asset_no+1):
            rau_matrix[i-1][j-1]=0
    rau_matrix=np.mat(rau_matrix)
    #print("rau",rau_matrix)
    arithPayoff_seq=[]
    geoPayoff_seq=[]
    points=Quasi_Random_Sobol(asset_no,Path_No)
    for i in range (1,Path_No+1):
        #bgT=Bg0*exp((r-0.5*sigma_bg*sigma_bg)*(T-t)+sigma_bg*sqrt(T-t)*points[i-1][0])
        Spath=[]
        for j in range(1,asset_no+1):
            random_list=rau_matrix*(np.mat(points[i-1]).transpose())
            growthRatio=exp((r-0.5*sigma_respetive[j-1]*sigma_respetive[j-1])*(T-t)+sigma_respetive[j-1]*sqrt(T-t)*float(random_list[j-1][0]))
            Spath.append(S[j-1]*growthRatio)
        baT=np.mean(Spath)
        bgT=exp((1.0/asset_no)*sum(np.log(Spath)))
        if(CallPutFlag=='C'):
            arithPayoff=exp(-r*(T-t))*max(0,baT-K)
            geoPayoff=exp(-r*(T-t))*max(0,bgT-K)
            arithPayoff_seq.append(arithPayoff)
            geoPayoff_seq.append(geoPayoff)
        elif(CallPutFlag=='P'):
            arithPayoff=exp(-r*(T-t))*max(0,K-baT)
            geoPayoff=exp(-r*(T-t))*max(0,K-bgT)
            arithPayoff_seq.append(arithPayoff)
            geoPayoff_seq.append(geoPayoff)
    #print(geoPayoff_seq)
    BasketStandardMC=np.mean(geoPayoff_seq)
        #control variate version
    if(is_Control_Variate=="1"):
        theta=np.cov(arithPayoff_seq, geoPayoff_seq)[0][1]/np.var(geoPayoff_seq)
        Target=[]
        for i in range(0,len(arithPayoff_seq)):
            T_value=arithPayoff_seq[i] + theta * (geo_expected - geoPayoff_seq[i]) 
            Target.append(T_value)
        Tmean = np.mean(Target)
        Tstd = np.std(Target)
        confcv_lower= Tmean-1.96*Tstd/sqrt(Path_No)
        confcv_upper=Tmean+1.96*Tstd/sqrt(Path_No)
        return Tmean,confcv_lower,confcv_upper
    elif(is_Control_Variate=="0"):
        Tmean = np.mean(arithPayoff_seq)
        Tstd = np.std(arithPayoff_seq)
        confcv_lower= Tmean-1.96*Tstd/sqrt(Path_No)
        confcv_upper=Tmean+1.96*Tstd/sqrt(Path_No) 
        return Tmean,confcv_lower,confcv_upper,BasketStandardMC

#Binominal tree
def Europe_binomial_tree(CallPutFlag,S,K,r,T,t,step_no,sigma):
    t_itv=(T-t)/step_no
    u=exp(sigma*sqrt(t_itv))
    d=1.0/u
    p_up=(exp(r*t_itv)-d)/(u-d)
    q=0 #repo rate
    intermidiate=0
    for i in range(0,step_no+1):
        if(CallPutFlag=="C"):
            intermidiate=intermidiate+max(S*pow(u,i)*pow(d,step_no-i)-K,0)*pow(p_up,i)*pow((1-p_up),step_no-i)*combination(step_no,i)
        elif(CallPutFlag=="P"):
            intermidiate=intermidiate+max(K-S*pow(u,i)*pow(d,step_no-i),0)*pow(p_up,i)*pow((1-p_up),step_no-i)*combination(step_no,i)
    result=intermidiate*exp(-r*(T-t))
    return result

def combination(step_no,select_no):
    result=factorial(step_no)/(factorial(select_no)*factorial(step_no-select_no))
    return result

def build_tree(step_no,u,d,S,K):
    tree=np.zeros((step_no+1,step_no+1))
    tree[0][0]=S
    for i in range(1,step_no+1):
        for j in range(1,i+1):
            tree[j][i]=tree[j-1][i-1]*d
        tree[0][i]=tree[0][i-1]*u
    return tree

def American_binomial_tree_im(CallPutFlag,S,K,r,T,t,step_no,sigma):
    t_itv=(T-t)/step_no
    u=exp(sigma*sqrt(t_itv))
    d=1.0/u
    p_up=(exp(r*t_itv)-d)/(u-d)
    df=exp(-r*t_itv)
    stock_tree=build_tree(step_no,u,d,S,K)
    if(CallPutFlag=="C"):   
        ame=Europe_binomial_tree(CallPutFlag,S,K,r,T,t,step_no,sigma)
        result=ame
    elif(CallPutFlag=="P"):
        for i in range(0,step_no+1):
            for j in range(0,i+1):
                stock_tree[j][i]=max(K-stock_tree[j][i],0)
        for i in range(step_no,0,-1):
            for j in range(0,i):
                stock_tree[j][i-1]=max(df*(p_up*stock_tree[j][i]+(1-p_up)*stock_tree[j+1][i]),stock_tree[j][i-1])
        result=stock_tree[0][0]
    return result

def modifed_binomial(is_Control_Variate,CallPutFlag,S,K,r,T,t,step_no,sigma):
    q=0 #repo rate
    bs_value=Task12.BlackScholes(CallPutFlag, S, K, t, T, sigma, r, q)
    #print(bs_value)
    if(is_Control_Variate=="1"):
        target=[]
        average_target=[]
        for i in range(1,step_no+1):
            #print(American_binomial_tree(CallPutFlag,S,K,r,T,t,i,sigma))
            #print(Europe_binomial_tree(CallPutFlag,S,K,r,T,t,i,sigma))
            target_value=American_binomial_tree_im(CallPutFlag,S,K,r,T,t,i,sigma)-Europe_binomial_tree(CallPutFlag,S,K,r,T,t,i,sigma)+bs_value
            target.append(target_value)
            #if(abs(target[i]-target[i-1])<=0.00001):
            #    break
        for i in range(len(target)-1):
            meanValue=0.5*(target[i]+target[i+1])
            average_target.append(meanValue)
        result=average_target[-1]
    elif(is_Control_Variate=="0"):
        result=American_binomial_tree_im(CallPutFlag,S,K,r,T,t,step_no,sigma)
        #result2=American_binomial_tree_im(CallPutFlag,S,K,r,T,t,step_no-1,sigma)
        #result=0.5*(result1+result2)
    return result

#route control
#index.html
@control.route('/',methods=['GET', 'POST'])
def index():
    return render_template('index.html')

#European_Option.html
@control.route('/EuropeanOption', methods=['GET'])
def European_Option_Form():
    return render_template('European_Option.html')

@control.route('/EuropeanOption', methods=['POST'])
def European_Option():
    CallPutFlag = request.form['CallPutFlag']
    S = float(request.form['S'])
    K = float(request.form['K'])
    r = float(request.form['r'])
    q = float(request.form['q'])
    t = float(request.form['t'])
    T = float(request.form['T'])
    tau = float(request.form['volatility'])
    EuropeanPrice = Task12.BlackScholes(CallPutFlag, S, K, t, T, tau, r, q)
    return render_template('European_Option.html', result=EuropeanPrice)

#Implied_Volatility.html
@control.route('/ImpliedVolatility', methods=['GET'])
def Implied_Volatility_Form():
    return render_template('Implied_Volatility.html')

@control.route('/ImpliedVolatility', methods=['POST'])
def Implied_Volatility():
    CallPutFlag = request.form['CallPutFlag']
    S = float(request.form['S'])
    K = float(request.form['K'])
    r = float(request.form['r'])
    q = float(request.form['q'])
    t = float(request.form['t'])
    T = float(request.form['T'])
    market_value = float(request.form['market_value'])
    Implied_Volatility = Task12.Implied_Volatility_Calculation(CallPutFlag,S,K,T,t,r,q,market_value)
    return render_template('Implied_Volatility.html', result=Implied_Volatility)

#Geometric_Asian_Option.html
@control.route('/GeometricAsianOption', methods=['GET'])
def Geometric_Asian_Option_Form():
    return render_template('Geometric_Asian_Option.html')

@control.route('/GeometricAsianOption', methods=['POST'])
def Geometric_Asian_Option():
    CallPutFlag = request.form['CallPutFlag']
    CalculationType = request.form['Calculation_Type']
    S = float(request.form['S'])
    K = float(request.form['K'])
    r = float(request.form['r'])
    t = float(request.form['t'])
    T = float(request.form['T'])
    tau = float(request.form['volatility'])
    n = int(request.form['n'])
    if CalculationType == "0":
        GeometricAsiaPrice = Task3.Geometric_Asia_Option(CallPutFlag,S,K,tau,r,T,t,n)
        return render_template('Geometric_Asian_Option.html', result=GeometricAsiaPrice)
    elif CalculationType == "1":
        AsiaStandardMC = Arithmetic_Asia_Option(100000,"0",CallPutFlag,S,K,tau,r,T,t,n)
        return render_template('Geometric_Asian_Option.html', result=AsiaStandardMC[3])

#Geometric_Basket_Option_2assets.html
@control.route('/GeometricBasketOption2assets', methods=['GET'])
def Geometric_Basket_Option_2assets_Form():
    return render_template('Geometric_Basket_Option_2assets.html')

@control.route('/GeometricBasketOption2assets', methods=['POST'])
def Geometric_Basket_Option_2assets():
    CallPutFlag = request.form['CallPutFlag']
    S1 = float(request.form['S1'])
    S2 = float(request.form['S2'])
    K = float(request.form['K'])
    r = float(request.form['r'])
    t = float(request.form['t'])
    T = float(request.form['T'])
    tau1 = float(request.form['volatility1'])
    tau2 = float(request.form['volatility2'])
    rau = float(request.form['rau'])
    GeometricBasketPrice2assets = Task3.Geometric_Basket_Option(CallPutFlag,S1,S2,K,tau1,tau2,rau,r,T,t)
    return render_template('Geometric_Basket_Option_2assets.html', result=GeometricBasketPrice2assets)

#Geometric_Basket_Option_nAssets.html
@control.route('/GeometricBasketOptionnAssets', methods=['GET'])
def Geometric_Basket_Option_nAssets_Form():
    return render_template('Geometric_Basket_Option_nAssets.html')

@control.route('/GeometricBasketOptionnAssets', methods=['POST'])
def Geometric_Basket_Option_nAssets():
    CallPutFlag = request.form['CallPutFlag']
    CalculationType=request.form['Calculation_Type']
    x = int(request.form['addnumber'])
    Bg=1
    sigmaBg=0
    miuBg=0
    Spotprice=[]
    tauValue=[]
    a=1
    i=1
    while(i < (x+1)):
        if(request.form['Spotprice'+str(i)]):
            Spotprice.append(float(request.form['Spotprice'+str(i)]))
            tauValue.append(float(request.form['Volatility'+str(i)]))
        else:
            x=x+1
        i+=1
    x = int(request.form['addnumber'])
    rau=request.form.getlist('g')
    rau1=rau[0].split(",,")
    rau2=[]
    for i in range(1,len(rau1)):
        rau2.append(rau1[i].split(","))
    for i in range(len(rau2)):
        for j in range(len(rau2)):
            rau2[i][j]=float(rau2[i][j])
    K = float(request.form['K'])
    r = float(request.form['r'])
    t = float(request.form['t'])
    T = float(request.form['T'])
    for i in range(0,x):
        Bg=Bg*Spotprice[i]
    Bg0=pow(Bg,(1/x))
    for i in range(0,x):
        for j in range(0,x):
            a=tauValue[i]*tauValue[j]
            sigmaBg=sigmaBg+a*rau2[i][j]
            a=1
    sigmaBg=sqrt(sigmaBg)/x
    for i in range(0,x):
        miuBg=miuBg+tauValue[i]*tauValue[i]
    miuBg=r-(1/2)*(miuBg/x)+(1/2)*sigmaBg*sigmaBg
    if CalculationType=="0":
        GeometricBasketPricenassets = Task3.Geometric_Basket_Option2(CallPutFlag,sigmaBg,miuBg,Bg0,K,r,T,t)
        return render_template('Geometric_Basket_Option_nAssets.html', result=GeometricBasketPricenassets)
    elif CalculationType=="1":
        BasketStandardMC=Arithmetic_Mean_Basket_Option_im(100000,"0",CallPutFlag,Spotprice,K,r,T,t,x,tauValue,sigmaBg,miuBg,Bg0,rau2)
        return render_template('Geometric_Basket_Option_nAssets.html', result=BasketStandardMC[3])

#American_binomial_tree.html
@control.route('/AmericanBinominalTree', methods=['GET'])
def American_Binominal_Tree_Form():
    return render_template('American_Binominal_Tree.html')

@control.route('/AmericanBinominalTree', methods=['POST'])
def American_Binominal_Tree():
    is_Control_variate=request.form['Average_Tail_Method']
    CallPutFlag = request.form['CallPutFlag']
    S = float(request.form['S'])
    K = float(request.form['K'])
    r = float(request.form['r'])
    step = int(request.form['step'])
    t = float(request.form['t'])
    T = float(request.form['T'])
    tau = float(request.form['volatility'])
    target_value = modifed_binomial(is_Control_variate,CallPutFlag,S,K,r,T,t,step,tau)
    return render_template('American_Binominal_Tree.html', result=target_value)

#Arithmetic Asian Option.html
@control.route('/ArithmeticAsianOptions', methods=['GET'])
def Arithmetic_Asian_Option_Form():
    return render_template('Arithmetic_Asian_Option.html')

@control.route('/ArithmeticAsianOptions', methods=['POST'])
def Arithmetic_Asian_Option():
    CallPutFlag = request.form['CallPutFlag']
    is_Control_Variate = request.form['Control_Variate_Method']
    S = float(request.form['S'])
    K = float(request.form['K'])
    r = float(request.form['r'])
    t = float(request.form['t'])
    T = float(request.form['T'])
    tau = float(request.form['volatility'])
    n = int(request.form['n'])
    #paths= int(request.form['paths'])
    ArthmeticAsianPrice = Arithmetic_Asia_Option(100000,is_Control_Variate,CallPutFlag,S,K,tau,r,T,t,n)
    return render_template('Arithmetic_Asian_Option.html', result=ArthmeticAsianPrice[:3])

#Arithmetic Basket Option.html
@control.route('/Arithmetic_Basket_Options', methods=['GET'])
def Arithmetic_Basket_Options_Form():
    return render_template('Arithmetic_Basket_Option.html')

@control.route('/Arithmetic_Basket_Options', methods=['POST'])
def Arithmetic_Basket_Option():
    CallPutFlag = request.form['CallPutFlag']
    is_Control_Variate = request.form['Control_Variate_Method']
    x = int(request.form['addnumber'])
    Bg=1
    sigmaBg=0
    miuBg=0
    Spotprice=[]
    tauValue=[]
    a=1
    i=1
    while(i < (x+1)):
        if(request.form['Spotprice'+str(i)]):
            Spotprice.append(float(request.form['Spotprice'+str(i)]))
            tauValue.append(float(request.form['Volatility'+str(i)]))
        else:
            x=x+1
        i+=1
    x = int(request.form['addnumber'])
    rau=request.form.getlist('g')
    rau1=rau[0].split(",,")
    rau2=[]
    for i in range(1,len(rau1)):
        rau2.append(rau1[i].split(","))
    for i in range(len(rau2)):
        for j in range(len(rau2)):
            rau2[i][j]=float(rau2[i][j])
    K = float(request.form['K'])
    r = float(request.form['r'])
    t = float(request.form['t'])
    T = float(request.form['T'])
    for i in range(0,x):
        Bg=Bg*Spotprice[i]
    Bg0=pow(Bg,(1/x))
    for i in range(0,x):
        for j in range(0,x):
            a=tauValue[i]*tauValue[j]
            sigmaBg=sigmaBg+a*rau2[i][j]
            a=1
    sigmaBg=sqrt(sigmaBg)/x
    for i in range(0,x):
        miuBg=miuBg+tauValue[i]*tauValue[i]
    miuBg=r-(1/2)*(miuBg/x)+(1/2)*sigmaBg*sigmaBg
    ArithmeticBasketOptionsPrice = Arithmetic_Mean_Basket_Option_im(100000,is_Control_Variate,CallPutFlag,Spotprice,K,r,T,t,x,tauValue,sigmaBg,miuBg,Bg0,rau2)
    return render_template('Arithmetic_Basket_Option.html', result=ArithmeticBasketOptionsPrice[:3])

if __name__ == '__main__':
    control.run(debug=True)
