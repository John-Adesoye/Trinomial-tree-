#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 22:12:35 2019

@author: john
"""

import numpy as np
import statsmodels.api as sm
import datetime
import pandas_datareader.data as read

#constant maturity data
class shortR():
    '''calibrates vasicek model
        std = start date
        edt = end date
        src = fred
    '''
    def __init__(self,dat='v',typ='DGS3MO', sdt='2018426', edt='2019426',src='fred', days=252):
        self.typ = typ
        self.sdt = datetime.datetime.strptime(sdt,'%Y%m%d').date()
        self.edt = datetime.datetime.strptime(edt,'%Y%m%d').date()
        self.src = src
        self.days = days
        self.dat = dat
        self.dt = 1/days
        self.today = datetime.datetime.today().date()
    def editData(self):
        '''get and edit 3-month constant maturity treasury rate data'''
        rates = read.DataReader(self.typ,self.src,self.sdt,self.edt)
        rates.DGS3MO=rates.DGS3MO/100
        rates = rates.fillna(method='bfill')
        dsg3mo=np.matrix(rates.DGS3MO).T
        return dsg3mo
    def VAS_OLS(self):
        '''calibrate vasicek model paramteres using OLS'''
        data = self.editData()
        Y = data[1:len(data)]
        X = data[0:len(data)-1]
        X = sm.add_constant(X)
        model = sm.OLS(Y, X)
        fit = model.fit()
        B0 = fit.params[0]
        B1 = fit.params[1]
        dt = 1/self.days# the average time interval over all date points, given that the dates were business days
        khat = (1 - B1)/dt
        theta_hat = B0/(khat*dt)
        sigma_hat = np.sqrt(fit.mse_resid)/ dt**0.5
        return khat, theta_hat, sigma_hat

def get_params():
    model = shortR()
    vasicek = model.VAS_OLS()
    a = vasicek[0]
    theta = vasicek[1]
    sigma = vasicek[2]
    data = model.editData()
    r0 = data[len(data)-1,0]
    n= model.days
    dt = model.dt
    Mij = lambda x: x*(1-(a*dt)) + a*theta*dt
#    Vi = sigma * np.sqrt(dt)
    k = lambda x: int(round(Mij(x)/(sigma * np.sqrt(dt)*np.sqrt(3))))
    K= k(r0)
#    eta = lambda x: (x*(1-a) + a*theta) * dt - k(x)*sigma * np.sqrt(dt)* np.sqrt(3)
    return r0, sigma, K, n, dt

def draw_tree():
    r0, sigma, K, n, dt = get_params()
    tree=np.matrix(np.zeros([n+1,n+1]))
    v = 3 #intial vertical nodes
    for i in range(len(tree)):
        temp = v #temp to traverse tree vertically
        j = i
        if i==0 and j==0:
            tree[(0,0)] = r0
            continue
        start = 0 #commence tree traverse from 0th row
        while temp>0 and start <len(tree) and j < len(tree):
            tree[(start,j)] = (K+j-start)*sigma*np.sqrt(dt)*np.sqrt(3)
            temp -= 1
            start+=1
        v += 2
    return tree

def main():
    tree = draw_tree()[0:11,0:6]
    print(tree)

main()
