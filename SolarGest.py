# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 10:30:21 2022

@author: Lawrence
"""

from __future__ import division, print_function
from builtins import input

import numpy as np
import scipy.io
import pickle

from sporco.dictlrn import bpdndl
from sporco import util
from sporco import array
from sporco import plot
from sporco.admm import bpdn
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

fvsize = 128;
natoms = 150;
ndata = 100;
nclasses = 5;

trset0 = scipy.io.loadmat('solarset_128.mat');
trset = np.reshape(trset0['series_out'],(nclasses,ndata,fvsize));
trset = np.transpose(trset,(0,2,1));

#%% Load test set and dictionary

with open('solardic_128.pickle','rb') as f:
    z1 = pickle.load(f);

tstset0 = scipy.io.loadmat('solarset_128_500.mat');
tstset = np.reshape(tstset0['series_out'],(5,500,fvsize));
        

#%% LCA

feature_2code = tstset[1][474];

xbar = z1;
q = np.transpose(np.dot(np.transpose(feature_2code),xbar))
g = np.dot(np.transpose(xbar),xbar);
np.fill_diagonal(g,0);
thres = 2
tau = 70
n = 100

u = np.zeros(np.shape(q))
for i in range(n): #10 iterations
    a = np.multiply(u,(np.absolute(u) > thres))
    #u = 0.9 * u + 0.01 * (q - np.dot(g,a)) 
    u += (1/tau)*(-u + q - np.dot(g,a)) 
u = np.multiply(u,(np.absolute(u) > thres))
x=u

f2,ax2=plt.subplots(1,3,figsize=(13,4))
ax2[0].plot(feature_2code)
ax2[0].set_title(f'Input Feature')
ax2[1].plot(u)
ax2[1].set_title(f'LCA $\lambda$ = {thres} n={n} t={tau}')
ax2[2].plot(np.dot(xbar,u))
ax2[2].set_title(f"Recovered Feature")

print(f'gest: {np.argmax(abs(x))},{np.floor(np.argmax(abs(x))/150)}')

#%% bpdn accuracy testd

epsilon = 0.5;
opt = bpdn.MinL1InL2Ball.Options({'Verbose': False, 'MaxMainIter': 150,
                                  'RelStopTol': 1e-3, 'rho': 1e0,
                                  'AutoRho': {'Enabled': False}})
score=np.zeros(5);

for idgest,gest in enumerate(tstset):
    for idfv,fv in enumerate(gest):
        b = bpdn.MinL1InL2Ball(z1, fv.reshape(128,1), epsilon, opt)
        x = abs(b.solve())
        if(np.floor(np.argmax(x)/150) == idgest):
            score[idgest]=score[idgest]+1;
        else:
            print(f'{idfv}.\t {np.floor(np.argmax(x)/150)} vs {idgest} ({np.argmax(x)})')
            
acc = np.sum(score)/(500*5);
print(acc)


#%% LCA accuracy test

xbar = z1;
g = np.dot(np.transpose(xbar),xbar);
np.fill_diagonal(g,0);
thres = 2
tau = 60
n = 400

score=np.zeros(5);

for idgest,gest in enumerate(tstset):
    for idfv,fv in enumerate(gest):
        q = np.transpose(np.dot(fv,xbar))
        u = np.zeros(np.shape(q))
        for i in range(n): 
            a = np.multiply(u,(np.absolute(u) > thres))
            u += (1/tau)*(-u + q - np.dot(g,a)) 
        u = np.multiply(u,(np.absolute(u) > thres))
        x = abs(u)
        if(np.floor(np.argmax(x)/150) == idgest):
            score[idgest]=score[idgest]+1;
        else:
            print(f'{idfv}.\t {np.floor(np.argmax(x)/150)} vs {idgest} ({np.argmax(x)})')
            
acc = np.sum(score)/(500*5);
print(score)
print(acc)

            
#%% test vector

epsilon = 0.5;
opt = bpdn.MinL1InL2Ball.Options({'Verbose': False, 'MaxMainIter': 150,
                                  'RelStopTol': 1e-3, 'rho': 1e0,
                                  'AutoRho': {'Enabled': False}})

feature_2code = tstset[1][5,:];

b = bpdn.MinL1InL2Ball(z1, feature_2code.reshape(fvsize,1), epsilon, opt)
x = b.solve()

f2,ax2=plt.subplots(1,3,figsize=(13,4))
ax2[0].plot(feature_2code)
ax2[0].set_title(f'Input Feature')
ax2[1].plot(x)
ax2[1].set_title(f'Sparse Representation $\epsilon = 0.5$')
ax2[2].plot(np.dot(z1,x))
ax2[2].set_title(f"Recovered Feature")

print(f'gest: {np.argmax(abs(x))},{np.floor(np.argmax(abs(x))/150)}')








