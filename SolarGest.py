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

trset0 = scipy.io.loadmat('solarset_128_100.mat');
trset = np.reshape(trset0['series_out'],(nclasses,ndata,fvsize));
trset = np.transpose(trset,(0,2,1));


def get_t4_class(sc): #runs in O(n+k log k) apparently
    sparsecode = np.abs(sc)
    ind = np.argpartition(np.abs(sparsecode),-4)[-4:];
    wmult = sparsecode[ind]
    wmult = wmult/sum(wmult)
    wavg = np.sum(wmult*ind);    
    return np.floor(wavg/150);

def get_t1_class(sc):
    return np.floor(np.argmax(np.abs(sc))/150);

def quantize(val, nlevels):    
    to_values = np.linspace(np.min(val),np.max(val),nlevels);
    out = np.zeros(np.shape(val))
    for idx,x in enumerate(val):
        for idy,y in enumerate(x):
            best_match = None
            best_match_diff = None
            for other_val in to_values:
                diff = abs(other_val - y)
                if best_match is None or diff < best_match_diff:
                    best_match = other_val
                    best_match_diff = diff
            out[idx][idy] = best_match;
    return out

def lca_getsc(xbar,feature_2code,thres,tau,n):
    q = np.transpose(np.dot(np.transpose(feature_2code),xbar))
    g = np.dot(np.transpose(xbar),xbar)
    np.fill_diagonal(g,0)
    
    u = np.zeros(np.shape(q))
    for i in range(n):
        a = np.multiply(u,(np.absolute(u) > thres))
        u += (1/tau)*(-u + q - np.dot(g,a)) 
    return np.multiply(u,(np.absolute(u) > thres))

def lca_getsc_plot(xbar,feature_2code,thres,tau,n):
    x = lca_getsc(xbar,feature_2code,thres,tau,n)
    f2,ax2=plt.subplots(1,3,figsize=(13,4))
    ax2[0].plot(feature_2code)
    ax2[0].set_title('Input Feature')
    ax2[1].plot(x)
    ax2[1].set_title(f'LCA $\lambda$ = {thres} n={n} t={tau}')
    ax2[2].plot(np.dot(z1_quant,x))
    ax2[2].set_title(f"Recovered Feature Q = {nlevels}")
    print(f'feat_class: {get_t1_class(x)}')
    
def lca_testacc(xbar,tstset,thres,tau,n,verbose=False):
    score=np.zeros(5);
    for idgest,gest in enumerate(tstset):
        for idfv,fv in enumerate(gest):
            x = lca_getsc(xbar,fv,thres,tau,n)
            if(get_t1_class(x) == idgest):
                score[idgest]=score[idgest]+1;
            else:
                if(verbose): print(f'{idfv}.\t {get_t1_class(x)} vs {idgest} ({np.argmax(x)})')
    acc = np.sum(score)/(500*5);
    if(verbose): print(f'{acc},{score}')
    return score,acc

def bpdn_testacc(dic,tstset,epsilon,verbose=False):
    score=np.zeros(5);
    for idgest,gest in enumerate(tstset):
        for idfv,fv in enumerate(gest):
            x = bpdn_getsc(dic,fv,epsilon)
            if(get_t1_class(x) == idgest):
                score[idgest]=score[idgest]+1;
            else:
                if(verbose): print(f'{idfv}.\t {get_t1_class(x)} vs {idgest} ({np.argmax(x)})')
    acc = np.sum(score)/(500*5);
    if(verbose): print(f'{acc},{score}')
    return score,acc

def bpdn_getsc(dic,fv,epsilon):    
    opt = bpdn.MinL1InL2Ball.Options({'Verbose': False, 'MaxMainIter': 150,
                                  'RelStopTol': 1e-3, 'rho': 1e0,
                                  'AutoRho': {'Enabled': False}})
    b = bpdn.MinL1InL2Ball(dic, fv.reshape(len(fv),1), epsilon, opt)
    x = b.solve()
    return x

def bpdn_getsc_plot(dic,fv,epsilon):
    x = bpdn_getsc(dic,fv,epsilon);
    f2,ax2=plt.subplots(1,3,figsize=(13,4))
    ax2[0].plot(fv)
    ax2[0].set_title('Input Feature')
    ax2[1].plot(x)
    ax2[1].set_title(f'BPDN $\epsilon = {epsilon}$')
    ax2[2].plot(np.dot(dic,x))
    ax2[2].set_title(f"Recovered Feature Q = {nlevels}")
    print(f'gest: {np.argmax(abs(x))},{np.floor(np.argmax(abs(x))/150)}')
    return x
    
def showacc(matrix):
    fig, ax = plt.subplots()
    ax.imshow(matrix, cmap='YlOrBr_r',aspect='auto')
    for i,row in enumerate(matrix):
       for j,elem in enumerate(row):
          c = matrix[i, j]
          ax.text(j, i, str(c), va='center', ha='center')
    nox_labels = len(tau_list)
    x_positions = np.arange(0,nox_labels,1)
    plt.xticks(x_positions,tau_list)
    plt.xlabel('tau')
    noy_labels = len(thres_list)
    y_positions = np.arange(0,noy_labels,1)
    plt.yticks(y_positions,thres_list)
    plt.ylabel('thres')
          
def sliceacc(matrix):
    m = np.transpose(matrix,(2,0,1))
    for i,slc in enumerate(m):
        showacc(slc)
        plt.title(f'n={i*25+50}')
    return

#%% Load test set and dictionary

with open('solardic_128.pickle','rb') as f:
    z1 = pickle.load(f);

tstset0 = scipy.io.loadmat('solarset_128_500.mat');
tstset = np.reshape(tstset0['series_out'],(5,500,fvsize));
        

#%% LCA

feature_2code = tstset[4][498];
thres = 2
tau = 70
n = 100
x = lca_getsc_plot(z1_quant,feature_2code,thres,tau,n)

#%% bpdn accuracy test

epsilon = 0.5;
acc,score = bpdn_testacc(z1_quant,tstset,epsilon,verbose=True);

#%% Quantize the crossbar

nlevels = 32;
z1_quant = quantize(z1,nlevels);


#%% LCA accuracy test

xbar = z1_quant;
thres = 2
tau = 70
n = 100
score,acc = lca_testacc(z1_quant,tstset,thres,tau,n,verbose=True);

            
#%% BPDN

epsilon = 0.5;
feature_2code = tstset[3][498];

x = bpdn_getsc_plot(z1_quant,feature_2code,epsilon);

#%% Sweeping LCA test

thres_list = [0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25]
tau_list = [20,30,40,50,60,70,80]
n_list = [10,20,30,40]

lca_sweeptest_scores = np.zeros((len(thres_list),len(tau_list),len(n_list),5))
lca_sweeptest_accs = np.zeros((len(thres_list),len(tau_list),len(n_list)))
for idth,thres in enumerate(thres_list):
    for idta,tau in enumerate(tau_list):
        for idn,n in enumerate(n_list):
            lca_sweeptest_scores[idth,idta,idn],lca_sweeptest_accs[idth,idta,idn] = lca_testacc(z1_quant,tstset,thres,tau,n);
            print(f'ttn:({thres},{tau},{n})\t{lca_sweeptest_scores[idth,idta,idn]}\t{lca_sweeptest_accs[idth,idta,idn]}')







