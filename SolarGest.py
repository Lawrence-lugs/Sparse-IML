# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 10:30:21 2022

@author: Lawrence
"""

#%%

from __future__ import division, print_function
from builtins import input

import numpy as np
import scipy.io
import pickle
from datetime import datetime

from sporco.dictlrn import bpdndl
from sporco import util
from sporco import array
from sporco import plot
from sporco.admm import bpdn
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from matplotlib.ticker import FormatStrFormatter

fvsize = 128;
natoms = 150;
ndata = 100;
nclasses = 5;

trset0 = scipy.io.loadmat('solarset_128_100.mat');
trset = np.reshape(trset0['series_out'],(nclasses,ndata,fvsize));
trset = np.transpose(trset,(0,2,1));

tstset0 = scipy.io.loadmat('solarset_128_500.mat');
tstset = np.reshape(tstset0['series_out'],(5,500,fvsize));

trset0 = scipy.io.loadmat('solarset_128_100.mat');
z_untrained = np.reshape(trset0['series_out'],(500,fvsize));
z_untrained = np.transpose(z_untrained);

with open('solardic_128.pickle','rb') as f:
    z1 = pickle.load(f);
with open('solardic_128_tr100_atoms50.pickle','rb') as f:
    z50 = pickle.load(f);
with open('z25.pickle','rb') as f:
    z25 = pickle.load(f);
with open('z10.pickle','rb') as f:
    z10 = pickle.load(f);


def get_t4_class(sc): #runs in O(n+k log k) apparently
    sparsecode = np.abs(sc)
    ind = np.argpartition(np.abs(sparsecode),-4)[-4:];
    wmult = sparsecode[ind]
    wmult = wmult/sum(wmult)
    wavg = np.sum(wmult*ind);    
    return np.floor(wavg/150);

def get_t1_class(sc,elementsperclass):
    return np.floor(np.argmax(np.abs(sc))/elementsperclass);

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
    ax2[1].stem(x,markerfmt=" ")
    ax2[1].set_title(f'LCA $\lambda$ = {thres} n={n} t={tau}')
    ax2[2].plot(np.dot(xbar,x))
    ax2[2].set_title("Recovered Feature")
    print(f'feat_class: {get_t1_class(x,xbar.shape[1]/5)}')
    return x
    
def lca_testacc(xbar,tstset,thres,tau,n,verbose=False,batchsize=500):
    score=np.zeros(5);
    for idgest,gest in enumerate(tstset):
        for idfv,fv in enumerate(gest):
            x = lca_getsc(xbar,fv,thres,tau,n)
            if(get_t1_class(x,xbar.shape[1]/5) == idgest):
                score[idgest]=score[idgest]+1;
            else:
                if(verbose): print(f'{idfv}.\t {get_t1_class(x,xbar.shape[1]/5)} vs {idgest} ({np.argmax(x)})')
    acc = np.sum(score)/(batchsize*5);
    if(verbose): print(f'{acc},{score}')
    return score,acc

def lca_testreconstruction(xbar,tstset,thres,tau,n,verbose=False,batchsize=500):
    normsum=0;
    for idgest,gest in enumerate(tstset):
        for idfv,fv in enumerate(gest):
            x = lca_getsc(xbar,fv,thres,tau,n)
            nerm = np.linalg.norm(np.dot(xbar,x)-fv)
            normsum = normsum+nerm
    acc = normsum/(batchsize*5);
    if(verbose): print(f'{acc}')
    return acc

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
    
def showacc(matrix,labelled = True):
    fig, ax = plt.subplots()
    ax.imshow(matrix, cmap='YlOrBr_r',aspect='auto')
    if(labelled):
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

def plotdic(dic,title=''):
    plotshape = int(np.sqrt(dic.shape[1]))
    fig, ax = plt.subplots(plotshape,plotshape)
    for i,ax_row in enumerate(ax):
        for j,axes in enumerate(ax_row):
            axes.set_yticklabels([])
            axes.set_xticklabels([])
            axes.plot(dic[:,i*5+j]);
    fig.suptitle(title)
    
def quantizewithvariation(qdic,relative=1,nlevels=16):
    to_values = np.linspace(np.min(qdic),np.max(qdic),nlevels);
    out = np.zeros(np.shape(qdic))
    for idx,x in enumerate(qdic):
        for idy,y in enumerate(x):
            best_match = None
            best_match_diff = None
            for other_val in to_values:
                diff = abs(other_val - y)
                if best_match is None or diff < best_match_diff:
                    best_match = other_val
                    best_match_diff = diff
            out[idx][idy] = best_match;
    stdev = relative*(to_values[1] - to_values[0]);
    out = np.random.normal(out,stdev)
    return out
    
    
#%% 

z_random = np.random.randn(128,500)

#%% Get random variation on dictionary

z50_qvar = quantizewithvariation(z50,relative=1)
z25_qvar = quantizewithvariation(z25,relative=1)
z10_qvar = quantizewithvariation(z10,relative=1)
z1_qvar = quantizewithvariation(z1,relative=1)

#%% LCA

feature_2code = tstset[0][233];
feature_2code = tstset[np.random.randint(0,5)][np.random.randint(0,500)];
thres = 0.25
tau = 45
n = 10000
x = lca_getsc_plot(z25,feature_2code,thres,tau,n)

#%% bpdn accuracy test

epsilon = 0.5;
acc,score = bpdn_testacc(z1,tstset,epsilon,verbose=True);

#%% LCA accuracy test

z_lcaacctest = z25_quant;
thres = 0.2773298211064743
tau = 42.02236657044855
n = 100  
score,acc = lca_testacc(z_lcaacctest,tstset,thres,tau,n,verbose=True);

            
#%% BPDN

epsilon = 0.5;
feature_2code = tstset[np.random.randint(0,5)][np.random.randint(0,500)];

x = bpdn_getsc_plot(z_random,feature_2code,epsilon);

#%% Sweeping LCA test

#thres_list = np.linspace(asc_calcstep_thres*-8,7*asc_calcstep_thres,16) + asc_ttn_winner[0]
#tau_list = np.linspace(-8*asc_calcstep_tau,7*asc_calcstep_tau,16) + asc_ttn_winner[1]
tau_list = np.linspace(10,60,64)
thres_list = np.linspace(0.05,0.4,64)
dic_list = (z1_quant,z50_quant,z25_quant,z10_quant)

n_list = [2,5]

start = datetime.now()

lca_sweeptest_scores = np.zeros((len(thres_list),len(tau_list),len(n_list),5))
lca_sweeptest_accs = np.zeros((len(dic_list),len(thres_list),len(tau_list),len(n_list)))
lca_sweeptest_reconaccs = np.zeros((len(dic_list),len(thres_list),len(tau_list),len(n_list)))
for idic,dic in enumerate(dic_list):
    for idth,thres in enumerate(thres_list):
        for idta,tau in enumerate(tau_list):
            for idn,n in enumerate(n_list):
                lca_sweeptest_scores[idth,idta,idn],lca_sweeptest_accs[idic,idth,idta,idn] = lca_testacc(dic,tstset_stoch[0],thres,tau,n,batchsize=100);
                #lca_sweeptest_reconaccs[idic,idth,idta,idn] = lca_testreconstruction(dic,tstset_stoch[0],thres,tau,n,batchsize=100);
                #print(f'ttn:({thres},{tau},{n})\t{lca_sweeptest_reconaccs[idic,idth,idta,idn]}')
                print(f'ttn:({thres},{tau},{n})\t{lca_sweeptest_scores[idth,idta,idn]}\t{lca_sweeptest_accs[idic,idth,idta,idn]}')
print(f'Runtime: {datetime.now()-start}')

#%%

plt.xlabel('$\tau$');
plt.ylabel('$\lambda$')
plt.imshow(acc_array,cmap='YlOrBr_r',vmin=0.8,vmax=1,extent=(30,70,0.3,0),aspect=40/0.3);


#%% Quantize Levels

nlevels = 10
z1_quant = quantize(z1,nlevels);
z10_quant = quantize(z10,nlevels);
z25_quant = quantize(z25,nlevels);
z50_quant = quantize(z50,nlevels);

#%% Total gradient ascent LCA optimization (maybe work but i can't guarantee convergence

asc_calcstep_tau = 0.3;
asc_calcstep_thres = 0.001;
asc_n = 2;
asc_maxEpochs = 5;
asc_batchsize = 100;
asc_thres_lr_init = 0.005;
asc_tau_lr_init = 10;
asc_epsilon = 0.0001;

#initialization
asc_thres = np.random.random()*0.2+0.1;
asc_tau = np.random.random()*20+30;
#asc_thres = 0.2598389128954294;
#asc_tau = 46.00089902104527;
asc_taustep_acc = 0;
asc_thresstep_acc = 0;
asc_tauslope = 0;
asc_threslope = 0;
asc_win_thres,asc_win_tau = (asc_thres,asc_tau)
asc_thres_lr, asc_tau_lr = asc_thres_lr_init,asc_tau_lr_init

#highscores
asc_ttn_winner = (asc_thres,asc_tau,asc_n)
asc_acc_winner = 0;
asc_prevacc = 0;

#points traversed
ptstraversed = np.zeros((asc_maxEpochs,int(tstset.shape[1]/asc_batchsize),2))
pt_i = 0;

z_touse = z1;

    
for asc_i in range(asc_maxEpochs):
    
    #obtain subsampled tstset
    rng = np.random.default_rng()
    tstset_stoch = np.zeros((int(tstset.shape[1]/asc_batchsize),tstset.shape[0],asc_batchsize,tstset.shape[2]))
    for p in range(nclasses):
        idxrand = rng.choice(range(tstset.shape[1]),tstset.shape[1],replace=False) #recalculate another 500 numbers
        for pp in range(int(tstset.shape[1]/asc_batchsize)):
            #populate class p for all batches pp
            tstset_stoch[pp][p] = tstset[p][idxrand[asc_batchsize*pp:asc_batchsize*(pp+1)]]
    
    for batch in range(int(tstset.shape[1]/asc_batchsize)):
        s,asc_acc = lca_testacc(z_touse,tstset_stoch[batch],asc_thres,asc_tau,asc_n,batchsize=asc_batchsize);
        
        if(asc_acc < asc_prevacc):
            print(f'Backtracking:{asc_acc,asc_prevacc,asc_thres,asc_tau}')
            #asc_tau_lr = asc_tau_lr/2;
            #asc_thres_lr = asc_thres_lr/2;
            #asc_tau = asc_tau - asc_tau_lr*asc_tauslope;
            #asc_thres = asc_thres - asc_thres_lr*asc_tauslope;        
            #s,asc_acc = lca_testacc(z_touse,tstset_stoch[batch],asc_thres,asc_tau,asc_n,batchsize=100);
            asc_tau = asc_tau - asc_tau_lr*asc_tauslope + asc_calcstep_tau;
            asc_thres = asc_thres - asc_thres_lr*asc_threslope + asc_calcstep_thres;         
            #asc_acc = asc_prevacc        
            s,asc_acc = lca_testacc(z_touse,tstset_stoch[batch],asc_thres,asc_tau,asc_n,batchsize=asc_batchsize);
            print(f'Backtracked:{asc_acc,asc_thres,asc_tau}')
        asc_thres_lr, asc_tau_lr = asc_thres_lr_init,asc_tau_lr_init
        
        if(asc_acc > asc_acc_winner):
            asc_ttn_winner = (asc_thres,asc_tau,asc_n)
            asc_acc_winner = asc_acc;
            asc_i_winner = asc_i;
            asc_batch_winner = batch;
        
        #approximate the slope of the accuracy
        s,asc_thresstep_acc = lca_testacc(z_touse,tstset_stoch[batch],asc_thres + asc_calcstep_thres,asc_tau,asc_n,batchsize=asc_batchsize);
        s,asc_taustep_acc = lca_testacc(z_touse,tstset_stoch[batch],asc_thres,asc_tau + asc_calcstep_tau,asc_n,batchsize=asc_batchsize);
        asc_tauslope = (asc_taustep_acc - asc_acc)/asc_calcstep_tau;
        asc_threslope = (asc_thresstep_acc - asc_acc)/asc_calcstep_thres;
        
        print(f'Epoch {asc_i}\tBatch {batch}.\t{asc_acc}:\t[{asc_thres},{asc_tau}]\t[{asc_threslope},{asc_tauslope}]');
        
        #update the operating point
        pt_i = pt_i+1
        ptstraversed[pt_i] = (asc_thres,asc_tau)
        asc_tau = asc_tau + asc_tau_lr*asc_tauslope;
        asc_thres = asc_thres + asc_thres_lr*asc_threslope;
        asc_prevacc = asc_acc;
        
        #convergence condition
        #if(abs(asc_tauslope) < 0.0001):
            #if(abs(asc_threslope) < 0.0001):
                #break;

print(asc_i_winner,batch,asc_acc_winner)
s,asc_acc_winner = lca_testacc(z_touse,tstset,asc_ttn_winner[0],asc_ttn_winner[1],asc_ttn_winner[2]);
print(asc_acc_winner,s,asc_ttn_winner)

#%%

with open('q16n5sweep.pickle','rb') as f:
    acc_old = pickle.load(f);
    
    
#%%
fig,ax = plt.subplots(1,2,sharey=True,sharex=True,figsize=(4.5,1.5))
for i,n in enumerate(n_list):
    ax[i].set_title(f'N={n} K=150',fontsize=10);
    ax[i].set_xlabel(r'$\tau$');
    ax[i].set_ylabel('$\lambda$');
    polt = ax[i].imshow(lca_sweeptest_accs[0,:,:,i],vmin=0,vmax=1,cmap='magma',extent=(10,60,0.4,0.05),aspect=50/(0.4-0.05));
fig.colorbar(polt,ax=fig.get_axes());


#%% Variation Histogram Maker

nsamp = 200
results_hist = np.zeros((4,nsamp))
thres =  0.09386287243352415
tau = 59.346896752542456
n = 2
qlevels = 16
r = [1.5,1,0.5,0.25]

for rel_i,rel in enumerate(r):
    for var_i in range(nsamp):
        z_touse = quantizewithvariation(z50,relative=rel,nlevels=qlevels);
        s,acc = lca_testacc(z_touse, tstset_stoch[0], thres, tau, n,batchsize=100)
        print(s,acc)
        results_hist[rel_i][var_i] = acc
            
#%%

z1_results_hist = results_hist

#%%
i = 3;
plt.title(f'K=50 $\ttau={tau:.2f},\lambda={thres:.3f}$,n={n},Q={qlevels}')
plt.hist(results_hist[0],bins=20,alpha=0.7)
plt.hist(results_hist[1],bins=20,alpha=0.7)
plt.hist(results_hist[2],bins=20,alpha=0.7)
plt.hist(results_hist[3],bins=20,alpha=0.7)
plt.xlabel('Classification Accuracy')
plt.legend(['$\sigma=1.5\Delta L_Q$','$\sigma=\Delta L_Q$','$\sigma=0.5\Delta L_Q$','$\sigma=0.1\Delta L_Q$'])

#%%

qlist = range(4,33)
diclist = [z10,z25,z50,z1]

thres = 0.25;
tau = 40;
nlist = 45;
outputline = np.zeros((4,50-2))

for dic_i,dic in enumerate(diclist):
    for lawrence_i,q in enumerate(qlist):
        z_touse = quantize(dic,q);
        outputline[dic_i][lawrence_i] = lca_testreconstruction(z_touse,tstset_stoch[0],thres,tau,n,batchsize=100,verbose=True);


#%%
plt.plot(qlist,np.transpose(outputline[:,:29]))
plt.legend(['K=10','K=25','K=50','K=150'])
plt.xlabel('Q')
plt.ylabel('Reconstruction Error')

#%%
subject = lca_sweeptest_accs[3,:,:,1]
np.unravel_index(np.argmax(subject),(subject.shape))
anspair = thres_list[np.unravel_index(np.argmax(subject),subject.shape)[0]],tau_list[np.unravel_index(np.argmax(subject),subject.shape)[1]]
print(f'({anspair[0]:.3f},{anspair[1]:.3f}),{np.max(subject)}')

