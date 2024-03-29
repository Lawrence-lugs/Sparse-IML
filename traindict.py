# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 13:16:59 2022

@author: Lawrence
"""

#%%train dictionary

import scipy.io
import numpy as np
import pickle
from sporco.dictlrn import bpdndl
from matplotlib import pyplot as plt

def plotdic(dic,title=''):
    plotshape = int(np.sqrt(dic.shape[1]))
    fig, ax = plt.subplots(plotshape,plotshape)
    for i,ax_row in enumerate(ax):
        for j,axes in enumerate(ax_row):
            axes.set_yticklabels([])
            axes.set_xticklabels([])
            axes.plot(dic[:,i*5+j]);
    fig.suptitle(title)

fvsize = 128;
natoms = 10;

trset0 = scipy.io.loadmat('solarset_128_100.mat');
trset = np.reshape(trset0['series_out'],(5,100,fvsize));
trset = np.transpose(trset,(0,2,1));

np.random.seed(12345)
D0 = np.random.randn(fvsize, natoms) #better recognition accuracy from overcomplete dictionary, but 2500 atoms is a bit insane.

lmbda = 0.1;
opt = bpdndl.BPDNDictLearn.Options({'Verbose': True, 'MaxMainIter': 100,
                      'BPDN': {'rho': 10.0*lmbda + 0.1},
                      'CMOD': {'rho': 100 / 1e3}})

z = np.zeros((5,fvsize,natoms));

for n in range(5):
    d = bpdndl.BPDNDictLearn(D0, trset[n], lmbda, opt)
    d.solve()
    print("BPDNDictLearn solve time: %.2fs" % d.timer.elapsed('solve'))
    z[n]=d.getdict()

z1_newtrain = z.transpose(1,0,2)
z1_newtrain = z1_newtrain.reshape(fvsize,natoms*5)

plotdic(z1_newtrain,title=f'BPDN {natoms} atoms per class')

#%%

with open('z10.pickle','wb') as f:
    pickle.dump(z1_newtrain, f, protocol=pickle.HIGHEST_PROTOCOL)