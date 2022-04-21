# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 14:34:12 2022

@author: Lawrence
"""

from __future__ import division, print_function
from builtins import input

import numpy as np

from sporco.dictlrn import bpdndl
from sporco import util
from sporco import array
from sporco import plot
from sporco.admm import bpdn
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

plot.config_notebook_plotting()

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

#%% obtain the sample images

exim = util.ExampleImages(scaled=True, zoom=0.25, gray=True)
S1 = exim.image('barbara.png', idxexp=np.s_[10:522, 100:612])
S2 = exim.image('kodim23.png', idxexp=np.s_[:, 60:572])
S3 = exim.image('monarch.png', idxexp=np.s_[:, 160:672])
S4 = exim.image('sail.png', idxexp=np.s_[:, 210:722])
S5 = exim.image('tulips.png', idxexp=np.s_[:, 30:542])


#%% make S, which is a dictionary containing all of the above images
S = array.extract_blocks((S1, S2, S3, S4, S5), (8, 8))
#%% make the 128x128 squares into a line
S = np.reshape(S, (np.prod(S.shape[0:2]), S.shape[2]))
#zero mean
S -= np.mean(S, axis=0)

#%% initialize the dict (64 = vector length, 132 vectors in dictionary)
np.random.seed(12345)
D0 = np.random.randn(S.shape[0], 132)
D0_64 = D0

#%% init solver
lmbda = 0.1
opt = bpdndl.BPDNDictLearn.Options({'Verbose': True, 'MaxMainIter': 100,
                      'BPDN': {'rho': 10.0*lmbda + 0.1},
                      'CMOD': {'rho': S.shape[1] / 1e3}})

#%% solve

d = bpdndl.BPDNDictLearn(D0, S, lmbda, opt)
d.solve()
print("BPDNDictLearn solve time: %.2fs" % d.timer.elapsed('solve'))

#%% display initial and final dictionaries

z=d.getdict()
D1 = d.getdict().reshape((8, 8, D0.shape[1]))
D0 = D0.reshape(8, 8, D0.shape[-1])
fig = plot.figure(figsize=(14, 7))
plot.subplot(1, 2, 1)
plot.imview(util.tiledict(D0), title='D0', fig=fig)
plot.subplot(1, 2, 2)
plot.imview(util.tiledict(D1), title='D1', fig=fig)

#%% try to sparsely represent a noisy version of one of the training images
img = mpimg.imread('macaw_smol.png')
img = rgb2gray(img)
img = img[:128,:128]
plt.imshow(img)

#%%

epsilon = 1

#s11 = np.reshape(S1, (1,np.prod(S1.shape[0:2])),s11.shape[2]))
S_1 = exim.image('barbara.png', idxexp=np.s_[-512:,-512:])
#s11 = array.extract_blocks(img,(8,8))
s11 = array.extract_blocks(S_1,(8,8))
s11 = np.reshape(s11,(np.prod(s11.shape[0:2]),s11.shape[2]))
signalmean = np.mean(s11,axis=0)
s11 -= signalmean


#%%
opt = bpdn.MinL1InL2Ball.Options({'Verbose': True, 'MaxMainIter': 150,
                                  'RelStopTol': 1e-3, 'rho': 1e0,
                                  'AutoRho': {'Enabled': False}})

b = bpdn.MinL1InL2Ball(d.getdict(), s11, epsilon, opt)
x = b.solve()
print("MinL1InL2Ball solve time: %.2fs" % b.timer.elapsed('solve'))

#%%

blksz = (8,8)
imgout_mean = array.average_blocks(np.dot(d.getdict(),x).reshape(blksz+(-1,)), (128,128))
imgout_median = array.combine_blocks(np.dot(d.getdict(),x).reshape(blksz+(-1,))+signalmean, (128,128))

f,ax=plt.subplots(1,2,figsize=(13,4),gridspec_kw={'width_ratios':[2,1],'height_ratios':[1]})
ax[0].plot(np.hstack(x))
ax[0].set_title(f'Sparse Representation $\epsilon$ = {epsilon}')
ax[1].imshow(imgout_mean)
ax[1].set_title(f"Reconstructed Image $\epsilon$={epsilon}")

#%% LCA

xbar = d.getdict()
q = np.transpose(np.dot(np.transpose(s11),xbar))
g = np.dot(np.transpose(xbar),xbar) - np.identity(132)

thres = 1
tau = 10
u = np.zeros(np.shape(q))
n = 20
for i in range(n): #10 iterations
    a = np.multiply(u,(np.absolute(u) > thres))
    #u = 0.9 * u + 0.01 * (q - np.dot(g,a)) 
    u += (1/tau)*(-u + q - np.dot(g,a)) 
u = np.multiply(u,(np.absolute(u) > thres))
    
imgout_lca = array.average_blocks(np.dot(d.getdict(),u).reshape(blksz+(-1,)), (128,128))
f2,ax2=plt.subplots(1,2,figsize=(13,4),gridspec_kw={'width_ratios':[2,1],'height_ratios':[1]})
ax2[0].plot(np.hstack(u))
ax2[0].set_title(f'LCA Sparse Representation n={n} $\lambda$={thres} $t$={tau}')
ax2[1].imshow(imgout_lca)
ax2[1].set_title(f"LCA Reconstructed Image n={n} $\lambda$={thres} $t$={tau}")

print('done')
#LCA

#%% 

f2,ax2=plt.subplots(1,2,figsize=(13,4),gridspec_kw={'width_ratios':[2,1],'height_ratios':[1]})
ax2[0].plot(np.hstack(q))
ax2[0].set_title(f'forward-pass output')
ax2[1].imshow(array.average_blocks(np.dot(d.getdict(),q).reshape(blksz+(-1,)), (128,128)))
ax2[1].set_title(f"forward-backward recovery")












