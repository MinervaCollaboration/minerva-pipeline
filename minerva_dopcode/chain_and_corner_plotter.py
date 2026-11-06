# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 17:43:39 2017

@author: jgarciamejia
"""

#import four_chunk_jgm_jason as fch
#from timeit import default_timer as timer
import ipdb
import numpy as np
import corner
#import emcee
#import scipy.optimize as op
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator

# load the sampler np array, with dimensions (nwalkers, nsteps, npars) where npars = 88

sampler = np.load('chain_nw200_nb8000_nit_42000.npy')

# save chain values for file naming
nwalkers,niter,nparspertrace = sampler.shape
nburn = 8000

param_list=["z","w0","dw",'dw2','a','b','sigma','hp1','hp2','hp3','hp4','hp5','hp6','hp7','hp8','hp9','hp10','hp11','hp12','hp13','hp14','hp15']

# clear all previous plots
#==============================================================================
# pl.clf()
# ipdb.set_trace() 
# 
# jiggle_amp_list = []
# 
# for param in range(len(param_list)):
#     jiggle_amp = np.amax(np.amax(sampler[:,:,param], axis = 1)) - np.amin(np.amin(sampler[:,:,param], axis = 1))
#     pl.clf()
#     fig = pl.figure(figsize =(8,5))
#     ax = fig.add_axes([0.15,0.15,0.8,0.8])
#     pl.plot(sampler[:, :, param].T, color="k", alpha=0.4)
#     ax.yaxis.set_major_locator(MaxNLocator(5))
#     ax.set_title(param_list[param]+'_jiggle_amp = '+str(jiggle_amp))
#     #axes[param].axhline(bpar[0], color="#888888", lw=2)
#     ax.set_ylabel(param_list[param])
#     ax.set_xlabel('number of steps')
#     fig.savefig('chain_plot'+'_'+'param'+'_'+param_list[param]+'_'+'nw'+str(nwalkers)+'_'+'nb'+str(nburn)+'_'+'nit'+'_'+str(niter)+'.png')
#     jiggle_amp_list.append(jiggle_amp)
#==============================================================================


#print jiggle_amp_list
'[2.4028773331916456e-06, 0.0052472433189905132, 0.00014626249993209689, 1.123388742909088e-06, 422.30118315333675, 12.313890959022132, 0.075045288129322818, 0.86763688346227841, 0.36472839599215839, 1.6195316162194313, 0.39254848111894525, 0.75783014197613618, 1.6395100842436765, 1.5766357866600773, 1.9710283275674949, 2.7535913793248459, 0.33017326686250731, 3.368937274446075, 1.7686265400660397, 1.8973493742711716, 1.4210678458668102, 0.02790294896022457]'

samples = sampler.chain[:, nburn:,:4].reshape((-1, nparspertrace))
fig = corner.corner(samples, labels=["z","w0","dw",'dw2','a','b','sigma','hp1','hp2','hp3','hp4','hp5','hp6','hp7','hp8','hp9','hp10','hp11','hp12','hp13','hp14','hp15'])
fig.savefig('triangle_plot'+'_'+'nw'+str(nwalkers)+'_'+'nb'+str(nburn)+'_'+'nit'+'_'+str(niter)+'.png')

