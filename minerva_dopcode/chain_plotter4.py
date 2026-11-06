# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 17:43:39 2017

@author: jgarciamejia
"""

#import four_chunk_jgm_jason as fch
#from timeit import default_timer as timer
import ipdb
import numpy as np
#import corner
#import emcee
#import scipy.optimize as op

#import matplotlib.pyplot as pl
#from matplotlib.ticker import MaxNLocator

# load the sampler np array, with dimensions (nwalkers, nsteps, npars) where npars = 88

sampler = np.load('chain_nw200_nb8000_nit_42003.npy')

# save chain values for file naming
nwalkers,niter,nsteps = sampler.shape
# input manually
nburn = 8000
param_list=["z","w0","dw",'dw2','a','b','sigma','hp1','hp2','hp3','hp4','hp5','hp6','hp7','hp8','hp9','hp10','hp11','hp12','hp13','hp14','hp15']

# clear all previous plots
#pl.clf()
#ipdb.set_trace() 

jiggle_amp_list = []

# modified parameter jiggle amps tweaked by eye - only z through sigma 
max_list_byeye = [0.0002788,5288.6,0.01894,4.14e-6,99500,-84,0.28]
min_list_byeye = [0.0002772,5288.5,0.01892,4.1131e-6,98000,-93,0.23]

# instantiate lists where we will svae refined max and min values of chains 

max_val_real = []
min_val_real = []
for param in range(len(param_list)):
    if param >= 0 and  param <= 6 :
	max_val = np.amax(np.amax(np.ma.masked_outside(sampler[:,:,param],min_list_byeye[param],max_list_byeye[param]), axis = 1))
	min_val = np.amin(np.amin(np.ma.masked_outside(sampler[:,:,param],min_list_byeye[param],max_list_byeye[param]), axis = 1))
	max_val_real.append(max_val)
	min_val_real.append(min_val)
	jiggle_amp = max_val - min_val
    else:
      	jiggle_amp = np.amax(np.amax(sampler[:,:,param], axis = 1)) - np.amin(np.amin(sampler[:,:,param], axis = 1))
    #pl.clf()
    #fig = pl.figure(figsize =(8,5))
    #ax = fig.add_axes([0.15,0.15,0.8,0.8])
    #pl.plot(sampler[:, :, param].T, color="k", alpha=0.4)
    #ax.yaxis.set_major_locator(MaxNLocator(5))
    #ax.set_title(param_list[param]+'_jiggle_amp = '+str(jiggle_amp))
    #axes[param].axhline(bpar[0], color="#888888", lw=2)
    #ax.set_ylabel(param_list[param])
    #ax.set_xlabel('number of steps')
    #fig.savefig('chain_plot'+'_'+'param'+'_'+param_list[param]+'_'+'nw'+str(nwalkers)+'_'+'nb'+str(nburn)+'_'+'nit'+'_'+str(niter)+'.png')
    	    
    jiggle_amp_list.append(jiggle_amp)


print jiggle_amp_list, max_val_real, min_val_real
# first round of jiggle amp values - did not work so well
'[2.4028773331916456e-06, 0.0052472433189905132, 0.00014626249993209689, 1.123388742909088e-06, 422.30118315333675, 12.313890959022132, 0.075045288129322818, 0.86763688346227841, 0.36472839599215839, 1.6195316162194313, 0.39254848111894525, 0.75783014197613618, 1.6395100842436765, 1.5766357866600773, 1.9710283275674949, 2.7535913793248459, 0.33017326686250731, 3.368937274446075, 1.7686265400660397, 1.8973493742711716, 1.4210678458668102, 0.02790294896022457]'

#second round of values - polished  with the 42,000 chain
'[1.8920377861333547e-07, 0.29479483444811194, 8.2353351804748814e-05, 1.1270850136581661e-09, 7662.6497179784928, 111.03143141095558, 0.30633742467407221, 33.103406737207315, 7.7846134030109848, 76.884267999251051, 1.9911097882452222, 25.965911918349612, 17.269617700470739, 102.11332902138034, 22.499074912155343, 119.55657533819019, 3.1053907038544546, 115.74252063730951, 28.758077652226774, 43.671697325152458, 15.775407829817411, 0.023970345999241156]'
