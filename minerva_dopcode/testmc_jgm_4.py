# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 10:08:33 2016

@author: jgarciamejia
"""
#import four_chunk_jj16_jgm3_mc3_grindoff as fch
#import four_chunk_jj8 as fch
import four_chunk_jgm_jason_2_2 as fch
from timeit import default_timer as timer
import ipdb
import numpy as np
import corner
import emcee
import scipy.optimize as op
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
obfile = '/Data/kiwispec-proc/n20160626/n20160626.daytimeSkyExpmeter.0047.proc.fits'
#obfile = '/Data/kiwispec-proc/n20160619/n20160619.HD122064.0014.proc.fits'

spec = fch.Spectrum(obfile)
nparspertrace = 22
i,Ord,j,Pix = 6,6,14,1856 # 14th chunk in 6th order with chi^2~2- looks ok
#i,Ord,j,Pix = 8,8,10,1344 # 14th chunk in 6th order with chi^2~2- looks ok

dpix = 128
offset = 0.0 # hardcoded for now b/c I know it from preliminary mpfitting BUT I need to jiggle it [-0.5,0.0,0.5]
# jj16 has four_chunk that looks like:
#FourChunk(spectrum, Ord, Pix, dpix, offset=offset)
chunks = fch.FourChunk(spec, Ord, Pix, dpix, offset)  
startT = timer()

nwalkers = 200
niter = 42003 
nburn = 8000
sampler,bpar,p0arr = chunks.emcee_fitter(nwalkers,niter,nburn) #nwalkers,niter,nburn
endT = timer()
print 'testmc total time for offset '+str(offset)+': ', (endT-startT)/60., 'minutes'


# generate best fit values and vqlues of places where the first ten walkers start
print bpar
##    samples = sampler.chain[tel:tel+nparspertrace, 50:, :].reshape((-1, nparspertrace))

param_list=["z","w0","dw",'dw2','a','b','sigma','hp1','hp2','hp3','hp4','hp5','hp6','hp7','hp8','hp9','hp10','hp11','hp12','hp13','hp14','hp15']

#pl.clf()
#ipdb.set_trace()
#fig, axes = pl.subplots(22, 1, sharex=True, figsize=(8, 9))
#for param in range(len(param_list)):
    #axes[param].plot(sampler.chain[:, :, param].T, color="k", alpha=0.4)
    #axes[param].yaxis.set_major_locator(MaxNLocator(5))
    #axes[param].axhline(bpar[0], color="#888888", lw=2)
    #axes[param].set_ylabel(param_list[param])

#ipdb.set_trace()

#fig.tight_layout(h_pad=0.0)
#fig.savefig('chain_plot'+'_'+'nw'+str(nwalkers)+'_'+'nb'+str(nburn)+'_'+'nit'+'_'+str(niter)+'.png')

# save the chain as an np array
np.save('chain'+'_'+'nw'+str(nwalkers)+'_'+'nb'+str(nburn)+'_'+'nit'+'_'+str(niter)+'.npy',sampler.chain)
#ipdb.set_trace()
# save best-fit params 
np.save('chunkpar'+'_'+'nw'+str(nwalkers)+'_'+'nb'+str(nburn)+'_'+'nit'+'_'+str(niter)+'.npy',chunks.par)

# working triangle plot for first 22 params
#ipdb.set_trace()
#samples = sampler.chain[:, nburn:,:22].reshape((-1, nparspertrace))
#fig = corner.corner(samples, labels=["z","w0","dw",'dw2','a','b','sigma','hp1','hp2','hp3','hp4','hp5','hp6','hp7','hp8','hp9','hp10','hp11','hp12','hp13','hp14','hp15'])
#fig.savefig('triangle_plot'+'_'+'nw'+str(nwalkers)+'_'+'nb'+str(nburn)+'_'+'nit'+'_'+str(niter)+'.png')

#for tel in range(1):
 #   samples = sampler.chain[tel:tel+nparspertrace, 50:, :].reshape((-1, nparspertrace))
    #print len(samples)
    
  #  fig = corner.corner(samples, labels=["z","w0","dw",'dw2','a','b','sigma',
   # 'hp1','hp2','hp3','hp4','hp5','hp6','hp7','hp8','hp9','hp10','hp11','hp12','hp13','hp14',
    #'hp15'], truths=chunks.par[nparspertrace*tel:nparspertrace*(1+tel)])
    #fig.savefig('testmc_data/triangle'+str(int(offset))+'_'+'tel'+'_'+str(int(tel))+'.png')

#ls
#ipdb.set_trace()

#==============================================================================
###### FYI: jj8 params #####
# 
# par[0 + N*parspertrace] = zguess for telescope N
# par[1 + N*parspertrace] = w0 for telescope N
# par[2 + N*parspertrace] = dw/dx for telescope N
# par[3 + N*parspertrace] = continuum shape offset for telescope N
# par[4 + N*parspertrace] = continuum shape slope for telescope N
# par[5 + N*parspertrace] = IP broadening for telescope N
# par[6 + N*parspertrace] = IP skew for telescope N
#==============================================================================
#==============================================================================
##### FYI: jj16 params #####
# par[0 + N*parspertrace] = zguess for telescope N
# par[1 + N*parspertrace] = w0 for telescope N
# par[2 + N*parspertrace] = dw/dx for telescope N
# par[3 + N*parspertrace] = dw2/dx2 for telescope N
# par[4 + N*parspertrace] = continuum shape offset for telescope N
# par[5 + N*parspertrace] = continuum shape slope for telescope N
# par[6 + N*parspertrace] = IP Gaussian width telescope N
# par[7 + N*parspertrace] = IP Hermite Polynomial 1 coefficient
# par[8 + N*parspertrace] = IP Hermite Polynomial 2 coefficient
# par[9 + N*parspertrace] = IP Hermite Polynomial 3 coefficient
# par[10 + N*parspertrace] = IP Hermite Polynomial 4 coefficient
# par[11 + N*parspertrace] = IP Hermite Polynomial 5 coefficient
# par[12 + N*parspertrace] = IP Hermite Polynomial 6 coefficient
# par[13 + N*parspertrace] = IP Hermite Polynomial 7 coefficient
# par[14 + N*parspertrace] = IP Hermite Polynomial 8 coefficient
# par[15 + N*parspertrace] = IP Hermite Polynomial 9 coefficient
# par[16 + N*parspertrace] = IP Hermite Polynomial 10 coefficient
# par[17 + N*parspertrace] = IP Hermite Polynomial 11 coefficient
# par[18 + N*parspertrace] = IP Hermite Polynomial 12 coefficient
# par[19 + N*parspertrace] = IP Hermite Polynomial 13 coefficient
# par[20 + N*parspertrace] = IP Hermite Polynomial 14 coefficient
# par[21 + N*parspertrace] = IP Hermite Polynomial 15 coefficient
#==============================================================================





