#!/usr/bin/env python 2.7

#This code is written to find the cross-sectional fit to the slit flats

#Import all of the necessary packages (and maybe some unnecessary ones)
from __future__ import division
import pyfits
import os
import glob
#import sys
#import math
import time
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
#from matplotlib import cm
##import scipy
##import scipy.stats as stats
#import scipy.special as sp
##import scipy.optimize as opt
##import scipy.sparse as sparse
##import scipy.signal as sig
##import scipy.linalg as linalg
import special as sf
import bsplines as spline
#import modify_fits as mf #For MINERVA DATA ONLY

##########################################################
### open slit flat file, subtract bias, scale by median ##
##########################################################
spline_smooth = True
redux_dir = os.environ['MINERVA_REDUX_DIR']
#date = 'n20160115'
#date2 = 'n20160216'
date = 'n20161204'
bias_fits = pyfits.open(os.path.join(redux_dir,date,'{}.bias_avg.fits'.format(date)),uint=True)
bias = bias_fits[0].data
sflat_fits = pyfits.open(os.path.join(redux_dir,date,'{}.slit_flat_avg.fits'.format(date)))
sflat = sflat_fits[0].data
#sflat-=bias
readnoise = 3.63
sflat /= np.max(sflat)
#sflat/=np.median(sflat[sflat>5*readnoise]) #Probably a better way to normalize
shdr = sflat_fits[0].header

###clip overscan and weird edge effects
#sflat = sflat[1:2049,0:2048]
hpix = 2048
vpix = sflat.shape[0]
sflat = sflat[::-1,0:hpix]
#plt.plot(sflat[:,1000])
#plt.show()

#xpix = shdr['NAXIS2']
#hpix = shdr['NAXIS1']

##########################################################
################# fit to profile #########################
##########################################################

def get_new_inds(sss, idx1, idx2, fib, pad = 2, noise = 0.04):
    ssub = sss[idx1:idx2]
    ### primitive sigma clip
    smsk = (ssub > (np.median(ssub) - (3 - 2.3*fib/29)*noise))
    if len(smsk>0) == 0:
        print idx1, idx2
        print smsk
        exit(0)
    idx1n = idx1 + np.nonzero(smsk)[0][0]
    idx2n = idx1 + np.nonzero(smsk)[0][-1]
    il = max(0, idx1-pad)
    ih = min(sflat.shape[0],idx2+pad)
    if (idx1n == idx1 or idx2n == idx2) and il !=0 and ih !=sflat.shape[0]:
        idx1n, idx2n = get_new_inds(sss, il, ih, fib)
    ### TODO, put in logic to handle overlap...
    return idx1n, idx2n

num_fibers = 29 #change to user input
pord = 4
line_fits = np.zeros((num_fibers,hpix,pord+1))
idxones = np.zeros((num_fibers, hpix))
idxtwos = np.zeros((num_fibers, hpix))
spds = np.zeros((num_fibers, hpix))
slit_norm = np.ones(sflat.shape)
for col in range(hpix):
#    if col %100 == 0:
#        print "column", col
#    sss = ediff1d(sflat[:,col])
    sss = sflat[:,col]
    if col < 600:
        idx1 = int(40 - col/15.0)
    else:
        idx1 = 0
#    idx2 = 0
    pad = 5
    slit_width = 67-15 #change this later too
    dip_width = 58-22
    for i in range(num_fibers):
#        if i > 1:
#            continue
#        print "Iteration", i
        ### for first slit need to do some special treatment because we can
        ### start partway into slit, or into dip
        if i == 0:
            ssub = sss[idx1:idx1+slit_width+dip_width+2*pad]
            smsk = ssub > 0.5*np.mean(ssub) ## should robustly cut low vals
            ### first nonzero value is temp new idx1
            idx1t = idx1 + np.nonzero(smsk)[0][0]
            idx2t = idx1t + np.nonzero(ediff1d(smsk[np.nonzero(smsk)[0][0]:]))[0][0]
            ### Now run through to cut
            idx1, idx2 = get_new_inds(sss, idx1t, idx2t, i)
#            slit_width = idx2 - idx1
        else:
            idx2o = idx2
            if dip_width > 6:
                idx1, idx2 = get_new_inds(sss, idx2+dip_width, idx2+dip_width+slit_width, i)
                dip_width = idx1-idx2o
                slit_width = idx2-idx1
            else:
                idx1, idx2 = idx2+dip_width, idx2+dip_width+slit_width
                dip_width -= 1
                slit_width -= 1
        idx2mx = 2050 ### Some pathological behavior on edge...
        idx2 = min(idx2,idx2mx)
        ### Once indices of edges are identified, fit to those within
        ### use spd to come in from the edges a little, helps prevent bad fits
        spd = 1 if i < 27 else 3  ### Extra pad on last due to overlap
        ### additional override from some of the lower signal values
        if i > 24 and i <27 and col < 100:
            spd = 2
        ### save inds, spd for spline fitting later...
        idxones[i, col] = idx1
        idxtwos[i, col] = idx2
        spds[i, col] = spd
        slit_inds = np.arange(idx1+spd,idx2-spd)
        slit_inds = slit_inds[slit_inds < sflat.shape[0]]
        ### range of columns to take median over for empirical fits
        cpd = 3
        crange = np.arange(max(0,col-cpd),min(sflat.shape[1],col+cpd+1))
        ### if we don't have enough points to fit polynomial, just go empirical
        if len(slit_inds) == 0:
            pass
        elif len(slit_inds)<=pord+1:
            if len(slit_inds) == 1:
                meds = np.median(sflat[slit_inds, crange[0]:crange[-1]])
            else:
                meds = np.median(sflat[slit_inds, crange[0]:crange[-1]], axis=1)
            slit_norm[slit_inds,col] = meds
        else:
            ### Change scale to [-1, 1] then fit
            n_inds = 2*(slit_inds - vpix/2)/vpix
            line_fits[i,col] = np.polyfit(n_inds,sss[slit_inds],pord)
            fitted = np.poly1d(line_fits[i,col,:])(n_inds)
            slit_norm[slit_inds,col] = fitted
        ### include nearest points on each edge...
        pd = min(1,dip_width)
        slit_norm[idx1-pd:idx1+spd,col] = np.median(sflat[idx1-pd:idx1+spd,crange], axis=1)
        if idx2 < 2048:
            slit_norm[idx2-spd:idx2+pd,col] = np.median(sflat[idx2-spd:idx2+pd,crange], axis=1)

### Fit splines to polynomial coefficients...
if spline_smooth:
    #plt.plot(slit_norm[1000,:])
    spc = 5 ## try evenly spaced breakpoints
    breakpoints = np.arange(0,hpix,spc)
    ### Fit splines, no weighting
    ### This is actually pretty slow...look at speeding up spline_1D
    smooth_line_fits = np.zeros(line_fits.shape)
    for val in range(pord+1):
#        print "val:", val
        for fib in range(num_fibers):
            spl = spline.spline_1D(np.arange(hpix),line_fits[fib,:,val],breakpoints)
            smooth_line_fits[fib,:,val] = spl
    
    ### Now go back and put modified values into slit_norm
    for col in range(hpix):
#        if col %100 == 0:
#            print "column", col
        for fib in range(num_fibers):
            idx1 = idxones[fib, col]
            idx2 = idxtwos[fib, col]
            spd = spds[fib, col]
            slit_inds = np.arange(idx1+spd,idx2-spd)
            slit_inds = slit_inds[slit_inds < sflat.shape[0]].astype(int)
            if len(slit_inds) > pord+1:
                n_inds = 2*(slit_inds - vpix/2)/vpix
                fitted = np.poly1d(smooth_line_fits[fib,col,:])(n_inds)
                slit_norm[slit_inds,col] = fitted


### visually evaluate quality of slit_norm
if plot_results:
    smooth = sflat/slit_norm
    plt.imshow(smooth, interpolation='none')
    plt.show()
    for j in [0, 1000, 2000]:
        plt.plot(smooth[:,j], linewidth=2)
        plt.plot(sflat[:,j], linewidth=2)
        plt.plot(slit_norm[:,j], linewidth=2)
        plt.show()

### Save file
#redux_dir = os.environ['MINERVA_REDUX_DIR']    
hdu = pyfits.PrimaryHDU(slit_norm)
hdu.header.append(('POLYORD',pord,'Polynomial order used for fitting'))
hdulist = pyfits.HDUList([hdu])
hdulist.writeto(os.path.join(redux_dir,date,'{}.slit_flat_smooth.fits'.format(date)),clobber=True)