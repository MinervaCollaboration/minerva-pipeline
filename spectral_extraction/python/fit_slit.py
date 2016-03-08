#!/usr/bin/env python 2.7

#This code is written to find the cross-sectional fit to the slit flats

#Import all of the necessary packages (and maybe some unnecessary ones)
from __future__ import division
import pyfits
import os
import glob
#import sys
#import math
#import time
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
#import special as sf
#import modify_fits as mf #For MINERVA DATA ONLY

##########################################################
### open slit flat file, subtract bias, scale by median ##
##########################################################
redux_dir = os.environ['MINERVA_REDUX_DIR']
date = 'n20160115'
date2 = 'n20160216'
bias_fits = pyfits.open(os.path.join(redux_dir,date,'bias_avg.fits'),uint=True)
bias = bias_fits[0].data
sflat_fits = pyfits.open(os.path.join(redux_dir,date,'sflat_avg.fits'))
sflat = sflat_fits[0].data
sflat-=bias
sflat/=np.median(sflat) #Probably a better way to normalize
shdr = sflat_fits[0].header

###clip overscan and weird edge effects
sflat = sflat[1:2049,0:2048]

#xpix = shdr['NAXIS2']
#ypix = shdr['NAXIS1']
ypix = 2048

##########################################################
################# fit to profile #########################
##########################################################
num_fibers = 29 #change to user input
slit_width = 60 #change this later too
pord = 1
line_fits = np.zeros((num_fibers,ypix,pord+1))
for col in range(ypix):
    sss = ediff1d(sflat[:,col])
    idx1 = 0
    idx2 = 0
    pad = 5
    for i in range(num_fibers):
        ssub = sss[idx1+pad:-1]
        try:
            idx1tmp = idx1 + pad + np.nonzero(ssub>(0.75*np.std(sss)))[0][0]
        except IndexError:
            if idx1<1000:
                idx1tmp=0
            else:
                continue
        if (idx1tmp-slit_width+2*pad)<idx1 and i>0: #one level of recursion to catch errors, really need to improve
            ssub = sss[idx1tmp+pad:-1]
            try:
                idx1tmp = idx1tmp + pad + np.nonzero(ssub>(0.75*np.std(sss)))[0][0]
            except IndexError:
                if idx1<1000:
                    idx1tmp=0
                else:
                    continue
        idx1 = idx1tmp
        ssub = sss[idx2+pad:-1]
        try:
            idx2tmp = idx2 + pad + np.nonzero(ssub<(-0.75*np.std(sss)))[0][0]
        except IndexError:
            if idx2>1000:
                idx2tmp = 2048
            else:
                continue
        if (idx2tmp-slit_width+2*pad)<idx2 and i>0:
            ssub = sss[idx2tmp+pad:-1]
            try:
                idx2tmp = idx2tmp + pad + np.nonzero(ssub<(-0.75*np.std(sss)))[0][0]
            except IndexError:
                if idx2>1000:
                    idx2tmp = 2048
                else:
                    continue
        idx2 = idx2tmp
        if idx2 < idx1:
            ssub = sss[idx2+pad:-1]
            idx2 = idx2 + pad + np.nonzero(ssub<(-0.75*np.std(sss)))[0][0]
        slit_inds = np.arange(idx1+pad,idx2-pad)
#        print slit_inds
        if len(slit_inds)<=3:
            continue
        line_fits[i,col] = np.polyfit(slit_inds,sflat[slit_inds,col],pord)
#        if col == 7:
#            fitted = np.poly1d(line_fits[i,col])(slit_inds)
#            print i, ':'
#            print fitted[0]
#        if col == 8 or col == 9 or col == 10:
#            plt.plot(sflat[:,col-1])
#            plt.show()
#        if col==1 and i==1:
#            plt.plot(sflat[idx1+pad:idx2-pad,col])
#            plt.plot(slit_inds-idx1-pad,fitted,'k')
#            plt.show()
#            plt.close()
        
redux_dir = os.environ['MINERVA_REDUX_DIR']    
hdu = pyfits.PrimaryHDU(line_fits)
hdu.header.append(('POLYORD',pord,'Polynomial order used for fitting'))
hdulist = pyfits.HDUList([hdu])
hdulist.writeto(os.path.join(redux_dir,date2,'slit_approx.fits'),clobber=True)