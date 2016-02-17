#!/usr/bin/env python 2.7

#This code is written to average dark and slit flats and save into redux folder
#Also makes a master bias frame

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
import modify_fits as mf #For MINERVA DATA ONLY

### files to reference, plus environmental variables
date = 'n20160115'
date2 = 'n20160216'
data_dir = os.environ['MINERVA_DATA_DIR']
redux_dir = os.environ['MINERVA_REDUX_DIR']
dark_files = glob.glob(os.path.join(data_dir,date,'*Dark*.fits'))#assumes certain format of files
sflat_files = glob.glob(os.path.join(data_dir,date2,'*slitFlat*.fits'))
bias_files = glob.glob(os.path.join(data_dir,date,'*Bias*.fits'))

#### Header to get dimensions
py1 = pyfits.open(dark_files[0],uint=True)
hdr = py1[0].header
d1 = hdr['NAXIS1']
d2 = hdr['NAXIS2']

### go through dark files, average
dark_avg = np.zeros((d2,d1),dtype=np.uint16)
dark_ct = 0
for df in dark_files:
    dark_avg+=mf.unpack_minerva(df)
    dark_ct+=1
    
dark_avg/=dark_ct

### go through slitflat files, average
### Header to get dimensions
py2 = pyfits.open(sflat_files[0],uint=True)
hdr2 = py2[0].header
f1 = hdr2['NAXIS1']
f2 = hdr2['NAXIS2']
sflat_avg = np.zeros((f2,f1))
sflat_ct = 0
for slf in sflat_files:
    sflat_avg+=mf.unpack_minerva(slf)
    sflat_ct+=1
    
sflat_avg/=sflat_ct


#### Header to get dimensions
py3 = pyfits.open(bias_files[0],uint=True)
hdr = py3[0].header
b1 = hdr['NAXIS1']
b2 = hdr['NAXIS2']

### go through dark files, average
bias_avg = np.zeros((b2,b1),dtype=np.uint16)
bias_ct = 0
for bf in bias_files:
    bias_avg+=mf.unpack_minerva(bf)
    bias_ct+=1
    
bias_avg/=bias_ct
bias_mean = np.mean(bias_avg)
bias_std = np.std(bias_avg)
sclip = 3
bias_outlier = (bias_avg>(bias_mean+sclip*bias_std))*(bias_avg<(bias_mean-sclip*bias_std))
bias_avg[bias_outlier] = int(bias_mean) #Turn outliers into average
plt.ion()
plt.imshow(bias_avg)

### Save files to redux
hdu_dark = pyfits.PrimaryHDU(dark_avg,uint=True)
hdulist_dark = pyfits.HDUList([hdu_dark])
hdulist_dark.writeto(os.path.join(redux_dir,date,'dark_avg.fits'),clobber=True)

hdu_sflat = pyfits.PrimaryHDU(sflat_avg,uint=True)
hdulist_sflat = pyfits.HDUList([hdu_sflat])
hdulist_sflat.writeto(os.path.join(redux_dir,date2,'sflat_avg.fits'),clobber=True)

hdu_bias = pyfits.PrimaryHDU(bias_avg,uint=True)
hdulist_bias = pyfits.HDUList([hdu_bias])
hdulist_bias.writeto(os.path.join(redux_dir,date,'bias_avg.fits'),clobber=True)