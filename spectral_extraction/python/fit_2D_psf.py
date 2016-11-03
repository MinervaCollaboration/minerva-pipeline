#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 10:49:18 2016

@author: matt cornachione

2D PSF fitter for MINERVA - output to be used in SP_extract.py
"""

#Import all of the necessary packages
from __future__ import division
import pyfits
import os
import math
import time
import numpy as np
from numpy import pi, sin, cos, random, zeros, ones, ediff1d
#from numpy import *
import matplotlib.pyplot as plt
#from matplotlib import cm
#import scipy
#import scipy.stats as stats
#import scipy.special as sp
#import scipy.interpolate as si
import scipy.optimize as opt
#import scipy.sparse as sparse
#import scipy.signal as signal
import scipy.linalg as linalg
#import solar
import special as sf
import bsplines as spline
import argparse
import lmfit
import psf_utils as psf
import minerva_utils as utils

data_dir = os.environ['MINERVA_DATA_DIR']
redux_dir = os.environ['MINERVA_REDUX_DIR']

#########################################################
########### Allow input arguments #######################
#########################################################
parser = argparse.ArgumentParser()
parser.add_argument("-f","--filename",help="Name of image file (.fits) to extract",
                    default=os.path.join(data_dir,'n20160216','n20160216.HR2209.0025.fits'))
#                    default=os.path.join(data_dir,'n20160115','n20160115.daytimeSky.0006.fits'))
parser.add_argument("-fib","--num_fibers",help="Number of fibers to extract",
                    type=int,default=29)
parser.add_argument("-bs","--bundle_space",help="Minimum spacing (in pixels) between fiber bundles",
                    type=int,default=40)
parser.add_argument("-fs","--fiber_space",help="Minimum spacing (in pixels) between fibers within a bundle",
                    type=int,default=13)
parser.add_argument("-ts","--telescopes",help="Number of telescopes feeding spectrograph",
                    type=int,default=4) 
parser.add_argument("-np","--num_points",help="Number of trace points to fit on each fiber",
                    type=int,default=20)
parser.add_argument("-p","--psf",help="Type of model to be used for PSF fitting.",
                    type=str,default='bspline')
parser.add_argument("-par","--parallel",help="Run in parallel mode",action='store_true')
parser.add_argument("--par_index",help="Fiber index for parallel mode",
                    type=int)
#parser.add_argument("-ns","--nosave",help="Don't save results",
#                    action='store_true')
#parser.add_argument("-T","--tscopes",help="T1, T2, T3, and/or T4 (remove later)",
#                    type=str,default=['T1','T2','T3','T4'])
args = parser.parse_args()
num_fibers = (args.num_fibers-1)*args.telescopes

##############################################################################


##############################################################################
### Snipped from find_ip.py
### Should roughly be able to find good peaks on arc frames
### Add lines at the end to fit a 2Dspline PSF along a trace

#redux_dir = os.environ['MINERVA_REDUX_DIR']
#data_dir = os.environ['MINERVA_DATA_DIR']
#Tarc = pyfits.open(os.path.join(redux_dir,'n20160130','n20160130.thar_T1_i2test.0025.proc.fits'))
#Tarc = pyfits.open(os.path.join(redux_dir,'n20160130','n20160130.thar_T2_i2test.0020.proc.fits'))
#Tarc = pyfits.open(os.path.join(redux_dir,'n20160130','n20160130.thar_T3_i2test.0012.proc.fits'))


Traw = pyfits.open(os.path.join(data_dir,'n20160130','n20160130.thar_T4_i2test.0017.fits'))
raw_img = Traw[0].data
#
#
#plt.imshow(raw_img)
#plt.show()
#plt.close()

#########################################################
######## Find location and information on arc peaks #####
#########################################################

arc_pos = dict()
for ts in range(args.telescopes):
    if ts == 0:
        Tarc = pyfits.open(os.path.join(redux_dir,'n20160130','n20160130.thar_T1_i2test.0025.proc.fits'))
        data = Tarc[0].data
        wvln = Tarc[2].data
        invar = Tarc[1].data
        mask = Tarc[3].data        
    elif ts == 1:
        Tarc = pyfits.open(os.path.join(redux_dir,'n20160130','n20160130.thar_T2_i2test.0020.proc.fits'))
        data = Tarc[0].data
        wvln = Tarc[2].data
        invar = Tarc[1].data
        mask = Tarc[3].data        
    elif ts == 2:
        Tarc = pyfits.open(os.path.join(redux_dir,'n20160130','n20160130.thar_T3_i2test.0012.proc.fits'))
        data = Tarc[0].data
        wvln = Tarc[2].data
        invar = Tarc[1].data
        mask = Tarc[3].data        
    elif ts == 3:
        Tarc = pyfits.open(os.path.join(redux_dir,'n20160130','n20160130.thar_T4_i2test.0017.proc.fits'))
        data = Tarc[0].data
        wvln = Tarc[2].data
        invar = Tarc[1].data
        mask = Tarc[3].data
    pos_d, wl_d, mx_it_d, stddev_d, chi_d, err_d = psf.arc_peaks(data,wvln,invar,ts=ts)
    for i in range(len(pos_d)):
        ### slot pos_d from each telescope into master dictionary for use
        arc_pos[ts+4*i] = pos_d[i]

#########################################################
########### Load Background Requirments #################
#########################################################

#hardcode in n20160115 directory
filename = args.filename#os.path.join(data_dir,'n20160115',args.filename)
software_vers = 'v0.1.0' #Later grab this from somewhere else

gain = 1.3
readnoise = 3.63

spectrum = pyfits.open(filename,uint=True)
spec_hdr = spectrum[0].header
#ccd = spectrum[0].data
ccd = Traw[0].data

#####CONVERT NASTY FORMAT TO ONE THAT ACTUALLY WORKS#####
#Dimensions
ypix = spec_hdr['NAXIS1']
xpix = spec_hdr['NAXIS2']
### Next part checks if iodine cell is in, assumes keyword I2POSAS exists
try:
    if spec_hdr['I2POSAS']=='in':
        i2 = True
    else:
        i2 = False
except KeyError:
    i2 = False

actypix = 2048

#Test to make sure this logic is robust enough for varying inputs
if np.shape(ccd)[0] > xpix:
    ccd_new = np.resize(ccd,[xpix,ypix,2])
        
    #Data is split into two 8 bit sections (totalling 16 bit).  Need to join
    #these together to get 16bit number.  Probably a faster way than this.
    ccd_16bit = np.zeros((xpix,ypix))
    for row in range(xpix):
        for col in range(ypix):
            #Join binary strings
            binstr = "{:08d}{:08d}".format(int(bin(ccd_new[row,col,0])[2:]),
                      int(bin(ccd_new[row,col,1])[2:]))
            ccd_16bit[row,col] = int(binstr,base=2)

    ccd = ccd_16bit[::-1,0:actypix] #Remove overscan region
else:
    ccd = ccd[::-1,0:actypix] #Remove overscan region
    ccd = ccd.astype(np.float)

#########################################################
########### Fit traces to spectra #######################
#########################################################

### Possibly temporary, use fiber flats to build traces
trace_ccd = np.zeros((np.shape(ccd)))
arc_ccd = np.zeros((np.shape(ccd)))

#ts = args.tscopes

for tscope in ['T1','T2','T3','T4']:
    #Choose fiberflats with iodine cell in
    if tscope=='T1':
        flnmflat = 'n20160130.fiberflat_T1.0023.fits'
        flnmarc = 'n20160130.thar_T1_i2test.0025.fits'
    #        continue
    elif tscope=='T2':
        flnmflat = 'n20160130.fiberflat_T2.0022.fits'
        flnmarc = 'n20160130.thar_T2_i2test.0020.fits'
    #        continue
    elif tscope=='T3':
        flnmflat = 'n20160130.fiberflat_T3.0014.fits'
        flnmarc = 'n20160130.thar_T3_i2test.0012.fits'
    #        continue
    elif tscope=='T4':
        flnmflat = 'n20160130.fiberflat_T4.0015.fits'
        flnmarc = 'n20160130.thar_T4_i2test.0017.fits'
    else:
        print("{} is not a valid telescope".format(tscope))
        continue
    #Import tungsten fiberflat
    fileflat = os.path.join(data_dir,'n20160130',flnmflat)
    filearc = os.path.join(data_dir,'n20160130',flnmarc)
    #fileflat = os.path.join(paths,'minerva_flat.fits')
    ff = pyfits.open(fileflat,ignore_missing_end=True,uint=True)
    fa = pyfits.open(filearc,ignore_missing_end=True,uint=True)
    ccd_tmp = ff[0].data
    arc_tmp = fa[0].data
    trace_ccd += ccd_tmp[::-1,0:actypix]
    arc_ccd += arc_tmp[::-1,0:actypix]

raw_img = arc_ccd
#ccd = arc_ccd #Temporary, just to examine fiber vs. wavelength
#ccd -= np.median(ccd)
#ccd[ccd<0] = 0
#i2 = False


trace_coeffs, trace_intense_coeffs, trace_sig_coeffs, trace_pow_coeffs = utils.find_trace_coeffs(trace_ccd,2,args.fiber_space,num_points=args.num_points,num_fibers=28*args.telescopes,skip_peaks=1)

###Plot to visualize traces      
#fig,ax = plt.subplots()
#ax.pcolorfast(trace_ccd)
#for i in range(num_fibers-4):
#    ys = (np.arange(ypix)-ypix/2)/ypix
#    xs = trace_coeffs[2,i]*ys**2+trace_coeffs[1,i]*ys+trace_coeffs[0,i]
#    yp = np.arange(ypix)
#    plt.plot(yp,xs)
#plt.show()


#plt.imshow(raw_img,interpolation='none')
#plt.show()


#########################################################
########### Load Flat and Bias Frames ###################
#########################################################   
date = 'n20160115' #Fixed for now, late make this dynamic
bias_hdu = pyfits.open(os.path.join(redux_dir,date,'bias_avg.fits'),uint=True)
bias = bias_hdu[0].data
sflat_hdu = pyfits.open(os.path.join(redux_dir,date,'slit_approx.fits'),uint=True)
slit_coeffs = sflat_hdu[0].data
#slit_coeffs = slit_coeffs[::-1,:] #Re-order, make sure this is right
polyord = sflat_hdu[0].header['POLYORD'] #Order of polynomial for slit fitting

### subtract bias (slit will be handled in loop)
bias = bias[:,0:actypix] #Remove overscan
#raw_img = raw_img[::-1,0:actypix]
raw_img -= 4*bias #Note, if ccd is 16bit array, this operation can cause problems
#raw_img /= 10
#raw_img[raw_img<0] = 0 #Enforce positivity
### More robust 
cut = int(3*np.median(raw_img))
raw_mask = (raw_img < cut)*(raw_img > -cut)
masked_img = raw_img[raw_mask]
arr = plt.hist(masked_img,2*(cut-1))
hgt = arr[0]
xvl = arr[1][:-1]
xmsk = (xvl < np.median(masked_img))
hgts = hgt[xmsk]
xvls = xvl[xmsk]
pguess = (7,np.median(masked_img),np.max(hgt))
sigma = 1/np.sqrt(abs(hgts)+1)
params, errarr = opt.curve_fit(sf.gaussian,xvls,hgts,p0=pguess,sigma=sigma)
#plt.title("Number of pixels with certain count value")
#plt.ion()
#plt.show()
#print params
#htst = sf.gaussian(xvl, params[0], center=params[1], height=params[2],bg_mean=0,bg_slope=0,power=2)
#plt.plot(xvl,htst)
#plt.figure("Residual Distribution")
#plt.plot(xvl,hgt-htst)
#exit(0)
plt.close()
raw_img -= params[1] # mean
pix_err = params[0] # standard deviation - use in lieu of readnoise

#########################################################
################## PSF fitting ##########################
#########################################################

num_fibers = 28*args.telescopes
#fitted_psf_coeffs = np.array(())
if not args.parallel:
    for idx in range(num_fibers):
        hcenters = arc_pos[idx]
        hscale = (hcenters-actypix/2)/actypix
        vcenters = trace_coeffs[2,idx+1]*hscale**2+trace_coeffs[1,idx+1]*hscale+trace_coeffs[0,idx+1]
        sigmas = trace_sig_coeffs[2,idx]*hscale**2+trace_sig_coeffs[1,idx]*hscale+trace_sig_coeffs[0,idx]
        powers = trace_pow_coeffs[2,idx]*hscale**2+trace_pow_coeffs[1,idx]*hscale+trace_pow_coeffs[0,idx]
        
        print("Running PSF Fitting on trace {}".format(idx))
        
        if args.psf is 'bspline':
            if idx == 0:
                tmp_psf_coeffs = psf.fit_spline_psf(raw_img,hcenters,
                             vcenters,sigmas,powers,pix_err,gain,plot_results=True)
                num_coeffs = len(tmp_psf_coeffs)
                fitted_psf_coeffs = np.zeros((num_fibers,num_coeffs))
                fitted_psf_coeffs[idx,:] = tmp_psf_coeffs
            else:
                fitted_psf_coeffs[idx,:] = psf.fit_spline_psf(raw_img,hcenters,
                             vcenters,sigmas,powers,pix_err,gain)
        else:
            print("Invalid PSF selection")
            print("Choose from the following:")
            print("  bspline")
            exit(0)
            
    hdu1 = pyfits.PrimaryHDU(fitted_psf_coeffs)
    hdu1.header.append(('PSFTYPE',args.psf,'Model used for finding PSF'))
    if args.psf is 'bspline':   
        hdu1.header.comments['NAXIS1'] = 'Coefficients (see key - not yet made)'
        hdu1.header.comments['NAXIS2'] = 'Fiber (of those actually used)'
                
    hdulist = pyfits.HDUList([hdu1])
    redux_dir = os.environ['MINERVA_REDUX_DIR']
    psf_dir = 'psf'
    filename = 'psf_coeffs'
    if not os.path.isdir(os.path.join(redux_dir,psf_dir)):
        os.makedirs(os.path.join(redux_dir,psf_dir))
    #    print os.path.join(redux_dir,savedate,savefile+'.proc.fits')
    hdulist.writeto(os.path.join(redux_dir,psf_dir,filename+'.fits'),clobber=True)
    
else:
    try:
        idx = args.par_index
    except:
        print("Must set --par_index for parallel mode")
        print("This selects which fiber will be fitted")
        exit(0)
    if idx >= num_fibers or idx < 0:
        print("Input --par_index cannot be greater than the number of fibers or less than zero")
        exit(0)

    hcenters = arc_pos[idx]
    hscale = (hcenters-actypix/2)/actypix
    vcenters = trace_coeffs[2,idx+1]*hscale**2+trace_coeffs[1,idx+1]*hscale+trace_coeffs[0,idx+1]
    sigmas = trace_sig_coeffs[2,idx]*hscale**2+trace_sig_coeffs[1,idx]*hscale+trace_sig_coeffs[0,idx]
    powers = trace_pow_coeffs[2,idx]*hscale**2+trace_pow_coeffs[1,idx]*hscale+trace_pow_coeffs[0,idx]
    print("Running PSF Fitting on trace {}".format(idx))
    if args.psf is 'bspline':
        psf_coeffs = psf.fit_spline_psf(raw_img,hcenters,vcenters,sigmas,powers,pix_err,gain,plot_results=True)
    else:
        print("Invalid PSF selection")
        print("Choose from the following:")
        print("  bspline")
        exit(0)
        
    hdu1 = pyfits.PrimaryHDU(psf_coeffs)
    hdu1.header.append(('PSFTYPE',args.psf,'Model used for finding PSF'))
    if args.psf is 'bspline':   
        hdu1.header.comments['NAXIS1'] = 'Coefficients (see key - not yet made)'
        hdu1.header.comments['NAXIS2'] = 'Fiber (of those actually used)'
        hdu1.header.append(('FIBERNUM',idx,'Fiber number (starting with 0)'))
                
    hdulist = pyfits.HDUList([hdu1])
    redux_dir = os.environ['MINERVA_REDUX_DIR']
    psf_dir = 'psf'
    filename = 'psf_coeffs_{:03d}'.format(idx)
    if not os.path.isdir(os.path.join(redux_dir,psf_dir)):
        os.makedirs(os.path.join(redux_dir,psf_dir))
    #    print os.path.join(redux_dir,savedate,savefile+'.proc.fits')
    hdulist.writeto(os.path.join(redux_dir,psf_dir,filename+'.fits'),clobber=True)