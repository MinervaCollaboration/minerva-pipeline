#!/usr/bin/env python

#Implementation of 2D "Spectro-perfectionism" extraction for MINERVA

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
#import scipy.optimize as opt
#import scipy.sparse as sparse
#import scipy.signal as signal
import scipy.linalg as linalg
#import solar
import special as sf
import bsplines as spline
import argparse
import lmfit
import psf_utils as psf
import minerva_utils as m_utils

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
parser.add_argument("-ns","--nosave",help="Don't save results",
                    action='store_true')
#parser.add_argument("-T","--tscopes",help="T1, T2, T3, and/or T4 (remove later)",
#                    type=str,default=['T1','T2','T3','T4'])
args = parser.parse_args()
num_fibers = args.num_fibers*args.telescopes-4

### Snipped from find_ip.py
### Should roughly be able to find good peaks on arc frames
### Add lines at the end to fit a 2Dspline PSF along a trace

#redux_dir = os.environ['MINERVA_REDUX_DIR']
#data_dir = os.environ['MINERVA_DATA_DIR']
#Tarc = pyfits.open(os.path.join(redux_dir,'n20160130','n20160130.thar_T1_i2test.0025.proc.fits'))
#Tarc = pyfits.open(os.path.join(redux_dir,'n20160130','n20160130.thar_T2_i2test.0020.proc.fits'))
#Tarc = pyfits.open(os.path.join(redux_dir,'n20160130','n20160130.thar_T3_i2test.0012.proc.fits'))
Tarc = pyfits.open(os.path.join(redux_dir,'n20160130','n20160130.thar_T4_i2test.0017.proc.fits'))
data = Tarc[0].data
wvln = Tarc[2].data
invar = Tarc[1].data
mask = Tarc[3].data

Traw = pyfits.open(os.path.join(data_dir,'n20160130','n20160130.thar_T4_i2test.0017.fits'))
raw_img = Traw[0].data


#plt.plot(data[0,0,:])
#plt.show()
#plt.close()

#########################################################
######## Find location and information on arc peaks #####
#########################################################

ts=3
pos_d, wl_d, mx_it_d, stddev_d, chi_d, err_d = psf.arc_peaks(data,wvln,invar,ts=ts)

#########################################################
########### Load Background Requirments #################
#########################################################

#hardcode in n20160115 directory
filename = args.filename#os.path.join(data_dir,'n20160115',args.filename)
software_vers = 'v0.2.1' #Later grab this from somewhere else

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

#ccd = arc_ccd #Temporary, just to examine fiber vs. wavelength
#ccd -= np.median(ccd)
#ccd[ccd<0] = 0
#i2 = False


trace_coeffs, trace_intense_coeffs, trace_sig_coeffs, trace_pow_coeffs = m_utils.find_trace_coeffs(trace_ccd,2,args.fiber_space,num_points=args.num_points,num_fibers=28*args.telescopes,skip_peaks=1)

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
raw_img = raw_img[::-1,0:actypix]
raw_img -= bias #Note, if ccd is 16bit array, this operation can cause problems
### TODO check on whether this next part is valid...
#raw_img -= readnoise
raw_img[raw_img<0] = 0 #Enforce positivity

#########################################################
################## PSF fitting ##########################
#########################################################

#####################################################################
######### Now test 2D extraction with spline PSF ####################
#####################################################################


######################################################################
########## Load and calibrate data ###################################
######################################################################

### Load section to extract
testfits = pyfits.open('n20160323.HR4828.0020.fits')
ccd = testfits[0].data
ccd = ccd[::-1,0:2048]
date = 'n20160115' #Fixed for now, late make this dynamic
bias_hdu = pyfits.open(os.path.join(redux_dir,date,'bias_avg.fits'),uint=True)
bias = bias_hdu[0].data

### subtract bias (slit will be handled in loop)
bias = bias[::-1,0:2048] #Remove overscan
ccd -= bias #Note, if ccd is 16bit array, this operation can cause problems
#print np.median(ccd[ccd<np.mean(ccd)])
#ccd -= np.median(ccd[ccd<np.mean(ccd)])#+readnoise
#ccd[ccd<0] = 0 #Enforce positivity
ccd *= gain #include gain
#plt.imshow(ccd,interpolation='none')
#plt.show()
#plt.close()

### Fit input trace coeffs (from fiberflat) to this ccd
trace_coeffs = m_utils.refine_trace_centers(ccd, trace_coeffs, trace_intense_coeffs, trace_sig_coeffs, trace_pow_coeffs, fact=20, readnoise=readnoise, verbose=True)

### Import PSF coefficients
psf_coeffs = pyfits.open('/home/matt/software/minerva/redux/psf/psf_coeffs_063.fits')[0].data

spectrum_2D = m_utils.extract_2D(ccd, psf_coeffs, trace_coeffs, readnoise=readnoise, gain=gain, verbose=True)

waves = np.arange(spectrum_2D.shape[1])#*101/wls
#scale = np.mean(np.ediff1d(hcents))
spectrum_2D = spectrum_2D[0]
#plt.plot(waves,flux2)
plt.figure("2D vs. Optimal Extraction")
#fluxtilde2 = fluxtilde2[0:-1] ### remove last point - bg
plt.plot(waves,spectrum_2D,linewidth='2')
#plt.plot(hcents,2*fluxtilde3*scale)
#plt.show()
#plt.plot(hcents,3.7*flux2*scale)

###Compare to optimal extraction:
opt_fits = pyfits.open('n20160323.HR4828.0020.proc.fits')
opt_dat = opt_fits[0].data
opt_wav = opt_fits[1].data
opt_dat_sec = opt_dat[3,15,:]
scl = np.mean(opt_dat_sec)/np.mean(spectrum_2D)
plt.plot(waves[:-4],opt_dat_sec[4:]/scl,linewidth='2')
##plt.plot(yc,max(fluxtilde2)*np.ones(len(yc)),'ko')
plt.show()

#plt.imshow(fitted_image,interpolation='none')
#plt.show()

#img_opt = np.resize(np.dot(B,opt_dat_sec),(d1,d0)).T
#plt.imshow(img_opt,interpolation='none')
#plt.show()

ln = 0
#print(ccd[np.arange(481,496),60])
#print(ccd_small[:,ln])
#plt.plot(ccd_small[:,ln]/np.max(ccd_small[:,ln]))
#plt.plot(A[ln,:,0]/np.max(A[ln,:,0]))
#plt.show()

#xaxis = np.arange(21)-9.5
#yaxis = np.arange(21)-9.5
#sigma = 2
#gaussgrid = sf.gauss2d(xaxis,yaxis,sigma,sigma*1.2)
##plt.imshow(gaussgrid)
##plt.show()
#
#for order in range(6):
#    xherm = np.arange(-3,4,.01)
#    sig = 1
#    hout = sf.hermite(order,xherm,sig)
##    plt.plot(xherm,hout)
#    
##plt.show()
#sigx = 2
#sigy = 2.4
#hermgrid = np.zeros((np.shape(gaussgrid)))
#for ii in range(len(xaxis)):
#    for jj in range(len(yaxis)):
#        #Need to figure out a way to permute all order combinations
#        for orderx in range(3):
#            for ordery in range(3):
#                hermx = sf.hermite(orderx,ii,sigx)
#                hermy = sf.hermite(ordery,jj,sigy)
#                weight = 1/(1+orderx+ordery)
#                hermgrid[jj,ii] += weight*hermx*hermy
#            
#netgrid=gaussgrid*hermgrid
##plt.imshow(netgrid.T,interpolation='none')
##plt.show()
plt.close()