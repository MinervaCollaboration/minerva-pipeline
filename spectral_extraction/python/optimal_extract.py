#!/usr/bin/env python

#Start of a generic tracefit program.  Geared now toward MINERVA initial data

#Import all of the necessary packages
from __future__ import division
import pyfits
import os
import sys
import glob
#import math
import time
import numpy as np
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
#import scipy.linalg as linalg
#import solar
import special as sf
import argparse
import minerva_utils as m_utils

t0 = time.time()

######## Import environmental variables #################
#os.environ['MINERVA_DATA_DIR'] = "/uufs/chpc.utah.edu/common/home/bolton_data0/minerva/data"
#os.environ['MINERVA_REDUX_DIR'] = "/uufs/chpc.utah.edu/common/home/bolton_data0/minerva/redux"
#os.environ['MINERVA_SIM_DIR'] = "/uufs/chpc.utah.edu/common/home/bolton_data0/minerva/sim"

try:
    data_dir = os.environ['MINERVA_DATA_DIR']
except KeyError:
    print("Must set MINERVA_DATA_DIR")
    exit(0)

try:
    redux_dir = os.environ['MINERVA_REDUX_DIR']
except KeyError:
    print("Must set MINERVA_REDUX_DIR")
    exit(0)
    
try:
    sim_dir = os.environ['MINERVA_SIM_DIR']
except KeyError:
    print("Must set MINERVA_SIM_DIR")
    exit(0)
    
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
parser.add_argument("-d","--date",help="Date of arc exposure, format nYYYYMMDD",default=None)
#parser.add_argument("-T","--tscopes",help="T1, T2, T3, and/or T4 (remove later)",
#                    type=str,default=['T1','T2','T3','T4'])
args = parser.parse_args()
num_fibers = args.num_fibers*args.telescopes
num_points = args.num_points
fiber_space = args.fiber_space

#bias = m_utils.stack_bias(redux_dir,data_dir,'n20161204')
#print np.mean(bias), np.std(bias)
#exit(0)

#########################################################
########### Load Background Requirments #################
#########################################################

#hardcode in n20160115 directory
filename = args.filename#os.path.join(data_dir,'n20160115',args.filename)
software_vers = 'v0.4.1' #Later grab this from somewhere else

gain = 1.3
readnoise = 3.63

ccd, spec_hdr = m_utils.open_minerva_fits(filename,return_hdr=True)
actypix = ccd.shape[1]

### Next part checks if iodine cell is in, assumes keyword I2POSAS exists
try:
    if spec_hdr['I2POSAS']=='in':
        i2 = True
    else:
        i2 = False
except KeyError:
    i2 = False

#########################################################
########### Load Flat and Bias Frames ###################
#########################################################   
date = 'n20160115' #Fixed for now, later make this dynamic
bias_hdu = pyfits.open(os.path.join(redux_dir,date,'bias_avg.fits'),uint=True)
bias = bias_hdu[0].data
sflat_hdu = pyfits.open(os.path.join(redux_dir,date,'slit_approx.fits'),uint=True)
slit_coeffs = sflat_hdu[0].data
#slit_coeffs = slit_coeffs[::-1,:] #Re-order, make sure this is right
polyord = sflat_hdu[0].header['POLYORD'] #Order of polynomial for slit fitting

### subtract bias (slit will be handled in loop)
bias = bias[:,0:actypix] #Remove overscan
ccd -= bias #Note, if ccd is 16bit array, this operation can cause problems
ccd[ccd<0] = 0 #Enforce positivity

######################################
### Find or load trace information ###
######################################

### Dynamically search for most recent arc frames (unless a date is supplied)
if args.date is None:
    date = m_utils.find_most_recent_arc_date(data_dir)
else:
    date = args.date

### Assumes fiber flats are taken on same date as arcs
fiber_flat_files = glob.glob(os.path.join(data_dir,'*'+date,'*[fF]iber*[fF]lat*'))

if os.path.isfile(os.path.join(sim_dir,'trace_{}.fits'.format(date))):
    trace_fits = pyfits.open(os.path.join(sim_dir,'trace_{}.fits'.format(date)))
    trace_coeffs = trace_fits[0].data
    trace_intense_coeffs = trace_fits[1].data
    trace_sig_coeffs = trace_fits[2].data
    trace_pow_coeffs = trace_fits[3].data
else:
    ### Combine four fiber flats into one image
    trace_ccd = np.zeros((np.shape(ccd)))
    for ts in ['T1','T2','T3','T4']:
        flat = pyfits.open(os.path.join(redux_dir, date, 'combined_flat_{}.fits'.format(ts)))[0].data
        #Choose fiberflats with iodine cell in
        norm = 1000 #Arbitrary norm to match flats
        tmmx = np.median(np.sort(np.ravel(flat))[-100:])
        trace_ccd += flat[:,0:actypix].astype(float)*norm/tmmx
    ### Find traces and label for the rest of the code
    print("Searching for Traces")
    multi_coeffs = m_utils.find_trace_coeffs(trace_ccd,2,fiber_space,num_points=num_points,num_fibers=num_fibers,skip_peaks=1)
    trace_coeffs = multi_coeffs[0]
    trace_intense_coeffs = multi_coeffs[1]
    trace_sig_coeffs = multi_coeffs[2]
    trace_pow_coeffs = multi_coeffs[3]
    ### Save for future use
    hdu1 = pyfits.PrimaryHDU(trace_coeffs)
    hdu2 = pyfits.PrimaryHDU(trace_intense_coeffs)
    hdu3 = pyfits.PrimaryHDU(trace_sig_coeffs)
    hdu4 = pyfits.PrimaryHDU(trace_pow_coeffs)
    hdulist = pyfits.HDUList([hdu1])
    hdulist.append(hdu2)
    hdulist.append(hdu3)
    hdulist.append(hdu4)
    hdulist.writeto(os.path.join(sim_dir,'trace_{}.fits'.format(date)),clobber=True)
    

##############################
#### Do optimal extraction ###
##############################
spec, spec_invar, spec_mask, image_model = m_utils.extract_1D(ccd,trace_coeffs,i_coeffs=trace_intense_coeffs,s_coeffs=trace_sig_coeffs,p_coeffs=trace_pow_coeffs,readnoise=readnoise,gain=gain,return_model=True,verbose=True)
### Evaluate fit
chi2total = np.sum((image_model[image_model != 0]-ccd[image_model != 0])**2*(1/(ccd[image_model != 0]+readnoise**2)))/(np.size(ccd[image_model != 0])-actypix*num_fibers*3)
print("Reduced chi^2 of ccd vs. model is {}".format(chi2total))

############################################################
######### Import wavelength calibration ####################        
############################################################
        
i2coeffs = [3.48097e-4,2.11689] #shift in pixels due to iodine cell
i2shift = np.poly1d(i2coeffs)(np.arange(actypix))

arc_date = m_utils.find_most_recent_arc_date(data_dir)
ypx_mod = 2*(np.arange(0,actypix)-i2shift*i2-actypix/2)/actypix #includes iodine shift (if i2 is in)
wavelength_soln = np.zeros((args.telescopes,args.num_fibers,actypix))
for j in range(4):
    wl_hdu = pyfits.open(os.path.join(redux_dir,arc_date,'wavelength_soln_T{}.fits'.format(j+1)))
    wl_coeffs =  wl_hdu[0].data
    for i in range(args.num_fibers):
        wavelength_soln[j,i,:] = np.poly1d(wl_coeffs[i])(ypx_mod)

###############################################################
########### Stack extracted values into 3D arrays #############
###############################################################
### Would like to automate this (how?) right now, here's the fiber arrangement:
###    1st (0) - T4 from order "2" (by my minerva csv accounting)
###    2nd (1) - T1 from order "3"
###    3rd (2) - T2 from order "3"
### etc.  continues T1 through T4 and ascending orders
### right now I don't have wavelength soln for order 2, so I just eliminate
### that fiber and keep moving forward (fiber "0" isn't used)

spec3D = np.zeros((args.telescopes,args.num_fibers,actypix))
spec3D[0,:,:] = spec[np.arange(1,num_fibers,4),:]
spec3D[1,:,:] = spec[np.arange(2,num_fibers,4),:]
spec3D[2,:,:] = spec[np.arange(3,num_fibers,4),:]
spec3D[3,:,:] = np.vstack((spec[np.arange(4,num_fibers,4),:],np.ones(actypix)))

spec_invar3D = np.zeros((args.telescopes,args.num_fibers,actypix))
spec_invar3D[0,:,:] = spec_invar[np.arange(1,num_fibers,4),:]
spec_invar3D[1,:,:] = spec_invar[np.arange(2,num_fibers,4),:]
spec_invar3D[2,:,:] = spec_invar[np.arange(3,num_fibers,4),:]
spec_invar3D[3,:,:] = np.vstack((spec_invar[np.arange(4,num_fibers,4),:],np.zeros(actypix)))

spec_mask3D = np.zeros((args.telescopes,args.num_fibers,actypix))
spec_mask3D[0,:,:] = spec_mask[np.arange(1,num_fibers,4),:]
spec_mask3D[1,:,:] = spec_mask[np.arange(2,num_fibers,4),:]
spec_mask3D[2,:,:] = spec_mask[np.arange(3,num_fibers,4),:]
spec_mask3D[3,:,:] = np.vstack((spec_mask[np.arange(4,num_fibers,4),:],np.zeros(actypix)))

#############################################################
########### And finally, save spectrum ######################
#############################################################
root, savefile = os.path.split(filename)
savefile = savefile[:-5] #remove '.fits' for now
junk, savedate = os.path.split(root)
if not os.path.isdir(root):
    os.mkdir(root)

if not args.nosave:
    hdu1 = pyfits.PrimaryHDU(spec3D)
    hdu2 = pyfits.PrimaryHDU(spec_invar3D)
    hdu3 = pyfits.PrimaryHDU(wavelength_soln)
    hdu4 = pyfits.PrimaryHDU(spec_mask3D)
    hdu1.header.comments['NAXIS1'] = 'Pixel axis'
    hdu1.header.comments['NAXIS2'] = 'Fiber axis (blue to red)'
    hdu1.header.comments['NAXIS3'] = 'Telescope axis (T1, T2, T3, T4)'
    hdu2.header.comments['NAXIS1'] = 'Pixel axis'
    hdu2.header.comments['NAXIS2'] = 'Fiber axis (blue to red)'
    hdu2.header.comments['NAXIS3'] = 'Telescope axis (T1, T2, T3, T4)'
    hdu3.header.comments['NAXIS1'] = 'Pixel axis'
    hdu3.header.comments['NAXIS2'] = 'Fiber axis (blue to red)'
    hdu3.header.comments['NAXIS3'] = 'Telescope axis (T1, T2, T3, T4)'
    hdu4.header.comments['NAXIS1'] = 'Pixel axis'
    hdu4.header.comments['NAXIS2'] = 'Fiber axis (blue to red)'
    hdu4.header.comments['NAXIS3'] = 'Telescope axis (T1, T2, T3, T4)'
    ### Additional new header values
    hdu1.header.append(('UNITS','Flux','Relative photon counts (no flat fielding)'))
    hdu2.header.append(('UNITS','Inv. Var','Inverse variance'))
    hdu3.header.append(('UNITS','Wavelength','Wavelength solution lambda (Angstroms) vs px'))
    hdu4.header.append(('UNITS','Mask','True (1) or False (0) good data point'))
    hdu1.header.append(('VERSION',software_vers,'Reduction software version'))
    #### Include all old header values in new header for hdu1
    ### As usual, probably a better way, but I think this will work
    for key in spec_hdr.keys():
        if key not in hdu1.header and key != 'BSCALE' and key != 'BZERO':
            hdu1.header.append((key,spec_hdr[key],spec_hdr.comments[key]))              
    hdulist = pyfits.HDUList([hdu1])
    hdulist.append(hdu2)
    hdulist.append(hdu3)
    hdulist.append(hdu4)
    if not os.path.isdir(os.path.join(redux_dir,savedate)):
        os.makedirs(os.path.join(redux_dir,savedate))
#    print os.path.join(redux_dir,savedate,savefile+'.proc.fits')
    hdulist.writeto(os.path.join(redux_dir,savedate,savefile+'.proc.fits'),clobber=True)

### Now add Jason Eastman's custom headers for barycentric corrections  
import utils as dop_utils
dop_utils.addzb(os.path.join(redux_dir,savedate,savefile+'.proc.fits'),fau_dir=data_dir)

tf = time.time()
print("Total extraction time = {}s".format(tf-t0))