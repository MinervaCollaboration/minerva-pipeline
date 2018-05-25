#!/usr/bin/env python

#Optimal Extraction code for MINERVA pipeline

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
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
#import scipy
#import scipy.stats as stats
#import scipy.special as sp
import scipy.interpolate as si
#import scipy.optimize as opt
#import scipy.sparse as sparse
#import scipy.signal as signal
#import scipy.linalg as linalg
#import solar
import special as sf
import argparse
import minerva_utils as m_utils

### Set image plotting keywords
myargs = {'interpolation':'none'}

t0 = time.time()

######## Import environmental variables #################

try:
    data_dir = os.environ['MINERVA_DATA_DIR']
except KeyError:
    print("Must set MINERVA_DATA_DIR")
    exit(0)
#    data_dir = "/uufs/chpc.utah.edu/common/home/bolton_data0/minerva/data"

try:
    redux_dir = os.environ['MINERVA_REDUX_DIR']
except KeyError:
    print("Must set MINERVA_REDUX_DIR")
    exit(0)
#    redux_dir = "/uufs/chpc.utah.edu/common/home/bolton_data0/minerva/redux"
    
try:
    sim_dir = os.environ['MINERVA_SIM_DIR']
except KeyError:
    print("Must set MINERVA_SIM_DIR")
    exit(0)
#    sim_dir = "/uufs/chpc.utah.edu/common/home/bolton_data0/minerva/sim"
    
#try:
#    tag_dir = os.environ['MINERVA_TAG_DIR']
#except KeyError:
    
    
#########################################################
########### Allow input arguments #######################
#########################################################
parser = argparse.ArgumentParser()
parser.add_argument("-f","--filename",help="Name of image file (.fits) to extract",
                    default=os.path.join(data_dir,'n20160216','n20160216.HR2209.0025.fits'))
#                    default=os.path.join(data_dir,'n20160115','n20160115.daytimeSky.0006.fits'))
parser.add_argument("-i","--num_fibers",help="Number of fibers to extract",
                    type=int,default=29)
parser.add_argument("-b","--bundle_space",help="Minimum spacing (in pixels) between fiber bundles",
                    type=int,default=40)
parser.add_argument("-s","--fiber_space",help="Minimum spacing (in pixels) between fibers within a bundle",
                    type=int,default=13)
parser.add_argument("-t","--telescopes",help="Number of telescopes feeding spectrograph",
                    type=int,default=4)
parser.add_argument("-c","--profile",help="Cross-dispersion profile to use for optimal extraction", type=str,default='gaussian')
parser.add_argument("-F","--fast",help="Fast fitting for extraction (slightly less precise)",
                    action='store_true')
parser.add_argument("-n","--num_points",help="Number of trace points to fit on each fiberflat",
                    type=int,default=256)#2048)
parser.add_argument("-x","--nosave",help="Don't save results",
                    action='store_true')
parser.add_argument("-z","--testing",help="Runs in test mode (overwrites files, etc.)",
                    action='store_true')
parser.add_argument("-o","--boxcar",help="Do simple boxcar extraction",
                    action='store_true')
parser.add_argument("-d","--date",help="Date of arc exposure, format nYYYYMMDD",default=None)
#parser.add_argument("-T","--tscopes",help="T1, T2, T3, and/or T4 (remove later)",
#                    type=str,default=['T1','T2','T3','T4'])
args = parser.parse_args()
num_fibers = args.num_fibers*args.telescopes
num_points = args.num_points
fiber_space = args.fiber_space
boxcar = args.boxcar
fast = args.fast

#bias = m_utils.stack_bias(redux_dir,data_dir,'n20161204')
#print np.mean(bias), np.std(bias)
#exit(0)

#########################################################
########### Load Background Requirments #################
#########################################################

#hardcode in n20160115 directory
filename = args.filename#os.path.join(data_dir,'n20160115',args.filename)
software_vers = 'v0.6.0' #Later grab this from somewhere else

gain = 1.3 ### Based on MINERVA_CCD Datasheet
readnoise = 3.63 ### Based on MINERVA_CCD Datasheet

ccd, overscan, spec_hdr = m_utils.open_minerva_fits(filename,return_hdr=True)
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
#date = 'n20160115' #Fixed for now, later make this dynamic
#date = 'n20161123' #Fixed for now, later make this dynamic
date = os.path.split(os.path.split(filename)[0])[1]
### Load Bias
bias = m_utils.stack_calib(redux_dir, data_dir, date)
bias = bias[::-1,0:actypix] #Remove overscan
#print np.mean(bias)
#exit(0)
### Load Dark
dark, dhdr = m_utils.stack_calib(redux_dir, data_dir, date, frame='dark')
dark = dark[::-1,0:actypix]
try:
    dark *= spec_hdr['EXPTIME']/dhdr['EXPTIME'] ### Scales linearly with exposure time
except:
    ### if EXPTIMES are unavailable, can't reliably subtract dark, just turn it
    ### into zeros
    dark = np.zeros(ccd.shape)
### Analyze overscan (essentially flat, very minimal correction)
overscan_fit = m_utils.overscan_fit(overscan)
bias0 = 1.0*bias
bias -= np.median(bias) ### give zero mean overall - readjust by overscan
## Making up this method, so not sure if it's good, but if it works it should reduce ccd noise
bias = m_utils.bias_fit(bias, overscan_fit)

#sflat_hdu = pyfits.open(os.path.join(redux_dir,date,'slit_approx.fits'),uint=True)
#slit_coeffs = sflat_hdu[0].data
#slit_coeffs = slit_coeffs[::-1,:] #Re-order, make sure this is right
#polyord = sflat_hdu[0].header['POLYORD'] #Order of polynomial for slit fitting

## Make master slitFlats
sflat = m_utils.stack_flat(redux_dir, data_dir, date)
### If no slit flat, sflat returns all ones, don't do any flat fielding
if np.max(sflat) - np.min(sflat) == 0:
    norm_sflat = np.ones(ccd.shape)
    sflat_mask = 0.0*norm_sflat
else:
    norm_sflat, sflat_mask = m_utils.make_norm_sflat(sflat, redux_dir, date, spline_smooth=True, keep_blaze=True, plot_results=False, include_edge=False)
sflat_mask = sflat_mask.astype(bool)
    
### Calibrate ccd
ccd -= bias #Note, if ccd is 16bit array, this operation can cause problems

### Opting not to do dark subtraction for now.  Can lead to artifacts and does
### not show appreciable improvement in overall fit (adds noise) and hot/cold
### pixels are rare
#ccd -= dark
if np.mean(bias) < 10: ### If no bias is present, median is a close estimate of background level
    ccd -= np.median(ccd)

## Find new background level (now more than readnoise because of bias/dark)
## use bstd instead of readnoise in optimal extraction
#norm_sflat = np.ones(ccd.shape)
if (np.max(norm_sflat) == np.min(norm_sflat)):
    cut = int(10*readnoise)
    junk, bstd = m_utils.remove_ccd_background(ccd,cut=cut)
    rn_eff = bstd*gain
else:
    bstd = 4.4
#    ccd, bstd = m_utils.remove_scattered_light(ccd, sflat_mask, redux_dir, date, overwrite=True)
#    junk, bstd = m_utils.remove_scattered_light(ccd, sflat_mask, redux_dir, date, overwrite=False)
    rn_eff = bstd*gain

### Use this to find inverse variance:
#invar = 1/(abs(ccd) + bstd**2)

#norm_sflat = np.ones(norm_sflat.shape)
### flatten ccd
#norm_sflat = np.ones(ccd.shape)
ccd /= norm_sflat

### Apply gain
ccd *= gain

######################################
### Find or load trace information ###
######################################

### Dynamically search for most recent arc frames (unless a date is supplied)
if args.date is None:
    arc_date = m_utils.find_most_recent_frame_date('arc', data_dir)
else:
    arc_date = args.date
    
#multi_coeffs = pyfits.open(os.path.join(redux_dir,'n20170406','trace_{}_{}.fits'.format('gaussian', 'n20170406')))[0].data    
    
daytime_sky = m_utils.stack_daytime_sky(date, data_dir, redux_dir, bias)
daytime_sky, sky_std = m_utils.remove_scattered_light(daytime_sky, sflat_mask, redux_dir, date, overwrite=True)
#sky_std = bstd

### Precise peak finding with a modified gaussian profile - save
#if os.path.isfile(os.path.join(redux_dir,date,'trace_{}_{}.fits'.format('gaussian', date))) and not args.testing:
#    multi_coeffs = pyfits.open(os.path.join(redux_dir,date,'trace_{}_{}.fits'.format('gaussian', date)))[0].data
#else:
#    multi_coeffs = m_utils.find_trace_coeffs(daytime_sky,6,fiber_space,rn=sky_std*gain,num_points=num_points,num_fibers=num_fibers,skip_peaks=1, profile='gaussian')
#    hdu1 = pyfits.PrimaryHDU(multi_coeffs)
#    hdulist = pyfits.HDUList([hdu1])
#    hdu1.header.append(('PROFILE','gaussian','Cross-dispersion profile used for trace fitting'))
#    hdulist.writeto(os.path.join(redux_dir,date,'trace_{}_{}.fits'.format('gaussian', date)),clobber=True)

#if os.path.isfile(os.path.join(redux_dir,date,'trace_{}_{}.fits'.format('gauss_lor', date))) and not args.testing:
#    multi_coeffs = pyfits.open(os.path.join(redux_dir,date,'trace_{}_{}.fits'.format('gauss_lor', date)))[0].data
#else:
#    multi_coeffs = m_utils.find_trace_coeffs(daytime_sky,12,fiber_space,rn=sky_std*gain,num_points=int(num_points/5),num_fibers=num_fibers,skip_peaks=1, profile='gauss_lor')
#    hdu1 = pyfits.PrimaryHDU(multi_coeffs)
#    hdulist = pyfits.HDUList([hdu1])
#    hdu1.header.append(('PROFILE','gauss_lor','Cross-dispersion profile used for trace fitting'))
#    hdulist.writeto(os.path.join(redux_dir,date,'trace_{}_{}.fits'.format('gauss_lor', date)),clobber=True)

if os.path.isfile(os.path.join(redux_dir,date,'trace_{}_{}.fits'.format('bspline', date))) and not args.testing:
    multi_coeffs = pyfits.open(os.path.join(redux_dir,date,'trace_{}_{}.fits'.format('bspline', date)))[0].data
else:
    multi_coeffs = m_utils.find_trace_coeffs(daytime_sky,12,fiber_space,rn=sky_std*gain,num_points=int(num_points/5),num_fibers=num_fibers,skip_peaks=1, profile='bspline')
    hdu1 = pyfits.PrimaryHDU(multi_coeffs)
    hdulist = pyfits.HDUList([hdu1])
    hdu1.header.append(('PROFILE','bspline','Cross-dispersion profile used for trace fitting'))
    hdulist.writeto(os.path.join(redux_dir,date,'trace_{}_{}.fits'.format('bspline', date)),clobber=True)
#
#print "Done with Gauss-Lor Test"
#exit(0)

### Old traces for comparison
#arc_date = 'n20161123'
#trace_fits = pyfits.open(os.path.join(redux_dir,arc_date,'trace_{}.fits'.format(arc_date)))
#hdr = trace_fits[0].header
#profile = hdr['PROFILE']
#multi_coeffs = trace_fits[0].data

#plt.figure('Trace Check')
#plt.imshow(np.log(ccd),interpolation='none')
#ypix = ccd.shape[1]
#t_coeffs = multi_coeffs[0]
#for i in range(num_fibers):
#    if i < 0 or i > 110:
#        continue
#    ys = (np.arange(ypix)-ypix/2)/ypix
#    xs = np.poly1d(t_coeffs[:,i])(ys)
#    yp = np.arange(ypix)
#    plt.plot(yp,xs, 'b', linewidth=2)
#plt.show()
#plt.close() 
    
### Now find bspline profiles using the gaussian trace centers (this is saved to disk)
skip_fibs = [0, 111, 112, 113, 114, 115]
#skip_fibs = np.arange(116)
#skip_fibs = np.delete(skip_fibs,3)
#m_utils.bspline_pre(daytime_sky, multi_coeffs[0], redux_dir, date, rn=rn_eff, window=10, skip_fibs=skip_fibs)

''' # Trace finding with fiber flats
### Assumes fiber flats are taken on same date as arcs
fiber_flat_files = glob.glob(os.path.join(data_dir,'*'+arc_date,'*[fF]iber*[fF]lat*'))

if os.path.isfile(os.path.join(redux_dir,arc_date,'trace_{}_{}.fits'.format(args.profile,arc_date))):
    print "Loading Trace Frames" 
    trace_fits = pyfits.open(os.path.join(redux_dir,arc_date,'trace_{}_{}.fits'.format(args.profile,arc_date)))
    hdr = trace_fits[0].header
    profile = hdr['PROFILE']
    multi_coeffs = trace_fits[0].data
else:
    ### Combine four fiber flats into one image
    profile = args.profile
    if profile == 'moffat' or profile == 'gaussian' or profile == 'gauss_lor' or profile == 'bspline':
        pass
    else:
        print("Invalid profile choice ({})".format(profile))
        print("Available choices are:\n  moffat\n  gaussian\n  gauss_lor\n bspline")
        exit(0)
    if arc_date == 'n20161123':
        backup_files = ['a']*4
        for i in range(4):
            backup_files[i] = os.path.join(redux_dir,'n20161123','combined_flat_T{}.fits'.format(int(i+1)))
        backup_fiberflat = m_utils.build_trace_ccd(backup_files, ccd.shape, reverse_y=False)
        trace_ccd = 1.0*backup_fiberflat
    else:
        trace_ccd = m_utils.build_trace_ccd(fiber_flat_files, ccd.shape)
    ### Find traces and label for the rest of the code
    ### Pre-process trace frame
#    try:
#        t_bias = m_utils.stack_calib(redux_dir, data_dir, arc_date)
#        t_bias = t_bias[::-1,0:actypix] #Remove overscan
#        t_bias = m_utils.bias_fit(t_bias, np.zeros(t_bias.shape[0]))
#        trace_ccd -= t_bias
#
#        t_sflat = m_utils.stack_flat(redux_dir, data_dir, arc_date)
#        ### If no slit flat, sflat returns all ones, don't do any flat fielding
#        if np.max(t_sflat) - np.min(t_sflat) == 0:
#            t_norm_sflat = np.ones(trace_ccd.shape)
#        else:
#            t_norm_sflat = m_utils.make_norm_sflat(t_sflat, redux_dir, arc_date, spline_smooth=True, plot_results=False, include_edge=False)        
#        
#        t_bg = trace_ccd[t_norm_sflat==1]
#        cut = np.median(t_bg)
#        if cut < 15:
#            cut = 15 ### enforce minimum
#        junk, tbstd = m_utils.remove_ccd_background(t_bg,cut=cut)
#        rn_t = tbstd*gain # effective/empirical readnoise (including effects of bias/dark subtraction)
#        trace_ccd -= cut
#    except:
    trace_ccd -= np.median(trace_ccd)
    rn_t = 3.63
#    trace_ccd = pyfits.open(os.path.join(redux_dir,'n20161123','simulated_trace.fits'))[0].data
    print("Searching for Traces")
#    for i in np.arange(0,2000,100):
#        plt.plot(trace_ccd[i,:])
#        plt.show()
#        plt.close()
    if profile == 'bspline':
        ### Need first guess at trace from somewhere...
#        trace_ccd /= norm_sflat
        if os.path.isfile(os.path.join(redux_dir,arc_date,'trace_gaussian_{}.fits'.format(arc_date))):
            trace_fits = pyfits.open(os.path.join(redux_dir,arc_date,'trace_gaussian_{}.fits'.format(arc_date)))
            t_coeffs = trace_fits[0].data[0]
        else:
            num_pts_tmp = 100
            multi_coeffs = m_utils.find_trace_coeffs(trace_ccd,6,fiber_space,num_points=num_pts_tmp,num_fibers=num_fibers,skip_peaks=1, profile='gaussian')
            hdu1 = pyfits.PrimaryHDU(multi_coeffs)
            hdulist = pyfits.HDUList([hdu1])
            hdu1.header.append(('PROFILE',profile,'Cross-dispersion profile used for trace fitting'))
            hdulist.writeto(os.path.join(redux_dir,arc_date,'trace_gaussian_{}.fits'.format(arc_date)),clobber=True)
            t_coeffs = multi_coeffs[0]
#        m_utils.bspline_pre(trace_ccd, t_coeffs, redux_dir, arc_date, rn=rn_t, window=5)
        num_points = actypix
        
    multi_coeffs = m_utils.find_trace_coeffs(trace_ccd,12,fiber_space,num_points=num_points,num_fibers=num_fibers,skip_peaks=1, profile=profile)
#    trace_coeffs = multi_coeffs[0]
#    trace_intense_coeffs = multi_coeffs[1]
#    trace_sig_coeffs = multi_coeffs[2]
#    trace_pow_coeffs = multi_coeffs[3]
#    ### Save for future use
#    hdu1 = pyfits.PrimaryHDU(trace_coeffs)
#    hdu2 = pyfits.PrimaryHDU(trace_intense_coeffs)
#    hdu3 = pyfits.PrimaryHDU(trace_sig_coeffs)
#    hdu4 = pyfits.PrimaryHDU(trace_pow_coeffs)
#    hdulist = pyfits.HDUList([hdu1])
#    hdulist.append(hdu2)
#    hdulist.append(hdu3)
#    hdulist.append(hdu4)
    hdu1 = pyfits.PrimaryHDU(multi_coeffs)
    hdulist = pyfits.HDUList([hdu1])
    hdu1.header.append(('PROFILE',profile,'Cross-dispersion profile used for trace fitting'))
    hdulist.writeto(os.path.join(redux_dir,arc_date,'trace_{}_{}.fits'.format(profile, arc_date)),clobber=True)
#'''
##############################
#### Do optimal extraction ###
##############################
#spec, spec_invar, spec_mask, image_model = m_utils.extract_1D(ccd, norm_sflat, trace_coeffs,i_coeffs=trace_intense_coeffs,s_coeffs=trace_sig_coeffs,p_coeffs=trace_pow_coeffs,readnoise=bstd*gain,gain=gain,return_model=True,verbose=True)
#norm_sflat = np.ones(ccd.shape)
profile = args.profile
#profile = 'bspline'
spec, spec_invar, spec_mask, image_model, image_mask, chi2_array = m_utils.extract_1D(ccd, norm_sflat, multi_coeffs, profile, date=date, readnoise=rn_eff, gain=gain, return_model=True, verbose=True, boxcar=boxcar, fast=fast)
### Evaluate fit
invar = (1/(abs(ccd)*gain + (rn_eff)**2))
resid = (ccd-image_model)*np.sqrt(invar)
chi2tot = np.sum((ccd-image_model)**2*invar)/(np.size(ccd-actypix*num_fibers*3))
chi2total = np.sum((image_model[image_model != 0]-ccd[image_model != 0])**2*(1/(ccd[image_model != 0]+readnoise**2)))/(np.size(ccd[image_model != 0])-actypix*num_fibers*3)
print("Reduced chi^2 of ccd vs. model is {}".format(chi2total))

#chir = np.ravel(chi2_array)#/7#(fiber_space+1-5) #[itest,:]
#print np.nanmedian(chir[chir!=0])
#msk = (abs(chir) < 100)*(~np.isnan(chir))*(chir > 0)
#print np.sum(msk)/len(chir)*100, "% good"
#plt.hist(chir[msk], bins=50)                
#plt.show()
#plt.close()

image_mask = (image_mask == 0) #Invert T/F
#bg_cr_mask = m_utils.simple_sig_clip((ccd-image_model)*image_mask, sig=7) != 0

bg_cr_mask = m_utils.simple_sig_clip((ccd-image_model)*np.sqrt(invar), sig=7) != 0

chi2tot = np.sum(((ccd-image_model)**2*invar)*bg_cr_mask)
dof = ccd.size - np.sum(bg_cr_mask==0)
print "Chi2/dof = {}/{}".format(chi2tot, dof)


save_model = False
if save_model:
    np.save(os.path.join(redux_dir,'n20170307','n20170307.HR3799.0015.1dproc.npy'), image_model)
### Evaluate plots
#plt.imshow(np.hstack((ccd,image_model,ccd-image_model)), interpolation='none', cmap=cm.hot)
#plt.show()
#plt.close()
resid = (ccd-image_model)*np.sqrt(invar)
plt.imshow(resid*bg_cr_mask, vmin=-np.max(resid*bg_cr_mask), interpolation='none', cmap=cm.hot)
plt.show()
plt.close()
#plt.plot(resid[:,1000])
#plt.show()
#plt.close()
#exit(0)

############################################################
######### Import wavelength calibration ####################        
############################################################
        
i2coeffs = [3.48097e-4,2.11689] #shift in pixels due to iodine cell
i2shift = np.poly1d(i2coeffs)(np.arange(actypix))

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
#spec3D = np.zeros((args.telescopes,args.num_fibers,actypix))
#spec3D[0,:,:] = spec[np.arange(1,num_fibers,4),:]
#spec3D[1,:,:] = spec[np.arange(2,num_fibers,4),:]
#spec3D[2,:,:] = spec[np.arange(3,num_fibers,4),:]
#spec3D[3,:,:] = np.vstack((spec[np.arange(4,num_fibers,4),:],np.ones(actypix)))
#
#spec_invar3D = np.zeros((args.telescopes,args.num_fibers,actypix))
#spec_invar3D[0,:,:] = spec_invar[np.arange(1,num_fibers,4),:]
#spec_invar3D[1,:,:] = spec_invar[np.arange(2,num_fibers,4),:]
#spec_invar3D[2,:,:] = spec_invar[np.arange(3,num_fibers,4),:]
#spec_invar3D[3,:,:] = np.vstack((spec_invar[np.arange(4,num_fibers,4),:],np.zeros(actypix)))
#
#spec_mask3D = np.zeros((args.telescopes,args.num_fibers,actypix))
#spec_mask3D[0,:,:] = spec_mask[np.arange(1,num_fibers,4),:]
#spec_mask3D[1,:,:] = spec_mask[np.arange(2,num_fibers,4),:]
#spec_mask3D[2,:,:] = spec_mask[np.arange(3,num_fibers,4),:]
#spec_mask3D[3,:,:] = np.vstack((spec_mask[np.arange(4,num_fibers,4),:],np.zeros(actypix)))


spec3D = np.zeros((args.telescopes,args.num_fibers,actypix))
spec3D[0,:,:] = spec[np.arange(0,num_fibers,4),:]
spec3D[1,:,:] = spec[np.arange(1,num_fibers,4),:]
spec3D[2,:,:] = spec[np.arange(2,num_fibers,4),:]
spec3D[3,:,:] = spec[np.arange(3,num_fibers,4),:]

spec_invar3D = np.zeros((args.telescopes,args.num_fibers,actypix))
spec_invar3D[0,:,:] = spec_invar[np.arange(0,num_fibers,4),:]
spec_invar3D[1,:,:] = spec_invar[np.arange(1,num_fibers,4),:]
spec_invar3D[2,:,:] = spec_invar[np.arange(2,num_fibers,4),:]
spec_invar3D[3,:,:] = spec_invar[np.arange(3,num_fibers,4),:]

spec_mask3D = np.zeros((args.telescopes,args.num_fibers,actypix))
spec_mask3D[0,:,:] = spec_mask[np.arange(0,num_fibers,4),:]
spec_mask3D[1,:,:] = spec_mask[np.arange(1,num_fibers,4),:]
spec_mask3D[2,:,:] = spec_mask[np.arange(2,num_fibers,4),:]
spec_mask3D[3,:,:] = spec_mask[np.arange(3,num_fibers,4),:]


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
    hdu1.header.append(('UNITS','Counts','Relative photon counts (no flat fielding)'))
    hdu2.header.append(('UNITS','Inv. Var','Inverse variance'))
    hdu3.header.append(('UNITS','Wavelength','Wavelength solution lambda (Angstroms) vs px'))
    hdu4.header.append(('UNITS','Mask','True (1) or False (0) good data point'))
    hdu1.header.append(('VERSION',software_vers,'Reduction software version'))
    #### Include all old header values in new header for hdu1
    ### As usual, probably a better way, but I think this will work
    for key in spec_hdr.keys():
        if key not in hdu1.header and key != 'BSCALE' and key != 'BZERO' and key != 'COMMENT':
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
try: 
    import utils as dop_utils
    dop_utils.addzb(os.path.join(redux_dir,savedate,savefile+'.proc.fits'),fau_dir=data_dir)
    barycentric = True
except:
    barycentric = False


#############################################################
######### Log files and email alerts for low counts #########
#############################################################

T_cnts = dict()
good = 2*np.ones((4), dtype=int)
low_cut = 100
moderate_cut = 1000
log_file = os.path.join(redux_dir, savedate, savefile+'.log')
lf = open(log_file, "w")
lf.write("Extraction summary for exposure {}\n\n".format(savefile+'.fits'))
if barycentric:
    lf.write("Barycentric corrections applied")
else:
    lf.write("WARNING: Barycentric corrections not applied")
lf.write("\n\n")
lf.write("Photon count check:\n")
for i in range(4):
    Tnum = 'T{}'.format(i+1)
    T_cnts[i] = np.mean(spec3D[i,:,:][spec_mask3D[i,:,:]==2])
    if np.mean(T_cnts[i]) < low_cut:
        good[i] = 0 #Bad exposure on this telescope
        lf.write("  {}: WARNING - very low counts".format(Tnum))
    elif np.mean(T_cnts[i]) < moderate_cut:
        good[i] = 1 #Mediocre exposure on this telescope
        lf.write("  {}: Notice - moderately low counts".format(Tnum))
    else:
        lf.write("  {}: Good exposure".format(Tnum))
    lf.write("\n")
lf.write("\n")
lf.write("Mean photon count > {} on {}/4 telescopes\n".format(moderate_cut,np.sum(good==2)))
lf.write("Mean photon count > {} on {}/4 telescopes\n".format(low_cut,np.sum(good>=1)))
lf.write("{} {} {} {}".format(good[0], good[1], good[2], good[3]))
lf.close()

tf = time.time()
dws = np.ediff1d(wavelength_soln[0,1,:])
dws = np.append(dws, dws[-1])
a1p = spec3D[0,1,:]/dws
errs = 1/np.sqrt(spec_invar3D[0,0,:])
xsxn = np.sum(ccd, axis=0)
#plt.plot(wavelength_soln[0,1,:],a1p,wavelength_soln[0,1,:],xsxn/dws)
#plt.show()
#plt.close()
print("Total extraction time = {}s".format(tf-t0))