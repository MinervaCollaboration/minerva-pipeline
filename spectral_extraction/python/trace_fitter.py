#!/usr/bin/env python

'''
# Fits traces for MINERVA
# Uses fiber flats to get cross-dispersion profile as a function of
# trace and ccd column (in pixels)
# Fits model coefficients to either a polynomial or a spline

# This can also be used with daytime sky spectra to find the cross-
# dispersion profile, but I have found that tends to return worse
# fits (judged by chi^2) than the profiles found from fiber flats

# This code should be run BEFORE doing any extractions, but it only
# needs to be re-run if new fiber flats are taken (which is rare)

INPUTS:
    Defaults are usually fine.
    May want to set the fiber flat date (-d)
    Can also try a few different profiles (though only gaussian and gauss_lor
        are fully supported)
    Can experiment using daytime sky if wanted through the -c option
    
OUTPUTS:
    In MINERVA_REDUX_DIR/flat_trace (or sky_trace) saves
        trace_{profile}.fits
'''

#Import all of the necessary packages
from __future__ import division
import pyfits
import os
import glob
import datetime
#import math
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import special as sf
import argparse
import minerva_utils as m_utils
from scipy.interpolate import interp1d

t0 = time.time()

######## Import environmental variables #################

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
########### Function for spline fitting #################
#########################################################
def get_bspline_profiles(fiber_ccd):
    sp_profs = {}
    for i in range(fiber_ccd.shape[1]):
        sp_profs[i] = interp1d(np.arange(fiber_ccd.shape[0]),fiber_ccd[:,i],kind='cubic')
    return sp_profs
    
#########################################################
########### Allow input arguments #######################
#########################################################
parser = argparse.ArgumentParser()
parser.add_argument("-i","--num_fibers",help="Number of fibers to extract", type=int,default=29)
parser.add_argument("-b","--bundle_space",help="Minimum spacing (in pixels) between fiber bundles", type=int,default=40)
parser.add_argument("-s","--fiber_space",help="Minimum spacing (in pixels) between fibers within a bundle", type=int,default=13)
parser.add_argument("-n","--num_points",help="Number of trace points to fit on each fiberflat",
                    type=int,default=128)
parser.add_argument("-c","--calibration",help="Type of calibration.  Either 'fiber_flat' or 'daytime_sky'", default="fiber_flat")
parser.add_argument("-m","--trace_method",help="Trace fitting method.  Can be: 'polynomial' 'spline'", default="spline")
parser.add_argument("-d","--date",help="Date of calibration exposure, format nYYYYMMDD",default=None)
parser.add_argument("-p","--profile",help="Cross-dispersion profile to use for optimal extraction.  Options: gaussian, bspline", type=str,default='gaussian')
parser.add_argument("-z","--testing",help="Runs in test mode (overwrites files, etc.)",
                    action='store_true')
args_in = parser.parse_args()
date = args_in.date
num_fibers = args_in.num_fibers*4
if date is None:
    if args_in.calibration == 'daytime_sky':
        date = datetime.datetime.today().strftime('n%Y%m%d')
        date2 = date
        if args_in.testing:
            date = 'n20170415'
            date2 = 'n20170406'
    elif args_in.calibration == 'fiber_flat':
        ### Update default if new fiber flats are taken
        date = 'n20161123'
        date2 = date
    else:
        print "Calibration argument must be either:\n  fiber_flat\n  daytime_sky"
        exit(0)

#########################################################
######## Get most recent daytime sky stack ##############
#########################################################

gain = 1.3

### Load bias frame
bias = m_utils.stack_calib(redux_dir, data_dir, date2)
bias = bias[::-1,0:2048] #Remove overscan
### Analyze exposure overscan (essentially flat, very minimal correction)
bias -= np.median(bias) ### give zero mean overall - readjust by overscan
bias = m_utils.bias_fit(bias, np.zeros((bias.shape[0])))

### load slit flat mask
sflat = m_utils.stack_flat(redux_dir, data_dir, date2)
### If no slit flat, sflat returns all ones, don't do any flat fielding
if np.max(sflat) - np.min(sflat) == 0:
    norm_sflat = np.ones(bias.shape)
    sflat_mask = 0.0*norm_sflat
else:
    norm_sflat, sflat_mask = m_utils.make_norm_sflat(sflat, redux_dir, date2, spline_smooth=True, keep_blaze=True, plot_results=False, include_edge=False)
sflat_mask = sflat_mask.astype(bool)  

if args_in.calibration == 'daytime_sky':
    ### Load daytime sky
    daytime_sky = m_utils.stack_daytime_sky(date, data_dir, redux_dir, bias, overwrite=args_in.testing)
    daytime_sky, sky_std = m_utils.remove_scattered_light(daytime_sky, sflat_mask, redux_dir, date, overwrite=True)
    
    #########################################################
    ######## Get traces with Gaussian profile ###############
    #########################################################
    
    
    ### Precise peak finding with a modified gaussian profile - save
    if os.path.isfile(os.path.join(redux_dir,"sky_trace",'trace_{}.fits'.format('gaussian'))) and not args_in.testing:
        multi_coeffs = pyfits.open(os.path.join(redux_dir,"sky_trace",'trace_{}.fits'.format('gaussian')))[0].data
    else:
        multi_coeffs = m_utils.find_trace_coeffs(daytime_sky,6,args_in.fiber_space,rn=sky_std*gain,num_points=args_in.num_points,num_fibers=num_fibers,skip_peaks=1, profile='gaussian')
        hdu1 = pyfits.PrimaryHDU(multi_coeffs)
        hdulist = pyfits.HDUList([hdu1])
        hdu1.header.append(('PROFILE','gaussian','Cross-dispersion profile used for trace fitting'))
        hdulist.writeto(os.path.join(redux_dir,"sky_trace",'trace_{}.fits'.format('gaussian')),clobber=True)
        
elif args_in.calibration == 'fiber_flat':
    ### Check if trace exists
    if os.path.isfile(os.path.join(redux_dir,"flat_trace",'trace_{}.fits'.format(args_in.profile))) and not args_in.testing:    
        print "Trace fits file already exists."
        exit(0)
    fiber_flat_files = glob.glob(os.path.join(data_dir,'*'+date,'*[fF]iber*[fF]lat*'))
    ### Quantities to save:
    hpix = bias.shape[1]
    trace_arr = np.zeros((num_fibers, hpix))
    sigma_arr = np.zeros((num_fibers, hpix))
    power_arr = np.zeros((num_fibers, hpix))
    if args_in.profile == 'gauss_lor':
        ratio_arr = np.zeros((num_fibers, hpix))
        siglr_arr = np.zeros((num_fibers, hpix))
    telescopes = ['T1','T2','T3','T4']
    for ts in telescopes:
        ### Set mask for which fibers are active (T4 = 0/4/8..., T1=1/5/9..., ...)
        ts_mask = np.zeros((num_fibers), dtype=bool)
        tint = int(int(ts[1])%len(telescopes))
        ts_mask[tint::4] = True
        method = 'median'
        m_utils.save_comb_arc_flat(fiber_flat_files, 'flat', ts, redux_dir, date, method=method)
        flat = pyfits.open(os.path.join(redux_dir,date,'combined_flat_{}.fits'.format(ts)))[0].data
        ### Invert ccd top/bottom orientation
#        flat = flat[::-1,:]
        flat -= bias
        ## If bias is empty, subtract an approximate bias
        if np.mean(bias) < 100:
            flat -= 515
        flat, bg_std = m_utils.remove_ccd_background(flat, plot=False)
        rn_eff = bg_std/2.5 ## Manual hack, gets close to the true value
        
        ### Get results for this fiber
        tr_tuple = m_utils.get_trace_arrs(flat, args_in.fiber_space, rn=rn_eff, pord=12, num_points=args_in.num_points, num_fibers=int(num_fibers/4), skip_peaks=1, method=args_in.trace_method, profile=args_in.profile, plot=False)
        trace_arr[ts_mask,:] = tr_tuple[0]
        sigma_arr[ts_mask,:] = tr_tuple[1]
        power_arr[ts_mask,:] = tr_tuple[2]
        if args_in.profile == 'gauss_lor':
            ratio_arr[ts_mask,:] = tr_tuple[3]
            siglr_arr[ts_mask,:] = tr_tuple[4]
    ### Save results into fits file, each with its own extension
    hdu1 = pyfits.PrimaryHDU(trace_arr)
    hdu2 = pyfits.PrimaryHDU(sigma_arr)
    hdu3 = pyfits.PrimaryHDU(power_arr)
    hdu1.header.append(('QUANTITY','Traces','Vertical trace position'))
    hdu2.header.append(('QUANTITY','Sigma','Gaussian width'))
    hdu3.header.append(('QUANTITY','Power','Modified Gaussian power'))
    if args_in.profile == 'gauss_lor':
        hdu4 = pyfits.PrimaryHDU(ratio_arr)
        hdu5 = pyfits.PrimaryHDU(siglr_arr)
        hdu4.header.append(('QUANTITY','Ratio','Ratio of Gauss and Lorentz'))
        hdu5.header.append(('QUANTITY','Sigma_l','Width of Lorentzian'))
    hdulist = pyfits.HDUList([hdu1])
    hdulist.append(hdu2)
    hdulist.append(hdu3)
    if args_in.profile == 'gauss_lor':
        hdulist.append(hdu4)
        hdulist.append(hdu5)
    hdu1.header.append(('PROFILE','{}'.format(args_in.profile),'Cross-dispersion profile for trace fitting'))
    hdulist.writeto(os.path.join(redux_dir,"flat_trace",'trace_{}.fits'.format(args_in.profile)),clobber=True)
else:
    print "Calibration argument must be either:\n  fiber_flat\n  daytime_sky"
    exit(0)