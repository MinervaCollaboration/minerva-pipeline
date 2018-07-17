#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 10:49:18 2016

@author: matt cornachione

2D PSF fitter for MINERVA - output to be used in SP_extract.py

Made for bsplines or Gauss-Hermite Polynomials w/ Lorentzian wings (see Pandey
2010, thesis).  The bsplines code is older and less cleanly written.  Also
bsplines depends on choices of breakpoints and a few other user decisions.
Gauss-Hermite/Lorentzian is the primary method employed by this code.

Set up to find the latest arc frames and fit to these.  Allows for stacking
of multiple arcs on the same day (recommend mean stacking, not median
to cut down noise + cosmic rays are rare here).  Assumes arcs have already
been reduced through optimal extract - used to identify peak locations

Also assumes traces are already fit to fiberflat
"""

#Import all of the necessary packages
from __future__ import division
import pyfits
import os
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
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
parser.add_argument("-fib","--num_fibers",help="Number of fibers to extract",
                    type=int,default=29)
parser.add_argument("-bs","--bundle_space",help="Minimum spacing (in pixels) between fiber bundles",
                    type=int,default=40)
parser.add_argument("-fs","--fiber_space",help="Minimum spacing (in pixels) between fibers within a bundle",
                    type=int,default=13)
parser.add_argument("-ts","--telescope",help="Telescope to analyze",
                    type=str,default='T1') 
parser.add_argument("-np","--num_points",help="Number of trace points to fit on each fiber",
                    type=int,default=20)
parser.add_argument("-p","--psfmodel",help="Type of model to be used for PSF fitting.",
                    type=str,default='ghl')
parser.add_argument("-par","--parallel",help="Run in parallel mode",action='store_true')
parser.add_argument("--par_index",help="Fiber index for parallel mode",
                    type=int)
parser.add_argument("-v","--verbose",help="Prints messages to standard output",
                    action='store_true')
parser.add_argument("-P","--plot_fitted",help="Plots individual PSF fits + gives X^2",
                    action='store_true')
parser.add_argument("-o","--overwrite",help="overwrites stacked arc frames",
                    action='store_true')
args = parser.parse_args()
num_fibers = (args.num_fibers-1)#*4
verbose = args.verbose
#filename = args.filename
ts = args.telescope
ts_num = int(ts[1])-1 #integer version of string ts.  T1 = 0, T2 = 1, etc.
software_vers = 'v0.2.0' #Later grab this from somewhere else

gain = 1.3
readnoise = 3.63

##############################################################################

#########################################################
###  
###    #      ###     #     ###
###    #     #   #   # #    #  #
###    #     #   #  #####   #  #
###    ####   ###  #     #  ###
###
#########################################################
date = 'n20161123'
date2 = m_utils.find_most_recent_frame_date('arc',data_dir,return_filenames=False,date_format='nYYYYMMDD', before_date=None)
arc_files = glob.glob(os.path.join(data_dir,'*'+date,'*[tT][hH][aA][rR]*'))
no_overwrite = not args.overwrite

method = 'mean'
m_utils.save_comb_arc_flat(arc_files, 'arc', ts, redux_dir, date, no_overwrite=no_overwrite, verbose=verbose, method=method)
if method == 'mean':
    arc = pyfits.open(os.path.join(redux_dir,date,'mean_combined_arc_{}.fits'.format(ts)))[0].data
else:
    arc = pyfits.open(os.path.join(redux_dir,date,'combined_arc_{}.fits'.format(ts)))[0].data
    arc = arc[::-1,:]

if os.path.isfile(os.path.join(redux_dir,date,'trace_{}_2.fits'.format(date))):
    print "Loading Trace Frames" 
    trace_fits = pyfits.open(os.path.join(redux_dir,date,'trace_{}_2.fits'.format(date)))
    hdr = trace_fits[0].header
    profile = hdr['PROFILE']
    multi_coeffs = trace_fits[0].data
else:
    print "No reduced trace file found on date {}".format(date)
trace_coeffs, trace_sig_coeffs, trace_pow_coeffs = multi_coeffs[0], multi_coeffs[2], multi_coeffs[3]

arc_pos = dict()
Tarc = pyfits.open(os.path.join(redux_dir,date,'combined_arc_{}.proc.fits'.format(ts)))
data = Tarc[0].data
wvln = Tarc[2].data
invar = Tarc[1].data
mask = Tarc[3].data        
pos_d, wl_d, mx_it_d, stddev_d, chi_d, err_d = psf.arc_peaks(data,wvln,invar,ts=ts_num)
for i in range(len(pos_d)):
    ### slot pos_d from each telescope into master dictionary for use
    arc_pos[i] = pos_d[i]
    print "fiber {}, good peaks = {}".format(i,len(arc_pos[i]))

actypix = arc.shape[1]
### Remove bias from frame
bias = m_utils.stack_calib(redux_dir, data_dir, date)
bias = bias[::-1,0:actypix] #Remove overscan
bias_mean = np.mean(bias, axis=0)
bias = np.tile(bias_mean, (bias.shape[0],1))
arc -= bias

arc, pix_err = m_utils.remove_ccd_background(arc, cut=abs(np.min(arc)))
### pix_err is the effective readnoise of the stacked, bias-subtracted frame

#########################################################
###
###     #####   #####   #######
###     #         #        #
###     ###       #        #
###     #         #        #
###     #       #####      #
###
#########################################################
################## PSF fitting ##########################
#########################################################

if not args.parallel:
    for idx in range(num_fibers):
        hcenters = arc_pos[idx].astype(float)
        hscale = (hcenters-actypix/2)/actypix
        vcenters = np.poly1d(trace_coeffs[:,4*idx])(hscale)
        sigmas = np.poly1d(trace_sig_coeffs[:,4*idx])(hscale)
        powers = np.poly1d(trace_pow_coeffs[:,4*idx])(hscale)
        if args.verbose:
            print("Running PSF Fitting on trace {}".format(idx))
        
        if args.psfmodel is 'bspline':
            if idx == 0:
                tmp_psf_coeffs = psf.fit_spline_psf(arc,hcenters,
                             vcenters,sigmas,powers,pix_err,gain,verbose=args.verbose)
                num_coeffs = len(tmp_psf_coeffs)
                fitted_psf_coeffs = np.zeros((num_fibers,num_coeffs))
                fitted_psf_coeffs[idx,:] = tmp_psf_coeffs
            else:
                fitted_psf_coeffs[idx,:] = psf.fit_spline_psf(arc,hcenters,
                             vcenters,sigmas,powers,pix_err,gain,verbose=args.verbose)
                             
        elif args.psfmodel is 'ghl':
            print("Non-parallel version not yet set up for GHL psf")
            exit(0)
        else:
            print("Invalid PSF selection")
            print("Choose from the following:")
            print("  bspline")
            print("  ghl")
            exit(0)
            
    hdu1 = pyfits.PrimaryHDU(fitted_psf_coeffs)
    hdu1.header.append(('PSFTYPE',args.psfmodel,'Model used for finding PSF'))
    if args.psfmodel is 'bspline':   
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
    ### Get saved fits for this telescope/fiber index based on trace information
    hcenters = arc_pos[idx].astype(float)
    hscale = (hcenters-actypix/2)/actypix
    shft = int(np.mod(ts_num+1,4))
    if ts_num == 3:
        idx += 1 ### shift for traces, skip first for T4
    vcenters = np.poly1d(trace_coeffs[:,4*idx+shft])(hscale) + 2 ##have some mistake messing up vcenters, need offset of 2, but don't know why :(
    sigmas = np.poly1d(trace_sig_coeffs[:,4*idx+shft])(hscale)
    powers = np.poly1d(trace_pow_coeffs[:,4*idx+shft])(hscale)
    s_coeffs = trace_sig_coeffs[:,4*idx+shft]
    p_coeffs = trace_pow_coeffs[:,4*idx+shft]
    if ts_num == 3:
        idx -= 1 ### shift index back so it matches other telescope formats
    if args.verbose:
        print("Running PSF Fitting on trace {}".format(idx))
    if args.psfmodel is 'bspline':
        r_breakpoints = np.hstack(([0, 1.5, 2.4, 3],np.arange(3.5,8.6,1))) #For cpad=6
        #r_breakpoints = np.hstack(([0, 1.2, 2.3, 3],np.arange(3.5,10,0.5))) #For cpad=8
        theta_orders = [0]
        cpad = 6
        bp_space = 2 #beakpoint spacing in pixels
        invar = 1/(abs(arc) + readnoise**2)
        hcenters = [hcenters[0]]
        vcenters = [vcenters[0]]
        sm_arc = arc[int(vcenters[0])-cpad:int(vcenters[0])+cpad, int(hcenters[0])-cpad:int(hcenters[0])+cpad]
        sm_invar = invar[int(vcenters[0])-cpad:int(vcenters[0])+cpad, int(hcenters[0])-cpad:int(hcenters[0])+cpad]
        spline_coeffs, scale_fit, fit_params, hc, vc = psf.spline_coeff_fit(arc,hcenters,vcenters,invar,r_breakpoints,sigmas,powers,theta_orders=[0],cpad=cpad,bp_space=2,return_new_centers=True)
        params = lmfit.Parameters()
        params.add('hc', value = hc[0])#-int(hcenters[0]) + cpad)
        params.add('vc', value = vc[0])#-int(np.floor(vcenters[0])) + cpad -1)
        params.add('q', value = fit_params[0,0])
        params.add('PA', value = fit_params[1,0])
        spline_fit = spline.spline_2D_radial(sm_arc, sm_invar, r_breakpoints, params, theta_orders=[0], order=4, return_coeffs=False, spline_coeffs=spline_coeffs.T, sscale=scale_fit, fit_bg=False, pts_per_px=1)
        print np.sum((sm_arc-spline_fit)**2*sm_invar)/(np.size(sm_arc-len(spline_coeffs)-4))
        plt.imshow(np.hstack((sm_arc,spline_fit,(sm_arc-spline_fit)*25)),interpolation='none')
        plt.figure('Residuals')
        plt.imshow((sm_arc-spline_fit)*sm_invar, interpolation='none')
        plt.show()
        plt.close()
        psf_coeffs = psf.fit_spline_psf(arc,hcenters,vcenters,sigmas,powers,pix_err,gain,verbose=args.verbose, plot_results=args.plot_fitted)
        hdu1 = pyfits.PrimaryHDU(psf_coeffs)
        hdu1.header.append(('PSFTYPE',args.psfmodel,'Model used for finding PSF'))
        hdu1.header.comments['NAXIS1'] = 'Coefficients (see key - not yet made)'
        hdu1.header.append(('FIBERNUM',idx,'Fiber number (starting with 0)'))
        hdulist = pyfits.HDUList([hdu1])
        redux_dir = os.environ['MINERVA_REDUX_DIR']
        psf_dir = 'psf'
        filename = 'psf_coeffs_{:03d}'.format(idx)
        if not os.path.isdir(os.path.join(redux_dir,psf_dir)):
            os.makedirs(os.path.join(redux_dir,psf_dir))
        hdulist.writeto(os.path.join(redux_dir,psf_dir,filename+'.fits'),clobber=True)
    elif args.psfmodel == 'ghl':
        invar = 1/(abs(arc) + pix_err**2)
        ### can change cpad, pord to input arguments if desired
        cpad = 4 ### padding from center of array (makes 2*cpad+1^2 box)
        pord = 2 ### polynomial order for GHL interpolation
        return_centers = True ### Use only for testing, centers don't matter later
        other_weights, lor_params, hcenters, vcenters, chi2r = psf.fit_ghl_psf(arc, hcenters, vcenters, s_coeffs, p_coeffs, pix_err, gain, pord=2, cpad=cpad, plot_results=True, verbose=True, return_centers=return_centers)
#        hdu1 = pyfits.PrimaryHDU(norm_weights)
        hdu1 = pyfits.PrimaryHDU(other_weights)
        hdu2 = pyfits.PrimaryHDU(lor_params)
        hdu3 = pyfits.PrimaryHDU(s_coeffs)
        hdu4 = pyfits.PrimaryHDU(p_coeffs)
        if return_centers:
            hdu5 = pyfits.PrimaryHDU(hcenters)
            hdu6 = pyfits.PrimaryHDU(vcenters)
        hdu1.header.append(('PSFTYPE',args.psfmodel,'Model used for finding PSF'))
#        hdu1.header.comments['NAXIS1'] = 'GH normalization weights'
        hdu1.header.comments['NAXIS1'] = 'GH shape weights'
        hdu2.header.comments['NAXIS1'] = 'Lorentz width and ratio'
        hdu3.header.comments['NAXIS1'] = 'Gaussian sigma coefficients'
        hdu4.header.comments['NAXIS1'] = 'Gaussian power coefficients'
        if return_centers:
            hdu5.header.comments['NAXIS1'] = 'Best-fit horizontal (x) centers of arc lines'
            hdu6.header.comments['NAXIS1'] = 'Best-fit vertical (y) centers of arc lines'
        hdu1.header.append(('FIBERNUM',idx,'Fiber number (starting with 0)'))
        hdu1.header.append(('TSCOPE',ts,'Telescope (T1, T2, T3, or T4'))
        hdu1.header.append(('CHI2R', chi2r[0], 'Reduced chi^2 from this order'))
        hdulist = pyfits.HDUList([hdu1])
        hdulist.append(hdu2)
        hdulist.append(hdu3)
        hdulist.append(hdu4)
#        hdulist.append(hdu5)
        if return_centers:
            hdulist.append(hdu5)
            hdulist.append(hdu6)
        redux_dir = os.environ['MINERVA_REDUX_DIR']
        psf_dir = 'psf'
        filename = '{}_psf_coeffs_{}_{:03d}.fits'.format(args.psfmodel,ts,idx)
        if not os.path.isdir(os.path.join(redux_dir,psf_dir)):
            os.makedirs(os.path.join(redux_dir,psf_dir))
        hdulist.writeto(os.path.join(redux_dir,psf_dir,filename),clobber=True)
    else:
        print("Invalid PSF selection {}".format(args.psfmodel))
        print("Choose from the following:")
        print("  bspline")
        print("  ghl")
        exit(0)