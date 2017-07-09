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
import minerva_utils as m_utils

data_dir = os.environ['MINERVA_DATA_DIR']
redux_dir = os.environ['MINERVA_REDUX_DIR']

#########################################################
########### Allow input arguments #######################
#########################################################
parser = argparse.ArgumentParser()
parser.add_argument("-f","--filename",help="Name of arc file (.fits) to use for PSF fitting")
                    #default=os.path.join(data_dir,'n20160216','n20160216.HR2209.0025.fits'))
#                    default=os.path.join(data_dir,'n20160115','n20160115.daytimeSky.0006.fits'))
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
parser.add_argument("-p","--psf",help="Type of model to be used for PSF fitting.",
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
filename = args.filename
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

if os.path.isfile(os.path.join(redux_dir,date,'trace_{}.fits'.format(date))):
    print "Loading Trace Frames" 
    trace_fits = pyfits.open(os.path.join(redux_dir,date,'trace_{}.fits'.format(date)))
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

#'''
sig = 1
power = 4
def find_norm(sig, power):
#    params = np.array(([0, 0, sig, power]))
#    weights = np.ones((12))
#    xarr = np.linspace(-5,5,2000)
#    yarr = np.linspace(-5,5,2000)
    xarr = np.linspace(-10,10,2000)
#    yarr = np.linspace(-10,10,2000)
    xd = np.ediff1d(xarr)[0]
#    norm = np.sum(sf.gauss_herm2d(xarr, yarr, params, weights))*xd**2
    norm = np.sum(sf.gaussian(xarr, sig, power=power))*xd
    return norm
    
norm = find_norm(sig,power)
print norm
#norm2 = find_norm(0.5, power)
#print norm2
#exit(0)
def alt_norm(sig, power):
#    anorm = (2/power)**2*(2*sig**2)**(2/power)*(np.pi)**(power/2)
#    anorm = (2/power)**2*(2**(2/power))*sig**(4/power)
#    anorm = sig**2*(2*np.pi)**(power/2)*(2/power)**2
    anorm = sig**2*(2/power)**2*(2**(2/power))
    return anorm
anorm = alt_norm(sig, power)
print anorm
nr = norm/anorm
print nr
#print nr**-(2/power)
print nr**(2/power)
exit(0)

## Remove when done - testing how well these can be loaded and re-used
idx = args.par_index
#hcenters = arc_pos[idx].astype(float)
#hcenters[0] = 35.193330142114498
#hscale = (hcenters-actypix/2)/(actypix)
shft = int(np.mod(ts_num+1,4))
#vcenters = np.poly1d(trace_coeffs[:,4*idx+shft])(hscale) + 2 ##have some mistake messing up vcenters, need offset of 2, but don't know why :(
#vcenters[0] = 638.82966131049409
#sigmas = np.poly1d(trace_sig_coeffs[:,4*idx+shft])(hscale)
#powers = np.poly1d(trace_pow_coeffs[:,4*idx+shft])(hscale)
s_coeffs = trace_sig_coeffs[:,4*idx+shft]
p_coeffs = trace_pow_coeffs[:,4*idx+shft]
cpad = 4
#hcenters += 0.15
#vcenters += 0.15



f_vals = pyfits.open('/home/matt/software/minerva/redux/psf/ghl_psf_coeffs_T1_010.fits')
norm_weights = f_vals[0].data
other_weights = f_vals[1].data
lorentz_fits = f_vals[2].data
hcenters = f_vals[5].data
vcenters = f_vals[6].data
pord = 2
params = psf.init_params(hcenters, vcenters, s_coeffs, p_coeffs, pord, r_guess=lorentz_fits[:,1], s_guess = lorentz_fits[:,0])
ncenters = len(hcenters)

#i = 0 ### Just look at one for now
for i in range(len(hcenters)):
#    data = arc[int(vcenters[i])-cpad:int(vcenters[i])+cpad+1, int(hcenters[i])-cpad:int(hcenters[i])+cpad+1]
#    invar = 1/(abs(data)+pix_err**2)
#    nw = norm_weights[i]
#    aws = np.zeros((11))
    #for j in range(len(aws)):
    #    aws[j] = np.poly1d(other_weights[j])(hscale[i])
    #lvls = np.zeros((2))
    #for k in range(len(lvls)):
    #    lvls[k] = np.poly1d(lorentz_fits[k])(hscale[i])
    
    #params = psf.convert_params(params, hcenters, i, pord)
    #for o in range(pord+1):
    #    params['sigl{}'.format(o)].value=lorentz_fits[o,0]
    #    params['ratio{}'.format(o)].value=lorentz_fits[o,1]
    
#    data, invar, scale_arr = psf.get_fitting_arrays(arc, hcenters, vcenters-1, pord, cpad, readnoise, scale_arr=None, return_scale_arr=True)
    data, invar, = psf.get_fitting_arrays(arc, hcenters, vcenters, pord, cpad, readnoise, scale_arr=np.ones((len(hcenters))), return_scale_arr=False)
    weights = np.zeros((ncenters+11*(pord+1)))
#    weights[0:ncenters], weights[ncenters:] = norm_weights[0:ncenters], np.ravel(other_weights)
    weights[0:ncenters], weights[ncenters:] = np.ones((ncenters)), np.ravel(other_weights)
#    lweights = np.ones((ncenters))
    lweights = 1.0*weights[0:ncenters]
    #ows = np.reshape(other_weights, (11,pord+1))
    ows = 1.0*other_weights
    #hci = np.array(([hcenters[i]]))
    #vci = np.array(([vcenters[i]]))
    #gh_model = psf.get_ghl_profile(hci, vci, params, weights, pord, cpad, return_model=True, no_convert=True)[0]
    #lorentz = psf.get_data_minus_lorentz(data, hci, vci, params, weights, pord, cpad, return_lorentz=True, no_convert=True)
    #model = gh_model + lorentz
    #plt.imshow(np.hstack((data, model*np.sum(data)*3, data-model*np.sum(data))), interpolation = 'none')
    #plt.show()
    #plt.close()
    #lweights = np.ones((len(hcenters)))
    #lweights = 1.0*norm_weights
    #print data
    #print hcenters[i]
    #print vcenters[i]
    #print params
    #print weights
    #print pord
    #print cpad
    lorentzs = psf.get_data_minus_lorentz(data, ncenters, params, weights, pord, cpad, return_lorentz=True)
    gh_models = psf.get_ghl_profile(ncenters, params, weights, pord, cpad, return_model=True)
    model = gh_models[i] + lorentzs[i] + params['bg{}'.format(i)].value
    params1 = psf.convert_params(params, i, pord)
#    print np.sum(model)
    norm_model = model/np.sum(model)
    profile = np.ravel(norm_model).reshape((norm_model.size,1))
    noise = np.diag(np.ravel(invar[i]))
    D = np.ravel(data[i])
    coeff, chi = sf.chi_fit(D, profile, noise)
    print coeff[0], np.sum(data[i])#, chi[0]/(norm_model.size-1)
    #print gh_models[i]
    #print lorentzs[i]
    print np.sum(model), np.sum(gh_models[i])/np.sum(model)
#    plt.imshow(np.hstack((data[i], norm_model*coeff[0], data[i]-norm_model*coeff[0])), interpolation='none')    
#    hc = hcenters[i] - int(np.round(hcenters[i])) + cpad
#    vc = vcenters[i] - int(np.round(vcenters[i])) + cpad
#    plt.plot(hc, vc, 'b.')
#    plt.plot(hc+2*cpad+1, vc, 'b.')
    plt.imshow((data[i]-norm_model*coeff[0])*invar[i],interpolation = 'none')
    plt.show()
    plt.close()
exit(0)

#'''

if not args.parallel:
    for idx in range(num_fibers):
        hcenters = arc_pos[idx].astype(float)
        hscale = (hcenters-actypix/2)/actypix
        vcenters = np.poly1d(trace_coeffs[:,4*idx])(hscale)
        sigmas = np.poly1d(trace_sig_coeffs[:,4*idx])(hscale)
        powers = np.poly1d(trace_pow_coeffs[:,4*idx])(hscale)
        if args.verbose:
            print("Running PSF Fitting on trace {}".format(idx))
        
        if args.psf is 'bspline':
            if idx == 0:
                tmp_psf_coeffs = psf.fit_spline_psf(arc,hcenters,
                             vcenters,sigmas,powers,pix_err,gain,verbose=args.verbose)
                num_coeffs = len(tmp_psf_coeffs)
                fitted_psf_coeffs = np.zeros((num_fibers,num_coeffs))
                fitted_psf_coeffs[idx,:] = tmp_psf_coeffs
            else:
                fitted_psf_coeffs[idx,:] = psf.fit_spline_psf(arc,hcenters,
                             vcenters,sigmas,powers,pix_err,gain,verbose=args.verbose)
                             
        elif args.psf is 'ghl':
            print("Non-parallel version not yet set up for GHL psf")
            exit(0)
        else:
            print("Invalid PSF selection")
            print("Choose from the following:")
            print("  bspline")
            print("  ghl")
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
    ### Get saved fits for this telescope/fiber index based on trace information
    hcenters = arc_pos[idx].astype(float)
    hscale = (hcenters-actypix/2)/actypix
    shft = int(np.mod(ts_num+1,4))
    vcenters = np.poly1d(trace_coeffs[:,4*idx+shft])(hscale) + 2 ##have some mistake messing up vcenters, need offset of 2, but don't know why :(
    sigmas = np.poly1d(trace_sig_coeffs[:,4*idx+shft])(hscale)
    powers = np.poly1d(trace_pow_coeffs[:,4*idx+shft])(hscale)
    s_coeffs = trace_sig_coeffs[:,4*idx+shft]
    p_coeffs = trace_pow_coeffs[:,4*idx+shft]
    if args.verbose:
        print("Running PSF Fitting on trace {}".format(idx))
    if args.psf is 'bspline':
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
#        print spline_fit
#        print scale_fit
#        spline_fit *= np.sum(sm_arc)/np.sum(spline_fit)
        print np.sum((sm_arc-spline_fit)**2*sm_invar)/(np.size(sm_arc-len(spline_coeffs)-4))
        plt.imshow(np.hstack((sm_arc,spline_fit,(sm_arc-spline_fit)*25)),interpolation='none')
#        plt.imshow(sm_arc, interpolation='none')
#        sf.plot_3D(np.hstack((sm_arc,spline_fit,(sm_arc-spline_fit)*25)))
        plt.figure('Residuals')
        plt.imshow((sm_arc-spline_fit)*sm_invar, interpolation='none')
        plt.show()
        plt.close()
        psf_coeffs = psf.fit_spline_psf(arc,hcenters,vcenters,sigmas,powers,pix_err,gain,verbose=args.verbose, plot_results=args.plot_fitted)
        hdu1 = pyfits.PrimaryHDU(psf_coeffs)
        hdu1.header.append(('PSFTYPE',args.psf,'Model used for finding PSF'))
        hdu1.header.comments['NAXIS1'] = 'Coefficients (see key - not yet made)'
        hdu1.header.append(('FIBERNUM',idx,'Fiber number (starting with 0)'))
        hdulist = pyfits.HDUList([hdu1])
        redux_dir = os.environ['MINERVA_REDUX_DIR']
        psf_dir = 'psf'
        filename = 'psf_coeffs_{:03d}'.format(idx)
        if not os.path.isdir(os.path.join(redux_dir,psf_dir)):
            os.makedirs(os.path.join(redux_dir,psf_dir))
        hdulist.writeto(os.path.join(redux_dir,psf_dir,filename+'.fits'),clobber=True)
    elif args.psf == 'ghl':
        invar = 1/(abs(arc) + pix_err**2)
        ### can change cpad, pord to input arguments if desired
        cpad = 4 ### padding from center of array (makes 2*cpad+1^2 box)
        pord = 2 ### polynomial order for GHL interpolation
        return_centers = True ### Use only for testing, centers don't matter later
        norm_weights, other_weights, lor_params, hcenters, vcenters = psf.fit_ghl_psf(arc, hcenters, vcenters, s_coeffs, p_coeffs, pix_err, gain, pord=2, cpad=cpad, plot_results=True, verbose=True, return_centers=return_centers)
        hdu1 = pyfits.PrimaryHDU(norm_weights)
        hdu2 = pyfits.PrimaryHDU(other_weights)
        hdu3 = pyfits.PrimaryHDU(lor_params)
        hdu4 = pyfits.PrimaryHDU(s_coeffs)
        hdu5 = pyfits.PrimaryHDU(p_coeffs)
        if return_centers:
            hdu6 = pyfits.PrimaryHDU(hcenters)
            hdu7 = pyfits.PrimaryHDU(vcenters)
        hdu1.header.append(('PSFTYPE',args.psf,'Model used for finding PSF'))
        hdu1.header.comments['NAXIS1'] = 'GH normalization weights'
        hdu2.header.comments['NAXIS1'] = 'GH shape weights'
        hdu3.header.comments['NAXIS1'] = 'Lorentz width and ratio'
        hdu4.header.comments['NAXIS1'] = 'Gaussian sigma coefficients'
        hdu5.header.comments['NAXIS1'] = 'Gaussian power coefficients'
        if return_centers:
            hdu6.header.comments['NAXIS1'] = 'Best-fit horizontal (x) centers of arc lines'
            hdu7.header.comments['NAXIS1'] = 'Best-fit vertical (y) centers of arc lines'
        hdu1.header.append(('FIBERNUM',idx,'Fiber number (starting with 0)'))
        hdu1.header.append(('TSCOPE',ts,'Telescope (T1, T2, T3, or T4'))
        hdulist = pyfits.HDUList([hdu1])
        hdulist.append(hdu2)
        hdulist.append(hdu3)
        hdulist.append(hdu4)
        hdulist.append(hdu5)
        if return_centers:
            hdulist.append(hdu6)
            hdulist.append(hdu7)
        redux_dir = os.environ['MINERVA_REDUX_DIR']
        psf_dir = 'psf'
        filename = '{}_psf_coeffs_{}_{:03d}.fits'.format(args.psf,ts,idx)
        if not os.path.isdir(os.path.join(redux_dir,psf_dir)):
            os.makedirs(os.path.join(redux_dir,psf_dir))
        hdulist.writeto(os.path.join(redux_dir,psf_dir,filename),clobber=True)
    else:
        print("Invalid PSF selection {}".format(args.psf))
        print("Choose from the following:")
        print("  bspline")
        print("  ghl")
        exit(0)
        
   
    
    
    
''' START OF OLD CODE


        all_weights = np.zeros((len(hcenters),12))
        all_params = np.zeros((len(hcenters),2))
        for i in range(len(hcenters)):
            sm_arc = arc[int(vcenters[i])-cpad:int(vcenters[i])+cpad+1, int(hcenters[i])-cpad:int(hcenters[i])+cpad+1]
#            sm_invar = invar[int(vcenters[i])-cpad:int(vcenters[i])+cpad, int(hcenters[i])-cpad:int(hcenters[i])+cpad+1]
            sm_invar = 1/(abs(sm_arc) + pix_err**2)
            params = lmfit.Parameters()
            params.add('xc', value = np.modf(hcenters[i])[0]+cpad, min=cpad-1, max=cpad+1)#-int(hcenters[0]) + cpad)
            params.add('yc', value = np.modf(vcenters[i])[0]+cpad, min=cpad-1, max=cpad+1)#-int(np.floor(vcenters[0])) + cpad -1)
            params.add('sig', value = sigmas[i])
            params.add('power', value = powers[i])
            params.add('sigl', value = 2*sigmas[i], min=0.5*sigmas[i])
            params.add('ratio', value = 0.95, min = 0.9, max = 1.0)
            params.add('bg', value = 0)
            weights = np.zeros(12)
            weights[0] = np.sum(sm_arc)
            chi2 = 0
            chi2old = 1
            mx_iter = 200
            itr = 0
#            if i < 5:
#                continue
            while abs(chi2-chi2old) > 0.01 and itr < mx_iter:
                chi2old = chi2
                params = sf.ghl_nonlin_fit(sm_arc, sm_invar, params, weights)
                model = sf.gh_lorentz(np.arange(sm_arc.shape[1]), np.arange(sm_arc.shape[0]), params, weights)
                chi2 = np.sum((sm_arc-model)**2*sm_invar)
#                if i == 5:
#                    print weights[0]
#                    print chi2
#                    plt.imshow(np.hstack((sm_arc,model,(sm_arc-model))),interpolation='none')
#                    plt.show()
#                    plt.close()
                weights = sf.ghl_linear_fit(sm_arc, sm_invar, params, weights)
                ### Enforce positivity in magnitude
                weights[0] = abs(weights[0])
                ### prevent runaway magnitudes
                if weights[0] > 1.5*np.sum(sm_arc):
                    print "runaway"
                    weights /= (weights[0]/np.sum(sm_arc))
                model = sf.gh_lorentz(np.arange(sm_arc.shape[1]), np.arange(sm_arc.shape[0]), params, weights)
                chi2 = np.sum((sm_arc-model)**2*sm_invar)
#                if i == 5:
#                    print weights[0]
#                    print chi2
#                    plt.imshow(np.hstack((sm_arc,model,(sm_arc-model))),interpolation='none')
#                    plt.show()
#                    plt.close()
                itr += 1
            print "Iter {}, arc/model diff = {}, Chi^2 reduced = {}, weights[0]={}".format(i, np.sum(sm_arc-model), chi2/(sm_arc.size-len(params)-weights.size),weights[0])
            all_weights[i] = weights/weights[0]
            all_params[i] = [params['sigl'].value, params['ratio'].value]
            plt.imshow(np.hstack((sm_arc,model,(sm_arc-model))),interpolation='none')
            plt.show()
            plt.close()
        msk = np.ones(len(hcenters),dtype=bool)
        inds = np.array(([1,3,4,5,7,10,13,14,16,17,23,25]))
        msk[inds] = False
        for j in range(12):
            plt.plot(hcenters[msk],all_weights[:,j][msk])
            plt.show()
            plt.close()

##############################################################################
### Snipped from find_ip.py
### Should roughly be able to find good peaks on arc frames
### Add lines at the end to fit a 2Dspline PSF along a trace

#redux_dir = os.environ['MINERVA_REDUX_DIR']
#data_dir = os.environ['MINERVA_DATA_DIR']
#Tarc = pyfits.open(os.path.join(redux_dir,'n20160130','n20160130.thar_T1_i2test.0025.proc.fits'))
#Tarc = pyfits.open(os.path.join(redux_dir,'n20160130','n20160130.thar_T2_i2test.0020.proc.fits'))
#Tarc = pyfits.open(os.path.join(redux_dir,'n20160130','n20160130.thar_T3_i2test.0012.proc.fits'))


#Traw = pyfits.open(os.path.join(data_dir,'n20160130','n20160130.thar_T4_i2test.0017.fits'))
Traw = pyfits.open('/home/matt/software/minerva/redux/n20161123/combined_arc_T1.fits')
raw_img = Traw[0].data-530

data = np.load('/home/matt/software/minerva/redux/n20161123/line_gauss_T1.npy')[10,:]

pos_d = find_peaks(data)
#pos_d, wl_d, mx_it_d, stddev_d, chi_d, err_d = psf.arc_peaks(data,data,np.ones(data.shape),ts=0)
tc = pyfits.open('/home/matt/software/minerva/redux/n20161123/trace_n20161123.fits')[0].data
hcenters = pos_d
hscl = (hcenters-2048/2)/2048
vcenters = np.poly1d(tc[0,:,41])(hscl)+1
sigmas = np.poly1d(tc[2,:,41])(hscl)
powers = np.poly1d(tc[3,:,41])(hscl)
plt.imshow(np.log10(raw_img))
plt.plot(hcenters, vcenters,'bo')
plt.show()
tmp_psf_coeffs = psf.fit_spline_psf(raw_img, hcenters, vcenters, sigmas, powers, 3.63, 1.3, plot_results=True, verbose=args.verbose)
exit(0)
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
#raw_img = raw_img[::-1,0:actypix]
raw_img -= 4*bias #Note, if ccd is 16bit array, this operation can cause problems
#raw_img /= 10
#raw_img[raw_img<0] = 0 #Enforce positivity
### More robust
raw_img, pix_err = m_utils.remove_ccd_background(raw_img)
# pix_err, bg standard deviation - use in lieu of readnoise
# END OF OLD CODE'''