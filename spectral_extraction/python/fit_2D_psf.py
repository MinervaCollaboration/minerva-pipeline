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
#parser.add_argument("-f","--filename",help="Name of arc file (.fits) to use for PSF fitting")
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
        other_weights, lor_params, hcenters, vcenters, chi2r = psf.fit_ghl_psf(arc, hcenters, vcenters, s_coeffs, p_coeffs, pix_err, gain, pord=2, cpad=cpad, plot_results=False, verbose=True, return_centers=return_centers)
#        hdu1 = pyfits.PrimaryHDU(norm_weights)
        hdu1 = pyfits.PrimaryHDU(other_weights)
        hdu2 = pyfits.PrimaryHDU(lor_params)
        hdu3 = pyfits.PrimaryHDU(s_coeffs)
        hdu4 = pyfits.PrimaryHDU(p_coeffs)
        if return_centers:
            hdu5 = pyfits.PrimaryHDU(hcenters)
            hdu6 = pyfits.PrimaryHDU(vcenters)
        hdu1.header.append(('PSFTYPE',args.psf,'Model used for finding PSF'))
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
        
   
'''
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
#norm_weights = f_vals[0].data
weights = f_vals[0].data.ravel()
lorentz_fits = f_vals[1].data
hcenters = f_vals[4].data
vcenters = f_vals[5].data
pord = 2
params = psf.init_params(hcenters, vcenters, s_coeffs, p_coeffs, pord, r_guess=lorentz_fits[:,1], s_guess = lorentz_fits[:,0])
ncenters = len(hcenters)
icenters = np.vstack((hcenters, vcenters))

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
#    weights = np.zeros((ncenters+11*(pord+1)))
#    weights[0:ncenters], weights[ncenters:] = norm_weights[0:ncenters], np.ravel(other_weights)
#    weights[0:ncenters], weights[ncenters:] = np.ones((ncenters)), np.ravel(other_weights)
#    lweights = np.ones((ncenters))
#    lweights = 1.0*weights[0:ncenters]
    #ows = np.reshape(other_weights, (11,pord+1))
#    ows = 1.0*other_weights
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
    lorentzs = psf.get_data_minus_lorentz(data, icenters, params, weights, pord, cpad, return_lorentz=True)
    gh_models = psf.get_ghl_profile(icenters, params, weights, pord, cpad, return_model=True)
    model = gh_models[i] + lorentzs[i] + params['bg{}'.format(i)].value
    params1 = psf.convert_params(params, i, pord)
    print "Model sum:", np.sum(model)
    norm_model = model/np.sum(model)
    profile = np.ravel(norm_model).reshape((norm_model.size,1))
    noise = np.diag(np.ravel(invar[i]))
    D = np.ravel(data[i])
    coeff, chi = sf.chi_fit(D, profile, noise)
    print "fit vs. sum:", coeff[0], np.sum(data[i])#, chi[0]/(norm_model.size-1)
    print "chi2r:", np.sum((data[i]-norm_model*coeff[0])**2*invar[i])/(data[i].size-3-12)
    #print gh_models[i]
    #print lorentzs[i]
#    print np.sum(model), np.sum(gh_models[i])/np.sum(model)
    plt.imshow(np.hstack((data[i], norm_model*coeff[0], data[i]-norm_model*coeff[0])), interpolation='none')    
#    hc = hcenters[i] - int(np.round(hcenters[i])) + cpad
#    vc = vcenters[i] - int(np.round(vcenters[i])) + cpad
#    plt.plot(hc, vc, 'b.')
#    plt.plot(hc+2*cpad+1, vc, 'b.')
#    plt.imshow((data[i]-norm_model*coeff[0])*invar[i],interpolation = 'none')
    plt.show()
    plt.close()
exit(0)
#'''




''' 
############################################################################
#RANDOM TESTING OF NORMALIZATION SUCCESS, MAY NEED TO RE-USE
############################################################################
sig = 1
power = 2
params = np.array(([0, 0, sig, power]))
weights = np.ones((12))
xarr = np.linspace(-10,10,2000)
yarr = np.linspace(-10,10,2000)
ghp = sf.gauss_herm2d(xarr, yarr, params, weights)
dx = xarr[1]-xarr[0]
print np.sum(ghp)*dx**2
exit(0)

def find_norm(sig, power):
    
#    xarr = np.linspace(-5,5,2000)
#    yarr = np.linspace(-5,5,2000)
    
    xd = np.ediff1d(xarr)[0]
#    norm = np.sum(sf.gauss_herm2d(xarr, yarr, params, weights))*xd**2
    norm = np.sum(sf.gauss_herm2d(xarr, yarr, params, weights))*xd**2
    exit(0)
    norm = np.sum(sf.gaussian(xarr, sig, height=None, power=power)*sf.hermite(2,xarr,sig,0))*xd
    plt.plot(xarr, sf.gaussian(xarr, sig, height=None, power=power))#*sf.hermite(2,xarr,sig,0), linewidth = 2)
#    h2 = sf.hermite(2,xarr,sig,0)
#    heff = xarr**2 - 1
#    plt.plot(xarr, h2, xarr, heff)
#    plt.show()
#    plt.close()
#    norm1 = np.sum(sf.gaussian(xarr, sig, height=None, power=power))*xd
#    norm2 = np.sum(sf.gaussian(xarr, sig, height=None, power=power)*xarr**2)*xd
#    print "norms:", norm, norm1
#    print "ratio=", norm/norm1, 1-norm/norm1
#    print norm
    return norm
    
norm = find_norm(sig,power)
#print norm
#plt.show()
#plt.close()
##norm2 = find_norm(0.5, power)
#print norm2
#exit(0)
from scipy.special import gamma
#from scipy.special import gammaincc as gamup
def alt_norm(sig, power):
#    anorm = (2/power)**2*(2*sig**2)**(2/power)*(np.pi)**(power/2)
#    anorm = (2/power)**2*(2**(2/power))*sig**(4/power)
#    anorm = sig**2*(2*np.pi)**(power/2)*(2/power)**2
#    anorm = sig**2*(2/power)**2*(2**(2/power))
#    anorm = 2*(2*sig**power)**(1/power)*np.sqrt(np.pi)*gamma(power)/gamma(power+1)
#    anorm = (2*sig**power)**(1/power)*(2/power)*gamma(1/power)
    a = 2*sig**2
    anorm = (2/power)*a**(1/power)*(a**(2/power)*gamma(3/power) - gamma(1/power))
    return anorm
anorm = alt_norm(sig, power)
print anorm**2
#nr = norm/anorm
#print nr
#print nr**-(2/power)
#print nr**(2/power)
exit(0)


#'''