#!/usr/bin/env python

#Functions for PSF fitting

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


########################################################
#########  Find peaks along arc frame ##################
########################################################

def arc_peaks(data,wvln,invar,ts=3,sampling_est=3,pad=4):
    """ Finds the position, wavelength, amplitude, width, etc. of distinct
        peaks along an extracted arc frame.
        INPUTS:
            data - extracted arc frame (a x b x c) a=telescope, b=fiber,
                   c=pixel
            wvln - wavelength corresponding to each point in 'data'
            invar - inverse variance of each point in 'data'
            ts - telescope number to use ('a' in data)
            sampling_est - estimated FWHM in pixels (must be integer)
            pad - width to fit to either side of gaussian max
        OUTPUTS:
            pos_d - position in pixels of each peak
            wl_d, mx_it_d, stddev_d, chi_d, err_d
        
    """
    #use dictionaries since number of peaks per fiber varies
    mx_it_d = dict() #max intensities of each peak
    stddev_d = dict() #FWHM of each peak
    wl_d = dict() #position of each peak in wavelength space
#    herm3_d = dict() #amplitude of third order hermite polynomial
#    herm4_d = dict() #amplitude of fourth order hermite polynomial
    pos_d = dict() #position of each peak in pixel space
    chi_d = dict() #reduced chi squared
    err_d = dict() #stddev parameter error
    for i in range(len(data[0,:,0])):
        ##Optional - use scipy to get initial guesses of peak locations
        ##Problem is this misses many of the low amplitude peaks.
        #pos_est = np.array(sig.find_peaks_cwt(data[ts,i,:],np.arange(3,4)))
        #Since spectrum has ~no background, can use my own peak finder.
        pos_est = np.zeros((len(data[ts,i,:])),dtype=int)
        for j in range(2*sampling_est,len(data[ts,i,:])-2*sampling_est):
            #if point is above both its neighbors, call it a peak
            if data[ts,i,j]>data[ts,i,j-1] and data[ts,i,j]>data[ts,i,j+1]:
                pos_est[j] = j
        #Then remove extra elements from pos_est
        pos_est = pos_est[np.nonzero(pos_est)[0]]
        #Cut out any that are within 2*sampling of each other (won't be able to fit well)
        pos_diff = ediff1d(pos_est)
        if np.count_nonzero(pos_diff<(2*sampling_est))>0:
            close_inds = np.nonzero(pos_diff<(2*sampling_est))[0]
        ### Try 1x sampling and see if that gives any more peaks in the end...
#        if np.count_nonzero(pos_diff<(1*sampling_est))>0:
#            close_inds = np.nonzero(pos_diff<(1*sampling_est))[0]
            close_inds = np.concatenate((close_inds,close_inds+1))
            close_inds = np.unique(close_inds)
            close_inds = np.sort(close_inds)
            pos_est = np.delete(pos_est,close_inds)
        #Also cut out any with a zero 1 pixels or less to either side
    #    tspl = pos_est-2
        ospl = pos_est-1
        ospr = pos_est+1
    #    tspr = pos_est+2
        zero_inds = np.zeros((len(pos_est)))
        for tt in range(len(zero_inds)):
    #        if tt < 6:
    #            print tt, ":"
    #            print data[ts,i,tspl[tt]]==0
    #            print data[ts,i,ospl[tt]]==0
    #            print data[ts,i,ospr[tt]]==0
    #            print data[ts,i,tspr[tt]]==0
            if data[ts,i,ospl[tt]]==0 or data[ts,i,ospr[tt]]==0:# or data[ts,i,tspl[tt]]==0 or data[ts,i,tspr[tt]]==0:
                zero_inds[tt]=1
    #    print zero_inds
    #    print pos_est
    #    plt.plot(data[0,0,:])
    #    plt.figure()
    #    plt.plot(pos_est)
    #    plt.show()
        pos_est = pos_est[zero_inds==0]
    #    if i == 0:
    #        print pos_est
        #variable length arrays to dump into dictionary
        num_pks = len(pos_est)
        mx_it = zeros((num_pks))
        stddev = zeros((num_pks))
#        herm3 = zeros((num_pks))
#        herm4 = zeros((num_pks))
        pos = zeros((num_pks))
        chi = zeros((num_pks))
        err = zeros((num_pks))
        pos_idx = zeros((num_pks),dtype=int)
    #    slp = zeros((num_pks))
        #Now fit gaussian with background to each (can improve function later)
        for j in range(num_pks):
            pos_idx[j] = pos_est[j]
            xarr = pos_est[j] + np.arange(-pad,pad,1)
            xarr = xarr[(xarr>0)*(xarr<2048)]
            yarr = data[ts,i,:][xarr]
            wlarr = wvln[ts,i,:][xarr]
            invarr = invar[ts,i,:][xarr]
            try:
                params, errarr = sf.gauss_fit(wlarr,yarr,invr=invarr,fit_background='n')
    #            params = sf.fit_gauss_herm1d(wlarr,yarr,invarr)
    #            errarr = np.diag(np.ones(len(params)))
            except RuntimeError:
                params = np.zeros(3)
    #            params = np.zeros(5)
                pos_idx[j] = 0
            tot = sf.gaussian(wlarr,abs(params[0]),params[1],params[2])#,params[3],params[4])
    #        tot = sf.gauss_herm1d(wlarr,abs(params[0]),params[1],params[2],params[3],params[4])
            chi_sq = sum((yarr-tot)**2*invarr)
            chi[j] = chi_sq/len(yarr)
            if chi_sq/len(yarr) > 10: #arbitrary cutoff
                params = np.zeros(5)
                pos_idx[j] = 0
            mx_it[j] = params[2] #height
            stddev[j] = params[0]#*2*sqrt(2*log(2)) #converted from std dev
    #        herm3[j] = params[3]
    #        herm4[j] = params[4]
            err[j] = np.sqrt(errarr[0,0])
            pos[j] = params[1] #center
    #        slp[j] = params[4] #bg_slope
        mx_it_d[i] = mx_it[np.nonzero(pos)[0]] #Remove zero value points
        stddev_d[i] = stddev[np.nonzero(pos)[0]]
    #    herm3_d[i] = herm3[np.nonzero(pos)[0]]
    #    herm4_d[i] = herm4[np.nonzero(pos)[0]]
        wl_d[i] = pos[np.nonzero(pos)[0]]
        pos_d[i] = pos_idx[np.nonzero(pos)[0]]
        plt.show()
        chi_d[i] = chi[np.nonzero(pos)[0]]
        err_d[i] = err[np.nonzero(pos)[0]]
    #    if i == 0:
    #        plt.plot(pos_idx,data[ts,i,:][pos_idx],'ks')
    #        plt.plot(data[ts,i,:])
    #        plt.show()
    return pos_d, wl_d, mx_it_d, stddev_d, chi_d, err_d
    
########################################################    
####### PSF fitting specific to b-splines ##############
########################################################

def spline_coeff_fit(raw_img,hcenters,vcenters,invar,r_breakpoints,sigmas,powers,theta_orders=[0],cpad=5,bp_space=2,return_new_centers=False):
    """ Highly specialized function.  Might want to re-write later for
        generality.  Takes an arc image and the horizontal/vertical centers
        of known "good" peaks with known std (sigmas) and gauss power (powers)
        Fits a spline to each peak and returns and array with the spline
        coefficients (which can later be used for 2D extraction)
    """
    new_hcenters = np.zeros(len(hcenters))
    new_vcenters = np.zeros(len(vcenters))
    fit_params = np.zeros((2,len(vcenters)))
    scale_fit = zeros((len(vcenters)))
    voff = 1
    for k in range(len(vcenters)):
        harr = np.arange(-cpad,cpad+1)+hcenters[k]
        varr = np.arange(-cpad,cpad+1)+int(np.floor(vcenters[k]))
        small_img = raw_img[varr[0]:varr[-1]+1,harr[0]:harr[-1]+1]
        small_inv = invar[varr[0]:varr[-1]+1,harr[0]:harr[-1]+1]
        hmean, hheight, hbg = sf.fit_mn_hght_bg(harr,small_img[cpad,:],small_inv[cpad,:],sigmas[k],hcenters[k],sigmas[k],powj=powers[k])
        vmean, vheight, vbg = sf.fit_mn_hght_bg(varr+voff,small_img[:,cpad],small_inv[:,cpad],sigmas[k],vcenters[k],sigmas[k],powj=powers[k])
        hdec, hint = math.modf(hmean)
        vdec, vint = math.modf(vmean)
#        small_img = recenter_img(small_img,[vmean,hmean],[varr[0]+cpad,harr[0]+cpad])
    #    print "Mean is [", vmean, hmean, "]"
    #    plt.imshow(small_img,extent=(-cpad+hcenters[k],cpad+hcenters[k]+1,-cpad+vcenters[k],cpad+vcenters[k]+1),interpolation='none')
    #    plt.show()
    #    plt.close()
#        r_breakpoints = [0, 1, 2, 2.5, 3, 3.5, 4, 5, 10]
#        r_breakpoints = [0, 1, 2, 3, 4, 5, 9]
#        theta_orders=[0,-2,2]
        args = (small_img,small_inv,r_breakpoints)
        kws = dict()
        kws['theta_orders'] = theta_orders
        params = lmfit.Parameters()
        params.add('vc', value = vmean-varr[0])
        params.add('hc', value = hmean-harr[0])
        params.add('q',value = 0.85, min=0)
        params.add('PA',value = 0)
        minimizer_results = lmfit.minimize(spline.spline_residuals,params,args=args,kws=kws)
        ecc = minimizer_results.params['q'].value
        pos_ang = minimizer_results.params['PA'].value
        if ecc > 1:
            ecc = 1/ecc
            pos_ang -= (np.pi/2)
        pos_ang = pos_ang % (2*np.pi)
#        print "Eccentricity = ", ecc
#        print "Position Angle = ", pos_ang*180/(np.pi)
#        center=[vmean-varr[0],hmean-harr[0]]
#        print center
        spline_fit, s_coeffs, s_scale = spline.spline_2D_radial(small_img,small_inv,r_breakpoints,minimizer_results.params,theta_orders=theta_orders,return_coeffs=True)
#        v_bpts = varr[np.mod(np.arange(len(varr)),bp_space)==0]-vcenters[k]
#        h_bpts = harr[np.mod(np.arange(len(harr)),bp_space)==0]-hcenters[k]
#        spline_fit, s_coeffs, s_scale = spline.spline_2D(small_img,1/(small_img+readnoise**2),h_bpts,v_bpts,return_coeffs=True)
#        if k == 0:            
#            vis = np.hstack((small_img,spline_fit,small_img-spline_fit))
#            plt.imshow(vis,interpolation='none')
##    #        plt.figure()
##    #        plt.hist(np.ravel((small_img-spline_fit)*small_inv))
#            plt.show()
#            plt.close()
        if k==0:
                spline_coeffs = zeros((len(vcenters),len(s_coeffs)))
        spline_coeffs[k] = s_coeffs
        scale_fit[k] = s_scale
        new_hcenters[k] = hmean
        new_vcenters[k] = vmean
        fit_params[:,k] = np.array(([ecc,pos_ang]))
    if return_new_centers:
        return spline_coeffs, scale_fit, fit_params, new_hcenters, new_vcenters
    else:
        return spline_coeffs, scale_fit, fit_params

def spline_coeff_eval(raw_img,hcenters,hc_ref,vcenters,vc_ref,invar,r_breakpoints,spline_poly_coeffs,s_scale,sigmas,powers,cpad=5,bp_space=2,full_image=True,view_plot=False,sp_coeffs=None,ecc_pa_coeffs=None):
    """ Another highly specialized function. Evaluates coeffs found in
        sf.interpolate_coeffs.  Right now, displays images of the fit
        compared to data.
        Returns spline_fit
    """
    voff = 1
    for k in range(len(vcenters)):
        if full_image:
            hmean = hcenters[k]
            vmean = vcenters[k]
            small_img = raw_img
            small_inv = invar
        else:
            harr = np.arange(-cpad,cpad+1)+hcenters[k]
            varr = np.arange(-cpad,cpad+1)+vcenters[k]
            small_img = raw_img[varr[0]:varr[-1]+1,harr[0]:harr[-1]+1]
            small_inv = invar[varr[0]:varr[-1]+1,harr[0]:harr[-1]+1]
            hmean, hheight, hbg = sf.fit_mn_hght_bg(harr,small_img[cpad,:],small_inv[cpad,:],sigmas[k],hcenters[k],sigmas[k],powj=powers[k])
            vmean, vheight, vbg = sf.fit_mn_hght_bg(varr+voff,small_img[:,cpad],small_inv[:,cpad],sigmas[k],vcenters[k],sigmas[k],powj=powers[k])
        hdec, hint = math.modf(hmean)
        vdec, vint = math.modf(vmean)
#        small_img = recenter_img(small_img,[vmean,hmean],[varr[0]+cpad,harr[0]+cpad])
    #    print "Mean is [", vmean, hmean, "]"
    #    plt.imshow(small_img,extent=(-cpad+hcenters[k],cpad+hcenters[k]+1,-cpad+vcenters[k],cpad+vcenters[k]+1),interpolation='none')
    #    plt.show()
    #    plt.close()
#        r_breakpoints = [0, 1, 2, 3, 4, 5, 9]
#        r_breakpoints = [0, 1.2, 2.5, 3.5, 5, 9]
#        theta_orders=[0,-2,2]
        theta_orders = [0]
#        spline_fit = spline.spline_2D_radial(small_img,small_inv,)
#        v_bpts = varr[np.mod(np.arange(len(varr)),bp_space)==0]-vcenters[k]
#        h_bpts = harr[np.mod(np.arange(len(harr)),bp_space)==0]-hcenters[k]
        spline_coeffs = np.zeros((len(spline_poly_coeffs[0])))
        for l in range(len(spline_poly_coeffs[0])):
            spline_coeffs[l] = sf.eval_polynomial_coeffs(hcenters[k],spline_poly_coeffs[:,l])
        ecc_pa = np.ones((2))
        if ecc_pa_coeffs is not None:
            for m in range(2):
                ecc_pa[m] = sf.eval_polynomial_coeffs(hcenters[k],ecc_pa_coeffs[m])
                
#        if k==0:
#            print spline_coeffs
        params = lmfit.Parameters()
        params.add('vc', value = vmean-vc_ref)
        params.add('hc', value = hmean-hc_ref)
        params.add('q',ecc_pa[0])
        params.add('PA',ecc_pa[1])
        spline_fit = spline.spline_2D_radial(small_img,small_inv,r_breakpoints,params,theta_orders=theta_orders,spline_coeffs=spline_coeffs,sscale=s_scale[k])
        if view_plot:
            plt.close()
            res = (small_img-spline_fit)*np.sqrt(small_inv)
            print("Chi^2 reduced = {}".format(np.sum(res**2)/(np.size(res)-2-len(spline_poly_coeffs[0]))))
            vis = np.vstack((small_img,spline_fit,res))
            plt.imshow(vis,interpolation='none')
#            plt.imshow(spline_fit,interpolation='none')
            plt.show()
            plt.close()
            plt.plot(spline_fit[:,5])
            plt.show()
            plt.close()
##        plt.figure()
##        plt.hist(np.ravel((small_img-spline_fit)*np.sqrt(small_inv)))
##        plt.imshow((small_img-spline_fit)*small_inv,interpolation='none')
#        chi2 = np.sum((small_img-spline_fit)**2*small_inv)/(np.size(small_img)-len(spline_coeffs))
#        print("Chi^2 = {}".format(chi2))
#        plt.show()
#        plt.close()
        return spline_fit