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

def arc_peaks(data,wvln,invar,ts,sampling_est=3,pad=4):
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
        harr = harr[harr>=0]
        harr = harr[harr<raw_img.shape[1]]
        varr = varr[varr>=0]
        varr = varr[varr<raw_img.shape[0]]
        small_img = raw_img[varr[0]:varr[-1]+1,harr[0]:harr[-1]+1]
        small_inv = invar[varr[0]:varr[-1]+1,harr[0]:harr[-1]+1]
        hcut = int(np.mod(np.argmax(small_img),small_img.shape[1]))
        vcut = int(np.floor(np.argmax(small_img)/small_img.shape[1]))
        hmean, hheight, hbg = sf.fit_mn_hght_bg(harr,small_img[vcut,:],small_inv[vcut,:],sigmas[k],hcenters[k],sigmas[k],powj=powers[k])
        vmean, vheight, vbg = sf.fit_mn_hght_bg(varr+voff,small_img[:,hcut],small_inv[:,hcut],sigmas[k],vcenters[k],sigmas[k],powj=powers[k])
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
        
def remove_bg(image,mask=None,sigma=3):
    """ Another attempt to remove a constant background using sigma clipping
        If needed, expand to 2D linear, maybe quadratic...
    """ 
    if mask is not None:
        image[mask] = np.nan
    im_mask = sf.sigma_clip(image,sigma=sigma,max_iters=np.size(image))
    im_mask = np.reshape(im_mask,image.shape)
    bg = np.mean(image[im_mask])
    bg_std = np.std(image[im_mask])
#    print bg, bg_std
    bg_mask = image < (bg+2*sigma/3*bg_std)
#    plt.imshow(bg_mask,interpolation='none')
#    plt.show()
    image[bg_mask] = 0
    return image
        
def poisson_bg(data,mask=None):
    """ Maybe a little simplistic, but generally should work if most data
        points are background.  If that isn't true, you can include a mask
    """
    if mask is not None:
        data = data[mask]
    ### First get very rough estimate of the background
    lam = np.mean(data[data<np.mean(data)])
    prob = sf.poisson(data,lam)
    ### Then do very weak cut
    mask = (prob > (1/data.size/1000))
    data = data[mask]
    ### Repeat with moderate cut
    lam = np.mean(data)
    prob = sf.poisson(data,lam)
    mask = (prob > (1/data.size/100))
    data = data[mask]
    ### Then iterate with stronger cut until settled on mean
    while np.sum(mask) < mask.size:
        lam = np.mean(data)
        prob = sf.poisson(data,lam)
        mask = (prob > (1.0/data.size/10))
        data = data[mask]
    return lam
    
def convolve_image(image,kernel):
    """ Runs a 2D convolution.  Returns conv_image - same dimension as image.
        Equivalent to using zero padding for edges.
    """
    kernel = kernel[::-1,::-1]  ### flip kernel orientation
    khc = np.floor(kernel.shape[1]/2)
    kvc = np.floor(kernel.shape[0]/2)
    padl = int(khc)
    padr = int(kernel.shape[1]-khc-1)
    padt = int(kvc)
    padb = int(kernel.shape[0]-kvc-1)
    pad_image = np.pad(image,np.array(([padt,padb],[padl,padr])),mode='constant')
    conv_image = np.zeros(pad_image.shape)
    for i in range(padl,image.shape[1]+padr):
        for j in range(padt,image.shape[0]+padb):
#            ### set edges
            lb = int(i-khc)
            rb = int(i+kernel.shape[1]-padl)
            tb = int(j-kvc)
            bb = int(j+kernel.shape[0]-padt)
            conv_image[j,i] = np.sum(pad_image[tb:bb,lb:rb]*kernel)
    conv_image = conv_image[padt:int(pad_image.shape[0])-padb,padl:int(pad_image.shape[1])-padr]
    return conv_image

def params_to_array(params):
    """ Converts lmfit parameters with hc0...n, vc0..n, q0/1/2, PA0/1/2 to
        numpy array for saving.
        OUTPUTS:
            centers - 2D array, row1 = hc's, row2 = vc's
            ellipse - 2D array, row1 = q0/1/2, row2 = PA0/1/2
    """
    ellipse = np.zeros((2,3))
    for i in range(3):
        ellipse[0,i] = params['q{}'.format(i)].value
        ellipse[1,i] = params['PA{}'.format(i)].value
    centers = np.zeros((2,1000)) ### second value is arbitrary, but large
    for i in range(np.shape(centers)[1]):
        try:
            centers[0,i] = params['hc{}'.format(i)].value
            centers[1,i] = params['vc{}'.format(i)].value
        except KeyError:
            break
    centers = centers[:,0:i-1] ### remove padded zeros
    return centers, ellipse
    
def fit_spline_psf(raw_img,hcenters,vcenters,sigmas,powers,readnoise,
                   gain,plot_results=False):
    """ function to fit parameters for radial bspline psf.
    """
    ### 1. Estimate spline amplitudes, centers, w/ circular model
    actypix = raw_img.shape[1]
    #r_breakpoints = [0, 1.2, 2.5, 3.7, 5, 8, 10]
    ## 2.3, 3
    #r_breakpoints = np.hstack(([0, 1.5, 2.4, 3],np.arange(3.5,10,0.5))) #For cpad=8
    
    ##########################################################################
    ############## All this hardcoded stuff should be flexible (TODO) ########    
    ##########################################################################
    
#    r_breakpoints = np.hstack(([0, 1.5, 2.4, 3],np.arange(3.5,6.6,1))) #For cpad=5
    r_breakpoints = np.hstack(([0, 1.5, 2.4, 3],np.arange(3.5,8.6,1))) #For cpad=6
    #r_breakpoints = np.hstack(([0, 1.2, 2.3, 3],np.arange(3.5,10,0.5))) #For cpad=8
    theta_orders = [0]
    cpad = 6
    bp_space = 2 #beakpoint spacing in pixels
    invar = 1/(raw_img+readnoise**2)
    ### Initial spline coeff guess
    
    spl_coeffs, s_scale, fit_params, new_hcenters, new_vcenters = spline_coeff_fit(raw_img,hcenters,vcenters,invar,r_breakpoints,sigmas,powers,theta_orders=theta_orders,cpad=cpad,bp_space=bp_space,return_new_centers=True)
    
    #'''
    ### 2. Set up and initialize while loop (other steps embedded within loop)
    num_bases = spl_coeffs.shape[1]
    new_hscale = (new_hcenters-actypix/2)/actypix
    peak_mask = np.ones((len(new_hscale)),dtype=bool) #Can be used to mask "bad" peaks
    params1 = lmfit.Parameters()
    ### Loop to add horizontal/vertical centers
    for j in range(len(new_hscale)):
        harr = np.arange(-cpad,cpad+1)+int(np.floor(new_hcenters[j]))
        varr = np.arange(-cpad,cpad+1)+int(np.floor(new_vcenters[j])) ### Shouldn't need +1...
        params1.add('vc{}'.format(j), value = new_vcenters[j]-varr[0])
        params1.add('hc{}'.format(j), value = new_hcenters[j]-harr[0])
    ### and add initial ellitical parameter guesses (for quadratic variation)
    params1.add('q0', value=0.9, min=0, max=1)
    params1.add('PA0', value=0, min=-np.pi, max=np.pi)
    params1.add('q1', value=0, min=-1, max=1)
    params1.add('PA1', value=0, min=-np.pi, max=np.pi)
    params1.add('q2', value=0, min=-1, max=1)
    params1.add('PA2', value=0, min=-np.pi, max=np.pi)
    params = lmfit.Parameters()
    params.add('hc', value = params1['hc0'].value)
    params.add('vc', value = params1['vc0'].value)
    params.add('q', value = 1, min=0, max=1)
    params.add('PA', value=0, min=-np.pi, max=np.pi)
    
    ### Start while loop - iterate until convergence
    chi_new = np.ones((sum(peak_mask))) #Can build this from first fit if desired
    chi_old = np.zeros((sum(peak_mask)))
    chi_min = 100
    coeff_matrix_min = np.zeros((3,np.shape(spl_coeffs)[1])).T
    params_min = lmfit.Parameters()
    dlt_chi = 1e-3 #difference between successive chi_squared values to cut off
    mx_loops = 50 #eventually must cutoff
    loop_cnt = 0
    fit_bg = False ## True fits a constant background at each subimage
    while abs(np.sum(chi_new)-np.sum(chi_old)) > dlt_chi and loop_cnt < mx_loops:
        print("starting loop {}".format(loop_cnt))
        print("  chi_old mean = {}".format(np.mean(chi_old)))
        print("  chi_new mean = {}".format(np.mean(chi_new)))
        print("  delta_chi = {}".format((np.sum(chi_new)-np.sum(chi_old))))
        chi_old = np.copy(chi_new)
    ### 3. Build profile, data, and noise matrices at each pixel point and sum
        dim_s = (2*cpad+1)**2
        dim_h = sum(peak_mask)*dim_s
        profile_matrix = np.zeros((dim_h,3*num_bases+fit_bg*len(new_hscale))) #hardcoded for quadratic
    #    last_profile = np.zeros((dim_s,3*num_bases+fit_bg))
        data_array = np.zeros((dim_h))
        noise_array = np.zeros((dim_h))
        data_for_fitting = np.zeros((2*cpad+1,2*cpad+1,len(new_hscale)))
        invar_for_fitting = np.zeros((2*cpad+1,2*cpad+1,len(new_hscale)))
        d_scale = np.zeros(len(new_hscale)) # Will build from data
#        bg_data = np.zeros(len(new_hscale))
        for k in range(len(new_hscale)):
            ### Slice subset of image data around each peak
            harr = np.arange(-cpad,cpad+1)+int(np.floor(new_hcenters[k]))
            varr = np.arange(-cpad,cpad+1)+int(np.floor(new_vcenters[k]))
            harr = harr[harr>=0]
            harr = harr[harr<raw_img.shape[1]]
            varr = varr[varr>=0]
            varr = varr[varr<raw_img.shape[0]]
            data_for_fitting[:,:,k] = raw_img[varr[0]:varr[-1]+1,harr[0]:harr[-1]+1]#/s_scale[k]
#            invar_for_fitting[:,:,k] = invar[varr[0]:varr[-1]+1,harr[0]:harr[-1]+1]#/s_scale[k]
            d_scale[k] = np.sum(data_for_fitting[:,:,k])
            invar_for_fitting[:,:,k] = s_scale[k]/(abs(data_for_fitting[:,:,k])+readnoise**2/s_scale[k])
#            rarr = sf.make_rarr(np.arange(2*cpad+1),np.arange(2*cpad+1),cpad,cpad)
#            bg_mask = rarr > 3
#            bg_data[k] = poisson_bg(data_for_fitting[:,:,k],mask=bg_mask)
        ### bound s_scale to (hopefully) prevent runaway growth
    #    for k in range(len(new_hscale)):
    #        sig_factor = 1 #Constrain s_scale to be within this man stddevs
    #        d_min = d_scale[k]-np.sqrt(d_scale[k])*sig_factor
    #        d_max = d_scale[k]+np.sqrt(d_scale[k])*sig_factor
    #        if s_scale[k] < d_min:
    #            s_scale[k] = d_min
    #        elif s_scale[k] > d_max:
    #            s_scale[k] = d_max
    #    s_scale *= np.sum(d_scale)/np.sum(s_scale)
        for k in range(len(new_hscale)):
            ### Pull in best center estimates
            params['hc'].value = params1['hc{}'.format(k)].value
            params['vc'].value = params1['vc{}'.format(k)].value
            ### Pull in best elliptical parameter estimates
            if loop_cnt == 0:
                params['q'].value = 1
            else:            
                params['q'].value = params1['q0'].value + params1['q1'].value*new_hscale[k] + params1['q2'].value*new_hscale[k]**2
            params['PA'].value = params1['PA0'].value + params1['PA1'].value*new_hscale[k] + params1['PA2'].value*new_hscale[k]**2
            ### Scale data
#            data_for_fitting[:,:,k] -= bg_data[k] ### remove bg first
            data_for_fitting[:,:,k] /= s_scale[k]
#            invar_for_fitting[:,:,k] *= s_scale[k]
            ### Setup arrays for spline analysis
            r_arr, theta_arr, dim1, r_inds = spline.build_rarr_thetaarr(data_for_fitting[:,:,k],params)
            ### Build data, noise, and profile array
            data_array[k*dim_s:(k+1)*dim_s] = np.ravel(data_for_fitting[:,:,k])[r_inds] #scaled, sorted data array
            noise_array[k*dim_s:(k+1)*dim_s] = np.ravel(invar_for_fitting[:,:,k])[r_inds]
            profile_base = spline.build_radial_profile(r_arr,theta_arr,r_breakpoints,theta_orders,(2*cpad+1)**2,order=4)
            profile_matrix[k*dim_s:(k+1)*dim_s,0:num_bases] = profile_base
            profile_matrix[k*dim_s:(k+1)*dim_s,num_bases:2*num_bases] = profile_base*new_hscale[k]
            profile_matrix[k*dim_s:(k+1)*dim_s,2*num_bases:3*num_bases] = profile_base*(new_hscale[k]**2)
            if fit_bg:
                profile_matrix[k*dim_s:(k+1)*dim_s,3*num_bases+k*fit_bg] = 1
    #    plt.imshow(profile_matrix,interpolation='none')
    #    plt.show()
        ### 4. Using matrices from step 3. perform chi^2 fitting for coefficients
        next_coeffs, next_chi = sf.chi_fit(data_array,profile_matrix,np.diag(noise_array))
        if fit_bg:
            bg_array = next_coeffs[3*num_bases:]
#            print bg_array*s_scale
            trunc_coeffs = next_coeffs[0:3*num_bases]
        else:
            trunc_coeffs = np.copy(next_coeffs)
        dd2 = int(np.size(trunc_coeffs)/3)
        coeff_matrix = trunc_coeffs.reshape(3,dd2).T
    #    if fit_bg: ### Don't save background fit term
    #        bg_array = coeff_matrix[:,-1]
    #        print bg_array*s_scale
    #        coeff_matrix = coeff_matrix[:,:-1]
    #    last_coeffs = np.dot(coeff_matrix,(np.vstack((ones(len(new_hscale)),new_hscale,new_hscale**2))))
        ### Check each of the profiles with next_coeffs + adjust scale factor
        profile_matrix = np.zeros((dim_s,3*num_bases+fit_bg*len(new_hscale))) #hardcoded for quadratic
        data_array = np.zeros((dim_h))
        noise_array = np.zeros((dim_h))
        chi2_first = np.zeros(len(new_hscale))
    #    fit_sums = 0
    #    print("Temp fit sums:")
        for k in range(len(new_hscale)):
            ### Pull in best center estimates
            params['hc'].value = params1['hc{}'.format(k)].value
            params['vc'].value = params1['vc{}'.format(k)].value
            ### Pull in best elliptical parameter estimates
            if loop_cnt == 0:
                params['q'].value = 1
            else:
                params['q'].value = params1['q0'].value + params1['q1'].value*new_hscale[k] + params1['q2'].value*new_hscale[k]**2
            params['PA'].value = params1['PA0'].value + params1['PA1'].value*new_hscale[k] + params1['PA2'].value*new_hscale[k]**2
            ### Setup arrays for spline analysis
            r_arr, theta_arr, dim1, r_inds = spline.build_rarr_thetaarr(data_for_fitting[:,:,k],params)
            ### Build data, noise, and profile array
            data_array[k*dim_s:(k+1)*dim_s] = np.ravel(data_for_fitting[:,:,k])[r_inds] #scaled, sorted data array
            noise_array[k*dim_s:(k+1)*dim_s] = np.ravel(invar_for_fitting[:,:,k])[r_inds]
            profile_base = spline.build_radial_profile(r_arr,theta_arr,r_breakpoints,theta_orders,(2*cpad+1)**2,order=4)
            profile_matrix[:,0:num_bases] = profile_base
            profile_matrix[:,num_bases:2*num_bases] = profile_base*new_hscale[k]
            profile_matrix[:,2*num_bases:3*num_bases] = profile_base*(new_hscale[k]**2)        
            if fit_bg:
                profile_matrix[:,3*num_bases:] = 0
                profile_matrix[:,3*num_bases+k*fit_bg] = 1
            tmp_fit = np.dot(profile_matrix,next_coeffs)
    #        print np.sum(tmp_fit)
    #        fit_sums += np.sum(tmp_fit)
            resort_inds = np.argsort(r_inds)
            tmp_fit = np.reshape(tmp_fit[resort_inds],data_for_fitting[:,:,k].shape)
        #    plt.figure("Arc, iteration {}".format(k))
        ##    plt.imshow(np.hstack((tmp_fit,small_img/s_scale[k])),interpolation='none')
            chi2_first[k] = np.sum(((tmp_fit-data_for_fitting[:,:,k])**2)*invar_for_fitting[:,:,k])#*s_scale[k]**2
        #    plt.imshow((tmp_fit-small_img/s_scale[k])*small_inv,interpolation='none')
        #    plt.show()
        #    plt.close()
    #    print "chi2 first:", chi2_first
    #    next_coeffs *= fit_sums/(k+1)
    #    s_scale /= fit_sums/(k+1)
        
        
        ### Optional place to check coefficients variation over order    
        #for i in range(8):
        #    plt.plot(new_hscale,last_coeffs[i])
        #
        #plt.show()
        #plt.close()
        
        #first_fit = np.dot(last_profile,next_coeffs)
        #print next_coeffs
        #print params['vc'].value
        #print params['hc'].value
        #print r_arr[0:10]
        #print profile_base[0]
        #print profile_matrix[0,:]/(k+1)
        #print last_profile[0]
        #print first_fit[0]
        #resort_inds = np.argsort(r_inds)
        #scale1 = np.max(small_img)/np.max(first_fit)
        ##print scale1, scale, scale1/scale
        #first_fit = np.reshape(first_fit[resort_inds],small_img.shape)
        #print np.sum(first_fit), k, scale1, s_scale[k]
        #first_fit /= np.sum(first_fit)
        ##plt.imshow(first_fit,interpolation='none')
        #plt.imshow(np.hstack((small_img/s_scale[k],first_fit,(small_img/s_scale[k]-first_fit)*small_inv)),interpolation='none')
        #plt.show()
        #plt.imshow((small_img/s_scale[k]-first_fit)*small_inv,interpolation='none')
        #plt.show()
        
        #test_xs = (np.arange(xpix)-xpix/2)/xpix
        #for i in range(num_bases):
        #    test_ys = next_coeffs[i]+next_coeffs[num_bases+i]*test_xs+next_coeffs[2*num_bases+i]*test_xs**2
        #    plt.plot(test_xs,test_ys)
        #plt.show()
        
    ### 5. Now do a nonlinear fit for hc, vc, q, and PA
        #data_for_lmfit = np.zeros((np.size(small_img),len(new_hscale)))
        #invar_for_lmfit = np.zeros((np.size(small_img),len(new_hscale)))
    #    for k in range(len(new_hscale)):
    #        harr = np.arange(-cpad,cpad+1)+int(np.floor(new_hcenters[k]))
    #        varr = np.arange(-cpad,cpad+1)+int(np.floor(new_vcenters[k]))
    #        data_for_lmfit[:,:,k] = raw_img[varr[0]:varr[-1]+1,harr[0]:harr[-1]+1]/s_scale[k]
    #        invar_for_lmfit[:,:,k] = invar[varr[0]:varr[-1]+1,harr[0]:harr[-1]+1]*(s_scale[k])
        #    r_arr, theta_arr, dim1, r_inds = spline.build_rarr_thetaarr(small_img,params)
        #    data_for_lmfit[:,k] = np.ravel(small_img)[r_inds]/s_scale[k]
        #    invar_for_lmfit[:,k] = np.ravel(small_inv)[r_inds]/np.sqrt(s_scale[k])
        #    resort_inds = np.argsort(r_inds)
        #    plt.imshow(np.resize(data_for_lmfit[:,k][resort_inds],np.shape(small_img)))
        #    plt.show()
        #    plt.close()
            
        ### Make proper inputs for minimizer function
        #centers = np.vstack((new_hcenters,new_vcenters)).T
        args = (data_for_fitting,invar_for_fitting,r_breakpoints,new_hscale,next_coeffs)
        kws = dict()
        kws['theta_orders'] = theta_orders
        kws['fit_bg'] = fit_bg
        minimizer_results = lmfit.minimize(spline.spline_poly_residuals,params1,args=args,kws=kws)
        ### Re-initialize params1, put in elliptical values.  Will add hc/vc at end
        ### (using mask, so #of values for centers will differ)
        params1['q0'].value = minimizer_results.params['q0'].value
        params1['q1'].value = minimizer_results.params['q1'].value
        params1['q2'].value = minimizer_results.params['q2'].value
        params1['PA0'].value = minimizer_results.params['PA0'].value
        params1['PA1'].value = minimizer_results.params['PA1'].value
        params1['PA2'].value = minimizer_results.params['PA2'].value
        #hc_ck = minimizer_results.params['hc0'].value + minimizer_results.params['hc1'].value*new_hscale + minimizer_results.params['hc2'].value*new_hscale**2
        #vc_ck = minimizer_results.params['vc0'].value + minimizer_results.params['vc1'].value*new_hscale + minimizer_results.params['vc2'].value*new_hscale**2
        q_ck = minimizer_results.params['q0'].value + minimizer_results.params['q1'].value*new_hscale + minimizer_results.params['q2'].value*new_hscale**2
        PA_ck = minimizer_results.params['PA0'].value + minimizer_results.params['PA1'].value*new_hscale + minimizer_results.params['PA2'].value*new_hscale**2
    #    print q_ck
    #    print PA_ck
        ### Convert so q is less than 1
        if np.max(q_ck) > 1:
            q_ck_tmp = 1/q_ck #change axis definition
            if np.max(q_ck_tmp) > 1:
                print "q array always over 1!"
            else:
                q_ck = q_ck_tmp
                PA_ck = PA_ck + np.pi/2 #change axis definition
        q_coeffs = np.polyfit(new_hscale,q_ck,2)
        PA_coeffs = np.polyfit(new_hscale,PA_ck,2)
        params1['q0'].value = q_coeffs[2]
        params1['q1'].value = q_coeffs[1]
        params1['q2'].value = q_coeffs[0]
        params1['PA0'].value = PA_coeffs[2]
        params1['PA1'].value = PA_coeffs[1]
        params1['PA2'].value = PA_coeffs[0]
    #    print q_ck
    #    print PA_ck
        #plt.plot(np.arange(5),np.arange(5))
        #plt.show()
        #plt.plot(hc_ck,vc_ck,new_hcenters,new_vcenters)
        #plt.show()
        #ecc = minimizer_results.params['q'].value
        #pos_ang = minimizer_results.params['PA'].value
        
        
        ### Check to see if elliptical values worked out well
        chi_new = np.zeros(len(new_hscale))
        for i in range(len(new_hscale)):
            params['vc'].value = minimizer_results.params['vc{}'.format(i)].value
            params['hc'].value = minimizer_results.params['hc{}'.format(i)].value
        #    harr = np.arange(-cpad,cpad+1)+int(np.floor(new_hcenters[i]))
        #    varr = np.arange(-cpad,cpad+1)+int(np.floor(new_vcenters[i]))
        #    params['vc'].value = new_vcenters[i]-varr[0]+1
        #    params['hc'].value = new_hcenters[i]-harr[0]
            x_coord = new_hscale[i]
            img_matrix = data_for_fitting[:,:,i]
            invar_matrix = invar_for_fitting[:,:,i]
            q = params1['q0'].value + params1['q1'].value*x_coord + params1['q2'].value*x_coord**2
            PA = params1['PA0'].value + params1['PA1'].value*x_coord + params1['PA2'].value*x_coord**2
            params['q'].value = q
            params['PA'].value = PA
            sp_coeffs = np.dot(coeff_matrix,np.array(([1,new_hscale[i],new_hscale[i]**2])))
            if fit_bg:
                sp_coeffs = np.hstack((sp_coeffs,bg_array[i]))
        #    r_arr, theta_arr, dim1, r_inds = spline.build_rarr_thetaarr(small_img,params)
        #    profile_base = spline.build_radial_profile(r_arr,theta_arr,r_breakpoints,theta_orders,(2*cpad+1)**2,order=4)
        
            fitted_image = spline.spline_2D_radial(img_matrix,invar_matrix,r_breakpoints,params,theta_orders,order=4,return_coeffs=False,spline_coeffs=sp_coeffs,sscale=None,fit_bg=fit_bg)
            ### Update s_scale
            chi_new[i] = np.sum(((img_matrix-fitted_image)**2)*invar_matrix)*s_scale[i]/(np.size(img_matrix)-len(sp_coeffs)-2)#*s_scale[i]**2
#            print chi_new[i]
#            print s_scale[i]
#            print np.max(invar_matrix)*3.63**2/s_scale[i]
    #        print chi_new[i]*s_scale[i]
    #        print chi_new[i]*s_scale[i]**2
            ### Set new scale - drive sum of image toward unity
            s_scale[i] = s_scale[i]*np.sum(fitted_image)
#            plt.imshow(np.hstack((img_matrix,fitted_image)),interpolation='none')#,(img_matrix-fitted_image)*invar_matrix)),interpolation='none')
#            plt.imshow(invar_matrix,interpolation='none')
    #        plt.plot(img_matrix[:,5])
    #        plt.plot(fitted_image[:,5])
#            plt.show()
#            plt.close()
        
        #print chi2_first
        #print chi2_second
        #print s_scale
        #print s_scale2
        
        ### Mask/eliminate points with high chi2
        peak_mask = sf.sigma_clip(chi_new,sigma=3,max_iters=1)
        if sum(peak_mask) < 4:
            print("Too few peaks for fitting")
            exit(0)
    #        break
        ### Update new_hscale, s_scale, new_h/vcenters
        s_scale = s_scale[peak_mask]
        cnts = len(new_hscale)
        new_hscale = np.zeros((sum(peak_mask)))
        lp_idx = 0
        for j in range(cnts):
            if not peak_mask[j]:
                print "skipping point {}".format(j)
                continue
            else:
                harr = np.arange(-cpad,cpad+1)+int(np.floor(new_hcenters[j]))
                params1.add('hc{}'.format(lp_idx), value = minimizer_results.params['hc{}'.format(j)].value)
                params1.add('vc{}'.format(lp_idx), value = minimizer_results.params['vc{}'.format(j)].value)
                new_hscale[lp_idx] = (params1['hc{}'.format(lp_idx)].value+harr[0]-1-actypix/2)/actypix
                lp_idx += 1
        new_hcenters = new_hcenters[peak_mask]
        new_vcenters = new_vcenters[peak_mask]    
        ### Record minimum values (some subsequent iterations give higher chi2)
        if loop_cnt == 0:
            coeff_matrix_min = np.copy(coeff_matrix)
            params_min = lmfit.Parameters(params1)
        if np.sum(chi_new) < chi_min:
            print "Better fit on loop ", loop_cnt
            chi_min = np.sum(chi_new)
            coeff_matrix_min = np.copy(coeff_matrix)
            params_min = lmfit.Parameters(params1)
        loop_cnt += 1
    
    ### End of loop
    print("End of Loop")
    ### Check that q, PA, aren't driving toward unphysical answers
#    test_hscale = np.arange(-1,1,0.01)
    #q = params_min['q0'].value + params_min['q1'].value*test_hscale + params_min['q2'].value*test_hscale**2
    #PA = params_min['PA0'].value + params_min['PA1'].value*test_hscale + params_min['PA2'].value*test_hscale**2
    #bg = coeff_matrix_min[0,-1] + coeff_matrix_min[1,-1]*test_hscale + coeff_matrix_min[2,-1]*test_hscale**2
    #plt.plot(test_hscale,q)
    #plt.show()
    #plt.plot(test_hscale,PA)
    #plt.show()
    #plt.plot(test_hscale,bg)
    #plt.show()
    #plt.close()    
    
    if plot_results:
        ### Plot final answers for evaluation
        for i in range(len(new_hscale)):
            params['vc'].value = minimizer_results.params['vc{}'.format(i)].value
            params['hc'].value = minimizer_results.params['hc{}'.format(i)].value
        #    harr = np.arange(-cpad,cpad+1)+int(np.floor(new_hcenters[i]))
        #    varr = np.arange(-cpad,cpad+1)+int(np.floor(new_vcenters[i]))
        #    params['vc'].value = new_vcenters[i]-varr[0]+1
        #    params['hc'].value = new_hcenters[i]-harr[0]
            x_coord = new_hscale[i]
            img_matrix = data_for_fitting[:,:,i]
            invar_matrix = invar_for_fitting[:,:,i]
            q = params_min['q0'].value + params_min['q1'].value*x_coord + params_min['q2'].value*x_coord**2
            PA = params_min['PA0'].value + params_min['PA1'].value*x_coord + params_min['PA2'].value*x_coord**2
            params['q'].value = q
            params['PA'].value = PA
            sp_coeffs = np.dot(coeff_matrix_min,np.array(([1,new_hscale[i],new_hscale[i]**2])))
            if fit_bg:
                sp_coeffs = np.hstack((sp_coeffs,bg_array[i]))
        #    r_arr, theta_arr, dim1, r_inds = spline.build_rarr_thetaarr(small_img,params)
        #    profile_base = spline.build_radial_profile(r_arr,theta_arr,r_breakpoints,theta_orders,(2*cpad+1)**2,order=4)
        
            fitted_image = spline.spline_2D_radial(img_matrix,invar_matrix,r_breakpoints,params,theta_orders,order=4,return_coeffs=False,spline_coeffs=sp_coeffs,sscale=None,fit_bg=fit_bg)
            ### Update s_scale
        #        print chi_new[i]
        #        print chi_new[i]*s_scale[i]
        #        print chi_new[i]*s_scale[i]**2
            chi_sq_red = np.sum(((img_matrix-fitted_image))**2*invar_matrix)/(np.size(img_matrix)-len(sp_coeffs)-2)*(s_scale[i])
            print "Reduced Chi^2 on iteration ", i, " is: ", chi_sq_red
#            plt.plot(fitted_image[:,cpad]/np.max(fitted_image[:,cpad]))
#            plt.plot(np.sum(fitted_image,axis=1)/np.max(np.sum(fitted_image,axis=1)))
#            plt.show()
#            plt.imshow(np.hstack((img_matrix,fitted_image,(img_matrix-fitted_image))),interpolation='none')
            plt.imshow((img_matrix-fitted_image)*invar_matrix,interpolation='none')
        #    plt.imshow((img_matrix-fitted_image)*invar_matrix,interpolation='none')
            plt.show()
            plt.close()
            
    centers, ellipse = params_to_array(params_min)
    results = np.hstack((np.ravel(coeff_matrix_min),np.ravel(ellipse)))    
    return results