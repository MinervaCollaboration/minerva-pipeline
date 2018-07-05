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
from scipy.interpolate import interp1d
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
        #Cut out any that are within 3*sampling of each other (won't be able to fit well)
        pos_diff = ediff1d(pos_est)
        if np.count_nonzero(pos_diff<(3*sampling_est))>0:
            close_inds = np.nonzero(pos_diff<(3*sampling_est))[0]
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
#    print "scfit"
    new_hcenters = np.zeros(len(hcenters))
    new_vcenters = np.zeros(len(vcenters))
    fit_params = np.zeros((2,len(vcenters)))
    scale_fit = zeros((len(vcenters)))
#    voff = 1
#    print len(vcenters)
    for k in range(len(vcenters)):
        chi2old = 1e10
        chi2 = 0
        mx = 50
        itr = 0
        while abs(chi2-chi2old) > 0.001 and itr < mx:
            chi2old = 1.0*chi2
            harr = np.arange(-cpad,cpad+1)+hcenters[k]
            varr = np.arange(-cpad,cpad+1)+int(np.floor(vcenters[k]))
            harr = harr[harr>=0]
            harr = harr[harr<raw_img.shape[1]]
            varr = varr[varr>=0]
            varr = varr[varr<raw_img.shape[0]]
            small_img = raw_img[varr[0]:varr[-1]+1,harr[0]:harr[-1]+1]
    #        plt.imshow(small_img, interpolation='none')
    ##        plt.plot(harr, varr, 'bo')
    #        plt.show()
    #        plt.close()
            small_inv = invar[varr[0]:varr[-1]+1,harr[0]:harr[-1]+1]
#            hcut = int(np.mod(np.argmax(small_img),small_img.shape[1]))
#            vcut = int(np.floor(np.argmax(small_img)/small_img.shape[1]))
#            hmean, hheight, hbg = sf.fit_mn_hght_bg(harr,small_img[vcut,:],small_inv[vcut,:],sigmas[k],hcenters[k],sigmas[k],powj=powers[k])
#            vmean, vheight, vbg = sf.fit_mn_hght_bg(varr+voff,small_img[:,hcut],small_inv[:,hcut],sigmas[k],vcenters[k],sigmas[k],powj=powers[k])
#            hdec, hint = math.modf(hmean)
#            vdec, vint = math.modf(vmean)
            if itr == 0:
                params = lmfit.Parameters()
                vo = cpad#vmean-varr[0]
                ho = cpad#hmean-harr[0]
                params.add('vc', value = vo, min=vo-0.5, max=vo+0.5)
                params.add('hc', value = ho, min=ho-0.5, max=ho+0.5)
                params.add('q',value = 0.95, min=0)
                params.add('PA',value = 0)
                print "minimizing", k
            spline_fit, s_coeffs, s_scale = spline.spline_2D_radial(small_img,small_inv,r_breakpoints,params,theta_orders=theta_orders,return_coeffs=True)
            args = (small_img,small_inv,r_breakpoints)
            kws = dict()
            kws['theta_orders'] = theta_orders
            kws['spline_coeffs'] = s_coeffs
            kws['sscale'] = s_scale
            minimizer_results = lmfit.minimize(spline.spline_residuals,params,args=args,kws=kws)
            ecc = minimizer_results.params['q'].value
            pos_ang = minimizer_results.params['PA'].value
            if ecc > 1:
                ecc = 1/ecc
                pos_ang -= (np.pi/2)
            pos_ang = pos_ang % (2*np.pi)
            minimizer_results.params['q'].value = ecc
            minimizer_results.params['PA'].value = pos_ang
            params = minimizer_results.params
            print params['hc']
            spline_fit = spline.spline_2D_radial(small_img, small_inv, r_breakpoints, minimizer_results.params, theta_orders=[0], order=4, return_coeffs=False, spline_coeffs=s_coeffs.T, sscale=s_scale, fit_bg=False, pts_per_px=1) 
            itr += 1
            chi2 = np.sum((small_img-spline_fit)**2*small_inv)
            print chi2/(np.size(small_img)-20)
    #        print "Eccentricity = ", ecc
    #        print "Position Angle = ", pos_ang*180/(np.pi)
    #        center=[vmean-varr[0],hmean-harr[0]]
    #        print center
            
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
        new_hcenters[k] = params['hc'].value
        new_vcenters[k] = params['vc'].value
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
                   gain,plot_results=False,verbose=False):
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
        if verbose:
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
            trunc_coeffs = 1.0*next_coeffs
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
#            plt.imshow(np.hstack((data_for_fitting[:,:,k],tmp_fit)),interpolation='none')
            chi2_first[k] = np.sum(((tmp_fit-data_for_fitting[:,:,k])**2)*invar_for_fitting[:,:,k])#*s_scale[k]**2
        #    plt.imshow((tmp_fit-small_img/s_scale[k])*small_inv,interpolation='none')
#            plt.show()
#            plt.close()
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
                if verbose:
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
            coeff_matrix_min = 1.0*coeff_matrix
            params_min = lmfit.Parameters(params1)
        if np.sum(chi_new) < chi_min:
            if verbose:
                print "Better fit on loop ", loop_cnt
            chi_min = np.sum(chi_new)
            coeff_matrix_min = 1.0*coeff_matrix
            params_min = lmfit.Parameters(params1)
        loop_cnt += 1
    
    ### End of loop
    if verbose:
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
            plt.imshow(np.hstack((img_matrix, fitted_image, (img_matrix-fitted_image))),interpolation='none')
        #    plt.imshow((img_matrix-fitted_image)*invar_matrix,interpolation='none')
            plt.show()
            plt.close()
            
    centers, ellipse = params_to_array(params_min)
    results = np.hstack((np.ravel(coeff_matrix_min),np.ravel(ellipse)))    
    return results

def fixed_psf(hcent, vcent, cpad, redux_dir, ratio=0.7):
    """ makes a fixed PSF with GHL in variable GH/(GHL) ratios
        Use for simulation/testing only
    """
    psf_hdu = pyfits.open(os.path.join(redux_dir,'psf','ghl_psf_coeffs_{ts}_{fnum:03d}.fits'.format(ts='T1',fnum=10)))
    weights = psf_hdu[0].data.ravel()
    lorentz_fits = psf_hdu[1].data
    s_coeffs = psf_hdu[2].data
    p_coeffs = psf_hdu[3].data
    hcenters_dummy = psf_hdu[4].data
    vcenters_dummy = psf_hdu[5].data
    pord = len(s_coeffs)-1
    params = init_params(hcenters_dummy, vcenters_dummy, s_coeffs, p_coeffs, pord, r_guess=lorentz_fits[:,1], s_guess = lorentz_fits[:,0])
    params1 = convert_params(params, 0, pord, hscale=0)
    params1['xc'].value = hcent
    params1['yc'].value = vcent
    params1['bg'].value = 0
    params1['ratio'].value = ratio
    icent = np.array(([hcent],[vcent]))
    gh_model = get_ghl_profile(icent, params1, weights, pord, cpad, return_model=True, no_convert=True)[0]
    params = lmfit.Parameters()
    params.add('xc', value=hcent)
    params.add('yc', value=hcent)
    params.add('sigl', value=2)
    params.add('ratio', value=params1['ratio'].value)
    params.add('q', value=0.7)
    params.add('PA', value=np.pi/4)
    center_ref = icent
    lorentz = sf.lorentz_ellipse(center_ref, cpad, params)
    psf_fixed = gh_model+lorentz
    psf_fixed /= np.sum(psf_fixed)
    return psf_fixed
    
def get_fitting_arrays(arc, hcenters, vcenters, pord, cpad, readnoise, scale_arr=None, return_scale_arr=False):
    data = np.zeros((len(hcenters),2*cpad+1,2*cpad+1))
    invar = np.zeros((len(hcenters),2*cpad+1,2*cpad+1))
    if return_scale_arr:
        new_scale = np.zeros((len(hcenters)))
    for i in range(len(hcenters)):
        vc = int(np.round(vcenters[i]))
        hc = int(np.round(hcenters[i]))
        data[i] = arc[vc-cpad:vc+cpad+1, hc-cpad:hc+cpad+1]
        if scale_arr is None:
            scl = np.sum(data[i])
            if return_scale_arr:
                new_scale[i] = scl
        else:
            scl = scale_arr[i]#np.sum(data[i])/scale_arr[i]
        data[i] /= scl
        invar[i] = scl/(abs(data[i]) + readnoise**2/scl)
    if return_scale_arr:
        return data, invar, new_scale
    else:
        return data, invar
    
def convert_params(params, idx, pord, hscale=None):
    params_out = lmfit.Parameters()
    if hscale is None:
        hscale = (params['xc{}'.format(idx)].value-1024)/2048
    params_out.add('xc', value = params['xc{}'.format(idx)].value)#, min=params['xc{}'.format(idx)].value-0.5, max=params['xc{}'.format(idx)].value+0.5)
    params_out.add('yc', value = params['yc{}'.format(idx)].value)#, min=params['yc{}'.format(idx)].value-0.5, max=params['yc{}'.format(idx)].value+0.5)
    params_out.add('bg', value = params['bg{}'.format(idx)].value)
    ### Unpack polynomial fitted parameters
    sigs = np.zeros((pord+1))
    pows = np.zeros((pord+1))
    sigls = np.zeros((pord+1))
    rats = np.zeros((pord+1))
    for o in range(pord+1):
        sigs[o] = params['sigma{}'.format(o)].value
        pows[o] = params['power{}'.format(o)].value
        sigls[o] = params['sigl{}'.format(o)].value
        rats[o] = params['ratio{}'.format(o)].value
    params_out.add('sig', value = np.poly1d(sigs)(hscale), vary=0)
    params_out.add('power', value = np.poly1d(pows)(hscale), vary=0)
    params_out.add('sigl', value = np.poly1d(sigls)(hscale), min=params_out['sig'].value/2, max=params_out['sig'].value*10)
    params_out.add('ratio', value = np.poly1d(rats)(hscale), min=0, max=1)
    return params_out

def init_params(hcenters, vcenters, s_coeffs, p_coeffs, pord, r_guess=None, s_guess=None, bg_arr=None):
    params_in = lmfit.Parameters()
    if len(hcenters) == 0:
        hcenters, vcenters = [hcenters], [vcenters]
    for i in range(len(hcenters)):
        params_in.add('xc{}'.format(i), value = hcenters[i])#, min=hcenters[i]-1.0, max=hcenters[i]+1.0)
        params_in.add('yc{}'.format(i), value = vcenters[i])#, min=vcenters[i]-1.0, max=vcenters[i]+1.0)
        if bg_arr is None:
            params_in.add('bg{}'.format(i), value = 1e-6, min=0)
        else:
            params_in.add('bg{}'.format(i), value = bg_arr[i])
    if r_guess is None:
        r_guess = [0, 0, 1]#fix at 10.95]
    if s_guess is None:
        s_guess = [0, 0, 2*s_coeffs[-1]]
    for o in range(pord+1):
        params_in.add('sigma{}'.format(o), value=s_coeffs[o], vary=0)
        params_in.add('power{}'.format(o), value=p_coeffs[o], vary=0)
        ### For now, fix lorentz at zero (ratio=1)
        params_in.add('sigl{}'.format(o), value=s_guess[o], vary=0)
        params_in.add('ratio{}'.format(o), value=r_guess[o], vary=0)
    return params_in
    
def get_ghl_profile(icenters, params, weights, pord, cpad, return_model=False, no_convert=False, force_norm=False):
    ncenters = len(icenters[0])
    ghl_profile = np.zeros(((2*cpad+1)**2*ncenters,weights.size))
    hscale = (icenters[0]-1024)/2048
    ws = 12
    for i in range(ncenters):
        hc = int(np.round(icenters[0][i]))
        vc = int(np.round(icenters[1][i]))
        xarr = np.arange(hc-cpad,hc+cpad+1)
        yarr = np.arange(vc-cpad,vc+cpad+1)
        ### Assign nonlinear parameters
        if no_convert:
            hc = int(icenters[0][i])
            vc = int(icenters[1][i])
            xarr = np.arange(hc-cpad,hc+cpad+1)
            yarr = np.arange(vc-cpad,vc+cpad+1)
            params1 = params
        else:
            params1 = convert_params(params, i, pord)
        sub_weights = update_weights_arr(weights, hscale[i], pord)
        profile = params1['ratio'].value*sf.gauss_herm2d(xarr, yarr, params1, sub_weights, return_profile=True, force_norm=force_norm)
        for p in range(pord+1):
            ghl_profile[(2*cpad+1)**2*i:(2*cpad+1)**2*(i+1),ws*p:ws*(p+1)] = profile*(hscale[i]**(pord-p))
    if return_model:
        return np.dot(ghl_profile,weights).reshape(ncenters,(2*cpad+1),(2*cpad+1))
    else:
        return ghl_profile
    
def get_data_minus_lorentz(data, icenters, params, weights, pord, cpad, return_lorentz=False, no_convert=False):
    """ data is from get_arrays_for_fitting"""
    ncenters = len(icenters[0])
    lorentz_arr = np.zeros((data.shape))
    data_minus_lorentz = 1.0*data
    for i in range(ncenters):
        if not no_convert:
            params1 = convert_params(params, i, pord)
#            if i == 0:
#                print params1, weights[i]
            icenter = (icenters[0][i], icenters[1][i])
            lorentz = sf.lorentz_for_ghl(data[i], icenter, params1, 1, cpad)
            data_minus_lorentz[i] -= (lorentz+params1['bg'].value)
            lorentz_arr[i] = lorentz
        else:
            params1 = params
            icenter = (icenters[0][i], icenters[1][i])
            lorentz = sf.lorentz_for_ghl(data, icenter, params1, 1, cpad)
            data_minus_lorentz -= (lorentz+params1['bg'].value)
            lorentz_arr = lorentz
    if return_lorentz:
        return lorentz_arr
    else:
        return data_minus_lorentz

def ghl_residuals(params, arc, readnoise, scale_arr, icenters, weights, pord, cpad):
#    try:
#        itr += 1
#    except:
#        global itr
#        itr = 0
#    ta = time.time()
#    hcenters, vcenters = update_centers(params, ncenters)
    data, invar = get_fitting_arrays(arc, icenters[0], icenters[1], pord, cpad, readnoise, scale_arr=scale_arr)
#    tb = time.time()
#    print "ITERATION", itr
#    print "get_fitting_arrays time=", tb-ta
    gh_model = get_ghl_profile(icenters, params, weights, pord, cpad, return_model=True, force_norm=True)
#    tc = time.time()
#    print "get_ghl_profile time =", tc-tb
    data_lorentz = get_data_minus_lorentz(data, icenters, params, weights, pord, cpad)
#    td = time.time()
#    print "get_data_minus_lorentz time =", td-tc, "\n"
#    print np.sum(np.ravel((gh_model-data_lorentz)**2*invar))
    return np.ravel((data_lorentz-gh_model)**2*invar)
    
def update_centers(params, cnt):
    hcenters = np.zeros((cnt))
    vcenters = np.zeros((cnt))
    for i in range(cnt):
        hcenters[i] = params['xc{}'.format(i)].value
        vcenters[i] = params['yc{}'.format(i)].value
    return hcenters, vcenters

#def update_scale_arr(scale_old, params, weights, pord, cpad):
#    scale_arr = np.ones((len(scale_old)))
#    hcenters, vcenters = update_centers(params, len(scale_old))
#    for i in range(len(scale_arr)):
#        params1 = convert_params(params, i, pord)
#        hscale = (hcenters[i]-1024)/2048
#        weightsi = update_weights_arr(weights, hscale, pord)
#        hc = int(np.round(hcenters[i]))
#        vc = int(np.round(vcenters[i]))
#        xarr = np.arange(hc-cpad,hc+cpad+1)
#        yarr = np.arange(vc-cpad,vc+cpad+1)
#        scale_arr[i] = sf.gauss_herm2d(xarr, yarr, params1, weightsi, return_norm=True)
#    scale_arr = scale_old*scale_arr
##    print scale_old[0], scale_arr[0]
#    return scale_arr
    
def update_weights_arr(weights, hscale, pord, ypix=2048):
    """ Finds weights at a particular point from polynomial fit array
    """
    weightsi = np.zeros(12)
    ows = np.reshape(weights, (pord+1, 12)).T
    for k in range(0,12):
        weightsi[k] = np.poly1d(ows[k])(hscale)
    return weightsi
        
def fit_ghl_psf(arc, hcenters, vcenters, s_coeffs, p_coeffs, readnoise, gain, pord=2, cpad=4, plot_results=False, verbose=False, return_centers=False):
    """ Fits the PSF for the raw image (should be an arc frame or similar) to
        2D Gauss-Hermite polynomials with a Lorentzian term.
        Interpolates a polynomial along each order.
        Takes inputs for arc centers to evaluate, best fit sigma and power
        for Gaussian envelope, CCD gain and readnoise, and polynomial order
        (2 = quadratic), and image padding from h/vcenter
    """
    ### Initialize arrays to house fitted parameters
    ncenters = len(hcenters)
    other_weights = np.zeros((12, pord+1)) ### GH weights
    all_params = np.zeros((pord+1,2)) ### Lorentz nonlinear parameters
    ### Initialize while loop - iterate between nonlinear, linear fits until convergence
    chi2 = 0
    chi2r = 0
    chi2old = 1
    chi2_min = 9999999
    mx_iter = 200
    itr = 0
    params = init_params(hcenters, vcenters, s_coeffs, p_coeffs, pord)
    data, invar, scale_arr = get_fitting_arrays(arc, hcenters, vcenters, pord, cpad, readnoise, return_scale_arr=True)
    ### fix initial centers in a new variable
    ihcenters, ivcenters = 1.0*hcenters, 1.0*vcenters
    icenters = np.vstack((ihcenters, ivcenters))
    weights = np.zeros(((pord+1)*12,1))
    weights[12*2] = 1 ### initially set all to a pure Gaussian
    t0 = time.time()
    kws = dict()
    nvars = 3*len(hcenters)+6 #+6 for lorentz components
    kws['maxfev'] = 1*(nvars+1)
    while abs(chi2-chi2old) > 0.001 and itr < mx_iter:
        if verbose:
            print "Iteration {}".format(itr)
#            print "  Delta chi^2:", abs(chi2-chi2old)
        chi2old = 1.0*chi2
        args = (arc, readnoise, scale_arr, icenters, weights, pord, cpad)
        ta = time.time()
        results = lmfit.minimize(ghl_residuals, params, args=args, **kws)
        if verbose:
            tb = time.time()
            print "  Nonlinear time = {}".format(tb-ta)
        params = results.params
        data, invar = get_fitting_arrays(arc, ihcenters, ivcenters, pord, cpad, readnoise, scale_arr=scale_arr)
        data_lorentz = get_data_minus_lorentz(data, icenters, params, weights, pord, cpad)
        profile = get_ghl_profile(icenters, params, weights, pord, cpad, force_norm=True)
        model = np.dot(profile, weights)
        weights, chi2 = sf.chi_fit(np.ravel(data_lorentz),profile,np.ravel(invar))
        profile_norm = get_ghl_profile(icenters, params, weights, pord, cpad, force_norm=True)
        model = np.dot(profile_norm, weights)
        model = model.reshape((len(hcenters),(2*cpad+1),(2*cpad+1)))
        lorentzs = get_data_minus_lorentz(data, icenters, params, weights, pord, cpad, return_lorentz=True)
        model += lorentzs
        datans, invarns = get_fitting_arrays(arc, ihcenters, ivcenters, pord, cpad, readnoise, scale_arr=np.ones(hcenters.shape))
        mprof = np.zeros(((2*cpad+1)**2*len(hcenters),len(hcenters)))
        for k in range(len(hcenters)):
            ind1 = (2*cpad+1)**2*k
            ind2 = (2*cpad+1)**2*(k+1)
            mprof[ind1:ind2,k] = np.ravel(model[k])
        scale_arr, jnkchi2 = sf.chi_fit(np.ravel(datans), mprof, np.ravel(invarns))
        if verbose:
            tc = time.time()
            print "  Linear time = {}".format(tc-tb)
            print "  chi^2 new = {}".format(chi2)
        if chi2 < chi2_min and chi2 > 0:
            chi2_min = chi2
            other_weights = weights.reshape((12,pord+1))
            for o in range(pord+1):
                all_params[o,0] = params['sigl{}'.format(o)].value
                all_params[o,1] = params['ratio{}'.format(o)].value
            chi2r = chi2_min/(data.size-3*len(hcenters)-6-12*(pord+1))
            print "  Reduced chi2", chi2r
        itr += 1
    tf = time.time()
    if verbose:
        print "Total fitting time = {}s".format(tf-t0)
    hcenters, vcenters = update_centers(params, ncenters)
    if plot_results:
#        hscale = (hcenters-1024)/2048
#        print "Plotting weight fits"
#        for j in range(other_weights.shape[0]):            
#            wfit = np.poly1d(other_weights[j])(hscale)
#            plt.plot(hcenters, wfit)
#            plt.show()
#            plt.close()
#        print "Plotting sigl, ratio"
#        slfit = np.poly1d(all_params[:,0])(hscale)
#        rfit = np.poly1d(all_params[:,1])(hscale)
#        plt.plot(hcenters, slfit)
#        plt.show()
#        plt.close()
#        plt.plot(hcenters, rfit)
#        plt.show()
#        plt.close()
#        print data
#        print hcenters, vcenters
#        print params
#        print weights
#        print pord
#        print cpad
        lorentzs = get_data_minus_lorentz(data, icenters, params, weights, pord, cpad, return_lorentz=True)
        for i in range(ncenters):
            gh_model = get_ghl_profile(icenters, params, weights, pord, cpad, return_model=True, force_norm=True)[i]
            model = gh_model + lorentzs[i] + params['bg{}'.format(i)].value
            plt.imshow(np.hstack((data[i], model, data[i]-model)), interpolation='none')
            ax = plt.gca()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
#            plt.savefig('/home/matt/Documents/Research/Thesis/latex/mattthesis/thar_psf.pdf', bbox_inches='tight')
#            plt.show()
#            plt.close()
    if return_centers:
        return other_weights, all_params, hcenters, vcenters, chi2r
    else:
        return other_weights, all_params, chi2r
  
#########################################################################
################ Functions for fitting 1D PSF ###########################
#########################################################################
def get_tc_box(ccd,t_coeffs,trace,width,h0,h1):
    """
    Cips a section of CCD for analysis
    Returns a box of +/- 2*width around the center of trace.
    Box is slanted in native CCD pixel coordinates based on trace curvature.
    Limits to horizonal coordinates between h0 and h1.
    """
    hpix = ccd.shape[1]
    hparr = np.arange(h0,h1)
    harr = (hparr - hpix/2)/hpix
    vtr = np.poly1d(t_coeffs[:,trace])(harr)
    if trace != 0:
        vtr1 = np.poly1d(t_coeffs[:,trace-1])(harr)
    vtr2 = np.poly1d(t_coeffs[:,trace+1])(harr)
    v0 = int(np.round(vtr[0]))
    if (h1-h0) < 20:
        voffset = (v0 - int(np.round(vtr[int((h1-h0)/2)])))*np.ones((h1-h0))
    else:
        voffset = np.round(v0-vtr).astype(int)
    ### Restrict range for traces that are extremely close together
    from minerva_utils import get_xlims
    vl0, vh0 = get_xlims(trace)
    if trace == 0:
        vl1 = -8
    else:
        vl1, vh1 = get_xlims(trace-1)
    vl2, vh2 = get_xlims(trace+1)
    low_off = 8+vl0+8+vl1
    high_off = 8-vh0+8-vh2
    data = np.zeros((4*width + 1 - low_off - high_off,h1-h0))
    for i in range(h1-h0):
        vli = max(0,v0-voffset[i]-2*width+low_off)
        vhi = min(ccd.shape[0],v0-voffset[i]+2*width+1-high_off)
        data[:,i] = ccd[vli:vhi,hparr[i]]
    ### Get coordinates for the 3 (or two if trace==0) traces in the 
    ### modified data coordinates
    vl_arr = v0-voffset-2*width+low_off
    vl_arr[vl_arr<0] = 0
    mod_traces = np.zeros((3,len(hparr)))
    ### Set off-box traces to -1
    mod_traces[0] = vtr1-vl_arr
    if trace == 0:
        mod_traces[0] = -1*np.ones(vtr1.shape)
    mod_traces[1] = vtr-vl_arr
    mod_traces[2] = vtr2-vl_arr
    ### Re-write with constant mod_trace
#    mod_traces[0] = vtr1[int((h1-h0)/2)]-vl_arr[int((h1-h0)/2)]
#    mod_traces[1] = vtr[int((h1-h0)/2)]-vl_arr[int((h1-h0)/2)]
#    mod_traces[2] = vtr2[int((h1-h0)/2)]-vl_arr[int((h1-h0)/2)]
    mod_traces -= 0.5 ### Best with half pixel shift, but I'm not sure why...
    m1msk = (mod_traces[0] < 0)
    mod_traces[0][m1msk] = -100
    m2msk = (mod_traces[2] >= data.shape[0])
    mod_traces[2][m2msk] = -100
    return data, mod_traces

def get_1D_box_params0(data, width, num_poly, mod_traces, linfit=False, bgconst=False, fittrace=False):
    """
    Gets initial parameter estimates
    """
    params = lmfit.Parameters()
    hc = int(data.shape[1]/2)
    if mod_traces.shape[0] != 3:
        print "Need to modify psf_utils.get_1D_box_params0 to allow for traces on the edge of the ccd"
        exit(0)
    vc = int(np.round(mod_traces[1,hc]))
    ### Initial profile guess
    p0g = np.median(data[vc-width:vc+width+1,hc-2:hc+3], axis=1)
    p0g[np.arange(2*width+1) < width/2] = 0
    p0g[np.arange(2*width+1) > 3*width/2] = 0
    pinterp = interp1d(np.arange(2*width+1),p0g)
    pgrid = np.arange(2*width+1)-(mod_traces[1,hc]-vc)
    pmask = (pgrid >= 0) * (pgrid < 2*width)
    pgrid = pgrid[pmask]
    p0g[pmask] = pinterp(pgrid)
    p0g[pmask ==0] = 0
    p0g /= np.sum(p0g)
    phg = np.zeros(p0g.shape)
    ### Initial background and heights
    bgg = np.zeros((3,data.shape[1]))
    hgg = np.zeros((3,data.shape[1]))
    cgg = np.zeros((3,data.shape[1]))
    for i in range(data.shape[1]):
        for l in range(3):
            if mod_traces[l,i] == -100:
                bgg[l,i] = 0
                hgg[l,i] = 0
                cgg[l,i] = 0
            else:
                idx1 = max(0,int(np.round(mod_traces[l,i])-max(5,width/2)))
                idx2 = min(data.shape[0],int(np.round(mod_traces[l,i])+max(5,width/2)))
                if idx1 >= data.shape[0]:
                    bgg[l,i] = 0
                    hgg[l,i] = 1
                    cgg[l,i] = idx1
                else:
                    bgg[l,i] = min(data[idx1,i],data[idx2,i])
                    hgg[l,i] = np.sum(data[idx1:idx2,i] - bgg[l,i])
                    cgg[l,i] = np.argmax(data[idx1:idx2,i]) + idx1
    ### Put profile guesses into lmfit.Parameters object
    for m in range(2*width+1):
        for n in range(num_poly):
            if n == 0:
                params.add('p{}_{}'.format(n,m), value=p0g[m], min=0)
            else:
                params.add('p{}_{}'.format(n,m), value=phg[m])
    ### Put backgrounds and heights into lmfit.Parameters object
    ### If using linear fitting, heights and bgs don't vary
    for k in range(data.shape[1]):
        for j in range(3):
            if not bgconst:
                params.add('bg_{}_{}'.format(j,k), value = bgg[j,k], vary=((linfit==False)or(mod_traces[j,k] == -100)))
            params.add('hght_{}_{}'.format(j,k), value = hgg[j,k], vary=((linfit==False) or(mod_traces[j,k] == -100)))
            if fittrace:
                params.add('tr_{}_{}'.format(j,k), value = cgg[j,k], min=cgg[j,k]-1.5, max=cgg[j,k]+1.5, vary=(linfit==False))
    if bgconst:
        params.add('bg', value=np.median(bgg), vary=(linfit==False))
    return params

def params_to_pix_arr(params, width, num_poly):
    parrs = np.zeros((num_poly, 2*width+1))
    for i in range(num_poly):
        for j in range(2*width+1):
            try:
                parrs[i,j] = params['p{}_{}'.format(i,j)].value
            except:
                print params
    return parrs
    
def pix_arr_to_prof_arr(parrs, length, num_poly, norm=False):
    prof_arr = np.zeros((parrs.shape[1], length))
    for k in range(length):
        prof_k = np.zeros((parrs.shape[1]))
        for j in range(num_poly):
            prof_k += parrs[j]*(j-int(length/2))**j
        if norm:
            prof_k /= np.sum(prof_k)
        prof_arr[:,k] = prof_k
    return prof_arr
    
def update_h_bg(params, h_bg_arr, length, bgconst=False):
    for i in range(length):
        for j in range(3): #3 traces in each fitting box
            params['hght_{}_{}'.format(j,i)].value = h_bg_arr[(2-bgconst)*j+(6-3*bgconst)*i]
            if not bgconst:
                params['bg_{}_{}'.format(j,i)].value = h_bg_arr[2*j+1+6*i]
    if bgconst:
        params['bg'].value = h_bg_arr[-1]
    return params

def make_profile(params, data, mod_traces, num_poly, width, bgconst=False):
    Prof = np.zeros((data.shape[0]*data.shape[1],(6-3*bgconst)*data.shape[1]+bgconst))
    hc = int(data.shape[1]/2)
    vparr = np.arange(2*width+1)
    Parr = np.zeros((2*width+1,data.shape[1]))
    parrs = params_to_pix_arr(params, width, num_poly)
    ### Get spline interpolation of profile
    splf = dict()
    for i in range(data.shape[1]):
        Ptmp = np.zeros((2*width+1))
        for j in range(num_poly):
            Ptmp += parrs[j]*(i-hc)**j
        Parr[:,i] = Ptmp
        splf[i] = interp1d(vparr, Ptmp)
#    paramsn = lmfit.Parameters()
#    paramsn = params
#    for k in range(data.shape[1]):
#        for j in range(3):
#            paramsn['bg_{}_{}'.format(j,k)].value = 0
#            paramsn['hght_{}_{}'.format(j,k)].value = 1
#            print params['hght_{}_{}'.format(j,k)].value
#    for i in range(data.shape[1]):
#        Pi = box_model_1D(paramsn, data, mod_traces, num_poly, width):
#        Prof[data.shape[0]*i:data.shape[0]*(i+1),6] = Pi
    for j in range(data.shape[1]):
        for k in range(mod_traces.shape[0]):
            vc = mod_traces[k,j]
            vint = int(np.round(vc))
            vl = max(0, vint-width)
            vh = min(data.shape[0], vint+width+1)
            sparr = vparr-(vc-vint)
            sparr = sparr[vl-vint+width:vh-vint+width]
            if len(sparr) == 0:
                continue
            if sparr[0] < 0:
                sparr = sparr[1:]
                vl += 1
            if sparr[-1] > 2*width:
                sparr = sparr[:-1]
                vh -= 1
            spinterp = splf[j](sparr)
#            spinterp -= np.min(spinterp) #Set minimum to zero
            spinterp /= np.sum(spinterp) #Must be normalized
            Prof[vl+j*data.shape[0]:vh+j*data.shape[0],(2-bgconst)*k+(6-3*bgconst)*j] = spinterp
            if not bgconst:
                Prof[vl+j*data.shape[0]:vh+j*data.shape[0],(2-bgconst)*k+1+(6-3*bgconst)*j] = np.ones(spinterp.shape)
    if bgconst:
        Prof[:,-1] = np.ones(Prof.shape[0])
    return Prof

def box_model_1D(params, data, mod_traces, num_poly, width, bgconst=False, fittrace=False, verbose=False):
    """
    Builds residuals for lmfit.minimize
    Takes a profile and the profile slope, builds the profile at each cross-
    section.  Then using scipy spline interpolation, shift this to match the
    trace centroid.
    This profile is used for adjacent traces as well which are each multiplied
    by a height and added to a background level for the model
    """
    model = np.zeros(data.shape)
    hc = int(data.shape[1]/2)
    vparr = np.arange(2*width+1)
    Parr = np.zeros((2*width+1,data.shape[1]))
    parrs = params_to_pix_arr(params, width, num_poly)
    ### Get spline interpolation of profile
    splf = dict()
    for i in range(data.shape[1]):
        Ptmp = np.zeros((2*width+1))
        for j in range(num_poly):
            Ptmp += parrs[j]*(i-hc)**j
        Parr[:,i] = Ptmp
        splf[i] = interp1d(vparr, Ptmp)
    ### Using heights, background, trace centers, and interpolated
    ### profile, build the model
    for j in range(data.shape[1]):
        for k in range(mod_traces.shape[0]):
            if fittrace:
                vc = params['tr_{}_{}'.format(k,j)].value
            else:
                vc = mod_traces[k,j]
                ### Don't build profile for off-box traces
                if vc == -100:
                    pass
            vint = int(np.round(vc))
            if vint >= data.shape[0] or vint < 0:
                pass
            vl = max(0, vint-width)
            vh = min(data.shape[0], vint+width+1)         
            sparr = vparr-(vc-vint)
            sparr = sparr[vl-vint+width:vh-vint+width]
            if len(sparr) == 0:
                continue
            if sparr[0] < 0:
                sparr = sparr[1:]
                vl += 1
            if sparr[-1] > 2*width:
                sparr = sparr[:-1]
                vh -= 1
            spinterp = splf[j](sparr)
#            spinterp -= np.min(spinterp) #Set minimum to zero
            spinterp /= np.sum(spinterp) #Must be normalized
            if bgconst:
                model[vl:vh,j] += params['hght_{}_{}'.format(k,j)].value*spinterp
            else:
                model[vl:vh,j] += params['hght_{}_{}'.format(k,j)].value*spinterp + params['bg_{}_{}'.format(k,j)].value
    if bgconst:
        model += params['bg'].value
#        print mod_traces[:,j]
#        sf.plt_deltas(mod_traces[:,j], np.ones((3))*np.max(model[:,j]))
#        plt.plot(model[:,j])
#        plt.show()
#        plt.close()
    return model

def results_to_pix_arr(results, width, num_poly):
    parrs = np.zeros((num_poly))
    return parrs

def box_res_1D(params, data, invar, mod_traces, num_poly, width, bgconst=False, fittrace=False):
    model = box_model_1D(params, data, mod_traces, num_poly, width, bgconst=bgconst, fittrace=fittrace)
#    for j in range(data.shape[1]):
#        print mod_traces[:,j]
#        plt.plot(data[:,j],'k',linewidth=2)
#        plt.plot(model[:,j],'g',linewidth=2)
#        plt.show()
#        plt.close()
    big_res = (data-model)*np.sqrt(invar)
    ### Only fit the central trace, just use the others to model cross-talk
    vc = mod_traces[1,int(data.shape[1]/2)]
    vint = int(np.round(vc))
    vl = max(0, vint-width-5)
    vh = min(data.shape[0], vint+width+6)
    small_res = big_res[vl:vh,:]
#    small_res = big_res
    ### Enforce a positive profile (with tolerance for a little below zero)
#    parrs = params_to_pix_arr(params, width, num_poly)
#    prof_arr = pix_arr_to_prof_arr(parrs, data.shape[1], num_poly)
#    if np.min(prof_arr) < -0.001:
#        small_res = abs(small_res)*1e300
    return np.ravel(small_res)
    
def lin_nonlin_1D(params, data, invar, mod_traces, num_poly, width, bgconst=False, plot=False):
    """
    Iterates between linear and non-linear fitting for parameters in box_res
    """
    delta_chi_min = 1e-2 ## Set minimum threshold for subsequent delta chi^2
    delta_chi = 10
    chi_min = 1e10
    chi = 1e10
    max_iters = 24
    cnt = 0
    best_params = lmfit.Parameters()
    while (delta_chi > delta_chi_min) and (cnt < max_iters):
        print "starting iter {}".format(cnt)
        chi_old = 1.0*chi
        ### Do nonlinear fit (for profile shapes)
        fit_args = (data,invar, mod_traces, num_poly, width)
        kws = {'bgconst':bgconst}
        results = lmfit.minimize(box_res_1D,params,args=fit_args, kws=kws)
        params = results.params
        model = box_model_1D(params, data, mod_traces, num_poly, width, bgconst=bgconst, verbose=True)
        big_res = (data-model)**2*invar
        vc = mod_traces[1,int(data.shape[1]/2)]
        vint = int(np.round(vc))
        vl = max(0, vint-width-2)
        vh = min(data.shape[0], vint+width+3)
        small_res = big_res[vl:vh,:]
        chi = np.sum(small_res)
#        print "  Chi^2 reduced:", chi/np.size(small_res)
#        print "  Chi^2 big:", np.sum(big_res)
        if plot:
            print "After nonlinear fit"
            for i in range(1):#data.shape[1]):
                plt.plot(data[width:3*width,i], 'b', linewidth=2)
                plt.plot(model[width:3*width,i], 'k', linewidth=2)
            plt.show()
            plt.close()
        ### Do linear fit (for height and background)
#        h_bg_arr = h_bg_from_params(params, data.shape[1], bgconst=bgconst)
        Profile = make_profile(params, data, mod_traces, num_poly, width, bgconst=bgconst)
        hbga = np.zeros((3*data.shape[1]+1))
        for i in range(data.shape[1]):
            for j in range(3):
                hbga[j + 3*i] = params['hght_{}_{}'.format(j,i)].value
        hbga[-1] = params['bg'].value
        mprof = np.dot(Profile, hbga)
#        mprof = np.resize(mprof, (1, len(mprof)))
        mprof = mprof.reshape((data.shape[1],data.shape[0])).T
        chi2 = np.sum((mprof-data)**2*invar)
        h_bg_new, junk_chi = sf.chi_fit(np.ravel(data.T),Profile,np.diag(np.ravel(invar.T)))
        mprof = np.dot(Profile, h_bg_new)
#        mprof = np.resize(mprof, (1, len(mprof)))
        mprof = mprof.reshape((data.shape[1],data.shape[0])).T
        params = update_h_bg(params, h_bg_new, data.shape[1], bgconst=bgconst)
        model = box_model_1D(params, data, mod_traces, num_poly, width, bgconst=bgconst)
        big_res = (data-model)**2*invar
        big_res2 = (data-mprof)**2*invar
        vc = mod_traces[1,int(data.shape[1]/2)]
        vint = int(np.round(vc))
        vl = max(0, vint-width-2)
        vh = min(data.shape[0], vint+width+3)
        small_res = big_res[vl:vh,:]
        chi = np.sum(big_res)
        print "  Chi^2 reduced:", chi/np.size(small_res)
#        print "  Chi^2 big:", np.sum(big_res), np.sum(big_res2)
        if plot:
            print "After linear fit:"
            for i in range(1):#data.shape[1]):
                plt.plot(data[width:3*width,i], 'b', linewidth=2)
                plt.plot(model[width:3*width,i], 'k', linewidth=2)
#                plt.plot(((data-model)*np.sqrt(invar))[12:28,i], 'k', linewidth=2)
            plt.show()
            plt.close()
            plt.imshow(np.hstack((data, model, data-model)), interpolation='none')
            plt.show()
            plt.close()
            plt.imshow(small_res, interpolation='none')
            plt.show()
            plt.close()
        delta_chi = chi_old-chi
        if chi < chi_min:
            chi_min = chi
            best_params = params
        cnt += 1
    return best_params
    
def interp_res(params, spinterp, sparr, data, invar):
    model = params['hght'].value*spinterp(sparr-params['cent'].value) + params['bg'].value
    return (data-model)*np.sqrt(invar)
    
def find_trace(spinterp, sparr, data, invar):
    params = lmfit.Parameters()
    params.add('cent', value = 0, min=-1.5, max=1.5)
    params.add('hght', value = np.max(data))
    params.add('bg', value = 0)
    args = (spinterp, sparr, data, invar)
    results = lmfit.minimize(interp_res, params, args=args)
    model = results.params['hght'].value*spinterp(sparr-results.params['cent'].value) + results.params['bg'].value
    chi2 = np.sum(((data-model)*np.sqrt(invar))**2)
    print chi2/(len(sparr)-2)
    return results.params['cent'].value
        
    
def update_mod_traces(data, invar, params, mod_traces, width, num_poly):
    new_traces = np.zeros((mod_traces.shape))
    hc = int(data.shape[1]/2)
    parrs = params_to_pix_arr(params, width, num_poly)
    for i in range(data.shape[1]):
        for j in range(3):
            if mod_traces[j,i] == -100:
                new_traces[j,i] = -100
                continue
            Ptmp = np.zeros((2*width+1))
            for k in range(num_poly):
                Ptmp += parrs[k]*(i-hc)**k
            pad = 7
            if width < pad:
                print "Width too small to re-fit traces"
                exit(0)
            else:
                sparr = np.arange(-width, width+1)
                spinterp = interp1d(sparr, Ptmp)
                sparr = np.arange(-pad,pad, dtype=int)
                vc = int(mod_traces[j,i])
                spm = ((sparr+vc)>=0)*((sparr+vc)<data.shape[0])
                sparr = sparr[spm]
                spdat = data[sparr+vc, i]
                spinv = invar[sparr+vc, i]
                new_traces[j,i] = vc+find_trace(spinterp, sparr, spdat, spinv)
#    plt.imshow(data)
#    for i in range(3):
#        if new_traces[i,0] != -100:
#            plt.plot(new_traces[i], 'b', linewidth = 2)
#    plt.show()
#    plt.close()
    return new_traces
            