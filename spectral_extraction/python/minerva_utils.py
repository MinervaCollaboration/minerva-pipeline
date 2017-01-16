#!/usr/bin/env python

#Special functions for MINERVA data reduction

#Import all of the necessary packages
from __future__ import division
import pyfits
import os
import math
import time
import sys
import glob
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
#import scipy.linalg as linalg
#import solar
import special as sf
import bsplines as spline
#import argparse
import lmfit

def open_minerva_fits(fits, ext=0, return_hdr=False):
    """ Converts from kiwispec format (raveled array of 2 8bit images) to
        analysis format (2D array of 16bit int, converted to float)
    """
    spectrum = pyfits.open(fits,uint=True)
    hdr = spectrum[ext].header
    ccd = spectrum[ext].data
    #Dimensions
    ypix = hdr['NAXIS1']
    xpix = hdr['NAXIS2']
    
    actypix = 2048 ### Hardcoded to remove overscan, fix later if needed
    
    if np.shape(ccd)[0] > ypix:
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
#                ccd_16bit[row,col] = int(''.join(ccd_tmp[row,col]),base=2)
    
        ccd = ccd_16bit[::-1,0:actypix] #Remove overscan region
    else:
        ccd = ccd[::-1,0:actypix] #Remove overscan region
        ccd = ccd.astype(np.float)
    if return_hdr:
        return ccd, hdr
    else:
        return ccd

def fit_trace(x,y,ccd,form='gaussian'):
    """quadratic fit (in x) to trace around x,y in ccd
       x,y are integer pixel values
       input "form" can be set to quadratic or gaussian
    """
    x = int(x)
    y = int(y)
    maxx = np.shape(ccd)[0]
    if form=='quadratic':
        xpad = 2
        xvals = np.arange(-xpad,xpad+1)
        def make_chi_profile(x,y,ccd):
            xpad = 2
            xvals = np.arange(-xpad,xpad+1)
            xwindow = x+xvals
            xvals = xvals[(xwindow>=0)*(xwindow<maxx)]
            zvals = ccd[x+xvals,y]
            profile = np.ones((2*xpad+1,3)) #Quadratic fit
            profile[:,1] = xvals
            profile[:,2] = xvals**2
            noise = np.diag((1/zvals))
            return zvals, profile, noise
        zvals, profile, noise = make_chi_profile(x,y,ccd)
        coeffs, chi = sf.chi_fit(zvals,profile,noise)
    #    print x
    #    print xvals
    #    print x+xvals
    #    print zvals
    #    plt.errorbar(x+xvals,zvals,yerr=sqrt(zvals))
    #    plt.plot(x+xvals,coeffs[2]*xvals**2+coeffs[1]*xvals+coeffs[0])
    #    plt.show()
        chi_max = 100
        if chi>chi_max:
            #print("bad fit, chi^2 = {}".format(chi))
            #try adacent x
            xl = x-1
            xr = x+1
            zl, pl, nl = make_chi_profile(xl,y,ccd)
            zr, pr, nr = make_chi_profile(xr,y,ccd)
            cl, chil = sf.chi_fit(zl,pl,nl)
            cr, chir = sf.chi_fit(zr,pr,nr)
            if chil<chi and chil<chir:
    #            plt.errorbar(xvals-1,zl,yerr=sqrt(zl))
    #            plt.plot(xvals-1,cl[2]*(xvals-1)**2+cl[1]*(xvals-1)+cl[0])
    #            plt.show()
                xnl = -cl[1]/(2*cl[2])
                znl = cl[2]*xnl**2+cl[1]*xnl+cl[0]
                return xl+xnl, znl, chil
            elif chir<chi and chir<chil:
                xnr = -cr[1]/(2*cr[2])
                znr = cr[2]*xnr**2+cr[1]*xnr+cr[0]
    #            plt.errorbar(xvals+1,zr,yerr=sqrt(zr))
    #            plt.plot(xvals+1,cr[2]*(xvals+1)**2+cr[1]*(xvals+1)+cr[0])
    #            plt.show()
                return xr+xnr, znr, chir
            else:
                ca = coeffs[2]
                cb = coeffs[1]
                xc = -cb/(2*ca)
                zc = ca*xc**2+cb*xc+coeffs[0]
                return x+xc, zc, chi
        else:
            ca = coeffs[2]
            cb = coeffs[1]
            xc = -cb/(2*ca)
            zc = ca*xc**2+cb*xc+coeffs[0]
            return x+xc, zc, chi
    elif form=='gaussian':
        xpad = 6
        xvals = np.arange(-xpad,xpad+1)
        xwindow = x+xvals
        xvals = xvals[(xwindow>=0)*(xwindow<maxx)]
        zvals = ccd[x+xvals,y]
        params, errarr = sf.gauss_fit(xvals,zvals,fit_exp='y')
        xc = x+params[1] #offset plus center
        zc = params[2] #height (intensity)
        sig = params[0] #standard deviation
        power = params[5]
#        pxn = np.linspace(xvals[0],xvals[-1],1000)
        fit = sf.gaussian(xvals,abs(params[0]),params[1],params[2],params[3],params[4],params[5])
        chi = sum((fit-zvals)**2/zvals)
        return xc, zc, abs(sig), power, chi

def find_trace_coeffs(image,pord,fiber_space,num_points=None,num_fibers=None,vertical=False,return_all_coeffs=True,skip_peaks=0):
    """ Polynomial fitting for trace coefficients.  Packs into interval [-1,1]
        INPUTS:
            image - 2D ccd image on which you'd like to find traces
            pord - polynomial order to fit trace positions
            fiber_space - estimate of fiber spacing in pixels
            num_points - number of cross sections to average for trace (if None, will set to 1/20 length or 2*pord (whichever is greater))
            num_fibers - number of fibers (if None, will auto-detect)
            vertical - True if traces run vertical. False if horizontal.
            return_all_coeffs - if False, only returns trace_poly_coeffs
        OUTPUTS:
            t_coeffs - nx(pord+1) array where n is number of detected traces
                                gives fitted coeffs for each trace
            i_coeffs - intensity along each trace
            s_coeffs - sigma (for gaussian fit) along each trace
            p_coeffs - power (for pseudo gaussian fit) along each trace
    """
    def find_peaks(array,bg_cutoff=None,mx_peaks=None,skip_peaks=0):
        """ Finds peaks of a 1D array.
            Assumes decent signal to noise ratio, no anomalies
            Assumes separation of at least 5 units between peaks
        """
        ###find initial peaks (center is best in general, but edge is okay here)
        xpix = len(array)
        px = 2
        pcnt = 0
        skcnt = 0
        peaks = np.zeros(len(array))
        if mx_peaks is None:
            mx_peaks = len(array)
        if bg_cutoff is None:
            bg_cutoff = 0.5*np.mean(array)
        while px<xpix:
#            if trct>=num_fibers:
#                break
#            y = yvals[0]
            if array[px-1]>bg_cutoff and array[px]<array[px-1] and array[px-1]>array[px-2]: #not good for noisy
                if skcnt < skip_peaks:
                    skcnt += 1 #Increment skip counter
                    continue
                else:
                    peaks[pcnt] = px-1
                    px += 5 #jump past peak
                    pcnt+=1 #increment peak counts
                    skcnt += 1 #increment skip counter
                if pcnt >= mx_peaks:
                    break
            else:
                px+=1
        peaks = peaks[0:pcnt]
        return peaks
    
    def find_fibers(image):  
        """ Estimates number of fibers.
            Assumes roughly square image.
            THIS DOESN'T WORK RIGHT YET
        """
        shrt_ax = int(min(np.shape(image))/2)
        crsx_pts = min(10,shrt_ax)
        tr_ul_lr = np.zeros(crsx_pts) #traces cutting upper left to lower right
        tr_ur_ll = np.zeros(crsx_pts) #traces cutting upper right to lower left
        for i in range(crsx_pts):
            ul_lr = np.ravel(image)[2*i:np.size(image):(len(image[0])+1)]
            ur_ll = np.ravel(image)[(len(image[0])-1-2*i):np.size(image):(len(image[0])-1)]
            tr_ul_lr[i] = len(find_peaks(ul_lr))
            tr_ur_ll[i] = len(find_peaks(ur_ll))
        tr_ul_lr = int(np.median(tr_ul_lr))
        tr_ur_ll = int(np.median(tr_ur_ll))
        if tr_ul_lr > tr_ur_ll:
            return tr_ul_lr, True
        else:
            return tr_ur_ll, False
                  
    ### Force horizontal
    if vertical:
        image = image.T
    ### Find number of fibers and direction (upper left to lower right or opposite)
    tmp_num_fibers, fiber_dir = find_fibers(image)
    if num_fibers is None:
        num_fibers=tmp_num_fibers
    ### Select number of points to use in tracing
    if num_points is None:
        num_points = max(2*pord,int(np.shape(image)[1]/20))
    ### Make all empty arrays for trace finding
    xpix = np.shape(image)[0]
    ypix = np.shape(image)[1]
    yspace = int(np.floor(ypix/(num_points+1)))
    yvals = yspace*(1+np.arange(num_points))
    xtrace = np.nan*np.ones((num_fibers,num_points)) #xpositions of traces
    ytrace = np.zeros((num_fibers,num_points)) #ypositions of traces
    sigtrace = np.zeros((num_fibers,num_points)) #standard deviation along trace
    powtrace = np.zeros((num_fibers,num_points)) #pseudo-gaussian power along trace
    Itrace = np.zeros((num_fibers,num_points)) #relative intensity of flat at trace
    chi_vals = np.zeros((num_fibers,num_points)) #returned from fit_trace
    bg_cutoff = 1.05*np.median(image) #won't fit values below this intensity
    
    ### Put in initial peak guesses
    peaks = find_peaks(image[:,yvals[0]],bg_cutoff=bg_cutoff,mx_peaks=num_fibers,skip_peaks=skip_peaks)
    xtrace[:len(peaks)-1,0] = peaks[:-1] ### have to cut off last point - trace wanders off ccd
    ytrace[:,0] = yvals[0]*np.ones(len(ytrace[:,0]))   
    ###From initial peak guesses fit for more precise location
    for i in range(num_fibers):
        y = yvals[0]
        if not np.isnan(xtrace[i,0]):
            xtrace[i,0], Itrace[i,0], sigtrace[i,0], powtrace[i,0], chi_vals[i,0] = fit_trace(xtrace[i,0],y,image)
        else:
            Itrace[i,0], sigtrace[i,0], powtrace[i,0], chi_vals[i,0] = np.nan, np.nan, np.nan, np.nan    
    
    for i in range(1,len(yvals)):
        y = yvals[i]
        crsxn = image[:,y]
        ytrace[:,i] = y
        for j in range(num_fibers):
            if not np.isnan(xtrace[j,i-1]):
                #set boundaries
                lb = int(xtrace[j,i-1]-fiber_space/2)
                ub = int(xtrace[j,i-1]+fiber_space/2)
                #cutoff at edges
                if lb<0:
                    lb = 0
                if ub > xpix:
                    ub = xpix
                #set subregion
                xregion = crsxn[lb:ub]
                #only look at if max is reasonably high (don't try to fit background)
                if np.max(xregion)>bg_cutoff:
                    #estimate of trace position based on tallest peak
                    xtrace[j,i] = np.argmax(xregion)+lb
                    #quadratic fit for sub-pixel precision
                    try:
                        xtrace[j,i], Itrace[j,i], sigtrace[j,i], powtrace[j,i], chi_vals[j,i] = fit_trace(xtrace[j,i],y,image)
                        if xtrace[j, i] < xtrace[j, i-1]:
                            xtrace[j,i], Itrace[j,i], sigtrace[j,i], sigtrace[j,i], chi_vals[j,i] = np.nan, np.nan, np.nan, np.nan, np.nan
                    except RuntimeError:
                        xtrace[j,i], Itrace[j,i], sigtrace[j,i], sigtrace[j,i], chi_vals[j,i] = np.nan, np.nan, np.nan, np.nan, np.nan
                else:
                    xtrace[j,i], Itrace[j,i], sigtrace[j,i], sigtrace[j,i], chi_vals[j,i] = np.nan, np.nan, np.nan, np.nan, np.nan
            else:
                xtrace[j,i], Itrace[j,i], sigtrace[j,i], sigtrace[j,i], chi_vals[j,i] = np.nan, np.nan, np.nan, np.nan, np.nan
                
    Itrace /= np.median(Itrace) #Rescale intensities
        
    #Finally fit x vs. y on traces.  Start with quadratic for simple + close enough
    t_coeffs = np.zeros((3,num_fibers))
    i_coeffs = np.zeros((3,num_fibers))
    s_coeffs = np.zeros((3,num_fibers))
    p_coeffs = np.zeros((3,num_fibers))
    for i in range(num_fibers):
        #Given orientation makes more sense to swap x/y
        mask = ~np.isnan(xtrace[i,:])
        profile = np.ones((len(ytrace[i,:][mask]),3)) #Quadratic fit
        profile[:,1] = (ytrace[i,:][mask]-ypix/2)/ypix #scale data to get better fit
        profile[:,2] = ((ytrace[i,:][mask]-ypix/2)/ypix)**2
        noise = np.diag(chi_vals[i,:][mask])
        if len(xtrace[i,:][mask])>3:
            tmp_coeffs, junk = sf.chi_fit(xtrace[i,:][mask],profile,noise)
            tmp_coeffs2, junk = sf.chi_fit(Itrace[i,:][mask],profile,noise)
            tmp_coeffs3, junk = sf.chi_fit(sigtrace[i,:][mask],profile,noise)
            tmp_coeffs4, junk = sf.chi_fit(powtrace[i,:][mask],profile,noise)
        else:
            tmp_coeffs = np.nan*np.ones((3))
            tmp_coeffs2 = np.nan*np.ones((3))
            tmp_coeffs3 = np.nan*np.ones((3))
            tmp_coeffs4 = np.nan*np.ones((3))
        t_coeffs[0,i] = tmp_coeffs[2]
        t_coeffs[1,i] = tmp_coeffs[1]
        t_coeffs[2,i] = tmp_coeffs[0]
        i_coeffs[0,i] = tmp_coeffs2[2]
        i_coeffs[1,i] = tmp_coeffs2[1]
        i_coeffs[2,i] = tmp_coeffs2[0]
        s_coeffs[0,i] = tmp_coeffs3[2]
        s_coeffs[1,i] = tmp_coeffs3[1]
        s_coeffs[2,i] = tmp_coeffs3[0]      
        p_coeffs[0,i] = tmp_coeffs4[2]
        p_coeffs[1,i] = tmp_coeffs4[1]
        p_coeffs[2,i] = tmp_coeffs4[0]
        
    if return_all_coeffs:
        return t_coeffs, i_coeffs, s_coeffs, p_coeffs
    else:
        return t_coeffs

def remove_ccd_background(ccd,cut=None,plot=False):
    """ Use to remove diffuse background (not bias).
        Assumes a gaussian background.
        Returns ccd without zero mean background and
        the mean background error (1 sigma)
    """
    if cut is None:
        cut = 3*np.median(ccd)
    cut = int(cut)
    ccd_mask = (ccd < cut)*(ccd > -cut)
    masked_ccd = ccd[ccd_mask]
    arr = plt.hist(masked_ccd,2*(cut-1))
    hgt = arr[0]
    xvl = arr[1][:-1]
    ### Assume lower tail is a better indicator than upper tail
    xmsk = (xvl < np.median(masked_ccd))
    hgts = hgt[xmsk]
    xvls = xvl[xmsk]
    sig_est = 2/2.35*(xvls[np.argmax(hgts)] - xvls[np.argmax(hgts>np.max(hgts)/2)])
    pguess = (sig_est,np.median(masked_ccd),np.max(hgt))
    sigma = 1/np.sqrt(abs(hgts)+1)
    params, errarr = opt.curve_fit(sf.gaussian,xvls,hgts,p0=pguess,sigma=sigma)
    if plot:
        plt.title("Number of pixels with certain count value")
        htst = sf.gaussian(xvl, params[0], center=params[1], height=params[2],bg_mean=0,bg_slope=0,power=2)
        plt.plot(xvl,htst)
        plt.show()
    plt.close()
    ccd -= params[1] # mean
    bg_std = params[0]
    return ccd, bg_std

def cosmic_ray_reject(D,f,P,iV,S=0,threshhold=25,verbose=False):
    """ Procedure to reject cosmic rays in optimal extraction.
        This assumes your inputs are all at a given wavelength.
        Only rejects the worst point (loop to reject more)
        Inputs (based on nomenclature in Horne, 1986):
            D - image data (array)
            f - estimated flux/scale factor (scalar)
            P - model profile (array)
            S - sky profile (array)
            iV - inverse variance (array), slight departure from Horne
            threshhold - absolute value of max residual to reject (default=25)
        Output:
            pixel_reject - mask array of pixel to reject (1 = keep, 0 = reject)
    """
    pixel_reject = np.ones(len(D))
    if len(D)!=len(P) and len(D)!=len(iV):
        if verbose:
            print("Array lengths must all be equal")
        exit(0)
    else:
        pixel_residuals = (D-f*P-S)**2*iV
        #Include the sign so that only positive residuals are eliminated
        pixel_residuals*=np.sign(D-f*P-S)
        if np.max(pixel_residuals)>threshhold:
            pixel_reject[np.argmax(pixel_residuals)]=0
    return pixel_reject

def linear_mn_hght_bg(xvals,yvals,invals,sigma,mn_est,power=2):
    """ Find mean of gaussian with linearized procedure.  Use eqn
        dx = dh/h * sigma/2 * exp(1/2) where dh = max - min residual
        Once mean is found, determines best fit hght and bg
    """
    gauss_model = sf.gaussian(xvals,sigma,center=mn_est,height=np.max(yvals),power=power)
    mn_est_err = 100
    mn_est_err_old = 0
    loop_ct = 0
    while abs(mn_est_err-mn_est_err_old)>0.01 and loop_ct < 1:
        mn_est_err_old = np.copy(mn_est_err)
        mn_est_old = np.copy(mn_est)
        residuals = yvals-gauss_model
        dh = (np.max(residuals)-np.min(residuals))/np.max(yvals)
        sign = 1
        if np.argmax(residuals) < np.argmin(residuals):
            sign = -1
        dx = sign*sigma*dh*np.exp(1/2)/2
        mn_est += dx
        hght, bg = sf.best_linear_gauss(xvals,sigma,mn_est,yvals,invals,power=power)
        gauss_model = sf.gaussian(xvals,sigma,center=mn_est,height=hght,bg_mean=bg,power=power)
        mn_est_err = abs(mn_est_old - mn_est)
        loop_ct += 1
#    hght, bg = sf.best_linear_gauss(xvals,sigma,mn_est,yvals,invals,power=power)
    return mn_est, hght, bg
    
def fit_mn_hght_bg(xvals,zorig,invorig,sigj,mn_new,spread,powj=2):
    """ Fits mean, height, and background for a gaussian of known sigma.
        Height and background are fit linearly.  Mean is fit through a grid
        search algorithm (may be better to change to a nonlinear fitter?)
    """
#    ts = time.time()
    mn_old = -100
    lp_ct = 0
    while abs(mn_new-mn_old)>0.001:
        mn_old = np.copy(mn_new)
#        t1 = time.time()
        hght, bg = sf.best_linear_gauss(xvals,sigj,mn_old,zorig,invorig,power=powj)
#        t2 = time.time()
        mn_new, mn_new_std = sf.best_mean(xvals,sigj,mn_old,hght,bg,zorig,invorig,spread,power=powj)
#        t3 = time.time()
#        print("Linear time = {}s".format(t2-t1))
#        print("Nonlinear time = {}s".format(t3-t2))
#        time.sleep(5)
        lp_ct+=1
        if lp_ct>1e3: break
#    print "Loop count is ", lp_ct
#    print("Len xvals = {}".format(len(xvals)))
#    te = time.time()
#    print("Total fit time is {}s".format(te-ts))
#    time.sleep(5)
    return mn_new, hght,bg
    
def extract_1D(ccd, t_coeffs, i_coeffs=None, s_coeffs=None, p_coeffs=None, readnoise=1, gain=1, return_model=False, verbose=False):
    """ Function to extract using optimal extraction method.
        This could benefit from a lot of cleaning up
        INPUTS:
        ccd - ccd image to extract
        t_coeffs - estimate of trace coefficients (from 'find_t_coeffs')
        i/s/p_coeffs - optional intensity, sigma, power coefficients
        readnoise, gain - of the ccd
        return_model - set True to return model of image based on extraction
        OUTPUTS:
        spec - extracted spectrum (n x hpix) where n is number of traces
        spec_invar - inverse variance at each point in extracted spectrum
        spec_mask - mask for invalid/suspect points in spectrum
        image_model - only if return_model = True. 
    """
#    def extract(ccd,t_coeffs,i_coeffs=None,s_coeffs=None,p_coeffs=None,readnoise=1,gain=1,return_model=False,fact,verbose=False):
#        """ Extraction.
#        """
    
    ### t_coeffs are from fiber flat - need to shift based on actual exposure
        
    ####################################################
    ###   Prep Needed variables/empty arrays   #########
    ####################################################
    ### CCD dimensions and number of fibers
    hpix = np.shape(ccd)[1]
    vpix = np.shape(ccd)[0]
    num_fibers = np.shape(t_coeffs)[1]

    ####################################################    
    #####   First refine horizontal centers (fit   #####
    #####   traces from data ccd using fiber flat  #####
    #####   as initial estimate)                   #####
    ####################################################
    ta = time.time()  ### Start time of trace refinement
    fact = 20 #do 1/fact * available points
    ### Empty arrays
    rough_pts = int(np.ceil(hpix/fact))
    xc_ccd = np.zeros((num_fibers,rough_pts))
    yc_ccd = np.zeros((num_fibers,rough_pts))
    inv_chi = np.zeros((num_fibers,rough_pts))
    if verbose:
        print("Refining trace centers")
    for i in range(num_fibers):
#        plt.show()
#        plt.close()
        for j in range(0,hpix,fact):
            ### set coordinates, gaussian parameters from coeffs
            jadj = int(np.floor(j/fact))
            yj = (j-hpix/2)/hpix
            yc_ccd[i,jadj] = j
            xc = np.poly1d(t_coeffs[:,i])(yj)
#            plt.plot(j, xc, 'b.')
#            Ij = i_coeffs[2,i]*yj**2+i_coeffs[1,i]*yj+i_coeffs[0,i] #May use later for normalization
            sigj = np.poly1d(s_coeffs[:,i])(yj)
            powj = np.poly1d(p_coeffs[:,i])(yj)
#            sigj = s_coeffs[2,i]*yj**2+s_coeffs[1,i]*yj+s_coeffs[0,i]
#            powj = p_coeffs[2,i]*yj**2+p_coeffs[1,i]*yj+p_coeffs[0,i]
            ### Don't try to fit any bad trace sections
            if np.isnan(xc):
                xc_ccd[i,jadj] = np.nan
                inv_chi[i,jadj] = 0
            else:
                ### Take subset of ccd of interest, xpad pixels to each side of peak
                xpad = 7
                xvals = np.arange(-xpad,xpad+1)
                xj = int(xc)
                xwindow = xj+xvals
                xvals = xvals[(xwindow>=0)*(xwindow<vpix)]
                zorig = gain*ccd[xj+xvals,j]
                ### If empty slice, don't try to fit
                if len(zorig)<1:
                    xc_ccd[i,jadj] = np.nan
                    inv_chi[i,jadj] = 0
                    continue
                invorig = 1/(abs(zorig)+readnoise**2) ### inverse variance
                ### Don't try to fit profile for very low SNR peaks
                if np.max(zorig)<20:
                    xc_ccd[i,jadj] = np.nan
                    inv_chi[i,jadj] = 0
                else:
                    ### Fit for center (mn_new), amongst other values
#                    mn_new, hght, bg = fit_mn_hght_bg(xvals,zorig,invorig,sigj,xc-xj-1,sigj,powj=powj)
                    mn_new, hght, bg = linear_mn_hght_bg(xvals,zorig,invorig,sigj,xc-xj-1,power=powj)
                    fitorig = sf.gaussian(xvals,sigj,mn_new,hght,power=powj)
                    inv_chi[i,jadj] = 1/sum((zorig-fitorig)**2*invorig)
                    ### Shift from relative to absolute center
                    xc_ccd[i,jadj] = mn_new+xj+1
              
    #####################################################
    #### Now with new centers, refit trace coefficients #
    #####################################################
    tmp_poly_ord = 6  ### Use a higher order for a closer fit over entire trace
    t_coeffs_ccd = np.zeros((tmp_poly_ord+1,num_fibers))
    for i in range(num_fibers):
        #Given orientation makes more sense to swap x/y
        mask = ~np.isnan(xc_ccd[i,:]) ### Mask bad points
        ### build profile matrix over good points
        profile = np.ones((len(yc_ccd[i,:][mask]),tmp_poly_ord+1))
        for order in range(tmp_poly_ord):
            profile[:,order+1] = ((yc_ccd[i,:][mask]-hpix/2)/hpix)**(order+1)
        profile = profile[:,::-1]
        noise = np.diag(inv_chi[i,:][mask])
        if len(xc_ccd[i,:][mask])>(tmp_poly_ord+1):
            ### Chi^2 fit
            tmp_coeffs, junk = sf.chi_fit(xc_ccd[i,:][mask],profile,noise)
        else:
            ### if not enough points to fit, call entire trace bad
            tmp_coeffs = np.nan*np.ones((tmp_poly_ord+1))
        t_coeffs_ccd[:,i] = tmp_coeffs
#    t_coeffs_ccd[:,0] = np.pad(t_coeffs[:,0], ((tmp_poly_ord-2,0)), mode='constant')

    tb = time.time() ### Start time of extraction/end of trace refinement
    if verbose:
        print("Trace refinement time = {}s".format(tb-ta))
       
    ### Uncomment below to see plot of traces
#    plt.figure('inside extract 1d')
#    plt.imshow(ccd,interpolation='none')
#    for i in range(num_fibers):
#        ys = (np.arange(hpix)-hpix/2)/hpix
#        xs = np.poly1d(t_coeffs[:,i])(ys)
#        yp = np.arange(hpix)
#        plt.plot(yp,xs-1, 'b', linewidth=2)
#    plt.show()
#    plt.close()
    
    ###########################################################
    ##### Finally, full extraction with refined traces ########
    ###########################################################
    
    ### Make empty arrays for return values
    spec = np.zeros((num_fibers,hpix))
    spec_invar = np.zeros((num_fibers,hpix))
    spec_mask = np.ones((num_fibers,hpix),dtype=bool)
    chi2red_array = np.zeros((num_fibers,hpix))
    if return_model:
        image_model = np.zeros((np.shape(ccd))) ### Used for evaluation
    ### Run once for each fiber
    for i in range(num_fibers):
        if i == 0:
            #First fiber in n20161123 set is not available
            continue
        #slit_num = np.floor((i)/4)#args.telescopes) # Use with slit flats
        if verbose:
            print("extracting trace {}".format(i+1))
        ### in each fiber loop run through each trace
        for j in range(hpix):
#            if verbose:
#                sys.stdout.write("\r  " + str(int(np.round(100*j/hpix))) + "% done" + " " * 11)
#                sys.stdout.flush()
            yj = (j-hpix/2)/hpix
            xc = np.poly1d(t_coeffs_ccd[:,i])(yj)
#            Ij = i_coeffs[2,i]*yj**2+i_coeffs[1,i]*yj+i_coeffs[0,i]
            sigj = np.poly1d(s_coeffs[:,i])(yj)
            powj = np.poly1d(p_coeffs[:,i])(yj)
#            sigj = s_coeffs[2,i]*yj**2+s_coeffs[1,i]*yj+s_coeffs[0,i]
#            powj = p_coeffs[2,i]*yj**2+p_coeffs[1,i]*yj+p_coeffs[0,i]
            ### If trace center is undefined mask the point
            if np.isnan(xc):
                spec_mask[i,j] = False
            else:
                ### Set values to use in extraction
                xpad = 5  ### can't be too big or traces start to overlap
                xvals = np.arange(-xpad,xpad+1)
                xj = int(xc) - 1
                xwindow = xj+xvals
                xvals = xvals[(xwindow>=0)*(xwindow<vpix)]
                zorig = gain*ccd[xj+xvals, j]
                fitorig = sf.gaussian(xvals,sigj,xc-xj-1,hght,power=powj)
                ### If too short, don't fit, mask point
                if len(zorig)<1:
                    spec[i,j] = 0
                    spec_mask[i,j] = False
                    continue
                invorig = 1/(abs(zorig)+readnoise**2)
                ### don't try to extract for very low signal
                if np.max(zorig)<20:
                    continue
                else:
                    ### Do nonlinear fit for center, height, and background
                    mn_new, hght, bg = fit_mn_hght_bg(xvals,zorig,invorig,sigj,xc-xj-1,sigj/8,powj=powj)
#                    mn_new, hght, bg = linear_mn_hght_bg(xvals,zorig,invorig,sigj,xc-xj-1,power=powj)
                    ### Use fitted values to make best fit arrays
                    fitorig = sf.gaussian(xvals,sigj,mn_new,hght,power=powj)
                    xprecise = np.linspace(xvals[0],xvals[-1],100)
                    fitprecise = sf.gaussian(xprecise,sigj,mn_new,hght,power=powj)
                    ftmp = sum(fitprecise)*np.mean(np.ediff1d(xprecise))
                    #Following if/else handles failure to fit
                    if ftmp==0:
                        fitnorm = np.zeros(len(zorig))
                    else:
                        fitnorm = fitorig/ftmp
                    ### Get extracted flux and error
                    fstd = sum(fitnorm*zorig*invorig)/sum(fitnorm**2*invorig)
                    invorig = 1/(readnoise**2 + abs(fstd*fitnorm))
                    chi2red = np.sum((fstd*fitnorm+bg-zorig)**2*invorig)/(len(zorig)-3)
                    ### Now set up to do cosmic ray rejection
                    rej_min = 0
                    loop_count=0
                    while rej_min==0:
                        pixel_reject = cosmic_ray_reject(zorig,fstd,fitnorm,invorig,S=bg,threshhold=0.3*np.mean(zorig),verbose=True)
                        rej_min = np.min(pixel_reject)
                        ### Once no pixels are rejected, re-find extracted flux
                        if rej_min==0:
                            ### re-index arrays to remove rejected points
                            zorig = zorig[pixel_reject==1]
                            invorig = invorig[pixel_reject==1]
                            xvals = xvals[pixel_reject==1]
                            ### re-do fit (can later cast this into a separate function)
                            mn_new, hght, bg = fit_mn_hght_bg(xvals,zorig,invorig,sigj,xc-xj-1,sigj/8,powj=powj)
#                            mn_new, hght, bg = linear_mn_hght_bg(xvals,zorig,invorig,sigj,xc-xj-1,power=powj)
                            fitorig = sf.gaussian(xvals,sigj,mn_new,hght,power=powj)
                            xprecise = np.linspace(xvals[0],xvals[-1],100)
                            fitprecise = sf.gaussian(xprecise,sigj,mn_new,hght,power=powj)
                            ftmp = sum(fitprecise)*np.mean(np.ediff1d(xprecise))
                            fitnorm = fitorig/ftmp
                            fstd = sum(fitnorm*zorig*invorig)/sum(fitnorm**2*invorig)
                            invorig = 1/(readnoise**2 + abs(fstd*fitnorm))
                            chi2red = np.sum((fstd*fitnorm+bg-zorig)**2*invorig)/(len(zorig)-3)
                        ### if more than 3 points are rejected, mask the extracted flux
                        if loop_count>3:
                            spec_mask[i,j] = False
                            break
                        loop_count+=1
                    ### Set extracted spectrum value, inverse variance
                    spec[i,j] = fstd
                    spec_invar[i,j] = sum(fitnorm**2*invorig)
                    chi2red_array[i,j] = chi2red
                    if return_model and not np.isnan(fstd):
                        ### Build model, if desired
                        image_model[xj+xvals,j] += (fstd*fitnorm+bg)/gain
            ### If a nan came out of the above routine, zero it and mask
            if np.isnan(spec[i,j]):
                spec[i,j] = 0
                spec_mask[i,j] = False
#        if verbose:
#            print(" ")
    if verbose:
        chi2red = np.mean(chi2red_array[i])
        print("Average reduced chi^2 = {}".format(np.mean(chi2red)))
    if return_model:
        return spec, spec_invar, spec_mask, image_model
    else:
        return spec, spec_invar, spec_mask
        
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
        
def array_to_params(ellipse):
    """ Converts ellipse array to lmfit parameter object.
    """
    params = lmfit.Parameters()
#    for i in range(np.shape(centers)[1]):
#        params.add('hc{}'.format(i), value=centers[0,i])
#        params.add('vc{}'.format(i), value=centers[1,i])
    for j in range(np.shape(ellipse)[1]):
        params.add('q{}'.format(j), value=ellipse[0,j])
        params.add('PA{}'.format(j), value=ellipse[1,j])
    return params
    
def refine_trace_centers(ccd, t_coeffs, i_coeffs, s_coeffs, p_coeffs, fact=10, readnoise=3.63, verbose=False):
    """ Uses estimated centers from fibers flats as starting point, then
        fits from there to find traces based on science ccd frame.
        INPUTS:
            ccd - image on which to fit traces
            t/i/s/p_coeffs - modified gaussian coefficients from fiberflat
            fact - do 1/fact of the available points
    """
    num_fibers = t_coeffs.shape[1]
    hpix = ccd.shape[1]
    vpix = ccd.shape[0]
    ### First fit vc parameters for traces
    rough_pts = int(np.ceil(hpix/fact))
    vc_ccd = np.zeros((num_fibers,rough_pts))
    hc_ccd = np.zeros((num_fibers,rough_pts))
    inv_chi = np.zeros((num_fibers,rough_pts))
    yspec = np.arange(hpix)
    if verbose:
        print("Refining trace centers")
    for i in range(num_fibers):
        if i != 63:
            continue
        if verbose:
            print("Running on index {}".format(i))
    #    slit_num = np.floor((i)/args.telescopes)
        for j in range(0,hpix,fact):
            jadj = int(np.floor(j/fact))
            yj = (yspec[j]-hpix/2)/hpix
            hc_ccd[i,jadj] = yspec[j]
            vc = t_coeffs[2,i]*yj**2+t_coeffs[1,i]*yj+t_coeffs[0,i]
#            Ij = i_coeffs[2,i]*yj**2+i_coeffs[1,i]*yj+i_coeffs[0,i]
            sigj = s_coeffs[2,i]*yj**2+s_coeffs[1,i]*yj+s_coeffs[0,i]
            powj = s_coeffs[2,i]*yj**2+s_coeffs[1,i]*yj+s_coeffs[0,i]
            if np.isnan(vc):
                vc_ccd[i,jadj] = np.nan
                inv_chi[i,jadj] = 0
            else:
                xpad = 7
                xvals = np.arange(-xpad,xpad+1)
                xj = int(vc)
                xwindow = xj+xvals
                xvals = xvals[(xwindow>=0)*(xwindow<vpix)]
                zorig = ccd[xj+xvals,yspec[j]]
                if len(zorig)<1:
                    vc_ccd[i,jadj] = np.nan
                    inv_chi[i,jadj] = 0
                    continue
                invorig = 1/(abs(zorig)+readnoise**2)
                if np.max(zorig)<20:
                    vc_ccd[i,jadj] = np.nan
                    inv_chi[i,jadj] = 0
                else:
                    mn_new, hght, bg = fit_mn_hght_bg(xvals, zorig, invorig, sigj, vc-xj-1, sigj, powj=powj)
                    fitorig = sf.gaussian(xvals,sigj,mn_new,hght,power=powj)
                    if j == 715:
                        print mn_new
                        plt.plot(xvals,zorig,xvals,fitorig)
                        plt.show()
                        plt.close()
                    inv_chi[i,jadj] = 1/sum((zorig-fitorig)**2*invorig)
                    vc_ccd[i,jadj] = mn_new+xj+1
                    
    
    tmp_poly_ord = 10
    trace_coeffs_ccd = np.zeros((tmp_poly_ord+1,num_fibers))
    for i in range(num_fibers):
        mask = ~np.isnan(vc_ccd[i,:])
        profile = np.ones((len(hc_ccd[i,:][mask]),tmp_poly_ord+1)) #Quadratic fit
        for order in range(tmp_poly_ord):
            profile[:,order+1] = ((hc_ccd[i,:][mask]-hpix/2)/hpix)**(order+1)
        noise = np.diag(inv_chi[i,:][mask])
        if len(vc_ccd[i,:][mask])>3:
            tmp_coeffs, junk = sf.chi_fit(vc_ccd[i,:][mask],profile,noise)
        else:
            tmp_coeffs = np.nan*np.ones((tmp_poly_ord+1))
        trace_coeffs_ccd[:,i] = tmp_coeffs 
    return trace_coeffs_ccd
    
def extract_2D(ccd, psf_coeffs, t_coeffs, readnoise=1, gain=1, return_model=False, verbose=False):
    """ Code to perform 2D spectroperfectionism algorithm on MINERVA data.
    """
    if psf_coeffs.ndim == 1: 
        psf_coeffs = psf_coeffs.reshape((1,len(psf_coeffs)))
    elif psf_coeffs.ndim != 2:
        print("Invalid shape for psf_coeffs with ndim = {}".format(psf_coeffs.ndim))
        exit(0)
    ### Set shape variables based on inputs
    num_fibers = t_coeffs.shape[1]
    num_fibers = 1 ### Manual override for testing
    hpix = ccd.shape[1]
    hscale = (np.arange(hpix)-hpix/2)/hpix
    extracted_counts = np.zeros((num_fibers,hpix))
    ### Remove CCD diffuse background - cut value matters
    cut = np.median(np.median(ccd[ccd<np.median(ccd)]))
    ccd, bg_err = remove_ccd_background(ccd,cut=cut)
    ### Parameters for extraction box size - try various values
    ### For meaning, see documentation
    num_sections = 16
    len_section = 143
    fit_pad = 4
    v_pad = 6
    len_edge = fit_pad*2
    ### iterate over all fibers
    for fib in range(num_fibers):
        if verbose:
            print("Running 2D Extraction on fiber {}".format(fib))
        ### Trace parameters
        vcenters = sf.eval_polynomial_coeffs(hscale,t_coeffs[:,fib+63])### Manual override  
        ### PSF parameters
        ellipse = psf_coeffs[fib,-6:]
        ellipse = ellipse.reshape((2,3))
        params = array_to_params(ellipse)
        coeff_matrix = psf_coeffs[fib,:-6]
        coeff_matrix = coeff_matrix.reshape((int(coeff_matrix.size/3),3))
        for sec in range(num_sections):
            tstart = time.time()
            ### Get a small section of ccd to extract
            hsec = np.arange(sec*(len_section-2*len_edge), len_section+sec*(len_section-2*len_edge))
            hmin = hsec[0]
            vcentmn = np.mean(vcenters[hsec])
            vmin = max(int(vcentmn-v_pad),0)
            vmax = min(int(vcentmn+v_pad+1),ccd.shape[0])
            ccd_sec = ccd[vmin:vmax,hsec]
            ccd_sec_invar = 1/(ccd_sec + bg_err**2)
            d0 = ccd_sec.shape[0]
            d1 = ccd_sec.shape[1]
            ### Optional - test removing background again
            ccd_sec, sec_bg_err = remove_ccd_background(ccd_sec,cut=3*bg_err)
            ### number of wavelength points to extract, default 1/pixel
            wls = len_section
            hcents = np.linspace(hsec[0],hsec[-1],wls)
            vcents = vcenters[hsec]
            A = np.zeros((wls,d0,d1))
            for jj in range(wls):
                ### Commented lines are if wl_pad is used
#                if jj < 0:
#                    hcent = hcents[0]+jj*dlth
#                    vcent = sf.eval_polynomial_coeffs((hcent-hpix/2)/hpix, trace_coeffs_ccd[:,idx])[0]
#                elif jj >= wls:
#                    hcent = hcents[-1]+(jj-wls+1)*dlth
#                    vcent = sf.eval_polynomial_coeffs((hcent-hpix/2)/hpix, trace_coeffs_ccd[:,idx])[0]
#                else:
#                    hcent = hcents[jj]
#                    vcent = vcents[jj]
                hcent = hcents[jj]
                vcent = vcents[jj]
                vcent -= 1  ### Something is wrong above - shouldn't need this...
                center = [np.mod(hcent,1),np.mod(vcent,1)]
                hpoint = (hcent-hpix/2)/hpix
                ### Now build PSF model around center point
                psf_type = 'bspline'
                if psf_type == 'bspline':
                    ### TODO - revamp this to pull from input
                    r_breakpoints = np.hstack(([0, 1.5, 2.4, 3],np.arange(3.5,8.6,1)))         
                    theta_orders = [0]
                    psf_jj = spline.make_spline_model(params, coeff_matrix, center, hpoint, [2*fit_pad+1,2*fit_pad+1], r_breakpoints, theta_orders, fit_bg=False)
                    bg_lvl = np.median(psf_jj[psf_jj<np.mean(psf_jj)])
                    psf_jj -= bg_lvl  
                    psf_jj /= np.sum(psf_jj) # Normalize to 1
#                    if jj == 1:
#                        plt.imshow(psf_jj)
#                        plt.show()
#                        plt.close()
                ### set coordinates for opposite corners of box (for profile matrix)
                vtl = max(0,int(vcent)-fit_pad-vmin)
                htl = max(0,int(hcent)-fit_pad-hmin)
                vbr = min(A.shape[1],int(vcent)+fit_pad+1-vmin)
                hbr = min(A.shape[2],int(hcent)+fit_pad+1-hmin)
#                print("Top left = ({},{}), Bottom right = ({},{})".format(htl,vtl,hbr,vbr))
                ### Use to slice psf_jj
                sp_l = max(0,fit_pad+(htl-int(hcent-hmin))) #left edge
                sp_r = min(2*fit_pad+1,fit_pad+(hbr-int(hcent-hmin))) #right edge
                sp_t = max(0,fit_pad+(vtl-int(vcent-vmin))) #top edge
                sp_b = min(2*fit_pad+1,fit_pad+(vbr-int(vcent-vmin))) #bottom edge
                ### indices of A slice to use
                a_l = max(0,htl) # left edge
                a_r = min(A.shape[2],hbr) # right edge
                a_t = max(0,vtl) # top edge
                a_b = min(A.shape[1],vbr) # bottom edge    
                A[jj,a_t:a_b,a_l:a_r] = psf_jj[sp_t:sp_b,sp_l:sp_r]  
            ##Now using the full available data
            im = np.sum(A,axis=0)
#            plt.imshow(im)
#            plt.show()
            B = np.matrix(np.resize(A.T,(d0*d1,wls)))
#            B = np.hstack((B,np.ones((d0*d1,1)))) ### add background term
            p = np.matrix(np.resize(ccd_sec.T,(d0*d1,1)))
            n = np.diag(np.resize(ccd_sec_invar.T,(d0*d1,)))
            #print np.shape(B), np.shape(p), np.shape(n)
            text_sp_st = time.time()
            fluxtilde = sf.extract_2D_sparse(p,B,n)
            t_betw_ext = time.time()
            if sec == 0:
                extracted_counts[fib,0:len_section] = fluxtilde
            else:
                sec_inds = np.arange(len_edge,len_section,dtype=int)
                extracted_counts[fib,sec_inds+int(sec*(len_section-2*len_edge))] = fluxtilde[sec_inds]
            tfinish = time.time()
            if verbose:
                print("Section {} Time = {}".format(sec,tfinish-tstart))
                print("  PSF modeling took {}s".format(text_sp_st-tstart))
                print("  Sparse extraction took {}s".format(t_betw_ext-text_sp_st))
    return extracted_counts
#    flux2 = sf.extract_2D_sparse(p,B,n,return_no_conv=True)

'''
    ### Figure out way to include diagnositic plots in a sensible place
    img_est = np.dot(B,flux2)
    img_estrc = np.dot(B,fluxtilde2)
    img_recon = np.real(np.resize(img_estrc,(d1,d0)).T)
    plt.figure("Residuals of 2D fit")
    plt.imshow(np.vstack((ccd_small,img_recon,ccd_small-img_recon)),interpolation='none')
    chi_red = np.sum((ccd_small-img_recon)[:,fit_pad:-fit_pad]**2*ccd_small_invar[:,fit_pad:-fit_pad])/(np.size(ccd_small[:,fit_pad:-fit_pad])-jj+1)
    print("Reduced chi2 = {}".format(chi_red))
    #plt.figure()
    #plt.imshow(ccd_small,interpolation='none')
    plt.show()
    #img_raw = np.resize(np.dot(B,np.ones(len(fluxtilde2))),(d1,d0)).T
    #plt.imshow(img_raw,interpolation='none')
    #plt.show()
    plt.figure("Cross section of fit, residuals")
    for i in range(20,26):
    #    plt.plot(ccd_small[:,i])
        plt.plot(img_recon[:,i])
    #    plt.plot(final_centers[i,1],np.max(ccd_small[:,i]),'kd')
        plt.plot((ccd_small-img_recon)[:,i])#/np.sqrt(abs(ccd_small[:,i])))
    #    plt.show()
    #    plt.close()
    plt.show()
    plt.close()
#'''
    
def find_most_recent_arc_date(data_dir,return_filenames=False,date_format='nYYYYMMDD', before_date=None):
    """ Finds the date of most recent arc exposures.
        If desired, can return list of arc exposures on that date.
        date_format should show all positions of Y, M, and D (plus any /, -, etc)
    """
    ### Find files
    filenames = glob.glob(os.path.join(data_dir,'*','*[tT][hH][aA][rR]*'))
    filenames.sort()
    ### Set up dates, formatting
    roots = [os.path.split(f)[0] for f in filenames]
    dates = np.array(([os.path.split(r)[1] for r in roots]))
    dates = np.unique(dates)
    date_format = date_format.upper()
    def most_recent(D,dates,date_format,before_date=None):
        """ Intended to have D = 'Y', 'M', or 'D'
            Returns string of dates that have the most recent Y, M, or D.
        """
        D = D.upper()
        digits = date_format.count(D,0)
        st = date_format.find(D,0)
        ret = np.array(([dates[i][st:st+digits] for i in range(len(dates))]),dtype=int)
        recent = str(np.max(ret))
        return recent
#        dates_recent = np.array(([dates[i]]))
    ### Loop through, cull out
    for D in ['Y','M','D']:
        recent = most_recent(D,dates,date_format,before_date=before_date)
        date_inds = np.array(([i for i in range(len(dates)) if recent in dates[i]]),dtype=int)
        dates = dates[date_inds]
    date = dates[0]
    if return_filenames:
        return [f for f in filenames if date in f]
    else:
        return date
    
def fits_to_arrays(fits_files,ext=0,i2_in=False):
    """ Opens fits files from a list, stacks into an array.
        Build specifically for arc/flat frames with i2_in False
    """
    if type(fits_files) is str:
        fits_files = [fits_files,]
    idx = 0
    didx = 0
    for flnm in fits_files:
        img, hdr = open_minerva_fits(flnm, ext=ext, return_hdr=True)
        if idx == 0 and didx == 0:
            imgs = np.zeros((len(fits_files),img.shape[0],img.shape[1]))
        try:
            if hdr['I2POSAS']=='out':
                i2 = False
            else:
                i2 = True
        except KeyError:
            i2 = False
        if i2 == i2_in:
            imgs[idx] = img
            idx += 1
        else:
            imgs = np.delete(imgs,idx,axis=0)
            didx += 1
    return imgs
    
def stack_bias(redux_dir, data_dir, date, fname='bias_avg.fits'):
    """ Stacks bias frames from the day (median combined), saves result to
        disk.  Will load directly instead of re-doing if available.
        Returns the median bias
    """
    if os.path.isfile(os.path.join(redux_dir,date,fname)):
        bias_hdu = pyfits.open(os.path.join(redux_dir,date,fname),uint=True)
        bias = bias_hdu[0].data 
        return bias
    else:
        filenames = glob.glob(os.path.join(data_dir,date,'*[Bb]ias*.fits'))
        shape = np.shape(pyfits.open(filenames[0])[0].data)
        biases = np.zeros((len(filenames),shape[0],shape[1]))
        for i in range(len(filenames)):
            biases[i] = pyfits.open(filenames[i])[0].data
        bias = sf.combine(biases)
        hdu = pyfits.PrimaryHDU(bias)
        hdulist = pyfits.HDUList([hdu])
        if not os.path.isdir(os.path.join(redux_dir,date)):
            os.mkdir(os.path.join(redux_dir,date))
        hdulist.writeto(os.path.join(redux_dir,date,fname),clobber=True)
        return bias
    
def save_comb_arc_flat(filelist,frm,ts,redux_dir,date,no_overwrite=True, verbose=False, method='median'):
    """ frm must be 'arc' or 'flat'
    """
    if os.path.isfile(os.path.join(redux_dir,date,'combined_{}_{}.fits'.format(frm,ts))) and no_overwrite:
        return
    else:
        if verbose:
            print("Combining most recent {}s".format(frm))
        if frm == 'arc':
            frames = [f for f in filelist if ts in f.upper()]
        elif frm == 'flat':
            frames = [f for f in filelist if ts in f.upper()]
        frame_imgs = fits_to_arrays(frames)
        comb_img = sf.combine(frame_imgs, method=method)
        if not os.path.isdir(os.path.join(redux_dir,date)):
            os.makedirs(os.path.join(redux_dir,date))
        hdu = pyfits.PrimaryHDU(comb_img)
        hdu.header.append(('COMBMTHD',method,'Method used to combine frames'))
        hdulist = pyfits.HDUList([hdu])
        hdulist.writeto(os.path.join(redux_dir,date,'combined_{}_{}.fits'.format(frm,ts)),clobber=True)
        return