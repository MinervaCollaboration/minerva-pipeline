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
import psf_utils as psf
#import argparse
import lmfit

def open_minerva_fits(fits, ext=0, return_hdr=False):
    """ Converts from kiwispec format (raveled array of 2 8bit images) to
        analysis format (2D array of 16bit int, converted to float)
    """
    spectrum = pyfits.open(fits,uint=True, ignore_missing_end=True)
    hdr = spectrum[ext].header
    ccd = spectrum[ext].data
    #Dimensions
    ypix = hdr['NAXIS1']
    xpix = hdr['NAXIS2']
    
    actypix = 2048 ### Hardcoded to remove overscan, fix later if needed

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
#                ccd_16bit[row,col] = int(''.join(ccd_tmp[row,col]),base=2)
    
        overscan = ccd_16bit[::-1,actypix:]
        ccd = ccd_16bit[::-1,0:actypix] #Remove overscan region
    else:
        overscan = ccd[::-1,actypix:]
        ccd = ccd[::-1,0:actypix] #Remove overscan region
        ccd = ccd.astype(np.float)
    if return_hdr:
        return ccd, overscan, hdr
    else:
        return ccd, overscan

def fit_trace(x,y,ccd,rn,form='gaussian', model=None):
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
        def make_chi_profile(x,y,ccd,rn):
            xpad = 2
            xvals = np.arange(-xpad,xpad+1)
            xwindow = x+xvals
            xvals = xvals[(xwindow>=0)*(xwindow<maxx)]
            zvals = ccd[x+xvals,y]
            profile = np.ones((2*xpad+1,3)) #Quadratic fit
            profile[:,1] = xvals
            profile[:,2] = xvals**2
            noise = np.diag((1/(abs(zvals)+rn**2)))
            return zvals, profile, noise
        zvals, profile, noise = make_chi_profile(x,y,ccd,rn)
        coeffs, chi = sf.chi_fit(zvals,profile,noise)
        neg_chi = False
        if chi<0:
            neg_chi=True
            print chi
        fit = np.dot(profile, coeffs)
        chi = (zvals-fit)**2*np.diag(noise)
        if neg_chi:
            print chi
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
        invar = 1/(abs(zvals)+rn**2)
        params, errarr = sf.gauss_fit(xvals,zvals,invr=invar,fit_exp='y',fit_skew='n')
        xc = x+params[1] #offset plus center
        zc = params[2] #height (intensity)
        sig = params[0] #standard deviation
        power = params[5]
#        pxn = np.linspace(xvals[0],xvals[-1],1000)
        fit = sf.gaussian(xvals,abs(params[0]),params[1],params[2],params[3],params[4],params[5])
        chi = sum((fit-zvals)**2*invar)
#        if chi > 200:
#            print chi
#            plt.plot(xvals, zvals, xvals, fit)
#            plt.show()
#            plt.close()
        return xc, zc, abs(sig), power, chi
    elif form=='gauss_lor':
        def gauss_lor(params, xarr):
            xc = params['xc'].value
            sig = params['sig'].value
            hght = params['hght'].value
            ratio = params['rat'].value
            sigl = params['sigl'].value
            bg = params['bg'].value
            bg_slp = params['bg_slp'].value
            try:
                power = params['power'].value
            except:
                power = 2
            gauss = hght*ratio*np.exp(-abs(xarr-xc)**power/(2*(abs(sig)**power))) 
            lorentz = hght*(1-ratio)/(1+(xarr-xc)**2/sigl**2)
            return gauss+lorentz+bg+bg_slp*xarr
        def gl_res(params, xarr, yarr, invar):
            model = gauss_lor(params, xarr)
            return (yarr-model)**2*invar
        xpad = 7
        xvals = np.arange(-xpad,xpad+1)
        xwindow = x+xvals
        xvals = xvals[(xwindow>=0)*(xwindow<maxx)]
        zvals = ccd[x+xvals,y]
        invar = 1/(abs(zvals)+rn**2)
#        params, errarr = sf.gauss_fit(xvals,zvals,invr=invar,fit_exp='y')
#        xc = x+params[1] #offset plus center
#        zc = params[2] #height (intensity)
#        sig = params[0] #standard deviation
#        power = params[5]
        params0 = lmfit.Parameters()
        params0.add('xc', value = xvals[np.argmax(zvals)])
        params0.add('bg', value = zvals[0])
        params0.add('bg_slp', value = 0)
        params0.add('power', value = 2)
        params0.add('hght', value = np.max(zvals))
        params0.add('sig', value = 1.5)
        params0.add('rat', value = 0.9, max = 1.0)
        params0.add('sigl', value = 1.5, min = 1.0, max = 5)
#        pxn = np.linspace(xvals[0],xvals[-1],1000)
#        fit = sf.gaussian(xvals,abs(params[0]),params[1],params[2],params[3],params[4],params[5])
        largs = (xvals, zvals, invar)
        results = lmfit.minimize(gl_res, params0, args=largs)
        paramsf = results.params
        fit = gauss_lor(paramsf, xvals)
        chi = sum((fit-zvals)**2*invar)
        chir = chi/(len(xvals)-8)
#        if chir > 4:
#            print chir, paramsf['rat'].value, paramsf['sigl'].value
#            plt.plot(xvals, zvals, xvals, fit)
#            plt.show()
#            plt.close()
        return paramsf['xc'].value, paramsf['hght'].value, abs(paramsf['sig'].value), paramsf['power'].value, chi
    elif form == 'moffat':
        xpad = 6
        xvals = np.arange(-xpad,xpad+1)
        xwindow = x+xvals
        xvals = xvals[(xwindow>=0)*(xwindow<maxx)]
        zvals = ccd[x+xvals,y]
        def moff_res(params, xarr, yarr, invar):
            model = sf.moffat_lmfit(params, xarr)
            return (model-yarr)**2*invar
        params0 = lmfit.Parameters()
        params0.add('xc', value = xvals[np.argmax(zvals)])
        params0.add('alpha', value = 1)#, min=0.3, max=5)
        params0.add('beta', value = 2)
        params0.add('bg', value = np.min(zvals))
        params0.add('power', value = 2)
        params0.add('hght', value = (np.max(zvals)-np.min(zvals))*5)
        args_in = (xvals, zvals, 1/(abs(zvals)+0.0001))
        results = lmfit.minimize(moff_res,params0,args=args_in)
        paramsf = results.params
        xc = x+paramsf['xc'].value #offset plus center
        zc = paramsf['hght'].value #height (intensity)
        power = paramsf['power'].value
        alpha = paramsf['alpha'].value
        beta = paramsf['beta'].value
        fit = sf.moffat_lmfit(paramsf, xvals)
        chi = sum((fit-zvals)**2/(abs(zvals)+0.0001))
        return xc, zc, alpha, beta, power, chi
    elif form == 'bspline':
        xpad = 6
        xvals = np.arange(-xpad,xpad+1)
        xwindow = x+xvals
        xvals = xvals[(xwindow>=0)*(xwindow<maxx)]
        zvals = ccd[x+xvals,y]
        invar = 1/(abs(zvals)+rn**2)
        xc, hght, bg = spline_fast_fit(xvals, zvals, invar, model)
        xorig = np.linspace(-7,7,len(model))
        model /= np.sum(model)*(xorig[1]-xorig[0])
        sfit = hght*sf.re_interp(xvals, xc, xorig, model) + bg
        chi = np.sum((sfit-zvals)**2*invar)
#        if chi > 200:
#            print chi
#            plt.plot(xvals, zvals, xvals, sfit)
#            plt.show()
#            plt.close()
        return x+xc, chi

def find_trace_coeffs(image,pord,fiber_space,rn=3.63,num_points=None,num_fibers=None,vertical=False,return_all_coeffs=True,skip_peaks=0, profile='gaussian'):
    """ Polynomial fitting for trace coefficients.  Packs into interval [-1,1]
        INPUTS:
            image - 2D ccd image on which you'd like to find traces
            pord - polynomial order to fit trace positions
            fiber_space - estimate of fiber spacing in pixels
            num_points - number of cross sections to average for trace (if None, will set to 1/20 length or 2*pord (whichever is greater))
            num_fibers - number of fibers (if None, will auto-detect)
            vertical - True if traces run vertical. False if horizontal.
            return_all_coeffs - if False, only returns trace_poly_coeffs
            profile - shape of cross-dispersion profile to use for fitting
        OUTPUTS:
            (Depends on profile)
            multi_coeffs - array with pord+1 coeffs for each parameter of profile
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
#        plt.plot(array)
#        plt.plot(peaks,array[peaks.astype(int)],'ro')
#        plt.plot(bg_cutoff*np.ones((array.shape)),'k')
#        plt.show()
#        plt.close()
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
#    tmp_num_fibers, fiber_dir = find_fibers(image)
    if num_fibers is None:
        print "Please specify number of fibers"
#        num_fibers=tmp_num_fibers
    ### Select number of points to use in tracing
    if num_points is None:
        num_points = max(2*pord,int(np.shape(image)[1]/20))
    ### Make all empty arrays for trace finding
    itest = 80
    xpix = np.shape(image)[0]
    ypix = np.shape(image)[1]
    yspace = int(np.floor(ypix/(min(num_points+1,ypix))))
    plus = 1
    if num_points == ypix:
        plus = 0
    yvals = yspace*(plus+np.arange(num_points))
    xtrace = np.nan*np.ones((num_fibers,num_points)) #xpositions of traces
    ytrace = np.zeros((num_fibers,num_points)) #ypositions of traces
    powtrace = np.zeros((num_fibers,num_points)) #pseudo-gaussian power along trace
    Itrace = np.zeros((num_fibers,num_points)) #relative intensity of flat at trace
    chi_vals = np.zeros((num_fibers,num_points)) #returned from fit_trace
    if profile == 'gaussian' or profile == 'gauss_lor':
        sigtrace = np.zeros((num_fibers,num_points)) #standard deviation along trace
    elif profile == 'moffat':
        alphatrace = np.zeros((num_fibers,num_points))
        betatrace = np.zeros((num_fibers,num_points))
    elif profile == 'bspline':
#        spline_models = np.load(os.path.join(os.environ['MINERVA_REDUX_DIR'], 'n20170226','model_flat_bsplines','bspline_crsxn_models.npy'))
        spline_models = np.zeros((1000,num_fibers,ypix))
        for i in range(num_fibers):
            if os.path.isfile(os.path.join(os.environ['MINERVA_REDUX_DIR'], 'n20170226','bspline_crsxn_model_{}.npy'.format(i))):
                spline_models[:,i,:] = np.load(os.path.join(os.environ['MINERVA_REDUX_DIR'], 'n20170226','bspline_crsxn_model_{}.npy'.format(i)))
    else:
        print "Invalid choice for 'profile'"
        exit(0)
#    bg_cutoff = 1.05*np.median(image) #won't fit values below this intensity
    bg_cutoff = 0.2*np.mean(image[0:300]) #won't fit values below this intensity
    ### Put in initial peak guesses
    peaks = find_peaks(image[:,yvals[0]],bg_cutoff=bg_cutoff,mx_peaks=num_fibers,skip_peaks=skip_peaks)
    xtrace[:len(peaks)-1,0] = peaks[:-1] ### have to cut off last point - trace wanders off ccd
    ytrace[:,0] = yvals[0]*np.ones(len(ytrace[:,0]))
    ###From initial peak guesses fit for more precise location
#    plt.imshow(np.log(image))
    for i in range(num_fibers):
#        if i != itest:
#            continue
        y = yvals[0]
        if profile == 'gaussian' or profile == 'gauss_lor':
            if not np.isnan(xtrace[i,0]):
                xtrace[i,0], Itrace[i,0], sigtrace[i,0], powtrace[i,0], chi_vals[i,0] = fit_trace(xtrace[i,0],y,image,rn,form=profile)
            else:
                Itrace[i,0], sigtrace[i,0], powtrace[i,0], chi_vals[i,0] = np.nan, np.nan, np.nan, np.nan    
        elif profile == 'moffat':
            if not np.isnan(xtrace[i,0]):
                xtrace[i,0], Itrace[i,0], alphatrace[i,0], betatrace[i,0], powtrace[i,0], chi_vals[i,0] = fit_trace(xtrace[i,0],y,image,rn,form=profile)
            else:
                Itrace[i,0], alphatrace[i,0], betatrace[i,0], powtrace[i,0], chi_vals[i,0] = np.nan, np.nan, np.nan, np.nan, np.nan
        elif profile == 'bspline':
            if not np.isnan(xtrace[i,0]):
                xtrace[i,0], chi_vals[i,0] = fit_trace(xtrace[i,0],y,image,rn,form=profile, model=spline_models[:,i,0])
            else:
                xtrace[i,0], chi_vals[i,0] = np.nan, np.nan
#    plt.plot(ytrace[0,:],xtrace[0,:])
    for i in range(1,len(yvals)):
        sys.stdout.write("\r{:6.3f}%".format(i/len(yvals)*100))
        sys.stdout.flush()
        y = yvals[i]
        crsxn = image[:,y]
        ytrace[:,i] = y
        for j in range(num_fibers):
#            if j != itest:
#                continue
            if profile == 'gaussian' or profile == 'gauss_lor':
                if not np.isnan(xtrace[j,i-1]):
                    ### Add approximate shifts to account for curvature
                    ### Doesn't need to be precise, but I haven't tested at all num_points
                    if i < 400:
                        shift = int(6*(1-np.sqrt(num_points/ypix)))
                    elif 400 < i and i < 1000:
                        shift = int(3*(1-np.sqrt(num_points/ypix)))
                    else:
                        shift = int(1*(1-np.sqrt(num_points/ypix)))
                    #set boundaries
                    lb = int(xtrace[j,i-1]-fiber_space/2)+shift
                    ub = int(xtrace[j,i-1]+fiber_space/2)+shift
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
                            xtrace[j,i], Itrace[j,i], sigtrace[j,i], powtrace[j,i], chi_vals[j,i] = fit_trace(xtrace[j,i],y,image,rn,form=profile)
                            if xtrace[j, i] - xtrace[j, i-1] < -2:
                                xtrace[j,i], Itrace[j,i], sigtrace[j,i], powtrace[j,i], chi_vals[j,i] = np.nan, np.nan, np.nan, np.nan, np.nan
                        except RuntimeError:
                            xtrace[j,i], Itrace[j,i], sigtrace[j,i], powtrace[j,i], chi_vals[j,i] = np.nan, np.nan, np.nan, np.nan, np.nan
                    else:
                        xtrace[j,i], Itrace[j,i], sigtrace[j,i], powtrace[j,i], chi_vals[j,i] = np.nan, np.nan, np.nan, np.nan, np.nan
                else:
                    xtrace[j,i], Itrace[j,i], sigtrace[j,i], powtrace[j,i], chi_vals[j,i] = np.nan, np.nan, np.nan, np.nan, np.nan
    #            plt.plot(ytrace[j,:],xtrace[j,:])
            elif profile == 'moffat':
                if not np.isnan(xtrace[j,i-1]):
                    ### Add approximate shifts to account for curvature
                    ### Doesn't need to be precise, but I haven't tested at all num_points
                    if i < 400:
                        shift = int(6*(1-np.sqrt(num_points/ypix)))
                    elif 400 < i and i < 1000:
                        shift = int(3*(1-np.sqrt(num_points/ypix)))
                    else:
                        shift = int(1*(1-np.sqrt(num_points/ypix)))
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
                    if lb >= ub:
                        pk = 0
                    else:
                        pk = np.max(xregion)
                    if pk>bg_cutoff:
                        #estimate of trace position based on tallest peak
                        xtrace[j,i] = np.argmax(xregion)+lb
                        #quadratic fit for sub-pixel precision
                        try:
                            xtrace[j,i], Itrace[j,i], alphatrace[j,i], betatrace[j,i], powtrace[j,i], chi_vals[j,i] = fit_trace(xtrace[j,i],y,image,rn,form=profile)
                            if xtrace[j, i] - xtrace[j, i-1] < -0.5:
                                xtrace[j,i], Itrace[j,i], alphatrace[j,i], betatrace[j,i], powtrace[j,i], chi_vals[j,i] = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
                        except RuntimeError:
                            xtrace[j,i], Itrace[j,i], alphatrace[j,i], betatrace[j,i], powtrace[j,i], chi_vals[j,i] = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
                    else:
                        xtrace[j,i], Itrace[j,i], alphatrace[j,i], betatrace[j,i], powtrace[j,i], chi_vals[j,i] = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
                else:
                    xtrace[j,i], Itrace[j,i], alphatrace[j,i], betatrace[j,i], powtrace[j,i], chi_vals[j,i] = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    #            plt.plot(ytrace[j,:],xtrace[j,:])
            elif profile == 'bspline':
                if not np.isnan(xtrace[j,i-1]):
                    ### Add approximate shifts to account for curvature
                    ### Doesn't need to be precise, but I haven't tested at all num_points
                    if i < 400:
                        shift = int(6*(1-np.sqrt(num_points/ypix)))
                    elif 400 < i and i < 1000:
                        shift = int(3*(1-np.sqrt(num_points/ypix)))
                    else:
                        shift = int(1*(1-np.sqrt(num_points/ypix)))
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
                    if lb >= ub:
                        pk = 0
                    else:
                        pk = np.max(xregion)
                    if pk>bg_cutoff:
                        xtrace[j,i] = np.argmax(xregion)+lb
                        try:
                            xtrace[j,i], chi_vals[j,i] = fit_trace(xtrace[j,i],y,image,rn,form=profile,model=spline_models[:,j,i])
                        except RuntimeError:
                            xtrace[j,i], chi_vals[j,i] = np.nan, np.nan
                else:
                    xtrace[j,i], chi_vals[j,i] = np.nan, np.nan
    sys.stdout.write("\n")

    ### Check chi2
    chir = np.ravel(chi_vals)#/7#(fiber_space+1-5) #[itest,:]
    print np.nanmedian(chir)
    msk = (abs(chir) < 100)*(~np.isnan(chir))*(chir > 0)
    if profile == 'gaussian':
        print np.nanmean(sigtrace)
    print np.sum(msk)/len(chir)*100, "% good"
    plt.hist(chir[msk], bins=50)                
    plt.show()
    plt.close()
#    print "Plotting sigmas"
#    plt.plot(sigtrace[14,:])
#    plt.show()
#    print "Plotting powers"
#    plt.plot(powtrace[14,:])
#    plt.show()
                
    #Finally fit x vs. y on traces.  Start with quadratic for simple + close enough
    t_coeffs = np.zeros((pord+1,num_fibers))
    i_coeffs = np.zeros((pord+1,num_fibers))
    p_coeffs = np.zeros((pord+1,num_fibers))
    if profile == 'gaussian':
        s_coeffs = np.zeros((pord+1,num_fibers))
    elif profile == 'moffat':
        a_coeffs = np.zeros((pord+1,num_fibers))
        b_coeffs = np.zeros((pord+1,num_fibers))
    xtrace -= 2  #Manually added to fit n20161123 to n201703** and on
    for i in range(num_fibers):
        #Given orientation makes more sense to swap x/y
        mask = ~np.isnan(xtrace[i,:])
        yf = (ytrace[i,:]-ypix/2)/ypix
        if len(xtrace[i,:][mask])>pord+1:
            t_coeffs[:,i] = np.polyfit(yf[mask], xtrace[i,:][mask], pord)
            if profile != 'bspline':
                i_coeffs[:,i] = np.polyfit(yf[mask], Itrace[i,:][mask], pord)
                p_coeffs[:,i] = np.polyfit(yf[mask], powtrace[i,:][mask], pord)
            if profile == 'gaussian':
                s_coeffs[:,i] = np.polyfit(yf[mask], sigtrace[i,:][mask], pord)
            elif profile == 'moffat':
                a_coeffs[:,i] = np.polyfit(yf[mask], alphatrace[i,:][mask], pord)
                b_coeffs[:,i] = np.polyfit(yf[mask], betatrace[i,:][mask], pord)
            if i == itest and profile == 'gaussian':
                print "plotting center fit"
                plt.plot(yf[mask], xtrace[i,:][mask], yf[mask], np.poly1d(t_coeffs[:,i])(yf[mask]))
                plt.show()
                plt.close()
                print "plotting sigma fit"
                plt.plot(yf[mask], sigtrace[i,:][mask], yf[mask], np.poly1d(s_coeffs[:,i])(yf[mask]))
                plt.show()
                plt.close()
                print "plotting power fit"
                plt.plot(yf[mask], powtrace[i,:][mask], yf[mask], np.poly1d(p_coeffs[:,i])(yf[mask]))
                plt.show()
                plt.close()
        else:
            t_coeffs[:,i] = np.nan*np.ones((pord+1))
            if profile != 'bspline':
                i_coeffs[:,i] = np.nan*np.ones((pord+1))
                p_coeffs[:,i] = np.nan*np.ones((pord+1))
            if profile == 'gaussian':
                s_coeffs[:,i] = np.nan*np.ones((pord+1))
            elif profile == 'moffat':
                a_coeffs[:,i] = np.nan*np.ones((pord+1))
                b_coeffs[:,i] = np.nan*np.ones((pord+1))
        
    ### Uncomment below to see plot of traces
    plt.figure('inside find_trace_coeffs')
    plt.imshow(np.log(image),interpolation='none')
    for i in range(num_fibers):
        ys = (np.arange(ypix)-ypix/2)/ypix
        xs = np.poly1d(t_coeffs[:,i])(ys)
        yp = np.arange(ypix)
        plt.plot(yp,xs, 'b', linewidth=2)
    plt.show()
    plt.close()  
#
#    plt.plot(image[:,100], linewidth=2)
#    for i in range(num_fibers):
#        ys = (100-ypix/2)/ypix
#        xs = np.poly1d(t_coeffs[:,i])(ys)
#        yy = np.array((0, 20000))
#        xx = np.array((xs, xs))
#        plt.plot(xx, yy, 'k', linewidth=2)
#    plt.show()
#    plt.close()
        
    if return_all_coeffs:
        if profile == 'gaussian':
            return t_coeffs, i_coeffs, s_coeffs, p_coeffs
        elif profile == 'moffat':
            return t_coeffs, i_coeffs, a_coeffs, b_coeffs, p_coeffs
        elif profile == 'bspline':
            return t_coeffs
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
    ccd_mask = (ccd < cut)*(ccd > -cut)
    masked_ccd = ccd[ccd_mask]
    if ccd.size < 2:
        print("Cut too aggressive - no points left in ccd")
        exit(0)
#    arr = plt.hist(masked_ccd,2*(cut-1))
#    hgt = arr[0]
#    xvl = arr[1][:-1]
    hgt, xvls = np.histogram(masked_ccd,max(2*(int(cut)-1),20))
    xvl = xvls[:-1]
    ### Assume lower tail is a better indicator than upper tail
    xmsk = (xvl < np.median(masked_ccd))
    hgts = hgt[xmsk]
    xvls = xvl[xmsk]
    sig_est = 2/2.35*(xvls[np.argmax(hgts)] - xvls[np.argmax(hgts>np.max(hgts)/2)])
    pguess = (sig_est,np.median(masked_ccd),np.max(hgt))
    sigma = 1/np.sqrt(abs(hgts)+1)
    try:
        params, errarr = opt.curve_fit(sf.gaussian,xvls,hgts,p0=pguess,sigma=sigma)
        if plot:
            plt.hist(masked_ccd,max(2*(int(cut)-1),20))
            plt.title("Number of pixels with certain count value")
            htst = sf.gaussian(xvl, params[0], center=params[1], height=params[2],bg_mean=0,bg_slope=0,power=2)
            plt.plot(xvl,htst)
            plt.show()
            plt.close()
    except:
        ### Don't subtract any background and use spec readnoise
        params = np.zeros(2)
        params[1] = 0
        params[0] = 3.63
    ccd_new = ccd - params[1] # mean
    bg_std = params[0]
    return ccd_new, bg_std

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

def linear_mn_hght_bg(xvals,yvals,invals,sigma,mn_est,power=2,beta=2,profile='gaussian'):
    """ Find mean of gaussian with linearized procedure.  Use eqn
        dx = dh/h * sigma/2 * exp(1/2) where dh = max - min residual
        Once mean is found, determines best fit hght and bg
    """
#    gauss_model = sf.gaussian(xvals,sigma,center=mn_est,height=np.max(yvals),power=power)
    mn_est_err = 100
    mn_est_err_old = 0
    loop_ct = 0
    hght = np.max(yvals)
    mn_std = sigma#/3
    bg = np.min(yvals)
    while abs(mn_est_err-mn_est_err_old)>0.01 and loop_ct < 1:
        mn_est_err_old = np.copy(mn_est_err)
        mn_est_old = np.copy(mn_est)
        if profile == 'gaussian':
#            gauss_model = sf.gaussian(xvals,sigma,center=mn_est,height=hght,bg_mean=bg,power=power)
#            plt.plot(xvals, yvals, xvals, gauss_model)
#            plt.show()
#            plt.close()
            mn_est, mn_std = sf.best_mean(xvals,sigma,mn_est,hght,bg,yvals,invals,mn_std,power=power)
            hght, bg = sf.best_linear_gauss(xvals,sigma,mn_est,yvals,invals,power=power)
        elif profile == 'moffat':
            # TODO - finish this!
            mn_est, hght, bg = 1, 1, 1
        mn_est_err = abs(mn_est_old - mn_est)
        loop_ct += 1
    return mn_est, hght, bg
    
def fit_mn_hght_bg(xvals,zorig,invorig,sigj,mn_new,spread,powj=2):
    """ Fits mean, height, and background for a gaussian of known sigma.
        Height and background are fit linearly.  Mean is fit through a grid
        search algorithm (may be better to change to a nonlinear fitter?)
    """
#    ts = time.time()
#    mn_old = -100
#    lp_ct = 0
    params = lmfit.Parameters()
    params.add('hght', value=130000)#np.max(zorig))
    params.add('mean', value=-0.2)#mn_new)
    params.add('bg', value=0)
    params.add('sigma', value=sigj, vary=0)
    params.add('power', value=powj, vary=0)
    args = (xvals, zorig, invorig)
    result = lmfit.minimize(sf.gauss_residual, params, args=args)
#    gauss_residual(params,xvals,zvals,invals):
#    while abs(mn_new-mn_old)>0.001:
#        mn_old = 1.0*mn_new
##        t1 = time.time()
#        hght, bg = sf.best_linear_gauss(xvals,sigj,mn_old,zorig,invorig,power=powj)
##        t2 = time.time()
#        print mn_new
#        mn_new, mn_new_std = sf.best_mean(xvals,sigj,mn_old,hght,bg,zorig,invorig,spread,power=powj)
#        print mn_new
#        fit = sf.gaussian(xvals, sigj, center=mn_new, height=hght, power=powj) + bg
#        print lp_ct
#        print mn_new-mn_old
#        plt.plot(xvals, zorig, xvals, fit)
#        plt.show()
#        plt.close()
##        t3 = time.time()
##        print("Linear time = {}s".format(t2-t1))
##        print("Nonlinear time = {}s".format(t3-t2))
##        time.sleep(5)
##        mn_new = 1.0*params['mn'].value
#        lp_ct+=1
#        if lp_ct>1e3: break
#    print "Loop count is ", lp_ct
#    print("Len xvals = {}".format(len(xvals)))
#    te = time.time()
#    print("Total fit time is {}s".format(te-ts))
#    time.sleep(5)
    paramsf = result.params
    mn_new = paramsf['mean'].value
    hght = paramsf['hght'].value
    bg = paramsf['bg'].value
#    fit = sf.gaussian(xvals, sigj, center=mn_new, height=hght, power=powj) + bg
    return mn_new, hght,bg
    
def small_moffat_fit(xvals, zorig, invorig, xc, alphaj, betaj, powj,dx=1):
    def moff_res(params, xarr, yarr, invar):
        model = sf.moffat_lmfit(params, xarr)
        return (model-yarr)**2*invar
    params0 = lmfit.Parameters()
    params0.add('xc', value = xc, min=xc-dx, max=xc+dx)
    params0.add('alpha', value = alphaj, vary=0)
    params0.add('beta', value = betaj, vary=0)
    params0.add('bg', value = np.min(zorig))
    params0.add('power', value = powj, vary=0)
    params0.add('hght', value = (np.max(zorig)-np.min(zorig))*5)
    args = (xvals, zorig, invorig)
    results = lmfit.minimize(moff_res,params0,args=args)
    params = results.params
    mn_new = params['xc'].value
    hght = params['hght'].value
    bg = params['bg'].value
    return mn_new, hght, bg, params
    
def moffat_linear_fit(xvals, zorig, invorig, xc, alphaj, betaj, powj,dx=1):
    params0 = lmfit.Parameters()
    params0.add('xc', value = xc)
    params0.add('alpha', value = alphaj)
    params0.add('beta', value = betaj)
    params0.add('bg', value = 0)
    params0.add('power', value = powj)
    params0.add('hght', value = 1)
    model = sf.moffat_lmfit(params0, xvals)
    profile = np.vstack((model,np.ones(len(model)))).T
    noise = np.diag(invorig)
    coeffs, chi = sf.chi_fit(zorig, profile, noise)
    params0['hght'].value = coeffs[0]
    params0['bg'].value = coeffs[1]
    ### coeffs are height, background
    return coeffs[0], coeffs[1], params0
    
def extract_1D(ccd, norm_sflat, multi_coeffs, form, readnoise=1, gain=1, return_model=False, verbose=False, boxcar=False):
    """ Function to extract using optimal extraction method.
        This could benefit from a lot of cleaning up
        INPUTS:
        ccd - ccd image to extract
        norm_sflat - slit flat averages for normalization
        multi_coeffs - trace position, sigma, power, intensity, polynomial coefficients
        form - profile shape to use in optimal extraction (gaussian or moffat)
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
    num_fibers = np.shape(multi_coeffs[0])[1]
    
    ####################################################
    ###   Get coefficients for profile fitting   #######
    ####################################################
    if form == 'gaussian':
        t_coeffs = multi_coeffs[0]
        i_coeffs = multi_coeffs[1]
        s_coeffs = multi_coeffs[2]
        p_coeffs = multi_coeffs[3]
    elif form == 'moffat':
        t_coeffs = multi_coeffs[0]
        i_coeffs = multi_coeffs[1]
        a_coeffs = multi_coeffs[2]
        b_coeffs = multi_coeffs[3]
        p_coeffs = multi_coeffs[4]

    ####################################################    
    #####   First refine horizontal centers (fit   #####
    #####   traces from data ccd using fiber flat  #####
    #####   as initial estimate)                   #####
    ####################################################
#    t_coeffs_ccd = refine_trace_centers(ccd, t_coeffs, i_coeffs, s_coeffs, p_coeffs, fact=10, readnoise=readnoise, verbose=True)
    ###t_coeffs_ccd = refine_trace_centers(ccd, norm_sflat, t_coeffs, i_coeffs, s_coeffs, p_coeffs, a_coeffs=None, b_coeffs=None, fact=10, readnoise=readnoise, form='gaussian', verbose=True, plot_results=False)
    #'''
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
        for j in range(0,hpix,fact):
            ### set coordinates, gaussian parameters from coeffs
            jadj = int(np.floor(j/fact))
            yj = (j-hpix/2)/hpix
            yc_ccd[i,jadj] = j
            xc = np.poly1d(t_coeffs[:,i])(yj) + 2
            powj = np.poly1d(p_coeffs[:,i])(yj)
            if form == 'gaussian':
                sigj = np.poly1d(s_coeffs[:,i])(yj)
            elif form == 'moffat':
                alphaj = np.poly1d(a_coeffs[:,i])(yj)
                betaj = np.poly1d(b_coeffs[:,i])(yj)
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
                zorig = ccd[xj+xvals,j]
                flt = norm_sflat[xj+xvals,j]
                ### If too short slice, don't try to fit
                if len(zorig)<3:
                    xc_ccd[i,jadj] = np.nan
                    inv_chi[i,jadj] = 0
                    continue
                invorig = 1/(abs(zorig)/flt + (readnoise/flt)**2)
                ### Don't try to fit profile for very low SNR peaks
                if np.max(zorig)<20:
                    xc_ccd[i,jadj] = np.nan
                    inv_chi[i,jadj] = 0
                else:
                    ### Fit for center (mn_new), amongst other values
                    if form == 'gaussian':
    #                    mn_new, hght, bg = fit_mn_hght_bg(xvals,zorig,invorig,sigj,xc-xj-1,sigj,powj=powj)
                        mn_new, hght, bg = linear_mn_hght_bg(xvals,zorig,invorig,sigj,xc-xj-1,power=powj)
                        fitorig = sf.gaussian(xvals,sigj,mn_new,hght,power=powj)
                    elif form == 'moffat':
                        mn_new, hght, bg, params = small_moffat_fit(xvals, zorig, invorig, xc-xj-1, alphaj, betaj, powj, dx=1.5)
                        fitorig = sf.moffat_lmfit(params, xvals)
                    inv_chi[i,jadj] = 1/sum((zorig-bg-fitorig)**2*invorig)
                    ### Shift from relative to absolute center
                    xc_ccd[i,jadj] = mn_new+xj+1
              
    #####################################################
    #### Now with new centers, refit trace coefficients #
    #####################################################
    tmp_poly_ord = 6  ### Use a higher order for a closer fit over entire trace
    hscl = (np.arange(hpix)-hpix/2)/hpix
    t_coeffs_ccd = np.zeros((tmp_poly_ord+1,num_fibers))
    for i in range(num_fibers):
        #Given orientation makes more sense to swap x/y
        mask = ~np.isnan(xc_ccd[i,:])*(~np.isnan(inv_chi[i,:])) ### Mask bad points
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
            ### if not enough points to fit, use original trace
#            tmp_coeffs = np.nan*np.ones((tmp_poly_ord+1))
            tmp_coeffs = np.pad(t_coeffs[:,i], ((tmp_poly_ord-2,0)), mode='constant')
        ### Add quality check to prevent wild solutions:
        err_max = 2 #pixels
        ff_trace = np.poly1d(t_coeffs[:,i])(hscl)
        ccd_trace = np.poly1d(tmp_coeffs)(hscl)
        if np.max(abs(ff_trace-ccd_trace)) > err_max:
            tmp_coeffs = np.pad(t_coeffs[:,i], ((tmp_poly_ord-2,0)), mode='constant')
#            if verbose:
#                print("Ignoring bad fit for fiber {}".format(i))
        t_coeffs_ccd[:,i] = tmp_coeffs

    tb = time.time() ### Start time of extraction/end of trace refinement
    if verbose:
        print("Trace refinement time = {}s".format(tb-ta))
       
    ### Uncomment below to see plot of traces
#    plt.figure('inside extract 1d')
#    plt.imshow(np.log(ccd),interpolation='none')
#    for i in range(num_fibers):
#            ys = (np.arange(hpix)-hpix/2)/hpix
#            xs = np.poly1d(t_coeffs_ccd[:,i])(ys)
#            yp = np.arange(hpix)
#            plt.plot(yp,xs, 'b', linewidth=2)
#    plt.show()
#    plt.close()
       
#    plt.plot(ccd[:,100], linewidth=2)
#    for i in range(num_fibers):
#        ys = (100-hpix/2)/hpix
#        xs = np.poly1d(t_coeffs_ccd[:,i])(ys)
#        yy = np.array((0, 10000))
#        xx = np.array((xs, xs))
#        plt.plot(xx, yy, 'k', linewidth=2)
#    plt.show()
#    plt.close()
      #'''
    
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
        if i == 0 or i == 112:
            #First fiber in n20161123 set is not available
            #Fiber 112 isn't entirely on frame and trace isn't reliable
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
            powj = np.poly1d(p_coeffs[:,i])(yj)
#            Ij = i_coeffs[2,i]*yj**2+i_coeffs[1,i]*yj+i_coeffs[0,i]
            if form == 'gaussian':
                sigj = np.poly1d(s_coeffs[:,i])(yj)
            elif form == 'moffat':
                alphaj = np.poly1d(a_coeffs[:,i])(yj)
                betaj = np.poly1d(b_coeffs[:,i])(yj)
            ### If trace center is undefined mask the point
            if np.isnan(xc):
                spec_mask[i,j] = False
            else:
                ### Set values to use in extraction
                xpad = 5  ### can't be too big or traces start to overlap
                xvals = np.arange(-xpad,xpad+1)
                xj = int(xc)
                xwindow = xj+xvals
                xvals = xvals[(xwindow>=0)*(xwindow<vpix)]
                zorig = ccd[xj+xvals, j]
                flt = norm_sflat[xj+xvals,j]
                ### If too short, don't fit, mask point
                if len(zorig)<(xpad-1):
                    spec[i,j] = 0
                    spec_mask[i,j] = False
                    continue
                invorig = 1/(abs(zorig)/flt + (readnoise/flt)**2)
                ### don't try to extract for very low signal
                if boxcar:
                    spec[i,j] = np.sum(zorig)
                    spec_invar[i,j] = 1/np.sum(1/invorig)
                    continue
                if np.max(zorig)<20:
                    continue
                else:
                    ### Do nonlinear fit for center, height, and background
                    ### Use fitted values to make best fit arrays
                    if form == 'gaussian':
                        mn_new, hght, bg = fit_mn_hght_bg(xvals,zorig,invorig,sigj,xc-xj-1,sigj/8,powj=powj)
#                        mn_new, hght, bg = linear_mn_hght_bg(xvals,zorig,invorig,sigj,xc-xj-1,power=powj)
                        fitorig = sf.gaussian(xvals,sigj,mn_new,hght,power=powj)
                        xprecise = np.linspace(xvals[0],xvals[-1],100)
                        fitprecise = sf.gaussian(xprecise,sigj,mn_new,hght,power=powj)
                    elif form == 'moffat':
#                        mn_new, hght, bg, params = small_moffat_fit(xvals, zorig, invorig, xc-xj-1, alphaj, betaj, powj, dx=0.3)
                        hght, bg, params = moffat_linear_fit(xvals, zorig, invorig, xc-xj-1, alphaj, betaj, powj)
                        fitorig = sf.moffat_lmfit(params, xvals)
                        xprecise = np.linspace(xvals[0],xvals[-1],100)
                        fitprecise = sf.moffat_lmfit(params, xprecise)
                    ftmp = sum(fitprecise)*np.mean(np.ediff1d(xprecise))
                    #Following if/else handles failure to fit
                    if ftmp==0:
                        fitnorm = np.zeros(len(zorig))
                    else:
                        fitnorm = fitorig/ftmp
                    ### Get extracted flux and error
                    fstd = sum(fitnorm*zorig*invorig)/sum(fitnorm**2*invorig)
                    invorig = 1/((readnoise/flt)**2 + abs(fstd*fitnorm)/flt)
                    chi2red = np.sum((fstd*fitnorm+bg-zorig)**2*invorig)/(len(zorig)-5)
                    ### Now set up to do cosmic ray rejection
                    rej_min = 1
                    loop_count=0
                    while rej_min==0:
                        pixel_reject = cosmic_ray_reject(zorig,fstd,fitnorm,invorig,S=bg,threshhold=0.25*np.mean(zorig),verbose=True)
                        rej_min = np.min(pixel_reject)
                        ### Once no pixels are rejected, re-find extracted flux
                        if rej_min==0:
                            ### re-index arrays to remove rejected points
                            zorig = zorig[pixel_reject==1]
                            invorig = invorig[pixel_reject==1]
                            xvals = xvals[pixel_reject==1]
                            flt = flt[pixel_reject==1]
                            ### re-do fit (can later cast this into a separate function)
                            if form == 'gaussian':
                                mn_new, hght, bg = fit_mn_hght_bg(xvals,zorig,invorig,sigj,xc-xj-1,sigj/8,powj=powj)
    #                            mn_new, hght, bg = linear_mn_hght_bg(xvals,zorig,invorig,sigj,xc-xj-1,power=powj)
                                fitorig = sf.gaussian(xvals,sigj,mn_new,hght,power=powj)
                                xprecise = np.linspace(xvals[0],xvals[-1],100)
                                fitprecise = sf.gaussian(xprecise,sigj,mn_new,hght,power=powj)
                            elif form == 'moffat':
#                                mn_new, hght, bg, params = small_moffat_fit(xvals, zorig, invorig, xc-xj-1, alphaj, betaj, powj, dx=0.3)
                                hght, bg, params = moffat_linear_fit(xvals, zorig, invorig, xc-xj-1, alphaj, betaj, powj)
                                fitorig = sf.moffat_lmfit(params, xvals)
                                xprecise = np.linspace(xvals[0],xvals[-1],100)
                                fitorig = sf.moffat_lmfit(params, xprecise)
                            ftmp = sum(fitprecise)*np.mean(np.ediff1d(xprecise))
                            fitnorm = fitorig/ftmp
                            fstd = sum(fitnorm*zorig*invorig)/sum(fitnorm**2*invorig)
                            invorig = 1/((readnoise/flt)**2 + abs(fstd*fitnorm)/flt)
                            chi2red = np.sum((fstd*fitnorm+bg-zorig)**2*invorig)/(len(zorig)-5)
                        ### if more than 3 points are rejected, mask the extracted flux
                        if loop_count>3:
                            spec_mask[i,j] = False
                            break
                        loop_count+=1
#                    dinp, = plt.plot(xvals, zorig*2.2, 'k', linewidth = 2, label='Data')
#                    dfit, = plt.plot(xvals, fitorig*2.2, 'b--', linewidth = 2, label='Fit')
#                    ax = plt.gca()
#                    ax.set_xlabel('Pixels (relative)', fontsize=24)
#                    ax.set_ylabel('Counts', fontsize=24)
#                    plt.legend(fontsize=24)
#                    plt.show()
#                    plt.close()
                    ### Set extracted spectrum value, inverse variance
                    spec[i,j] = fstd
                    spec_invar[i,j] = sum(fitnorm**2*invorig)
                    chi2red_array[i,j] = chi2red
#                    print chi2red
#                    if chi2red > 10:
#                        plt.plot(xvals,zorig, xprecise, fitprecise+bg)
#                        plt.show()
#                        plt.close()
#                    plt.plot(xvals,(fstd*fitnorm+bg-zorig)**2*invorig)
#                    plt.show()
#                    plt.close()
                    if return_model and not np.isnan(fstd):
                        ### Build model, if desired
                        image_model[xj+xvals,j] += (fstd*fitnorm+bg)#/gain
            ### If a nan came out of the above routine, zero it and mask
            if np.isnan(spec[i,j][spec_mask[i,j]]):
                spec[i,j] = 0
                spec_mask[i,j] = False
#        plt.plot(spec[i,:])
#        plt.show()
#        plt.close()
#        if verbose:
#            print(" ")
    if verbose:
#        chi2red = np.median(chi2red_array[i])
        print("Median reduced chi^2 = {}".format(np.nanmedian(chi2red_array)))
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

def update_trace_coeffs(ccd, xc_ccd, yc_ccd, inv_chi, hpix, pord, num_fibers, t_coeffs, verbose=False, plot_results=False):
    #####################################################
    #### Now with new centers, refit trace coefficients #
    #####################################################
    tmp_poly_ord = pord  ### Use a higher order for a closer fit over entire trace
    mpix = 2048
    hscl = (np.arange(hpix)+yc_ccd[0,0]-mpix/2)/mpix
    t_coeffs_ccd = np.zeros((tmp_poly_ord+1,num_fibers))
    for i in range(num_fibers):
        #Given orientation makes more sense to swap x/y
        mask = ~np.isnan(xc_ccd[i,:])*(~np.isnan(inv_chi[i,:])) ### Mask bad points
        ### build profile matrix over good points
#        profile = np.ones((len(yc_ccd[i,:][mask]),tmp_poly_ord+1))
#        print (yc_ccd[i,:][mask]-mpix/2)/mpix
#        for order in range(tmp_poly_ord):
#            profile[:,order+1] = ((yc_ccd[i,:][mask]-mpix/2)/mpix)**(order+1)
#        profile = profile[:,::-1]
#        noise = np.diag(inv_chi[i,:][mask])
        if len(xc_ccd[i,:][mask])>(tmp_poly_ord+1):
            ### Chi^2 fit
#            tmp_coeffs, junk = sf.chi_fit(xc_ccd[i,:][mask],profile,noise)
            yscl = (yc_ccd[i,:][mask]-mpix/2)/mpix
            tmp_coeffs = np.polyfit(yscl, xc_ccd[i,:][mask], tmp_poly_ord, w=inv_chi[i,:][mask])
        else:
            ### if not enough points to fit, use original trace
#            tmp_coeffs = np.nan*np.ones((tmp_poly_ord+1))
            tmp_coeffs = np.pad(t_coeffs[:,i], ((tmp_poly_ord-2,0)), mode='constant')
        ### Add quality check to prevent wild solutions:
        err_max = 2 #pixels
        ff_trace = np.poly1d(t_coeffs[:,i])(hscl)
        ccd_trace = np.poly1d(tmp_coeffs)(hscl)
        if np.max(abs(ff_trace-ccd_trace)) > err_max:
            tmp_coeffs = np.pad(t_coeffs[:,i], ((tmp_poly_ord-2,0)), mode='constant')
#            if verbose:
#                print("Ignoring bad fit for fiber {}".format(i))
        t_coeffs_ccd[:,i] = tmp_coeffs
       
    if plot_results:
        plt.figure('inside extract 1d')
        plt.imshow(np.log(ccd),interpolation='none')
        for i in range(num_fibers):
                ys = (np.arange(hpix)+yc_ccd[0,0]-mpix/2)/mpix
                xs = np.poly1d(t_coeffs_ccd[:,i])(ys)
                yp = np.arange(hpix)
                plt.plot(yp,xs, 'b', linewidth=2)
        plt.show()
        plt.close()
        ### Following lines plot deltas at each point vs an arbitrary ccd cross-section
#        plt.plot(ccd[:,100], linewidth=2)
#        for i in range(num_fibers):
#            ys = (100-hpix/2)/hpix
#            xs = np.poly1d(t_coeffs_ccd[:,i])(ys)
#            yy = np.array((0, 10000))
#            xx = np.array((xs, xs))
#            plt.plot(xx, yy, 'k', linewidth=2)
#        plt.show()
#        plt.close()
    return t_coeffs_ccd
    
def refine_trace_centers(ccd, norm_sflat, t_coeffs, i_coeffs, s_coeffs, p_coeffs, a_coeffs=None, b_coeffs=None, pord=6, fact=10, readnoise=3.63, form='gaussian', verbose=False, plot_results=False):
    """ Uses estimated centers from fibers flats as starting point, then
        fits from there to find traces based on science ccd frame.
        INPUTS:
            ccd - image on which to fit traces
            t/i/s/p_coeffs - modified gaussian coefficients from fiberflat
            fact - do 1/fact of the available points
    """
    vpix, hpix = ccd.shape[0], ccd.shape[1]
    num_fibers = t_coeffs.shape[1]
    ta = time.time()  ### Start time of trace refinement
#    fact = 20 #do 1/fact * available points
    ### Empty arrays
    rough_pts = int(np.ceil(hpix/fact))
    xc_ccd = np.zeros((num_fibers,rough_pts))
    yc_ccd = np.zeros((num_fibers,rough_pts))
    inv_chi = np.zeros((num_fibers,rough_pts))
    if verbose:
        print("Refining trace centers")
    for i in range(num_fibers):
        for j in range(0,hpix,fact):
            ### set coordinates, gaussian parameters from coeffs
            jadj = int(np.floor(j/fact))
            yj = (j-hpix/2)/hpix
            yc_ccd[i,jadj] = j
            xc = np.poly1d(t_coeffs[:,i])(yj) + 2
            powj = np.poly1d(p_coeffs[:,i])(yj)
            if form == 'gaussian':
                sigj = np.poly1d(s_coeffs[:,i])(yj)
            elif form == 'moffat':
                alphaj = np.poly1d(a_coeffs[:,i])(yj)
                betaj = np.poly1d(b_coeffs[:,i])(yj)
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
                zorig = ccd[xj+xvals,j]
                flt = norm_sflat[xj+xvals,j]
                ### If too short slice, don't try to fit
                if len(zorig)<3:
                    xc_ccd[i,jadj] = np.nan
                    inv_chi[i,jadj] = 0
                    continue
                invorig = 1/(abs(zorig)/flt + (readnoise/flt)**2)
                ### Don't try to fit profile for very low SNR peaks
                if np.max(zorig)<20:
                    xc_ccd[i,jadj] = np.nan
                    inv_chi[i,jadj] = 0
                else:
                    ### Fit for center (mn_new), amongst other values
                    if form == 'gaussian':
    #                    mn_new, hght, bg = fit_mn_hght_bg(xvals,zorig,invorig,sigj,xc-xj-1,sigj,powj=powj)
                        mn_new, hght, bg = linear_mn_hght_bg(xvals,zorig,invorig,sigj,xc-xj-1,power=powj)
                        fitorig = sf.gaussian(xvals,sigj,mn_new,hght,power=powj)
                    elif form == 'moffat':
                        mn_new, hght, bg, params = small_moffat_fit(xvals, zorig, invorig, xc-xj-1, alphaj, betaj, powj, dx=1.5)
                        fitorig = sf.moffat_lmfit(params, xvals)
                    inv_chi[i,jadj] = 1/sum((zorig-bg-fitorig)**2*invorig)
                    ### Shift from relative to absolute center
                    xc_ccd[i,jadj] = mn_new+xj+1
              
    #####################################################
    #### Now with new centers, refit trace coefficients #
    #####################################################
    t_coeffs_ccd = update_trace_coeffs(ccd, xc_ccd, yc_ccd, inv_chi, hpix, pord, num_fibers, t_coeffs, verbose=verbose, plot_results=plot_results)
    tb = time.time() ### Start time of extraction/end of trace refinement
    if verbose:
        print("Trace refinement time = {}s".format(tb-ta))
    '''
    tmp_poly_ord = pord  ### Use a higher order for a closer fit over entire trace
    hscl = (np.arange(hpix)-hpix/2)/hpix
    t_coeffs_ccd = np.zeros((tmp_poly_ord+1,num_fibers))
    for i in range(num_fibers):
        #Given orientation makes more sense to swap x/y
        mask = ~np.isnan(xc_ccd[i,:])*(~np.isnan(inv_chi[i,:])) ### Mask bad points
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
            ### if not enough points to fit, use original trace
#            tmp_coeffs = np.nan*np.ones((tmp_poly_ord+1))
            tmp_coeffs = np.pad(t_coeffs[:,i], ((tmp_poly_ord-2,0)), mode='constant')
        ### Add quality check to prevent wild solutions:
        err_max = 2 #pixels
        ff_trace = np.poly1d(t_coeffs[:,i])(hscl)
        ccd_trace = np.poly1d(tmp_coeffs)(hscl)
        if np.max(abs(ff_trace-ccd_trace)) > err_max:
            tmp_coeffs = np.pad(t_coeffs[:,i], ((tmp_poly_ord-2,0)), mode='constant')
#            if verbose:
#                print("Ignoring bad fit for fiber {}".format(i))
        t_coeffs_ccd[:,i] = tmp_coeffs

    tb = time.time() ### Start time of extraction/end of trace refinement
    if verbose:
        print("Trace refinement time = {}s".format(tb-ta))
       
    ### Uncomment below to see plot of traces
    if plot_results:
        plt.figure('inside extract 1d')
        plt.imshow(np.log(ccd),interpolation='none')
        for i in range(num_fibers):
                ys = (np.arange(hpix)-hpix/2)/hpix
                xs = np.poly1d(t_coeffs_ccd[:,i])(ys)
                yp = np.arange(hpix)
                plt.plot(yp,xs, 'b', linewidth=2)
        plt.show()
        plt.close()
           
#        plt.plot(ccd[:,100], linewidth=2)
#        for i in range(num_fibers):
#            ys = (100-hpix/2)/hpix
#            xs = np.poly1d(t_coeffs_ccd[:,i])(ys)
#            yy = np.array((0, 10000))
#            xx = np.array((xs, xs))
#            plt.plot(xx, yy, 'k', linewidth=2)
#        plt.show()
#        plt.close()
    #'''
    return t_coeffs_ccd

def refine_trace_section(ccd_sec, hsec, vmn, norm_sflat_sec, t_coeffs, s_coeffs, p_coeffs, a_coeffs=None, b_coeffs=None, pord=4, fact=5, readnoise=3.63, form='gaussian', verbose=False, plot_results=False):
    ''' Refines trace centers, but only for a small ccd section
    '''
    vpix = ccd_sec.shape[0]
    ta = time.time()  ### Start time of trace refinement
#    fact = 20 #do 1/fact * available points
    ### Empty arrays
    hpix = len(hsec)
    rough_pts = int(np.ceil(hpix/fact))
    xc_ccd = np.zeros((1,rough_pts))
    yc_ccd = np.zeros((1,rough_pts))
    inv_chi = np.zeros((1,rough_pts))
    if verbose:
        print("Refining section trace centers")
    hrng = hsec[::fact]
    for j in hrng:
        ### set coordinates, gaussian parameters from coeffs
        jadj = int(np.floor((j-hsec[0])/fact))
        yj = (j-1024)/2048
        yc_ccd[0,jadj] = j
        xc = np.poly1d(t_coeffs)(yj)
        powj = np.poly1d(p_coeffs)(yj)
        if form == 'gaussian':
            sigj = np.poly1d(s_coeffs)(yj)
        elif form == 'moffat':
            alphaj = np.poly1d(a_coeffs)(yj)
            betaj = np.poly1d(b_coeffs)(yj)
        ### Don't try to fit any bad trace sections
        if np.isnan(xc):
            xc_ccd[0,jadj] = np.nan
            inv_chi[0,jadj] = 0
        else:
            ### Take subset of ccd of interest, xpad pixels to each side of peak
            xpad = 7
            xvals = np.arange(-xpad,xpad+1)
            xj = int(xc)-vmn
            xwindow = xj+xvals
            xvals = xvals[(xwindow>=0)*(xwindow<vpix)]
            zorig = ccd_sec[xj+xvals,j-hsec[0]]
            flt = norm_sflat_sec[xj+xvals,j-hsec[0]]
            xj += vmn
            ### If too short slice, don't try to fit
            if len(zorig)<3:
                xc_ccd[0,jadj] = np.nan
                inv_chi[0,jadj] = 0
                continue
            invorig = 1/(abs(zorig)/flt + (readnoise/flt)**2)
            ### Don't try to fit profile for very low SNR peaks
            if np.max(zorig)<20:
                xc_ccd[0,jadj] = np.nan
                inv_chi[0,jadj] = 0
            else:
                ### Fit for center (mn_new), amongst other values
                if form == 'gaussian':
#                    mn_new, hght, bg = fit_mn_hght_bg(xvals,zorig,invorig,sigj,xc-xj-1,sigj,powj=powj)
                    mn_new, hght, bg = linear_mn_hght_bg(xvals,zorig,invorig,sigj,xc-xj-1,power=powj)
                    fitorig = sf.gaussian(xvals,sigj,mn_new,hght,power=powj)
                elif form == 'moffat':
                    mn_new, hght, bg, params = small_moffat_fit(xvals, zorig, invorig, xc-xj-1, alphaj, betaj, powj, dx=1.5)
                    fitorig = sf.moffat_lmfit(params, xvals)
                inv_chi[0,jadj] = 1/sum((zorig-bg-fitorig)**2*invorig)
                ### Shift from relative to absolute center
                xc_ccd[0,jadj] = mn_new+xj+1
    t_coeffs_ccd = update_trace_coeffs(ccd_sec, xc_ccd, yc_ccd, inv_chi, hpix, pord, 1, t_coeffs.reshape((t_coeffs.size,1)), verbose=verbose, plot_results=plot_results) 
    tb = time.time() ### Start time of extraction/end of trace refinement
    if verbose:
        print("Trace refinement time = {}s".format(tb-ta))
    return t_coeffs_ccd
    
def extract_2D(ccd, norm_sflat, psfmodel, t_coeffs, sig_coeffs, pow_coeffs, redux_dir, readnoise=1, gain=1, return_model=False, verbose=False):
    """ Code to perform 2D spectroperfectionism algorithm on MINERVA data.
    """
#    if psf_coeffs.ndim == 1: 
#        psf_coeffs = psf_coeffs.reshape((1,len(psf_coeffs)))
#    elif psf_coeffs.ndim != 2:
#        print("Invalid shape for psf_coeffs with ndim = {}".format(psf_coeffs.ndim))
#        exit(0)
    ### Import PSF coefficients
#    psf_coeffs = pyfits.open('/home/matt/software/minerva/redux/psf/psf_coeffs_063.fits')[0].data
    ### Set shape variables based on inputs
    num_fibers = t_coeffs.shape[1]
    hpix = ccd.shape[1]
    hscale = (np.arange(hpix)-hpix/2)/hpix
    extracted_counts = np.zeros((num_fibers,hpix))
    extracted_covar = np.zeros((num_fibers,hpix))
    raw_ex_counts = np.zeros((num_fibers,hpix))
    if return_model:
        ccd_model = np.zeros(ccd.shape)
    ### Remove CCD diffuse background - cut value matters
#    cut = np.median(np.median(ccd[ccd<np.median(ccd)]))
#    cut = readnoise
#    ccd, bg_err = remove_ccd_background(ccd,cut=cut)
    bg_err = readnoise
    ### Parameters for extraction box size - try various values
    ### For meaning, see documentation
    '''
    num_sections = 16
    len_section = 143
    fit_pad = 4
    v_pad = 7
    len_edge = fit_pad*2
    #'''
    '''
    num_sections = 11
    len_section = 208
    fit_pad = 6
    v_pad = 7
    len_edge = fit_pad*2
    #'''
    #'''
    num_sections = 22
    len_section = 116
    fit_pad = 6
    v_pad = 7
    len_edge = fit_pad*2
    #'''
    '''
    num_sections = 24
    len_section = 116
    fit_pad = 8
    v_pad = 7
    len_edge = fit_pad*2
    #'''
    Rmat = np.zeros((num_sections,len_section,len_section))
    ### iterate over all fibers
    for fib in range(num_fibers):
        if verbose:
            print("Running 2D Extraction on fiber {}".format(fib))
            tf0 = time.time()
        ### Trace parameters
#        if fib > num_fibers-1-4:
        if fib == num_fibers-1:
            ### Have a 1 fiber offset so need to skip the last
            continue
#        if fib < 3:
#            continue
        vcenters = np.poly1d(t_coeffs[:,fib+1])(hscale) #fib-4?
        #sf.eval_polynomial_coeffs(hscale,t_coeffs[:,fib+1])
        ### PSF parameters -BSPLINES ONLY
#        ellipse = psf_coeffs[fib,-6:]
#        ellipse = ellipse.reshape((2,3))
#        params = array_to_params(ellipse)
#        coeff_matrix = psf_coeffs[fib,:-6]
#        coeff_matrix = coeff_matrix.reshape((int(coeff_matrix.size/3),3))
        for sec in range(num_sections):
            tstart = time.time()
            ### Get a small section of ccd to extract
            hsec = np.arange(sec*(len_section-2*len_edge), len_section+sec*(len_section-2*len_edge))
            hmin = hsec[0]
            vcentmn = np.mean(vcenters[hsec])
            if np.isnan(vcentmn):
                continue
            vmin = max(int(vcentmn-v_pad),0)
            vmax = min(int(vcentmn+v_pad+1),ccd.shape[0])
            ccd_sec = ccd[vmin:vmax,hsec]
            ccd_sec_invar = 1/(ccd_sec + bg_err**2)
            d0 = ccd_sec.shape[0]
            d1 = ccd_sec.shape[1]
            ### Optional - test removing background again
#            ccd_sec, sec_bg_err = remove_ccd_background(ccd_sec,cut=1*bg_err, plot=True)
            ### number of wavelength points to extract, default 1/pixel
            wls = len_section
            hcents = np.linspace(hsec[0],hsec[-1],wls)
            ### Update vertical centers for a better per-section fit
            try: # -4 to fib?
                nsflat_sec = norm_sflat[vmin:vmax,hsec]
    #                print t_coeffs.shape, s_coeffs.shape, p_coeffs.shape
                t_sec_coeffs = refine_trace_section(ccd_sec, hsec, vmin, nsflat_sec, t_coeffs[:,fib+1], sig_coeffs[:,fib+1], pow_coeffs[:,fib+1], pord=6, fact=1, readnoise=readnoise, form='gaussian', verbose=False, plot_results=False)
                vcents = np.poly1d(t_sec_coeffs[:,0])((hsec-hpix/2)/hpix)
            except ValueError:
                vcents = vcenters[hsec]
            A = np.zeros((wls,d0,d1))
            skip=False
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
                vcent -= 1 ### Something is wrong above - shouldn't need this...
                center = [np.mod(hcent,1),np.mod(vcent,1)]
                hpoint = (hcent-hpix/2)/hpix
                ### Now build PSF model around center point
                if psfmodel == 'bspline':
                    ### TODO - revamp this to pull from input
                    print "Need to update bspline psf generation"
                    exit(0)
                    r_breakpoints = np.hstack(([0, 1.5, 2.4, 3],np.arange(3.5,8.6,1)))         
                    theta_orders = [0]
                    psf_jj = spline.make_spline_model(params, coeff_matrix, center, hpoint, [2*fit_pad+1,2*fit_pad+1], r_breakpoints, theta_orders, fit_bg=False)
                    bg_lvl = np.median(psf_jj[psf_jj<np.mean(psf_jj)])
                    psf_jj -= bg_lvl  
                    psf_jj /= np.sum(psf_jj) # Normalize to 1
                elif psfmodel == 'ghl':
                    ts_names = {0:'T1', 1:'T2', 2:'T3', 3:'T4'}
                    ts_num = fib%4
                    ts = ts_names[ts_num]
                    fnum = int(fib/4)
                    ### skip first of T1, T2, T3 and last of T4 (no fit)
                    if ts_num < 3 and fnum ==0:
                        skip=True
                        continue
                    elif ts_num ==3 and fnum == 27:
                        skip=True
                        continue
                    psf_hdu = pyfits.open(os.path.join(redux_dir,'psf','ghl_psf_coeffs_{ts}_{fnum:03d}.fits'.format(ts=ts,fnum=fnum)))
                    weights = psf_hdu[0].data.ravel()
                    lorentz_fits = psf_hdu[1].data
                    s_coeffs = psf_hdu[2].data
                    p_coeffs = psf_hdu[3].data
                    hcenters_dummy = psf_hdu[4].data
                    vcenters_dummy = psf_hdu[5].data
                    pord = len(s_coeffs)-1
                    params = psf.init_params(hcenters_dummy, vcenters_dummy, s_coeffs, p_coeffs, pord, r_guess=lorentz_fits[:,1], s_guess = lorentz_fits[:,0])
                    params1 = psf.convert_params(params, 0, pord, hscale=hpoint)
                    params1['xc'].value = hcent
                    params1['yc'].value = vcent
                    params1['bg'].value = 0
                    cpad = fit_pad
                    icent = np.array(([hcent],[vcent]))
                    gh_model = psf.get_ghl_profile(icent, params1, weights, pord, cpad, return_model=True, no_convert=True)[0]
                    lorentz = psf.get_data_minus_lorentz(np.zeros((2*cpad+1,2*cpad+1)), icent, params1, weights, pord, cpad, return_lorentz=True, no_convert=True)
                    psf_jj = gh_model+lorentz
                    psf_jj /= np.sum(psf_jj) ### Empirically normalized - may introduce error, bias?
                elif psfmodel == 'fixed':
                    psf_jj = psf.fixed_psf(hcent, vcent, fit_pad, redux_dir)
#                    ncenters = len(hcenters)
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
            if skip:
                continue
            ##Now using the full available data
#            if sec > 4:
            im = np.sum(A,axis=0)
#            for i in range(3):
#                if i==0:
#                    idx = 10
#                elif i==1:
#                    idx = 12
#                elif i==2:
#                    idx = 20
#                print idx
#                plt.imshow(A[idx][:11,5:25], interpolation='none')
#                plt.show()
#                plt.close()
#            plt.imshow(im, interpolation='none')
#            plt.show()
#            plt.close()
            B = np.matrix(np.resize(A.T,(d0*d1,wls)))
#            B = np.hstack((B,np.ones((d0*d1,1)))) ### add background term
            p = np.matrix(np.resize(ccd_sec.T,(d0*d1,1)))
            n = np.diag(np.resize(ccd_sec_invar.T,(d0*d1,)))
#            print np.shape(B), np.shape(p), np.shape(n)
#            unity_model = np.dot(B, np.ones((wls,1))).reshape(ccd_sec.shape)
#            plt.imshow(unity_model,interpolation='none')
#            plt.show()
#            plt.close()
            text_sp_st = time.time()
            fluxtilde, ctilde, Rmat[sec] = sf.extract_2D_sparse(p,B,n, return_covar=True)
            fluxraw = sf.extract_2D_sparse(p,B,n,return_no_conv=True)
            t_betw_ext = time.time()
            if sec == 0:
                extracted_counts[fib,0:len_section] = fluxtilde
                extracted_covar[fib,0:len_section] = ctilde
            else:
                sec_inds = np.arange(len_edge,len_section,dtype=int)
                extracted_counts[fib,sec_inds+int(sec*(len_section-2*len_edge))] = fluxtilde[sec_inds]
                extracted_covar[fib,sec_inds+int(sec*(len_section-2*len_edge))] = ctilde[sec_inds]
                raw_ex_counts[fib,sec_inds+int(sec*(len_section-2*len_edge))] = fluxraw[sec_inds]
            tfinish = time.time()
#            if verbose:
#                print(" Section {} Time = {}".format(sec,tfinish-tstart))
#                print("  PSF modeling took {}s".format(text_sp_st-tstart))
#                print("  Sparse extraction took {}s".format(t_betw_ext-text_sp_st))
            
            sec_model = np.dot(B, fluxtilde).reshape((ccd_sec.shape[1], ccd_sec.shape[0])).T
            sec_model = np.asarray(sec_model)
            raw_model = np.dot(B, fluxraw).reshape((ccd_sec.shape[1], ccd_sec.shape[0])).T
            raw_model = np.asarray(raw_model)
            if return_model:
                if sec == 0:
                    hinds = hsec[0:-len_edge]-hsec[0]
                elif sec == num_sections-1:
                    hinds = hsec[len_edge:] - hsec[0]# + len_edge
                else:
                    hinds = hsec[len_edge:-len_edge] - hsec[0]# + len_edge
                shift = 0
#                if sec > 0:
#                    shift = 1 ### Not sure why this is happening, hack to correct
                ccd_model[vmin+shift:vmax+shift,hinds+hsec[0]] += raw_model[:,hinds]
            pd = 7
            scm = sec_model[:,pd:-pd]
            rcm = raw_model[:,pd:-pd]
            csm = ccd_sec[:,pd:-pd]
            ism = ccd_sec_invar[:,pd:-pd]
#            chi2r = np.sum(np.ravel(csm-scm)**2*np.ravel(ism))/(np.size(csm)-csm.shape[1])
#            print chi2r
            chi2r = np.sum(np.ravel(csm-rcm)**2*np.ravel(ism))/(np.size(csm)-csm.shape[1])
#            print chi2r
            plot_sec = False
            if plot_sec:
#                plt.imshow(np.vstack((ccd_sec, sec_model, ccd_sec-sec_model)), interpolation='none') #im*2*np.max(ccd_sec),
#                plt.imshow(ccd_sec[:,hinds-hinds[0]]-sec_model[:,hinds-hinds[0]], interpolation='none')
#                plt.plot(hsec-hsec[0], vcenters[hsec]+v_pad-vcentmn)
#                plt.plot(hsec-hsec[0], vcenters[hsec]+3*v_pad+1-vcentmn)
#                plt.plot(hsec-hsec[0], vcenters[hsec]+5*v_pad+1-vcentmn)
#                plt.show()
#                plt.close()
                plt.imshow(np.vstack((ccd_sec, raw_model, ccd_sec-raw_model)), interpolation='none')
                plt.show()
                plt.close()
#                plt.imshow((ccd_sec-sec_model)*np.sqrt(ccd_sec_invar), interpolation='none')
#                plt.show()
#                plt.close()
#                plt.imshow((ccd_sec-raw_model)*np.sqrt(ccd_sec_invar), interpolation='none')
#                plt.show()
#                plt.close()
        if verbose:
            tff = time.time()
            if fib > 3:
                plt.plot(extracted_counts[fib,:])
#                plt.plot(raw_ex_counts[fib,:])
                plt.show()
                plt.close()
            print("Fiber {} Time = {:06}s".format(fib,tff-tf0))
    np.save(os.path.join(redux_dir,'n20161123','rmat.npy'), Rmat)
    if return_model:
        return extracted_counts, extracted_covar, ccd_model
    else:
        return extracted_counts, extracted_covar#, raw_ex_counts
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
    
def find_most_recent_frame_date(ftype,data_dir,return_filenames=False,date_format='nYYYYMMDD', before_date=None):
    """ Finds the date of most recent arc exposures.
        If desired, can return list of arc exposures on that date.
        date_format should show all positions of Y, M, and D (plus any /, -, etc)
        TODO - make before_date option work
    """
    ### Find files
    if ftype == 'arc':
        filenames = glob.glob(os.path.join(data_dir,'*','*[tT][hH][aA][rR]*'))
    elif ftype == 'fiberflat':
        filenames = glob.glob(os.path.join(data_dir,'*','*[fF]iber[fF]lat*'))
    elif ftype == 'dark':
        filenames = glob.glob(os.path.join(data_dir,'*','*[dD]ark*'))
    elif ftype == 'bias':
        filenames = glob.glob(os.path.join(data_dir,'*','*[bB]ias*'))
    elif ftype == 'slitflat':
        filenames = glob.glob(os.path.join(data_dir,'*','*[sS]lit[fF]lat*'))
    else:
        print("'ftype' must be one of the following:")
        print("'arc'\n'fiberflat'\n'dark'\n'bias'\'slitflat'")
        exit(0)
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
        st_inds = -1*np.ones(len(date_format),dtype=int)
        for i in range(len(date_format)):
            if date_format[i] == D:
                st_inds[i] = i
        st_inds = st_inds[st_inds != -1].astype(int)
        date_inds = np.array(([i for i in range(len(dates)) if recent in dates[i][st_inds[0]:st_inds[-1]+1]]),dtype=int)
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
        img, oscan, hdr = open_minerva_fits(flnm, ext=ext, return_hdr=True)
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
    
def overscan_fit(overscan,pord=5,plot_fit=False):
    """ Assumes we are fitting along the vertical direction
    """
    if overscan.size <= 1:
        return 0
    overscan_avg = np.median(overscan, axis=1)
    vgrph = np.arange(len(overscan_avg))
    vfit = (vgrph-len(overscan_avg)/2)/len(overscan_avg)
    os_cff = np.polyfit(vfit, overscan_avg, pord)
    overscan_fit = np.poly1d(os_cff)(vfit)
    if plot_fit:
        plt.plot(vfit,overscan_fit)
        plt.show()
        plt.close()
    return overscan_fit
    
def bias_fit(bias, overscan_fit, plot_fit=False):
    bias_mean = np.mean(bias,axis=0)
    bm0 = np.mean(bias_mean)
    bias_fit = np.zeros((bias.shape))
    for i in range(len(overscan_fit)):
        bias_fit[i] = bias_mean + overscan_fit[i] - bm0
    if plot_fit:
        plt.imshow(bias_fit, interpolation='none')
        plt.show()
    return bias_fit
    
def stack_calib(redux_dir, data_dir, date, method='median', frame='bias'):
    """ Stacks bias frames from the day (median combined), saves result to
        disk.  Will load directly instead of re-doing if available.
        Returns the median bias
        Can do the same for dark frames
    """
    if frame == 'bias':
        fname='bias_avg.fits'
    elif frame == 'dark':
        fname = 'scalable_dark_avg.fits'
    else:
        print("Can only accommodate dark and bias frames right now")
        exit(0)
    fname = date + '.' + fname
    if os.path.isfile(os.path.join(redux_dir,date,fname)):
        calib, oscan, hdr = open_minerva_fits(os.path.join(redux_dir,date,fname), return_hdr=True)
#        calib_hdu = pyfits.open(os.path.join(redux_dir,date,fname),uint=True)
#        calib = calib_hdu[0].data
        if frame == 'dark':
            return calib, hdr
        else:
            return calib
    else:
        if frame == 'bias':
            filenamesb = glob.glob(os.path.join(data_dir,date,'*[Bb]ias*.fits'))
            if len(filenamesb) == 0:
                filenamesb = find_most_recent_frame_date('bias', data_dir, return_filenames=True)
#                print("ERROR: No bias frames available on date {}".format(date))
#                exit(0)
            
            b0, oscan = open_minerva_fits(filenamesb[0])
            biases = np.zeros((len(filenamesb),b0.shape[0],b0.shape[1]))
            for i in range(len(filenamesb)):
                biases[i], oscan = open_minerva_fits(filenamesb[i])
            bias = sf.combine(biases, method=method)
            hdu = pyfits.PrimaryHDU(bias)
        if frame == 'dark':
            bias = stack_calib(redux_dir, data_dir, date, frame='bias')
            filenamesd = glob.glob(os.path.join(data_dir,date,'*[Dd]ark*.fits'))
            if len(filenamesd) == 0:
                filenamesd = find_most_recent_frame_date('dark', data_dir, return_filenames=True)
#                print("ERROR: No dark frames available on date {}".format(date))
#                exit(0)
            d0, oscan = open_minerva_fits(filenamesd[0])
            darks = np.zeros((len(filenamesd),d0.shape[0],d0.shape[1]))
            exptimes = np.zeros((len(filenamesd)))
            for i in range(len(filenamesd)):
                darks[i], oscan, hdr = open_minerva_fits(filenamesd[i], return_hdr=True)
                oscan_fit = overscan_fit(oscan)
                bias_fiti = bias_fit(bias, oscan_fit)
                darks[i] -= bias_fiti
                exptimes[i] = hdr['EXPTIME']
            mean_exptime = np.mean(exptimes)
            dark = sf.combine(darks, method=method)
            hdu = pyfits.PrimaryHDU(dark)
            hdu.header.append(('EXPTIME', mean_exptime, 'Average exposure time'))
        hdulist = pyfits.HDUList([hdu])
        try:
            if not os.path.isdir(os.path.join(redux_dir,date)):
                os.mkdir(os.path.join(redux_dir,date))
            hdulist.writeto(os.path.join(redux_dir,date,fname),clobber=True)
        except:
            print("Directory/File already exists")
        if frame == 'dark':
            return dark, hdr
        elif frame == 'bias':
            return bias
            
def stack_flat(redux_dir, data_dir, date, method='median'):
    """ makes a master slit flat - stacks all and bias subtracts + dark corrects
    """
    fname = 'slit_flat_avg.fits'
    fname = date + '.' + fname
    if os.path.isfile(os.path.join(redux_dir,date,fname)):
        sflat, oscan = open_minerva_fits(os.path.join(redux_dir,date,fname))
        return sflat
    else:
        ### load bias and dark
        bias = stack_calib(redux_dir, data_dir, date, frame='bias')
        dark, dhdr = stack_calib(redux_dir, data_dir, date, frame='dark')
        filenames = glob.glob(os.path.join(data_dir,date,'*[Ss]lit[Ff]lat*.fits'))
        if len(filenames) == 0:
                print("Warning: No slit flats available on date {}".format(date))
                print("         CCD will not be flat fielded...")
#                exit(0)
                return np.ones(bias.shape)
        s0, oscan = open_minerva_fits(filenames[0])
        sflats = np.zeros((len(filenames),s0.shape[0],s0.shape[1]))
        keep_inds = []
        for i in range(len(filenames)):
            sflats[i], oscan, hdr = open_minerva_fits(filenames[i], return_hdr=True)
            oscan_fit = overscan_fit(oscan)
            bias_fiti = bias_fit(bias, oscan_fit)
            ## Subtract overscan corrected bias
            sflats[i] -= bias_fiti
            ## subtract scaled dark exposure
            try:
                sflats[i] -= dark*(hdr['EXPTIME']/dhdr['EXPTIME'])
            except:
                continue
            if np.mean(sflats[i]) > 100:
                ### Don't keep flats with no signal
                keep_inds += [i]
        sflat = sf.combine(sflats[keep_inds], method=method)
        ### Reverse orientation to match ccd, etc.
        sflat = sflat[::-1,:]
        hdu = pyfits.PrimaryHDU(sflat)
        hdulist = pyfits.HDUList([hdu])
        try:
            if not os.path.isdir(os.path.join(redux_dir,date)):
                os.mkdir(os.path.join(redux_dir,date))
            hdulist.writeto(os.path.join(redux_dir,date,fname),clobber=True)
        except:
            print("Directory/File already exists")
        return sflat
    
def make_norm_sflat(sflat, redux_dir, date, spline_smooth=True, include_edge=True, plot_results=False):
    """ Long function, originally fit_slit.py.  May rearrange later, but 
        not necessary.
    """
    ##########################################################
    ### open slit flat file, subtract bias, scale by median ##
    ##########################################################
    if os.path.isfile(os.path.join(redux_dir, date, '{}.slit_flat_smooth.fits'.format(date))):
        slit_norm = pyfits.open(os.path.join(redux_dir, date, '{}.slit_flat_smooth.fits'.format(date)))[0].data
        return slit_norm
    #spline_smooth = True
    #redux_dir = os.environ['MINERVA_REDUX_DIR']
    #date = 'n20160115'
    #date2 = 'n20160216'
    #date = 'n20161204'
    #bias_fits = pyfits.open(os.path.join(redux_dir,date,'{}.bias_avg.fits'.format(date)),uint=True)
    #bias = bias_fits[0].data
    #sflat_fits = pyfits.open(os.path.join(redux_dir,date,'{}.slit_flat_avg.fits'.format(date)))
    #sflat = sflat_fits[0].data
    #sflat-=bias
#    readnoise = 3.63
    sflat /= np.max(sflat)
    sflat = sflat[::-1,:] #I've confused myself on orientation...
    #sflat/=np.median(sflat[sflat>5*readnoise]) #Probably a better way to normalize
#    shdr = sflat_fits[0].header
    
    ###clip overscan and weird edge effects
    #sflat = sflat[1:2049,0:2048]
    hpix = sflat.shape[1]
    vpix = sflat.shape[0]
#    sflat = sflat[::-1,0:hpix]
    #plt.plot(sflat[:,1000])
    #plt.show()
    
    #xpix = shdr['NAXIS2']
    #hpix = shdr['NAXIS1']
    
    ##########################################################
    ################# fit to profile #########################
    ##########################################################
    
    def get_new_inds(sss, idx1, idx2, fib, pad = 2, noise = 0.04):
        ssub = sss[idx1:idx2]
        ### primitive sigma clip
        smsk = (ssub > (np.median(ssub) - (3 - 2.3*fib/29)*noise))
        if len(smsk>0) == 0:
            print "Indices:", idx1, idx2
#            print smsk
            exit(0)
#        if fib == 25:
#            print idx1, idx2
#            print smsk
#            plt.plot(ssub)
#            plt.show()
        idx1n = idx1 + np.nonzero(smsk)[0][0]
        idx2n = idx1 + np.nonzero(smsk)[0][-1]
        il = max(0, idx1-pad)
        ih = min(sflat.shape[0],idx2+pad)
        if (idx1n == idx1 or idx2n == idx2) and il !=0 and ih !=sflat.shape[0]:
            idx1n, idx2n = get_new_inds(sss, il, ih, fib)
        return idx1n, idx2n
    
    def get_rough_norm(sflat, num_fibers=29, pord=4, idxones=None, idxtwos=None, slit_widths=None, dip_widths=None, include_edge=True):
        hpix = sflat.shape[1]
        vpix = sflat.shape[0]
        line_fits = np.zeros((num_fibers,hpix,pord+1))
        find_inds = False
        if idxones is None: #Lazy, need to put in all four or none right now
            idxones = np.zeros((num_fibers, hpix))
            idxtwos = np.zeros((num_fibers, hpix))
            slit_widths = np.zeros((num_fibers, hpix))
            dip_widths = np.zeros((num_fibers, hpix))
            find_inds = True
        spds = np.zeros((num_fibers, hpix))
        slit_norm = np.ones(sflat.shape)
        for col in range(hpix):
        #    if col %100 == 0:
        #        print "column", col
        #    sss = ediff1d(sflat[:,col])
            sss = sflat[:,col]
            if col < 600:
                idx1 = int(40 - col/15.0)
            else:
                idx1 = 0
        #    idx2 = 0
            pad = 5
            slit_width = 67-15 #change this later too
            dip_width = 58-22
            for i in range(num_fibers):
        #        if i > 1:
        #            continue
    #            print "Iteration", i
                if find_inds:
                    ### for first slit need to do some special treatment because we can
                    ### start partway into slit, or into dip
                    if i == 0:
                        ssub = sss[idx1:idx1+slit_width+dip_width+2*pad]
                        smsk = ssub > 0.5*np.mean(ssub) ## should robustly cut low vals
                        ### first nonzero value is temp new idx1
                        idx1t = idx1 + np.nonzero(smsk)[0][0]
                        idx2t = idx1t + np.nonzero(ediff1d(smsk[np.nonzero(smsk)[0][0]:]))[0][0]
                        ### Now run through to cut
                        idx1, idx2 = get_new_inds(sss, idx1t, idx2t, i)
            #            slit_width = idx2 - idx1
                    else:
                        idx2o = idx2
                        if dip_width > 6:
                            idx1, idx2 = get_new_inds(sss, idx2+dip_width, idx2+dip_width+slit_width-1, i)
                            dip_width = idx1-idx2o
                            slit_width = idx2-idx1
                        else:
                            idx1, idx2 = idx2+dip_width, idx2+dip_width+slit_width
                            dip_width -= 1
                            slit_width -= 1
                    idx2mx = 2050 ### Some pathological behavior on edge...
                    idx2 = min(idx2,idx2mx)
                    ### Once indices of edges are identified, fit to those within
                    ### Constrain to be close to old indices (slit_width error propagates otherwise...)
                    if col > 0:
                        if abs(idxones[i, col-1]-idx1) > 1:
                            idx1 = int(idxones[i,col-1] + 1*(2*(idx1>idxones[i, col-1])-1))
                        if abs(idxtwos[i, col-1]-idx2) > 1:
                            idx2 = int(idxtwos[i,col-1] + 1*(2*(idx2>idxtwos[i, col-1])-1))
                    ### save inds, spd for spline fitting later...
                    idxones[i, col] = idx1
                    idxtwos[i, col] = idx2
                    slit_widths[i, col] = slit_width
                    dip_widths[i, col] = dip_width
                else:
                    idx1 = int(idxones[i, col])
                    idx2 = int(idxtwos[i, col])
                    slit_width = slit_widths[i, col]
                    dip_width = dip_widths[i, col]
                    if dip_width <= 6:
                        dip_width = np.median(dip_widths[i,:])
                        slit_width = np.median(slit_widths[i,:])
                        idx1 = int(idxtwos[i-1, col]+dip_width)
                        idx2 = int(idxtwos[i-1, col]+dip_width+slit_width)
                        ### Then update indices
                        idxones[i, col] = idx1
                        idxtwos[i, col] = idx2
                        slit_widths[i, col] = slit_width
                        dip_widths[i, col] = dip_width
                ### use spd to come in from the edges a little, helps prevent bad fits
                spd = 1 if i < 27 else 3  ### Extra pad on last due to overlap
                ### additional override from some of the lower signal values
                if i > 23 and i <27 and col < 100:
                    spd = 2
                spds[i, col] = spd
                slit_inds = np.arange(idx1+spd,idx2-spd)
                slit_inds = slit_inds[slit_inds < sflat.shape[0]]
                ### range of columns to take median over for empirical fits
                cpd = 1
                crange = np.arange(max(0,col-cpd),min(sflat.shape[1],col+cpd+1))
                ### if we don't have enough points to fit polynomial, just go empirical
                if len(slit_inds) == 0:
                    pass
                elif len(slit_inds)<=pord+1:
                    if len(slit_inds) == 1:
                        meds = np.median(sflat[slit_inds, crange[0]:crange[-1]])
                    else:
                        meds = np.median(sflat[slit_inds, crange[0]:crange[-1]], axis=1)
                    slit_norm[slit_inds,col] = meds
                else:
                    ### Change scale to [-1, 1] then fit
                    n_inds = 2*(slit_inds - vpix/2)/vpix
                    line_fits[i,col] = np.polyfit(n_inds,sss[slit_inds],pord)
                    fitted = np.poly1d(line_fits[i,col,:])(n_inds)
                    slit_norm[slit_inds,col] = fitted
                ### include nearest points on each edge...
                if include_edge:
                    pd = int(min(1,dip_width))
                    slit_norm[idx1-pd:idx1+spd,col] = np.median(sflat[idx1-pd:idx1+spd,crange], axis=1)
                    if idx2 < 2048:
                        slit_norm[idx2-spd:idx2+pd,col] = np.median(sflat[idx2-spd:idx2+pd,crange], axis=1)
        return slit_norm, idxones, idxtwos, slit_widths, dip_widths, line_fits, spds
       
    ### Initial norm fit...
    slit_norm, idxones, idxtwos, slit_widths, dip_widths, line_fits, spds = get_rough_norm(sflat, include_edge=include_edge)
    ### Refine with better slit_norm estimates
    slit_norm, idxones, idxtwos, slit_widths, dip_widths, line_fits, spds = get_rough_norm(sflat, idxones=idxones, idxtwos=idxtwos, slit_widths=slit_widths+1, dip_widths=dip_widths, include_edge=include_edge)
    pord = line_fits.shape[2]-1
    num_fibers = line_fits.shape[0]
    ### Fit splines to polynomial coefficients...
    if spline_smooth:
        #plt.plot(slit_norm[1000,:])
        spc = 5 ## try evenly spaced breakpoints
        breakpoints = np.arange(0,hpix,spc)
        ### Fit splines, no weighting
        ### This is actually pretty slow...look at speeding up spline_1D
        smooth_line_fits = np.zeros(line_fits.shape)
        for val in range(pord+1):
            print "val:", val
            for fib in range(num_fibers):
                spl = spline.spline_1D(np.arange(hpix),line_fits[fib,:,val],breakpoints, window = 1024, pad = 50)
                smooth_line_fits[fib,:,val] = spl
        
        ### Now go back and put modified values into slit_norm
        for col in range(hpix):
            for fib in range(num_fibers):
                idx1 = idxones[fib, col]
                idx2 = idxtwos[fib, col]
                spd = spds[fib, col]
                slit_inds = np.arange(idx1+spd,idx2-spd)
                slit_inds = slit_inds[slit_inds < sflat.shape[0]].astype(int)
                if len(slit_inds) > pord+1:
                    n_inds = 2*(slit_inds - vpix/2)/vpix
                    fitted = np.poly1d(smooth_line_fits[fib,col,:])(n_inds)
                    slit_norm[slit_inds,col] = fitted
    
    ### visually evaluate quality of slit_norm
    if plot_results:
        smooth = sflat/slit_norm
        plt.imshow(smooth, interpolation='none', vmax = 3)
        plt.show()
        for j in [0, 1000, 2000]:
            plt.plot(smooth[:,j], linewidth=2)
            plt.plot(sflat[:,j], linewidth=2)
            plt.plot(slit_norm[:,j], linewidth=2)
            plt.show()
    
    ### Save file
    slit_norm = slit_norm[::-1,:]
    #redux_dir = os.environ['MINERVA_REDUX_DIR']    
    hdu = pyfits.PrimaryHDU(slit_norm)
    hdu.header.append(('POLYORD',pord,'Polynomial order used for fitting'))
    hdulist = pyfits.HDUList([hdu])
    hdulist.writeto(os.path.join(redux_dir,date,'{}.slit_flat_smooth.fits'.format(date)),clobber=True)  
    return slit_norm
    
def cal_fiberflat(flat, data_dir, redux_dir, arc_date):
    """ calibrates fiber flat, right now omit dark (no exptime available)
        and overscan (miniscule correction anyway)
        Use only bias subtraction and slit flat correction
    """
    bdate = find_most_recent_frame_date('bias', data_dir)
#    ddate = find_most_recent_frame_date('dark', data_dir)
    sfdate = find_most_recent_frame_date('slitflat', data_dir)
    bias = stack_calib(redux_dir, data_dir, bdate)
    bias = bias[:,0:2048] #Remove overscan
    ### Load Dark
#    dark, dhdr = stack_calib(redux_dir, data_dir, ddate, frame='dark')
#    dark = dark[:,0:2048]
#    dark *= exptime/dhdr['EXPTIME'] ### Scales linearly with exposure time
    ### Analyze overscan (essentially flat, very minimal correction)
#    overscan_fit = m_utils.overscan_fit(overscan)
    
    #bias -= np.median(bias) ### give zero mean overall - readjust by overscan
    ### Making up this method, so not sure if it's good, but if it works it should reduce ccd noise
#    bias = bias_fit(bias, overscan_fit)
    
    
    ### Make master slitFlats
    sflat = stack_flat(redux_dir, data_dir, sfdate)
    norm_sflat = make_norm_sflat(sflat, redux_dir, sfdate)
    
    
    ### Calibrate ccd
    flat -= bias #Note, if ccd is 16bit array, this operation can cause problems
#    flat -= dark
    flat /= norm_sflat
    return flat
    
def save_comb_arc_flat(filelist,frm,ts,redux_dir,date,no_overwrite=True, verbose=False, method='median'):
    """ frm must be 'arc' or 'flat'
    """
    pre = ''
    if method == 'mean':
        pre = 'mean_'
    if os.path.isfile(os.path.join(redux_dir,date,'{}combined_{}_{}.fits'.format(pre,frm,ts))) and no_overwrite:
        return
    else:
        if verbose:
            print("Combining most recent {}s".format(frm))
        if frm == 'arc':
            frames = [f for f in filelist if ts in f.upper()]
        elif frm == 'flat':
            frames = [f for f in filelist if ts in f.upper()]
        frame_imgs = fits_to_arrays(frames)
        print 'ARC COMBINING'
        print method
        comb_img = sf.combine(frame_imgs, method=method)
        if not os.path.isdir(os.path.join(redux_dir,date)):
            os.makedirs(os.path.join(redux_dir,date))
        hdu = pyfits.PrimaryHDU(comb_img)
        hdu.header.append(('COMBMTHD',method,'Method used to combine frames'))
        hdulist = pyfits.HDUList([hdu])
        hdulist.writeto(os.path.join(redux_dir,date,'{}combined_{}_{}.fits'.format(pre,frm,ts)),clobber=True)
        return
        
def find_peaks(array,bg_cutoff=None,mx_peaks=None,skip_peaks=0, view_plot=False):
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
        if view_plot:
            plt.plot(array)
            plt.plot(peaks,array[peaks.astype(int)],'ro')
            plt.plot(bg_cutoff*np.ones((array.shape)),'k')
            plt.show()
            plt.close()
        return peaks

def build_trace_ccd(fiber_flat_files, ccd_shape, reverse_y=True, tscopes=['T1','T2','T3','T4']):
    trace_ccd = np.zeros((ccd_shape))
    for ts in tscopes:
        for j in range(4):
            if ts.lower() in fiber_flat_files[j].lower():
                find = j
        flatfits = pyfits.open(fiber_flat_files[find])
        if reverse_y:
            flat = flatfits[0].data[::-1,:]
        else:
            flat = flatfits[0].data
        #Choose fiberflats with iodine cell in
        norm = 10000 #Arbitrary norm to match flats
        tmmx = np.median(np.sort(np.ravel(flat))[-100:])
        if tmmx > 62e3:
            print("WARNING: Fiberflat counts may be saturated on {}".format(ts))
        trace_ccd += flat[:,0:ccd_shape[1]].astype(float)*norm/tmmx
    return trace_ccd

def bspline_pre(fiberflat, t_coeffs, redux_dir, date, window=10, skip_fibs=[0]):
    """ Precalculates the bspline model array for the given fiberflat.
    """
    ### Use 11/23/2016 fiber flats as backup if new fiberflats have overflow
    backup_files = ['a']*4
    for i in range(4):
        backup_files[i] = os.path.join(redux_dir,'n20161123','combined_flat_T{}.fits'.format(int(i+1)))
    backup_fiberflat = build_trace_ccd(backup_files, fiberflat.shape)
    t_backup = pyfits.open(os.path.join(redux_dir,'n20161123','trace_n20161123.fits'))[0].data[0]
    ### Initialize relevant variables, some are hardcoded here    
    pts_per_model = 1000
    breakpoints = np.array(([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]))
    ypix = fiberflat.shape[1]
    ys = (np.arange(ypix)-ypix/2)/ypix
    pd = 7
    ### Now fill in average model by order
    sfit_models = np.zeros((pts_per_model,t_coeffs.shape[1], ypix))    
    for fib in range(t_coeffs.shape[1]):
        if fib in skip_fibs:
            continue
        if fib != 5:
            continue
        print "Spline modeling on fiber {}/{}".format(fib+1,t_coeffs.shape[1])
        _sfit_model = np.zeros((pts_per_model, ypix))
        xs = np.poly1d(t_coeffs[:,fib])(ys) + 2
        xbak = np.poly1d(t_backup[:,fib])(ys) + 2
        coeffs = np.zeros((ypix, len(breakpoints)+2))
        ### Find coefficients
        for j in range(ypix):
            if np.isnan(xs[j]):
                continue
            xvals = np.arange(-pd,pd+1) + int(xs[j])
            xvals = xvals[(xvals>0)*(xvals<fiberflat.shape[0])]
            yarr = fiberflat[xvals,j]
            breakpoints_i = breakpoints + xs[j]
            coeffs[j], junk, junk = spline.spline_1D(xvals, yarr, breakpoints_i, invar=1/(abs(yarr)+3.63**2),return_coeffs=True)
        ### Build average model
        xfit = np.linspace(-pd,pd,pts_per_model)
        ydum = np.zeros(xfit.shape)
        for j in range(ypix):
            sys.stdout.write("\r{:6.3f}%".format(j/ypix*100))
            sys.stdout.flush()
            sfit_i = np.zeros((pts_per_model))
            skip_cnt = 0
            for i in range(-int(window/2), int(window/2)):
                if j+i < 0 or j+i >= ypix:
                    skip_cnt += 1
                    continue
                _sfit_i = spline.get_spline_1D_fit(xfit, ydum, breakpoints, coeffs[j+i])
                if fib == 5:
                    plt.plot(xfit, _sfit_i)
                sfit_i += _sfit_i
            sfit_i /= (window-skip_cnt)
            sfit_i /= np.sum(sfit_i) #Normalize
            _sfit_model[:,j] = sfit_i
            if fib == 5:
                plt.show()
                plt.close()
        sys.stdout.write("\n")
        sfit_models[:,fib,:] = _sfit_model
        if os.path.isdir(os.path.join(redux_dir, date)):
            np.save(os.path.join(redux_dir, date, 'bspline_crsxn_model_{}.npy'.format(fib)),_sfit_model)    
        else:
            os.mkdir(os.path.join(redux_dir, date))
    np.save(os.path.join(redux_dir, date, 'bspline_crsxn_models.npy'),sfit_models)
    
def spline_res(params, xvals, zvals, invar, model):
    xc = params['xc'].value
    hght = params['hght'].value
    bg = params['bg'].value
    xorig = np.linspace(-7,7,len(model))
    model /= np.sum(model)*(xorig[1]-xorig[0])
    sfit = hght*sf.re_interp(xvals, xc, xorig, model) + bg
    return (sfit-zvals)**2*invar
    
def spline_fast_fit(xvals, zvals, invar, model):
    Ng = 10
    ### Course
    xgrid = np.linspace(-0.5, 0.5, Ng)
    hght = np.sum(zvals)-len(zvals)*np.min(zvals)
    bg = np.min(zvals)
    xorig = np.linspace(-7,7,len(model))
    model /= np.sum(model)*(xorig[1]-xorig[0])
    chi2s = np.zeros((len(xgrid)))
    for i in range(len(xgrid)):
        sfit = hght*sf.re_interp(xvals, xgrid[i], xorig, model) + bg    
        chi2s[i] = np.sum((sfit-zvals)**2*invar)
    xmin = xgrid[np.argmin(chi2s)]

    ### Fine
    xgrid = np.linspace(xmin-0.05, xmin+0.05, Ng)
    hght = np.sum(zvals)-len(zvals)*np.min(zvals)
    bg = np.min(zvals)
    xorig = np.linspace(-7,7,len(model))
    model /= np.sum(model)*(xorig[1]-xorig[0])
    chi2s = np.zeros((len(xgrid)))
    for i in range(len(xgrid)):
        sfit = hght*sf.re_interp(xvals, xgrid[i], xorig, model) + bg    
        chi2s[i] = np.sum((sfit-zvals)**2*invar)
    poly_coeffs = np.polyfit(xgrid, chi2s, 2)
    
    xc, xc_std = sf.chi_coeffs_to_mn_std(poly_coeffs, sigma=1)
    sfit = sf.re_interp(xvals, xc, xorig, model)
    P = np.vstack((sfit,np.ones(len(sfit)))).T
    coeffs, chi_junk = sf.chi_fit(zvals, P, np.diag(invar))
    hght, bg = coeffs[0], coeffs[1]
    return xc, hght, bg