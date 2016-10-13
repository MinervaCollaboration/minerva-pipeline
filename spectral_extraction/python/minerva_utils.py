#!/usr/bin/env python

#Special functions for MINERVA data reduction

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
#import scipy.optimize as optimize
#import scipy.sparse as sparse
#import scipy.signal as signal
#import scipy.linalg as linalg
#import solar
import special as sf
#import bsplines as spline
#import argparse
import lmfit

#########JASON E CODE - from  minerva_dopcode/utils.py
#import datetime
#import glob
#from photutils import aperture_photometry, CircularAperture, CircularAnnulus
#
#
#def addzb(fitsname, redo=False, fau_dir=None):
#    telescopes = ['1','2','3','4']
#    night = (fitsname.split('/'))[3]
#
#    print 'beginning ' + fitsname
#
#
#    # 2D spectrum
#    h = pyfits.open(fitsname,mode='update')
#
#    if 'BARYSRC4' in h[0].header.keys() and not redo:
#        print fitsname + " already done"
#        h.flush()
#        h.close()
#        return
#
#    specstart = datetime.datetime.strptime(h[0].header['DATE-OBS'],"%Y-%m-%dT%H:%M:%S.%f")
#    specmid = specstart + datetime.timedelta(seconds = h[0].header['EXPTIME']/2.0)
#    specend = specstart + datetime.timedelta(seconds = h[0].header['EXPTIME'])
#
#    t0 = datetime.datetime(2000,1,1)
#    t0jd = 2451544.5
#
#    aperture_radius = 3.398 # fiber radius in pixels
#    annulus_inner = 2.0*aperture_radius
#    annulus_outer = 3.0*aperture_radius
#
#    for telescope in telescopes:
#        print 'beginning telescope ' + telescope + ' on ' + fitsname
#
#        # get the barycentric redshift for each time
#        ra = h[0].header['TARGRA' + telescope]
#        dec = h[0].header['TARGDEC' + telescope]
#        try: pmra = h[0].header['PMRA' + telescope]
#        except: pmra = 0.0
#        try: pmdec = h[0].header['PMDEC' + telescope]
#        except: pmdec = 0.0
#        try: parallax = h[0].header['PARLAX' + telescope]
#        except: parallax = 0.0
#        try: rv = h[0].header['RV' + telescope]
#        except: rv = 0.0
#
#        objname = h[0].header['OBJECT' + telescope]
#        if fau_dir is None:
#            faupath = '/Data/t' + telescope + '/' + night + '/' + night + '.T' + telescope + '.FAU.' + objname + '.????.fits'
#        else:
#            faupath = fau_dir + '/t' + telescope + '/' + night + '/' + night + '.T' + telescope + '.FAU.' + objname + '.????.fits'
#        guideimages = glob.glob(faupath)
#
##        if telescope == '2' and "HD62613" in fitsname: ipdb.set_trace()
#
#        times = []
#        fluxes = np.array([])
#
#        for guideimage in guideimages:
#            try:
#                fauimage = pyfits.open(guideimage)
#            except:
#                print "corrupt file for " + guideimage
#                continue
#
#            # midtime of the guide image (UTC)
#            midtime = datetime.datetime.strptime(fauimage[0].header['DATE-OBS'],"%Y-%m-%dT%H:%M:%S") +\
#                datetime.timedelta(seconds=fauimage[0].header['EXPTIME']/2.0)
#            
#            # convert to Julian date
#            midjd = t0jd + (midtime-t0).total_seconds()/86400.0
#
#            # only look at images during the spectrum
#            if midtime < specstart or midtime > specend: continue
#
#            # find the fiber position
#            try:
#                fiber_x = fauimage[0].header['XFIBER' + telescope]
#                fiber_y = fauimage[0].header['YFIBER' + telescope]
#            except:
#                print "keywords missing for " + guideimage
#                continue
#
#            # do aperture photometry
#            positions = [(fiber_x,fiber_y)]
#            apertures = CircularAperture(positions,r=aperture_radius)
#            annulus_apertures = CircularAnnulus(positions, r_in=annulus_inner, r_out=annulus_outer)
#
#            # calculate the background-subtracted flux at the fiber position
#            rawflux_table = aperture_photometry(fauimage[0].data, apertures)
#            bkgflux_table = aperture_photometry(fauimage[0].data, annulus_apertures)
#            bkg_mean = bkgflux_table['aperture_sum'].sum() / annulus_apertures.area()
#            bkg_sum = bkg_mean * apertures.area()
#            flux = rawflux_table['aperture_sum'].sum() - bkg_sum
#            
#            # append to the time and flux arrays
#            times.append(midjd)
#            fluxes = np.append(fluxes,flux)
#
#        if len(times) == 0:
#            print "No guider images for " + fitsname + " on Telescope " + telescope + "; assuming mid time"
#            if 'daytimeSky' in fitsname: 
#                h[0].header['BARYCOR' + telescope] = ('UNKNOWN','Barycentric redshift')
#                h[0].header['BARYSRC' + telescope] = ('UNKNOWN','Source for the barycentric redshift')
#                continue
#            
#            # convert specmid to Julian date
#            midjd = t0jd + (specmid-t0).total_seconds()/86400.0
#
#            print midjd
#            zb = barycorr(midjd, ra, dec, pmra=pmra, pmdec=pmdec, parallax=parallax, rv=rv)/2.99792458e8
#            h[0].header['BARYCOR' + telescope] = (zb,'Barycentric redshift')
#            h[0].header['BARYSRC' + telescope] = ('MIDTIME','Source for the barycentric redshift')
#            continue
#            
#
#        zb = np.asarray(barycorr(times, ra, dec, pmra=pmra, pmdec=pmdec, parallax=parallax, rv=rv))/2.99792458e8
#
#        # weight the barycentric correction by the flux
#        #*******************!*!*!*!*!*!*
#        #this assumes guider images were taken ~uniformly throughout the spectroscopic exposure!
#        #****************!*!*!*!*!**!*!*!*!**!*!*!*!*
#        wzb = np.sum(zb*fluxes)/np.sum(fluxes)
#        
#        # update the header to include aperture photometry and barycentric redshift
#        h[0].header['BARYCOR' + telescope] = (wzb,'Barycentric redshift')
#        h[0].header['BARYSRC' + telescope] = ('FAU Flux Weighted','Source for the barycentric redshift')
#        hdu = pyfits.PrimaryHDU(zip(times,fluxes))
#        hdu.header['TELESCOP'] = ('T' + telescope,'Telescope')
#        h.append(hdu)
#
#    # write updates to the disk
#    h.flush()
#    h.close()

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
#                    print xtrace[j,i]
                    xtrace[j,i], Itrace[j,i], sigtrace[j,i], powtrace[j,i], chi_vals[j,i] = fit_trace(xtrace[j,i],y,image)
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
        t_coeffs[0,i] = tmp_coeffs[0]
        t_coeffs[1,i] = tmp_coeffs[1]
        t_coeffs[2,i] = tmp_coeffs[2]
        i_coeffs[0,i] = tmp_coeffs2[0]
        i_coeffs[1,i] = tmp_coeffs2[1]
        i_coeffs[2,i] = tmp_coeffs2[2]
        s_coeffs[0,i] = tmp_coeffs3[0]
        s_coeffs[1,i] = tmp_coeffs3[1]
        s_coeffs[2,i] = tmp_coeffs3[2]      
        p_coeffs[0,i] = tmp_coeffs4[0]
        p_coeffs[1,i] = tmp_coeffs4[1]
        p_coeffs[2,i] = tmp_coeffs4[2]
        
    if return_all_coeffs:
        return t_coeffs, i_coeffs, s_coeffs, p_coeffs
    else:
        return t_coeffs

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
    
def extract_1D(ccd,t_coeffs,i_coeffs=None,s_coeffs=None,p_coeffs=None,readnoise=1,gain=1,return_model=False,verbose=False):
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
        for j in range(0,hpix,fact):
            ### set coordinates, gaussian parameters from coeffs
            jadj = int(np.floor(j/fact))
            yj = (j-hpix/2)/hpix
            yc_ccd[i,jadj] = j
            xc = t_coeffs[2,i]*yj**2+t_coeffs[1,i]*yj+t_coeffs[0,i]
#            Ij = i_coeffs[2,i]*yj**2+i_coeffs[1,i]*yj+i_coeffs[0,i] #May use later for normalization
            sigj = s_coeffs[2,i]*yj**2+s_coeffs[1,i]*yj+s_coeffs[0,i]
            powj = p_coeffs[2,i]*yj**2+p_coeffs[1,i]*yj+p_coeffs[0,i]
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
        noise = np.diag(inv_chi[i,:][mask])
        if len(xc_ccd[i,:][mask])>(tmp_poly_ord+1):
            ### Chi^2 fit
            tmp_coeffs, junk = sf.chi_fit(xc_ccd[i,:][mask],profile,noise)
        else:
            ### if not enough points to fit, call entire trace bad
            tmp_coeffs = np.nan*np.ones((tmp_poly_ord+1))
        t_coeffs_ccd[:,i] = tmp_coeffs

    tb = time.time() ### Start time of extraction/end of trace refinement
    if verbose:
        print("Trace refinement time = {}s".format(tb-ta))
       
    ### Uncomment below to see plot of traces
#    for i in range(num_fibers):
#        ys = (np.arange(hpix)-hpix/2)/hpix
#        xs = t_coeffs_ccd[2,i]*ys**2+t_coeffs_ccd[1,i]*ys+t_coeffs_ccd[0,i]
#        yp = np.arange(hpix)
#        plt.plot(yp,xs)
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
        #slit_num = np.floor((i)/4)#args.telescopes) # Use with slit flats
        if verbose:
            print("extracting trace {}".format(i+1))
        ### in each fiber loop run through each trace
        for j in range(hpix):
            yj = (j-hpix/2)/hpix
            xc = np.poly1d(t_coeffs_ccd[::-1,i])(yj)
#            Ij = i_coeffs[2,i]*yj**2+i_coeffs[1,i]*yj+i_coeffs[0,i]
            sigj = s_coeffs[2,i]*yj**2+s_coeffs[1,i]*yj+s_coeffs[0,i]
            powj = p_coeffs[2,i]*yj**2+p_coeffs[1,i]*yj+p_coeffs[0,i]
            ### If trace center is undefined mask the point
            if np.isnan(xc):
                spec_mask[i,j] = False
            else:
                tc = time.time() ### Start of extraction loop
                ### Set values to use in extraction
                xpad = 5  ### can't be too big or traces start to overlap
                xvals = np.arange(-xpad,xpad+1)
                xj = int(xc)
                xwindow = xj+xvals
                xvals = xvals[(xwindow>=0)*(xwindow<vpix)]
                zorig = gain*ccd[xj+xvals,j]
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
                    td = time.time() ### Time before nonlinear fit
                    mn_new, hght, bg = fit_mn_hght_bg(xvals,zorig,invorig,sigj,xc-xj-1,sigj/8,powj=powj)
#                    mn_new, hght, bg = linear_mn_hght_bg(xvals,zorig,invorig,sigj,xc-xj-1,power=powj)
                    te = time.time() ### Time after nonlinear fit
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
                    tf = time.time() ### Time before cosmic ray rejection
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
                    tg = time.time() ### Time after cosmic ray rejection
                    chi2red_array[i,j] = chi2red
#                    if verbose:
#                        print("Total fit time = {}s".format(tg-tc))
#                        print("Nonlinear fit time = {}s".format(te-td))
#                        print("First flux est time = {}s".format(tf-te))
#                        print("Cosmic ray reject time = {}s".format(tg-tf))
#                        print("Reduced chi^2 = {}".format(chi2red))
#                        if j > 300:
#                            plt.plot(xvals,zorig,xvals,fstd*fitnorm+bg,linewidth='2')
#                            plt.show()
#                            plt.close()
#                        time.sleep(2)
                    if return_model and not np.isnan(fstd):
                        ### Build model, if desired
                        image_model[xj+xvals,j] += (fstd*fitnorm+bg)/gain
            ### If a nan came out of the above routine, zero it and mask
            if np.isnan(spec[i,j]):
                spec[i,j] = 0
                spec_mask[i,j] = False
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
        
def arrays_to_params(centers,ellipse):
    """ Converts centers and ellipse arrays to lmfit parameter object.
    """
    params = lmfit.Parameters()
    for i in range(np.shape(centers)[1]):
        params.add('hc{}'.format(i), value=centers[0,i])
        params.add('vc{}'.format(i), value=centers[1,i])
    for j in range(np.shape(ellipse)[1]):
        params.add('q{}'.format(j), value=ellipse[0,j])
        params.add('PA{}'.format(j), value=ellipse[1,j])
    return params