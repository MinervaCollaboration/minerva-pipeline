#!/usr/bin/env python 2.7

#Start of a generic tracefit program.  Geared now toward MINERVA initial data

#Import all of the necessary packages
from __future__ import division
import pyfits
import os
import csv
import math
import time
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from matplotlib import cm
#import scipy
#import scipy.stats as stats
#import scipy.special as sp
#import scipy.interpolate as si
#import scipy.optimize as opt
import scipy.sparse as sparse
import scipy.signal as sig
#import scipy.linalg as linalg
import solar
import special as sf

def fit_trace(x,y,ccd,form='gaussian'):
    """quadratic fit (in x) to trace around x,y in ccd
       x,y are integer pixel values
       input "form" can be set to quadratic or gaussian
    """
    x = int(x)
    y = int(y)
    if form=='quadratic':
        xpad = 2
        xvals = np.arange(-xpad,xpad+1)
        def make_chi_profile(x,y,ccd):
            xpad = 2
            xvals = np.arange(-xpad,xpad+1)
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
        xpad = 7
        xvals = np.arange(-xpad,xpad+1)
        xinds = x+xvals
        xvals = xvals[(xinds>=0)*(xinds<np.shape(ccd)[0])]
        zvals = ccd[x+xvals,y]
        params = sf.gauss_fit(xvals,zvals)
        xc = x+params[1] #offset plus center
        zc = params[2] #height (intensity)
#        pxn = np.linspace(xvals[0],xvals[-1],1000)
        fit = sf.gaussian(xvals,abs(params[0]),params[1],params[2],params[3],params[4])
        chi = sum((fit-zvals)**2/zvals)
        return xc, zc, chi


#set paths
pathd = os.environ['MINERVA_DATA_DIR']
paths = os.environ['MINERVA_SIM_DIR']
date = 'n20160130'

telescopes = ['T1','T2','T3','T4']
for ts in telescopes:
#    #Import tungsten fiberflat
#    fileflat = os.path.join(pathd,'initial_spectro_runs','test4.FIT')
#    #fileflat = os.path.join(paths,'minerva_flat.fits')
#    ff = pyfits.open(fileflat,ignore_missing_end=True,uint=True)
#    hdr = ff[0].header
#    ccd = ff[0].data
#    
#    #Import thorium argon arc
#    filearc = os.path.join(pathd,'initial_spectro_runs','test11.FIT')
#    #filearc = os.path.join(paths,'minerva_arc.fits')
#    af = pyfits.open(filearc,ignore_missing_end=True,uint=True)
#    hdr_a = af[0].header
#    arc = af[0].data
#These are the best available frames (iodine cell out)
    if ts=='T1':
        flnmflat = 'n20160130.fiberflat_T1.0024.fits'
        flnmarc = 'n20160130.thar_T1_i2test.0025.fits'
#        continue
    elif ts=='T2':
        flnmflat = 'n20160130.fiberflat_T2.0021.fits'
        flnmarc = 'n20160130.thar_T2_i2test.0020.fits'
#        continue
    elif ts=='T3':
        flnmflat = 'n20160130.fiberflat_T3.0013.fits'
        flnmarc = 'n20160130.thar_T3_i2test.0012.fits'
#        continue
    elif ts=='T4':
        flnmflat = 'n20160130.fiberflat_T4.0016.fits'
        flnmarc = 'n20160130.thar_T4_i2test.0017.fits'
    else:
        print("{} is not a valid telescope".format(ts))
        continue

    #Import tungsten fiberflat
    fileflat = os.path.join(pathd,date,flnmflat)
    #fileflat = os.path.join(paths,'minerva_flat.fits')
    ff = pyfits.open(fileflat,ignore_missing_end=True,uint=True)
    hdr = ff[0].header
    ccd = ff[0].data
    
    #Import thorium argon arc
    filearc = os.path.join(pathd,date,flnmarc)
    #filearc = os.path.join(paths,'minerva_arc.fits')
    af = pyfits.open(filearc,ignore_missing_end=True,uint=True)
    hdr_a = af[0].header
    arc = af[0].data
    
    #Hardcoded characteristics of the trace
    num_fibers=29
    fiberspace = np.linspace(60,90,num_fibers)/2 #approximately half the fiber spacing
    fiberspace = fiberspace.astype(int)
    
    #Now to get traces
#    ypix = hdr['NAXIS1']  #Number of pixels
    ypix = 2048 #Manual, overscan is included in header value
    xpix = hdr['NAXIS2']
    num_points = 20 #2048= all pixels, probably overkill.  Maybe ~100 is good?
    yspace = int(floor(ypix/(num_points+1)))
    yvals = yspace*(1+np.arange(num_points))
    
    #Check lines to plot cross sections
    #for y in yvals:
    #    plt.plot(ccd[:,y])
    #    
    #plt.show()
    
    xtrace = np.zeros((num_fibers,num_points)) #xpositions of traces
    ytrace = np.zeros((num_fibers,num_points)) #ypositions of traces
    Itrace = np.zeros((num_fibers,num_points)) #relative intensity of flat at trace
    chi_vals = np.zeros((num_fibers,num_points)) #returned from fit_trace
#    bg_cutoff = 0.1*np.max(ccd) #won't work if have cosmic rays or similar
    bg_cutoff = 1.2*np.median(ccd)
    
    #find initial peaks (center is best in general, but edge is okay here)
    px = 1;
    trct = 0;
    while px<xpix:
        y = yvals[0]
        if ccd[px,y]>bg_cutoff and ccd[px,y]<ccd[px-1,y]: #not good for noisy
            xtrace[trct,0] = px-1
            ytrace[trct,0] = y
            px += 5 #jump past peak
            trct+=1
        else:
            px+=1
    if trct<num_fibers:
        xtrace[trct:,0] = np.nan*np.zeros((num_fibers-trct))
    
    for i in range(num_fibers):
        y = yvals[0]
        if not np.isnan(xtrace[i,0]):
            xtrace[i,0], Itrace[i,0], chi_vals[i,0] = fit_trace(xtrace[i,0],y,ccd)
        else:
            Itrace[i,0], chi_vals[i,0] = nan, nan  
    
    
    #Right now I miss one trace that starts off the edge, but okay enough for testing
    for i in range(1,len(yvals)):
        y = yvals[i]
        crsxn = ccd[:,y]
        ytrace[:,i] = y
        for j in range(num_fibers):
            if not np.isnan(xtrace[j,i-1]):
                #set boundaries
                lb = int(xtrace[j,i-1]-fiberspace[j])
                ub = int(xtrace[j,i-1]+fiberspace[j])
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
                    xtrace[j,i], Itrace[j,i], chi_vals[j,i] = fit_trace(xtrace[j,i],y,ccd)
                else:
                    xtrace[j,i], Itrace[j,i], chi_vals[j,i] = nan, nan, nan
            else:
                xtrace[j,i], Itrace[j,i], chi_vals[j,i] = nan, nan, nan
                
    #Finally fit x vs. y on traces.  Start with quadratic for simple + close enough
    trace_coeffs = np.zeros((3,num_fibers))
    trace_intense_coeffs = np.zeros((3,num_fibers))
    for i in range(num_fibers):
        #Given orientation makes more sense to swap x/y
        mask = ~np.isnan(xtrace[i,:])
        if len(mask[mask])<3:
            continue
        profile = np.ones((len(ytrace[i,:][mask]),3)) #Quadratic fit
        profile[:,1] = (ytrace[i,:][mask]-ypix/2)/ypix #scale data to get better fit
        profile[:,2] = ((ytrace[i,:][mask]-ypix/2)/ypix)**2
        noise = np.diag(chi_vals[i,:][mask])
        tmp_coeffs, junk = sf.chi_fit(xtrace[i,:][mask],profile,noise)
        trace_coeffs[0,i] = tmp_coeffs[0]
        trace_coeffs[1,i] = tmp_coeffs[1]
        trace_coeffs[2,i] = tmp_coeffs[2]
        tmp_coeffs2, junk = sf.chi_fit(Itrace[i,:][mask],profile,noise)
        trace_intense_coeffs[0,i] = tmp_coeffs2[0]
        trace_intense_coeffs[1,i] = tmp_coeffs2[1]
        trace_intense_coeffs[2,i] = tmp_coeffs2[2]
            
    #Plot to check traces
#    fig,ax = plt.subplots()
#    ax.pcolorfast(ccd)
#    for i in range(num_fibers):
#        ys = (np.arange(ypix)-ypix/2)/ypix
#        xs = trace_coeffs[2,i]*ys**2+trace_coeffs[1,i]*ys+trace_coeffs[0,i]
#        yp = np.arange(ypix)
#        plt.plot(yp,xs)
#        plt.show()
    #Check trace cross section
    #plt.plot(ccd[:,1000])
    #ys = (1000-ypix/2)/ypix
    #xs = zeros((num_fibers))
    #for i in range(num_fibers):
    #    xs[i] = trace_coeffs[2,i]*ys**2+trace_coeffs[1,i]*ys+trace_coeffs[0,i]
    #plt.plot(xs,np.max(ccd[:,1000])*ones((len(xs))),'r*')
    #plt.show()
             
             
    ############################################################
    #####Above is tracing - offsource to a different code#######
    ############ Below start arc calibration code ##############
    ############################################################
             
             
    ############################################################
    ### First use trace to find cross sectional intensity of 
    ### each arc (only those above ~1.5*background) as a
    ### function of pixel position.
    ############################################################        
             
    #Next  - need to assign a wavelength estimate to each point, then fit
    #legendre polynomials.  Then I should be able to input to specex (after
    #saving as a .fits file of course)
    ys = (np.arange(ypix)-ypix/2)/ypix
    xs = np.zeros((num_fibers,len(ys)))
    Is = np.zeros((num_fibers,len(ys)))
    arc_bg = np.median(arc) #approximate background estimate
    line_amps = np.zeros((num_fibers,ypix))
    nonzero_count = 0
    zero_count = 0
    line_gauss = np.zeros((num_fibers,ypix)) #intensity from gaussian fit
    for j in range(num_fibers):
        pad = 4
        for k in range(ypix):
            xs[j,k] = trace_coeffs[2,j]*ys[k]**2+trace_coeffs[1,j]*ys[k]+trace_coeffs[0,j]
            if np.isnan(xs[j,k]) or xs[j,k]<0:
                line_amps[j,k] = 0
                line_gauss[j,k] = 0
                continue
            xs[j,k] = int(xs[j,k])
            Is[j,k] = trace_intense_coeffs[2,j]*ys[k]**2+trace_intense_coeffs[1,j]*ys[k]+trace_intense_coeffs[0,j]
            xsub1 = np.arange(-pad,pad+1)+int(xs[j,k])
            xsub1 = xsub1[(xsub1>=0)*(xsub1<xpix)]
            zvals = arc[xsub1,k]
            if np.max(zvals)>(arc_bg+15): #Hack on setting background cutoff
                #Lines to find more precise zmax point
                zpeak_ind = np.argmax(zvals)
                zsub = zvals[zpeak_ind-2:zpeak_ind+3]
                xsub = np.arange(len(zsub))-2
                profile = np.ones((len(xsub),3)) #Quadratic fit
                profile[:,1] = xsub #scale data to get better fit
                profile[:,2] = xsub**2 
                noise = np.diag(1/zsub)
                coeffsn, junk = sf.chi_fit(zsub-arc_bg,profile,noise)
    #            if k==65:
    #                fit = coeffsn[2]*xsub**2 + coeffsn[1]*xsub + coeffsn[0] + arc_bg/1.5
    #                plt.plot(xsub,fit)
    #                plt.plot(xsub,zsub,'k')
    #                plt.show()
                xmx = -coeffsn[1]/(2*coeffsn[2])
                zmx = coeffsn[2]*xmx**2+coeffsn[1]*xmx+coeffsn[0]
                line_amps[j,k] = zmx[0]#/Is[j,k]
                '''
                #Lines to find sum of 10px square around the peak (total intensity)
                intensity = sum(arc[-5+xs[j,k]:6+xs[j,k],-5+k:6+k])
                line_amps[j,k] = intensity/Is[j,k]
                '''
                #print zmx[0]/Is
                pad2 = 7
                xsub = np.arange(-pad2,pad2+1)+int(xs[j,k])
                xsub = xsub[(xsub>=0)*(xsub<xpix)]
                zvals = arc[xsub,k]
                try:
                    params = sf.gauss_fit(xsub,zvals)
                except RuntimeError:
                    print("No fit for fiber {fib} at pixel {px}".format(fib=j,px=k))
                    line_gauss[j,k] = 0
                else:
                    line_gauss[j,k] = params[2] #-background?
    #            if j==0 and k==332:
    #                fit = sf.gaussian(xsub,abs(params[0]),params[1],params[2],params[3],params[4])
    #                plt.plot(xsub,zvals,xsub,fit)
    #                print xs[j,k]
    #                plt.show()
                nonzero_count+=1 
            else:
                line_amps[j,k] = 0
                line_gauss[j,k] = 0
                zero_count+=1
    #    if j==0:
    #        plt.plot(line_gauss[j,:])
    #        plt.show()
    #    if j==12:
    ##        plt.plot(line_gauss[j,:])
    ##        plt.show()
    #        for k in range(643,646):
    #            xregion = np.arange(-10,10+1)
    #            zvals = arc[xregion+int(xs[j,k]),k]
    #            plt.plot(xregion,zvals)
    #            pxn = np.linspace(xregion[0],xregion[-1],1000)
    #            fit = sf.gaussian(pxn,abs(params[0]),params[1],params[2],params[3],params[4])
    #            print(params)                  
    #            plt.plot(pxn,fit)
    #            plt.show()
                        
            
    
    #############################################
    ###INTENSITY AS SUM OF REGION AROUND PEAKS###
    ###Different method - probably will ignore###
    ### better just to use 2D formalism/fit   ###
    #############################################
    
    intensity = np.zeros(np.shape(line_amps))
    for j in range(num_fibers):
        pk_width = 6
        peaks = sig.find_peaks_cwt(line_amps[j,:],np.arange(4,5))
        pk_ind = 0
        for k in range(ypix):
            if pk_ind >= len(peaks):
                break
            elif peaks[pk_ind]==k:
                #if there is a peak here, sum in a box around it
                intensity[j,k] = sum(arc[-pk_width+xs[j,k]:pk_width+1+xs[j,k],
                                     -pk_width+k:pk_width+k])
                if j==0 and k==0:
                    print xs[j,k]
                    print k
                #subtract off background
                intensity[j,k]-=arc_bg*(2*pk_width)**2
                #scale by flat response (assume roughly constant over peak)
                intensity[j,k]/=Is[j,k]
                pk_ind+=1
            else:
                intensity[j,k] = 0
        
    
    #xtraces start at red, move toward blue.  Want to reverse from proper ordering
    #line_amps = line_amps[::-1,:]
    line_gauss = line_gauss[::-1,:]
    #Temporary hack to shift T3 lines to match T1 and T2
    if ts == 'T3' or ts == 'T4':
        line_gauss = np.vstack((line_gauss[1:29,:],np.zeros((1,np.shape(line_gauss)[1]))))
    intensity = intensity[::-1,:]/np.max(intensity)
#    if ts == 'T4':
#        for i in range(num_fibers):
#            plt.plot(line_gauss[i])
#            plt.show()
#            plt.close()
    
    ############################################################
    ### Open line list (hardcoded for now - user input later) ##
    ############################################################ 
    
    wl_min = 4874
    wl_max = 6466
    pathm = os.environ['MINERVA_SIM_DIR']
    rows = 8452
    cols = 2
    lamp_lines = np.zeros((rows,cols))
    line_names = np.chararray((rows,1),itemsize=6)
    with open(pathm+'table1.dat','r') as ll:
        ln = ll.readlines()
        for i in range(rows):
            lamp_lines[i,0] = ln[i][0:11]
            lamp_lines[i,1] = ln[i][33:40]
            line_names[i] = ln[i][54:-1]
        
    wls = lamp_lines[:,0] #wavelengths
    its = lamp_lines[:,1] #relative intensities (may not need)
    itmax = max(its)
    itssc = its/itmax
    
    ############################################################
    ###Make cuts to wls, its based on proximity and intensity###
    ############################################################
    
    wlref = wls
    itref = itssc
    
    #Don't do intensity masking for now
    itmask = itref>(0.01*np.max(itref)) #Mask bottom 1%
    wlref = wlref[itmask]
    itref = itref[itmask]
    
    blnd_wid = 0.5 #make user input
    idx = 0
    while idx < len(wlref)-1:
        if abs(wlref[idx]-wlref[idx+1])<blnd_wid:
            wlref = np.delete(wlref,idx)
            wlref = np.delete(wlref,idx)
            itref = np.delete(itref,idx)
            itref = np.delete(itref,idx)
        else:
            idx+=1
            
         
    ############################################################
    ### Now make cuts to "extracted" spectrum from arc frame.
    ### These can include close lines, anomalous FWHM, low 
    ### intensity, and large slope in the background.
    ### To start I will just go with "too close" lines
    ############################################################
       
    sampling_est = 3 #Whole number estimate of sampling
    #use dictionaries since number of peaks per fiber varies
    mx_it_d = dict() #max intensities of each peak
    fwhm_d = dict() #FWHM of each peak
    pos_d = dict() #position of each peak
    slp_d = dict() #background slope around each peak
    for i in range(num_fibers):
        ##Optional - use scipy to get initial guesses of peak locations
        ##Problem is this misses many of the low amplitude peaks.
        #pos_est = np.array(sig.find_peaks_cwt(line_gauss[i,:],np.arange(3,4)))
        #Since spectrum has ~no background, can use my own peak finder.
        pos_est = np.zeros((len(line_gauss[i,:])),dtype=int)
        for j in range(2*sampling_est,len(line_gauss[i,:])-2*sampling_est):
            #if point is above both its neighbors, call it a peak
            if line_gauss[i,j]>line_gauss[i,j-1] and line_gauss[i,j]>line_gauss[i,j+1]:
                pos_est[j] = j
        #Then remove extra elements from pos_est
        pos_est = pos_est[np.nonzero(pos_est)[0]]
        #Cut out any that are within 2*sampling of each other (won't be able to fit well)
        #Optional - add fitting algorithm for overlapping gaussians to recover these
        #blended lines
        pos_diff = ediff1d(pos_est)
        if np.count_nonzero(pos_diff<(2*sampling_est))>0:
            close_inds = np.nonzero(pos_diff<(2*sampling_est))[0]
            close_inds = np.concatenate((close_inds,close_inds+1))
            close_inds = np.unique(close_inds)
            close_inds = np.sort(close_inds)
            pos_est = np.delete(pos_est,close_inds)
        #variable length arrays to dump into dictionary
        num_pks = len(pos_est)
        mx_it = zeros((num_pks))
        fwhm = zeros((num_pks))
        pos = zeros((num_pks))
        slp = zeros((num_pks))
        #Now fit gaussian with background to each (can improve function later)
        for j in range(num_pks):
            xarr = pos_est[j] + np.arange(-(2*sampling_est),(2*sampling_est),1)
            xarr = xarr[(xarr>0)*(xarr<2048)]
            yarr = line_gauss[i,:][xarr]
            try:
                params = sf.gauss_fit(xarr,yarr)
            except RuntimeError:
                params = np.zeros(5)
            mx_it[j] = params[2] #height
            fwhm[j] = params[0]*2*sqrt(2*log(2)) #converted from std dev
            pos[j] = params[1] #center
            slp[j] = params[4] #bg_slope
        mx_it_d[i] = mx_it[np.nonzero(pos)[0]] #Remove zero value points
        fwhm_d[i] = fwhm[np.nonzero(pos)[0]]
        pos_d[i] = pos[np.nonzero(pos)[0]] 
        slp_d[i] = slp[np.nonzero(pos)[0]]
            
      
    #Find mean and std FWHM and slope
    fwhm_arr = np.array(())
    slp_arr = np.array(())
    for i in range(num_fibers):
        fwhms = fwhm_d[i][np.nonzero(fwhm_d[i])[0]]
        fwhm_arr = np.concatenate((fwhm_arr,fwhms))
        slps = slp_d[i][np.nonzero(slp_d[i])[0]]
        slp_arr = np.concatenate((slp_arr,slps))
        
    #Can use more advanced statistics if needed
    fwhm_mean = np.mean(fwhm_arr)
    fwhm_std = np.std(fwhm_arr)
    slp_mean = np.mean(slp_arr)
    slp_std = np.std(slp_arr)
    #plt.plot(slp_arr,'k.')
    
      
    ### Make cuts (for now don't do anything)
    #px_wid = blnd_wid/(45/2048) #Change this to user input also
    #print "px_wid = ", px_wid
    #
    #for i in range(num_fibers):
    #    pos = pos_d[i]
    #    if len(pos)>0:
    ##        fwhm = fwhm_d[i]
    #        mx_it = mx_it_d[i]
    ##        slp = slp_d[i]
    #        msk = np.ones((len(pos)),dtype=np.bool)
    ##        pos_diff = ediff1d(pos)
    ##        if np.count_nonzero(pos_diff<px_wid)>0:
    ##            close_inds = np.nonzero(pos_diff<px_wid)[0]
    ##            close_inds = np.concatenate((close_inds,close_inds+1))
    ##            close_inds = np.unique(close_inds)
    ##            close_inds = np.sort(close_inds)
    ##            msk[close_inds] = np.zeros((len(close_inds)))
    #        for j in range(len(pos)):
    ##            #FWHM cut
    ###            if fwhm[j]>(fwhm_mean+fwhm_std) or fwhm[j]<(fwhm_mean-fwhm_std):
    ###                msk[j] = 0
    #            #Weak lines cut
    #            if mx_it[j] < 0.01*np.max(line_gauss[i,:]):#6*arc_bg:
    #                msk[j] = 0
    ##            #Slope cut
    ###            if abs(slp[j]) > 10: #semi-arbitrarily set
    ###                msk[j] = 0
    #        pos_strong_lines = pos[msk]
    #        msk_strong_lines = np.ones((len(pos_strong_lines)),dtype=np.bool)
    #        pos_diff = ediff1d(pos_strong_lines)
    #        if np.count_nonzero(pos_diff<px_wid)>0:
    #            close_inds = np.nonzero(pos_diff<px_wid)[0]
    #            close_inds = np.concatenate((close_inds,close_inds+1))
    #            close_inds = np.unique(close_inds)
    #            close_inds = np.sort(close_inds)
    #            msk_strong_lines[close_inds] = np.zeros((len(close_inds)))
    #        #update postion array with only unmasked values
    #        pos_d[i] = pos_strong_lines[msk_strong_lines]
    ##        mx_it_d[i] = mx_it[msk==1]
    
    
    
    ############################################################
    ### Next, for each non-cut peak, search for nearby line from
    ### line list (must use inputs, lamba_0, lambda_f, px_0, px_f)
    ### Apply wavelength to pixel for all matches, then fit the
    ### result to a polynomial. If fit is poor (or maybe even if
    ### not), run iterative sigma clipping and re-fit.  Idea is
    ### that outliers come from unidentified lines and I can't
    ### yet think of a better way to remove these from consideration
    ############################################################
    
    lambdas = np.zeros((num_fibers,5)) #input 5 points by eye
    pixels = np.zeros((num_fibers,5))
    skipsets = 2 #skip the first two sets of 5 - these don't show on CCD
    ### Import estimates from file
    ### TODO should think of an easier format for simple imports
    with open(pathm + 'arc_order_estimates.csv') as arccsv:
        orders = csv.reader(arccsv,delimiter=',')
        ct = 0
        fib = 0
        pt = 0
        for row in orders:
            if ct/8 < skipsets:
                ct += 1
                continue
            elif ct%8 == 0 or (ct-1)%8 == 0:
                ct += 1
                continue            
            elif (ct+1)%8 == 0:
                pt = 0
                fib += 1
                ct += 1
            else:
                lambdas[fib,pt] = row[0]
                pixels[fib,pt] = row[1]
                ct += 1
                pt += 1
    #approximate correction to my initial estimates
    if ts == 'T1':
        coeffs = np.array(([8.9617e-3,9.9478e1]))
        pixels+=np.poly1d(coeffs)(pixels)
    elif ts == 'T2':
        coeffs = np.array(([9.412566e-3,1.0294e2]))
        pixels+=np.poly1d(coeffs)(pixels)
    elif ts == 'T3':
        coeffs = np.array(([9.68858e-3,1.062277e2]))
        pixels+=np.poly1d(coeffs)(pixels)
    elif ts == 'T4':
        coeffs = np.array(([1.0007e-2,1.093839e2]))
        pixels+=np.poly1d(coeffs)(pixels)
    ### Fit initial guesses to polynomial
    pixels = 2*(pixels-ypix/2)/ypix #change interval to [-1,1]
    init_poly_order = 2
    lam_vs_px_coeffs = np.zeros((num_fibers,init_poly_order+1))
    for i in range(num_fibers):
        if lambdas[i,0] != 0:
            lam_vs_px_coeffs[i] = np.polyfit(pixels[i],lambdas[i],init_poly_order)
#            fill = 2*(np.arange(ypix)-ypix/2)/ypix
#            lam_fill = np.poly1d(lam_vs_px_coeffs[i])(fill)
#            plt.plot(pixels[i],lambdas[i],fill,lam_fill)
#            plt.plot(lam_fill,line_gauss[i])
#            plt.show()
#            plt.close()
            
    ### Now we want to dial in a more precise wavelength solution
    ### TODO - figure out how to use the iodine values
    fin_poly_order = 6
    blnd_wid = 0.06 #Angstroms, allowable error from initial guess to reference, 0.06 is conservative, resistant to blends
    lam_vs_px_final = np.zeros((num_fibers,fin_poly_order+1))
    for i in range(num_fibers):
#        if ts == 'T1' and i == 0: #Hack, skip the first T1 fiber
#            continue
#        elif ts == 'T1' and i>0:  #Hack, re-index subsequent T1 fibers
#            i-=1
        line_px = 2*(pos_d[i]-ypix/2)/ypix
        line_lams = np.poly1d(lam_vs_px_coeffs[i])(line_px)
        ref_lams = np.zeros((len(line_lams)))
        for j in range(len(line_lams)):
            #find reference wavelengths in this range
            ref_wavelength = wls[(wls>(line_lams[j]-blnd_wid))*
                                   (wls<(line_lams[j]+blnd_wid))]
            if len(ref_wavelength)==1:
                ref_lams[j] = ref_wavelength
        lam_msk = np.nonzero(ref_lams)[0]
        ref_lams = ref_lams[lam_msk] #cut out any zero points
        line_px = line_px[lam_msk]
        if len(line_px) < fin_poly_order:
            #fin_poly_tmp = 2
            print("Fiber {} has too few lines to fit polynomial order {}".format(i,fin_poly_order))
            #For now, just use the initial fit
            lam_vs_px_final[i] = np.pad(lam_vs_px_coeffs[i],(fin_poly_order-init_poly_order,0),'constant',constant_values=(0,0))
            continue
        else:
            lam_vs_px_final[i] = np.polyfit(line_px,ref_lams,fin_poly_order)
#        if ts == 'T1':
##        #    print ypix*(line_px)/2 + ypix/2
##        #    print ref_lams
#            print ypix
#            fill = 2*(np.arange(ypix)-ypix/2)/ypix
#            lam_fill = np.poly1d(lam_vs_px_final[i])(fill)
#            lam_fit = np.poly1d(lam_vs_px_final[i])(line_px)
##    #        plt.plot(line_px,ref_lams-lam_fit,'r.',markersize=6)
##    #        plt.plot(line_px,np.zeros((len(line_px))),'k',linewidth=2)
#            plt.plot(lam_fill,line_gauss[i,:])
##    #        plt.plot(wls,np.zeros((len(wls))),'c.',markersize=10)
#            plt.show()
#            plt.close() 
        
    ###################### Save values to file ####################################
    redux_dir = os.environ['MINERVA_REDUX_DIR']    
    savedate = 'n20160216'
    hdu = pyfits.PrimaryHDU(lam_vs_px_final)
    hdu.header.append(('POLYORD',fin_poly_order,'Polynomial order used for fitting'))
    hdulist = pyfits.HDUList([hdu])
    hdulist.writeto(os.path.join(redux_dir,savedate,'wavelength_soln_'+ts+'.fits'),clobber=True)
            
    #lambda_0 = np.loadtxt(os.path.join(pathm,'lambda0est.txt')) + 4 #4 is estimated RV shift for solar spectrum
    #lambda_diff = np.loadtxt(os.path.join(pathm,'lambda_diff_est.txt'))
    #dispersion_est = lambda_diff/ypix
    #lambda_f = lambda_0+lambda_diff
    #px_0 = np.ones((num_fibers))
    #px_f = 2048*np.ones((num_fibers))
    #poly_order = 3 #Change to user input
    #wavelength_coeffs = np.zeros((num_fibers,poly_order+1))
    #for i in range(num_fibers):
    #    l0 = lambda_0[i]
    #    lf = lambda_f[i]
    #    p0 = px_0[i]
    #    pf = px_f[i]
    #    
    #    lambda_slope = (lf-l0)/(pf-p0)
    #    pos_lambda_est = lambda_slope*pos_d[i] + l0 #completely linear to start
    #    pos_msk = np.ones((len(pos_lambda_est)),dtype=bool)
    #    for j in range(len(pos_lambda_est)):
    #        #find reference wavelengths in this range
    #        ref_wavelength = wlref[(wlref>(pos_lambda_est[j]-blnd_wid))*
    #                               (wlref<(pos_lambda_est[j]+blnd_wid))]
    #        if len(ref_wavelength)==1:
    #            pos_lambda_est[j] = ref_wavelength
    #        else:
    #            pos_msk[j] = False
    #    #Now fit for polynomial coefficients (make separate function for this)
    #    pos_fit_est = pos_lambda_est[pos_msk]
    #    print("length pos_fit_est = {} and len pos_msk = {}".format(len(pos_fit_est),len(pos_msk)))
    #    #zero center and normalize pixels for fitting
    #    pix_est = (pos_d[i][pos_msk] - ypix/2)/ypix
    #    #Make generic for polynomial order
    #    P = np.vstack((np.ones(len(pix_est)),pix_est,pix_est**2,pix_est**3))
    #    P = P.T
    #    N = np.eye(len(pix_est)) #How should I actually estimate noise?
    #    coeffs, chi = sf.chi_fit(pos_fit_est,P,N)
    ##    print chi, len(pix_est)
    #    wavelength_coeffs[i] = coeffs.T
    #    #Now visualize to check
    ##    plt.figure("Order {}".format(i+1))
    ##    plt.plot((pix_est*ypix+ypix/2),pos_fit_est,'b.')
    #    fit_px = (np.arange(ypix)-ypix/2)/ypix
    #    fit_wl = np.zeros((ypix))
    #    for k in range(len(coeffs)):
    #        fit_wl+=coeffs[k]*fit_px**k
    ##    plt.plot(fit_px*ypix+ypix/2,fit_wl,'g')
    ##    plt.figure("Extracted Profile")
    ##    plt.plot(line_gauss[i,:])
    ##    plt.show()
    ##    plt.plot(fit_wl,line_gauss[i,:]/np.max(line_gauss))
    #    est_wl = lambda_slope*np.arange(ypix) + l0
    #    plt.plot(est_wl,line_gauss[i,:]/np.max(line_gauss[i,:]))
    #    sf.plt_deltas(wlref,itref/np.max(itref),'k')
    #    plt.show()
    #    plt.close()
    
    
    #SR = 3*80e3#fwhm_mean*80e3
    #wsc = 49
    #A = wsc/(np.exp((ypix-1)/SR)-1)
    #ws = 6407+21.49
    #wlg1 = A*(np.exp(np.arange(ypix)/SR)-1)+ws
    #wlg2 = np.linspace(ws,ws+wsc,ypix)
    
    
    
    
    
    
    #px = np.arange(1965,1980)
    #val = ccd[:,1000][px]
    #params = sf.gauss_fit(px,val,fit_background='n')
    #pxn = np.linspace(px[0],px[-1],1000)
    #fit = sf.gaussian(pxn,abs(params[0]),params[1],params[2],params[3],params[4])
    #plt.plot(px,val,pxn,fit)
    
    #for i in range(num_fibers):
    #    plt.figure(i)
    #    plt.plot(line_amps[i,:])
    #    plt.plot(line_gauss[i,:])
    #    plt.show()