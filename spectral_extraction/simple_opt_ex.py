#!/usr/bin/env python 2.7

#Start of a generic tracefit program.  Geared now toward MINERVA initial data

#Import all of the necessary packages
from __future__ import division
import pyfits
import os
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
import argparse

#########################################################
########### Allow input arguments #######################
#########################################################
parser = argparse.ArgumentParser()
parser.add_argument("-f","--filename",help="Name of image file (.fits) to extract",
                    default='n20160115.daytimeSky.0006.fits')
parser.add_argument("-fib","--num_fibers",help="Number of fibers to extract",
                    type=int,default=30*3)
parser.add_argument("-bs","--bundle_space",help="Minimum spacing (in pixels) between fiber bundles",
                    type=int,default=40)
parser.add_argument("-fs","--fiber_space",help="Minimum spacing (in pixels) between fibers within a bundle",
                    type=int,default=13)
parser.add_argument("-fb","--fibers_per_bundle",help="Number of fibers in a bundle",
                    type=int,default=3) 
parser.add_argument("-np","--num_points",help="Number of trace points to fit on each fiber",
                    type=int,default=20)
args = parser.parse_args()

#########################################################
##### Function for trace fitting (move later) ###########
#########################################################
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
        params = sf.gauss_fit(xvals,zvals)
        xc = x+params[1] #offset plus center
        zc = params[2] #height (intensity)
#        pxn = np.linspace(xvals[0],xvals[-1],1000)
        fit = sf.gaussian(xvals,abs(params[0]),params[1],params[2],params[3],params[4])
        chi = sum((fit-zvals)**2/zvals)
        return xc, zc, chi


#########################################################
########### Load Background Requirments #################
#########################################################

pathd = os.environ['MINERVA_DATA_DIR']
redux_dir = os.environ['MINERVA_REDUX_DIR']
#hardcode in n20160115 directory
filename = os.path.join(pathd,'n20160115',args.filename)



spectrum = pyfits.open(filename,uint=True)
hdr = spectrum[0].header
ccd = spectrum[0].data

#####CONVERT NASTY FORMAT TO ONE THAT ACTUALLY WORKS#####
#Dimensions
ypix = hdr['NAXIS1']
xpix = hdr['NAXIS2']

#Test to make sure this logic is robust enough for varying inputs
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

ccd = ccd_16bit[::-1,0:2048] #Remove overscan region

#########################################################
########### Fit traces to spectra #######################
#########################################################

print("Searching for Traces")

num_fibers = args.num_fibers
num_points = args.num_points
yspace = int(floor(ypix/(num_points+1)))
yvals = yspace*(1+np.arange(num_points))

xtrace = np.zeros((num_fibers,num_points)) #xpositions of traces
ytrace = np.zeros((num_fibers,num_points)) #ypositions of traces
Itrace = np.zeros((num_fibers,num_points)) #relative intensity of flat at trace
chi_vals = np.zeros((num_fibers,num_points)) #returned from fit_trace
bg_cutoff = 1.5*np.median(ccd) #won't fit values below this intensity

###find initial peaks (center is best in general, but edge is okay here)
px = 1;
trct = 0;
while px<xpix:
    if trct>=num_fibers:
        break
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


###From initial peak guesses fit for more precise location
for i in range(num_fibers):
    y = yvals[0]
    if not np.isnan(xtrace[i,0]):
        xtrace[i,0], Itrace[i,0], chi_vals[i,0] = fit_trace(xtrace[i,0],y,ccd)
    else:
        Itrace[i,0], chi_vals[i,0] = nan, nan  


for i in range(1,len(yvals)):
    y = yvals[i]
    crsxn = ccd[:,y]
    ytrace[:,i] = y
    for j in range(num_fibers):
        if not np.isnan(xtrace[j,i-1]):
            #set boundaries
            lb = int(xtrace[j,i-1]-args.fiber_space/2)
            ub = int(xtrace[j,i-1]+args.fiber_space/2)
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
    profile = np.ones((len(ytrace[i,:][mask]),3)) #Quadratic fit
    profile[:,1] = (ytrace[i,:][mask]-ypix/2)/ypix #scale data to get better fit
    profile[:,2] = ((ytrace[i,:][mask]-ypix/2)/ypix)**2
    noise = np.diag(chi_vals[i,:][mask])
    if len(xtrace[i,:][mask])>3:
        tmp_coeffs, junk = sf.chi_fit(xtrace[i,:][mask],profile,noise)
    else:
        tmp_coeffs = nan*np.ones((3))
    trace_coeffs[0,i] = tmp_coeffs[0]
    trace_coeffs[1,i] = tmp_coeffs[1]
    trace_coeffs[2,i] = tmp_coeffs[2]
#    tmp_coeffs2, junk = sf.chi_fit(Itrace[i,:][mask],profile,noise)
#    trace_intense_coeffs[0,i] = tmp_coeffs2[0]
#    trace_intense_coeffs[1,i] = tmp_coeffs2[1]
#    trace_intense_coeffs[2,i] = tmp_coeffs2[2]
#      

###Plot to visualize traces      
#fig,ax = plt.subplots()
#ax.pcolorfast(ccd)
#for i in range(num_fibers):
#    ys = (np.arange(ypix)-ypix/2)/ypix
#    xs = trace_coeffs[2,i]*ys**2+trace_coeffs[1,i]*ys+trace_coeffs[0,i]
#    yp = np.arange(ypix)
#    plt.plot(yp,xs)
#plt.show()
      
      
#########################################################
########### Load Flat and Bias Frames ###################
#########################################################   
date = 'n20160115'
bias_hdu = pyfits.open(os.path.join(redux_dir,date,'bias_avg.fits'),uint=True)
bias = bias_hdu[0].data
sflat_hdu = pyfits.open(os.path.join(redux_dir,date,'slit_approx.fits'),uint=True)
slit_coeffs = sflat_hdu[0].data
#slit_coeffs = slit_coeffs[::-1,:] #Re-order, make sure this is right
polyord = sflat_hdu[0].header['POLYORD'] #Order of polynomial for slit fitting

### subtract bias (slit will be handled in loop)
bias = bias[:,0:2048] #Remove overscan
ccd -= bias #Note, if ccd is 16bit array, this operation can cause problems
ccd[ccd<0] = 0 #Enforce positivity
      
#########################################################
########### Now Extract the Spectrum ####################
#########################################################

print("Starting Extraction")

yspec = np.arange(0,2048) #"Y" spectrum is just pixel values
zspec = np.zeros((num_fibers,2048)) #relative intensity at each point
zspec2 = np.zeros((num_fibers,2048))

for i in range(num_fibers):
#    if i < 82 or i >= 83: #Manually skip first two fibers since they don't work
#        continue 
    slit_num = np.floor((i)/3)
    print("starting on trace {}".format(i+1))
    for j in range(2048):
        yj = (yspec[j]-2048/2)/2048
        xj = trace_coeffs[2,i]*yj**2+trace_coeffs[1,i]*yj+trace_coeffs[0,i]
        if np.isnan(xj):
            zspec[i,j] = nan
        else:
            try:
                xpad = 7
                xvals = np.arange(-xpad,xpad+1)
                xj = int(xj)
                xwindow = xj+xvals
                xvals = xvals[(xwindow>=0)*(xwindow<xpix)]
                slitvals = np.poly1d(slit_coeffs[slit_num,j])(xj+xvals)
                zvals = ccd[xj+xvals,yspec[j]]/slitvals
                zorig = ccd[xj+xvals,yspec[j]]
#                if j>=698 and j<=700:
#                    plt.plot(zvals/0.33)
#                    print slitvals
#                    plt.plot(zvals/slitvals)
#                    plt.show()
#                    plt.close()
#                if np.max(zvals)>bg_cutoff:
                params = sf.gauss_fit(xvals,zvals)
                paramsorig = sf.gauss_fit(xvals,zorig)
                fit = sf.gaussian(xvals,abs(params[0]),params[1],params[2],params[3],params[4])
#                    plt.plot(xvals,zvals,xvals,fit)
#                    plt.show()
                zspec[i,j] = params[2]
                zspec2[i,j] = paramsorig[2]
#                if abs(params[2]/np.max(zvals)-paramsorig[2]/np.max(zorig))/(paramsorig[2]/np.max(zorig)) > 0.05:
#                    print("Bad fit on fiber {} at pixel {}".format(i,j))
#                    print params[2]
#                    print paramsorig[2]
#                    time.sleep(0.5)
#                print params[2]
#                else:
#                    zspec[i,j] = 0
            except RuntimeError:
                zspec[i,j] = nan
        
############################################################
######### Import wavelength calibration ####################        
############################################################
        
wl_hdu = pyfits.open(os.path.join(redux_dir,date,'wavelength_soln.fits'))
wl_coeffs =  wl_hdu[0].data
#wl_coeffs = wl_coeffs[::-1,:] #Re-order, need to be sure this is consistent
wl_polyord = wl_hdu[0].header['POLYORD']
ypx_mod = 2*(yspec-2048/2)/2048
wavelength_soln = np.zeros((num_fibers,2048))
for i in range(num_fibers):
#    if i < 2:
#        continue
    wl_fib = np.floor((i)/3)
    wavelength_soln[i] = np.poly1d(wl_coeffs[wl_fib])(ypx_mod)

###Plot orders for visualization
#for i in range(num_fibers):
#    if mod(i,3)==0:
#        ir = num_fibers-i-1
#    #    if ~np.isnan(np.max(zspec[ir,:])):
#        plt.plot(zspec[ir,:])#wavelength_soln[ir,:],zspec[ir,:])
#        plt.plot(zspec2[ir,:])#wavelength_soln[ir,:],zspec2[ir,:])
#        plt.show()
#        plt.close()
        
hdu1 = pyfits.PrimaryHDU(zspec)
hdu2 = pyfits.PrimaryHDU(zspec2)
hdu3 = pyfits.PrimaryHDU(wavelength_soln)
hdu1.header.append(('FIRSTFIB','T1','Telescope assocated with first fiber'))
hdu1.header.append(('DATA','Flux','Relative counts at each extracted pixel value'))
hdu2.header.append(('DATA','FluxNoFlat','Extracted counts without slit flat scaling'))
hdu3.header.append(('DATA','Wavelength','Approximate wavelength solution lambda vs px'))
hdulist = pyfits.HDUList([hdu1])
hdulist.append(hdu2)
hdulist.append(hdu3)
hdulist.writeto(os.path.join(redux_dir,date,'minerva1d_test.fits'),clobber=True)