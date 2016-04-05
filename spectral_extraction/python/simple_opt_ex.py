#!/usr/bin/env python

#Start of a generic tracefit program.  Geared now toward MINERVA initial data

#Import all of the necessary packages
from __future__ import division
import pyfits
import os
#import math
import time
import numpy as np
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
#import scipy.linalg as linalg
#import solar
import special as sf
import argparse


t0 = time.time()

######## Import environmental variables #################
data_dir = os.environ['MINERVA_DATA_DIR']
redux_dir = os.environ['MINERVA_REDUX_DIR']

#########################################################
########### Allow input arguments #######################
#########################################################
parser = argparse.ArgumentParser()
parser.add_argument("-f","--filename",help="Name of image file (.fits) to extract",
                    default=os.path.join(data_dir,'n20160216','n20160216.HR2209.0025.fits'))
#                    default=os.path.join(data_dir,'n20160115','n20160115.daytimeSky.0006.fits'))
parser.add_argument("-fib","--num_fibers",help="Number of fibers to extract",
                    type=int,default=29)
parser.add_argument("-bs","--bundle_space",help="Minimum spacing (in pixels) between fiber bundles",
                    type=int,default=40)
parser.add_argument("-fs","--fiber_space",help="Minimum spacing (in pixels) between fibers within a bundle",
                    type=int,default=13)
parser.add_argument("-ts","--telescopes",help="Number of telescopes feeding spectrograph",
                    type=int,default=4) 
parser.add_argument("-np","--num_points",help="Number of trace points to fit on each fiber",
                    type=int,default=20)
parser.add_argument("-ns","--nosave",help="Don't save results",
                    action='store_true')
#parser.add_argument("-T","--tscopes",help="T1, T2, T3, and/or T4 (remove later)",
#                    type=str,default=['T1','T2','T3','T4'])
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
        params, errarr = sf.gauss_fit(xvals,zvals,fit_exp='y')
        xc = x+params[1] #offset plus center
        zc = params[2] #height (intensity)
        sig = params[0] #standard deviation
        power = params[5]
#        pxn = np.linspace(xvals[0],xvals[-1],1000)
        fit = sf.gaussian(xvals,abs(params[0]),params[1],params[2],params[3],params[4],params[5])
        chi = sum((fit-zvals)**2/zvals)
        return xc, zc, abs(sig), power, chi


#########################################################
########### Load Background Requirments #################
#########################################################

#hardcode in n20160115 directory
filename = args.filename#os.path.join(data_dir,'n20160115',args.filename)
software_vers = 'v0.1' #Later grab this from somewhere else

gain = 1.3
readnoise = 3.63

spectrum = pyfits.open(filename,uint=True)
spec_hdr = spectrum[0].header
ccd = spectrum[0].data

#####CONVERT NASTY FORMAT TO ONE THAT ACTUALLY WORKS#####
#Dimensions
ypix = spec_hdr['NAXIS1']
xpix = spec_hdr['NAXIS2']
### Next part checks if iodine cell is in, assumes keyword I2POSAS exists
if spec_hdr['I2POSAS']=='in':
    i2 = True
else:
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

for ts in ['T1','T2','T3','T4']:
    #Choose fiberflats with iodine cell in
    if ts=='T1':
        flnmflat = 'n20160130.fiberflat_T1.0023.fits'
        flnmarc = 'n20160130.thar_T1_i2test.0025.fits'
    #        continue
    elif ts=='T2':
        flnmflat = 'n20160130.fiberflat_T2.0022.fits'
        flnmarc = 'n20160130.thar_T2_i2test.0020.fits'
    #        continue
    elif ts=='T3':
        flnmflat = 'n20160130.fiberflat_T3.0014.fits'
        flnmarc = 'n20160130.thar_T3_i2test.0012.fits'
    #        continue
    elif ts=='T4':
        flnmflat = 'n20160130.fiberflat_T4.0015.fits'
        flnmarc = 'n20160130.thar_T4_i2test.0017.fits'
    else:
        print("{} is not a valid telescope".format(ts))
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

#ccd = arc_ccd #Temporary, just to examine fiber vs. wavelength
#ccd -= np.median(ccd)
#ccd[ccd<0] = 0
#i2 = False

print("Searching for Traces")

num_fibers = args.num_fibers*args.telescopes
num_points = args.num_points
yspace = int(np.floor(ypix/(num_points+1)))
yvals = yspace*(1+np.arange(num_points))

xtrace = np.zeros((num_fibers,num_points)) #xpositions of traces
ytrace = np.zeros((num_fibers,num_points)) #ypositions of traces
sigtrace = np.zeros((num_fibers,num_points)) #standard deviation along trace
powtrace = np.zeros((num_fibers,num_points)) #pseudo-gaussian power along trace
Itrace = np.zeros((num_fibers,num_points)) #relative intensity of flat at trace
chi_vals = np.zeros((num_fibers,num_points)) #returned from fit_trace
bg_cutoff = 1.05*np.median(trace_ccd) #won't fit values below this intensity

###find initial peaks (center is best in general, but edge is okay here)
px = 1;
trct = 0;
while px<xpix:
    if trct>=num_fibers:
        break
    y = yvals[0]
    if trace_ccd[px,y]>bg_cutoff and trace_ccd[px,y]<trace_ccd[px-1,y] and trace_ccd[px-1,y]>trace_ccd[px-2,y]: #not good for noisy
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
        xtrace[i,0], Itrace[i,0], sigtrace[i,0], powtrace[i,0], chi_vals[i,0] = fit_trace(xtrace[i,0],y,trace_ccd)
    else:
        Itrace[i,0], sigtrace[i,0], powtrace[i,0], chi_vals[i,0] = np.nan, np.nan, np.nan, np.nan


for i in range(1,len(yvals)):
    y = yvals[i]
    crsxn = trace_ccd[:,y]
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
                xtrace[j,i], Itrace[j,i], sigtrace[j,i], powtrace[j,i], chi_vals[j,i] = fit_trace(xtrace[j,i],y,trace_ccd)
            else:
                xtrace[j,i], Itrace[j,i], sigtrace[j,i], sigtrace[j,i], chi_vals[j,i] = np.nan, np.nan, np.nan, np.nan, np.nan
        else:
            xtrace[j,i], Itrace[j,i], sigtrace[j,i], sigtrace[j,i], chi_vals[j,i] = np.nan, np.nan, np.nan, np.nan, np.nan
            
Itrace /= np.median(Itrace) #Rescale intensities

#Finally fit x vs. y on traces.  Start with quadratic for simple + close enough
trace_coeffs = np.zeros((3,num_fibers))
trace_intense_coeffs = np.zeros((3,num_fibers))
trace_sig_coeffs = np.zeros((3,num_fibers))
trace_pow_coeffs = np.zeros((3,num_fibers))
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
    trace_coeffs[0,i] = tmp_coeffs[0]
    trace_coeffs[1,i] = tmp_coeffs[1]
    trace_coeffs[2,i] = tmp_coeffs[2]
    trace_intense_coeffs[0,i] = tmp_coeffs2[0]
    trace_intense_coeffs[1,i] = tmp_coeffs2[1]
    trace_intense_coeffs[2,i] = tmp_coeffs2[2]
    trace_sig_coeffs[0,i] = tmp_coeffs3[0]
    trace_sig_coeffs[1,i] = tmp_coeffs3[1]
    trace_sig_coeffs[2,i] = tmp_coeffs3[2]      
    trace_pow_coeffs[0,i] = tmp_coeffs4[0]
    trace_pow_coeffs[1,i] = tmp_coeffs4[1]
    trace_pow_coeffs[2,i] = tmp_coeffs4[2]      


###Plot to visualize traces      
#fig,ax = plt.subplots()
#ax.pcolorfast(trace_ccd)
#for i in range(num_fibers):
#    ys = (np.arange(ypix)-ypix/2)/ypix
#    xs = trace_coeffs[2,i]*ys**2+trace_coeffs[1,i]*ys+trace_coeffs[0,i]
#    yp = np.arange(ypix)
#    plt.plot(yp,xs)
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
ccd -= bias #Note, if ccd is 16bit array, this operation can cause problems
ccd[ccd<0] = 0 #Enforce positivity
      
#########################################################
########### Now Extract the Spectrum ####################
#########################################################

print("Starting Extraction")

yspec = np.arange(0,actypix) #"Y" spectrum is just pixel values
zspec = np.zeros((num_fibers,actypix)) #relative intensity at each point
zspec2 = np.zeros((num_fibers,actypix))
zspecbox = np.zeros((num_fibers,actypix))
zspecbox2 = np.zeros((num_fibers,actypix))
zmask = np.ones((num_fibers,actypix),dtype=bool)
zinvar = np.zeros((num_fibers,actypix))

#plt.plot(ccd[:,0])
#plt.show()

#########################################################################
############# Try iterative fitting, enforcing smoothness ###############
#########################################################################

### First fit, find sigma, xc parameters for traces
### Adapt to use sig, xc from fiberflats as first guesses
### quantities to smooth (potentially everything except height/intensity)
#fact = 20 #do 1/fact of the available points
#rough_pts = int(np.ceil(actypix/fact))
#sigrough = np.zeros((num_fibers,rough_pts)) #std
#meanrough = np.zeros((num_fibers,rough_pts)) #mean
#bgmrough = np.zeros((num_fibers,rough_pts)) #background mean
#bgsrough = np.zeros((num_fibers,rough_pts)) #background slope
#powrough = np.zeros((num_fibers,rough_pts)) #exponential power
#for i in range(num_fibers):
##    if i > 3:
##        plt.plot(zspec2[i-1,:])
##        plt.show()
##        plt.close()
##    slit_num = np.floor((i)/args.telescopes)
#    print("estimating parameters on trace {}".format(i+1))
#    for j in range(0,actypix,fact):
##        if j != 1000:
##            continue
#        jadj = int(np.floor(j/fact))
#        yj = (yspec[j]-actypix/2)/actypix
#        xc = trace_coeffs[2,i]*yj**2+trace_coeffs[1,i]*yj+trace_coeffs[0,i]
#        Ij = trace_intense_coeffs[2,i]*yj**2+trace_intense_coeffs[1,i]*yj+trace_intense_coeffs[0,i]
#        sigj = trace_sig_coeffs[2,i]*yj**2+trace_sig_coeffs[1,i]*yj+trace_sig_coeffs[0,i]
#        if np.isnan(xc):
#            sigrough[i,jadj] = np.nan
#            meanrough[i,jadj] = np.nan
#            bgmrough[i,jadj] = np.nan
#            bgsrough[i,jadj] = np.nan
#            powrough[i,jadj] = np.nan
#        else:
#            try:
#                xpad = 7
#                xvals = np.arange(-xpad,xpad+1)
#                xj = int(xc)
#                xwindow = xj+xvals
#                xvals = xvals[(xwindow>=0)*(xwindow<xpix)]
##                slitvals = np.poly1d(slit_coeffs[slit_num,j])(xj+xvals)
##                zvals = ccd[xj+xvals,yspec[j]]/Ij
##                invvals = 1/(abs(zvals)+readnoise)
#                zorig = ccd[xj+xvals,yspec[j]]
##                plt.figure()
##                plt.plot(xj+xvals,zorig,xj+xvals,zvals)
##                plt.show()
##                plt.close()
##                time.sleep(0.5)
#                invorig = 1/(abs(zorig)+readnoise)
##                zspecbox[i,j] = gain*sum(zvals)
##                zspecbox2[i,j] = gain*sum(zorig)
#                if np.max(zorig)<readnoise:
#                    continue
##                    print("zorig max less than 10 for pixel {}".format(j))
##                    zspec[i,j] = gain*sum(zvals)
##                    zspec2[i,j] = gain*sum(zorig)
#                else:
##                    print "Actually doing fitting!"
#                    pguess = [sigj,xc-xj-1,np.max(zorig),0,0,2] #std, mean, height, bgmean, bgslope, power
##                    params, errarr = sf.gauss_fit(xvals,zvals,invr=invvals,pguess=pguess,fit_exp='y')
#                    paramsorig, errorig = sf.gauss_fit(xvals,zorig,invr=invorig,pguess=pguess,fit_exp='y')
##                    height1 = params[2]
#                    height2 = paramsorig[2]
##                    height1 = sf.fit_height(xvals,zvals,invvals,sigj,xc-xj-1)[0]
##                    height2 = sf.fit_height(xvals,zorig,invorig,sigj,xc-xj-1)[0]
#                    ### Now add first primitive cosmic ray masking
#                    ### Make this recursive!
##                    fit = sf.gaussian(xvals,sigj,xc-xj-1,height1,params[3],params[4],params[5])
#                    fitorig = sf.gaussian(xvals,sigj,xc-xj-1,height2,paramsorig[3],paramsorig[4],paramsorig[5])
##                    differ = fit-zvals
#                    difforig = fitorig-zorig
##                    refit = False
#                    refitorig = False
#                    for k in range(len(difforig)):
##                        if differ[k] > 3*np.std(differ): #primitive sigma clipping
##                            refit = True
##                            invvals[k] = 0.00000001
#                        if difforig[k] > 3*np.std(difforig): #primitive sigma clipping
#                            refitorig = True
#                            invorig[k] = 0.00000001
##                    if refit:
##                        params, errarr = sf.gauss_fit(xvals,zvals,invr=invvals,pguess=pguess,fit_exp='y')
##                        height1 = params[2]
##                        height1 = sf.fit_height(xvals,zvals,invvals,sigj,xc-xj-1)[0]
#                    if refitorig:
#                        paramsorig, errorig = sf.gauss_fit(xvals,zorig,invr=invorig,pguess=pguess,fit_exp='y')
#                        height2 = paramsorig[2]
##                        height2 = sf.fit_height(xvals,zorig,invorig,sigj,xc-xj-1)[0]
#                    sigrough[i,jadj] = paramsorig[0]
#                    meanrough[i,jadj] = paramsorig[1]+xj
#                    bgmrough[i,jadj] = paramsorig[3]
#                    bgsrough[i,jadj] = paramsorig[4]
#                    powrough[i,jadj] = paramsorig[5]
##                    if refit:
###                        print refit,i,j, invvals, height1
##                        fit = sf.gaussian(xvals,sigj,xc-xj-1,height1,0,0)
##                        fitorig = sf.gaussian(xvals,sigj,xc-xj-1,height2,0,0)
##                    print refit,height1,refitorig,height2
##                    if refitorig:
##                        print invorig
##                        print difforig
##                    plt.figure()
##                    plt.plot(xj+xvals,zvals,xj+xvals,fit)
##                    plt.figure()
##                    plt.plot(xj+xvals,zorig,xj+xvals,fitorig)
##                    plt.show()
##                    plt.close()
##                    params, errarr = sf.gauss_fit(xvals,zvals,invvals)
##                    paramsorig, errorig = sf.gauss_fit(xvals,zorig,invorig)
##                    zspec[i,j] = gain*params[2]*1.24*np.sqrt(2*np.pi)
##                    zspec2[i,j] = gain*paramsorig[2]*1.24*np.sqrt(2*np.pi)
##                    fit = sf.gaussian(xvals,abs(params[0]),params[1],params[2],params[3],params[4],params[5])
##                    if i>1:
###                        print xc, sigj, Ij, yj
##                        fitorig = sf.gaussian(xvals,abs(paramsorig[0]),paramsorig[1],paramsorig[2],paramsorig[3],paramsorig[4],paramsorig[5])
##                        print paramsorig
##                        plt.plot(xj+xvals,zorig,xj+xvals,fitorig)
##                        plt.show()
#            except RuntimeError:
#                sigrough[i,jadj] = np.nan
#                meanrough[i,jadj] = np.nan
#                bgmrough[i,jadj] = np.nan
#                bgsrough[i,jadj] = np.nan
#                powrough[i,jadj] = np.nan     
#
#
#polysmooth = 2 #order of smoothing polynomial
#### for now, don't enforce background smoothness
#sigsmooth = np.zeros((num_fibers,polysmooth+1))
#meansmooth = np.zeros((num_fibers,polysmooth+1))
#powsmooth = np.zeros((num_fibers,polysmooth+1))
#sigspec = (np.arange(0,actypix,fact)-actypix/2)/actypix
#meanspec = (np.arange(0,actypix,fact)-actypix/2)/actypix
#powspec = (np.arange(0,actypix,fact)-actypix/2)/actypix
#for k in range(num_fibers):
#    sigmsk = (~np.isnan(sigrough[k]))*(sigrough[k]<3)*(sigrough[k]>0.5)
#    meanmsk = ~np.isnan(meanrough[k])
#    powmsk = (~np.isnan(powrough[k]))*(powrough[k]<3)*(powrough[k]>1.5)
#    if len(sigmsk==0)>=(len(sigrough[k])+polysmooth):
#        continue
#    sigsmooth[k] = np.polyfit(sigspec[sigmsk],sigrough[k][sigmsk],polysmooth)
#    meansmooth[k] = np.polyfit(meanspec[meanmsk],meanrough[k][meanmsk],polysmooth)
#    powsmooth[k] = np.polyfit(powspec[powmsk],powrough[k][powmsk],polysmooth)
#    if 1<k<6:
#        plt.figure('Sigmas')
#        plt.plot(sigspec,np.poly1d(sigsmooth[k])(sigspec))
#        plt.plot(sigspec[sigmsk],sigrough[k][sigmsk],'k+')
#        plt.figure('Means')
#        plt.plot(meanspec,np.poly1d(meansmooth[k])(meanspec))
#        plt.plot(meanspec[meanmsk],meanrough[k][meanmsk],'k+')
#        plt.figure('Powers')
#        plt.plot(powspec,np.poly1d(powsmooth[k])(powspec))
#        plt.plot(powspec[powmsk],powrough[k][powmsk],'k+')
#        plt.show()
#        plt.close()


##### Now a second fit using smoothed values from first fit
##### Here we only allow height to vary

def cosmic_ray_reject(D,f,P,iV,S=0,threshhold=25,verbose=False):
    """ Procedure to reject cosmic rays in optimal extraction.
        This assumes your inputs are all at a given wavelength
        Inputs (based on nomenclature in Horne, 1986):
            D - image data (array)
            f - estimated flux/scale factor (scalar)
            P - model profile (array)
            S - sky profile (array)
            iV - inverse variance (array), slight departure from Horne
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
#        if verbose:
##            print np.max(pixel_residuals)
#            print threshhold
#            time.sleep(0.5)
#            print pixel_residuals
    #        print iV
    #        print (D-f*P)
    #        print (D-f*P)**2
    #        print sum(abs(pixel_residuals))/len(D)
    #        print np.max(pixel_residuals)
    #        plt.plot(D)
    #        plt.plot(f*P)
    #        plt.show()
    #        plt.close()
        if np.max(pixel_residuals)>threshhold:
            pixel_reject[np.argmax(pixel_residuals)]=0
    #        elif sum(abs(pixel_residuals))/len(D) >chi_sq_lim:
    #            pixel_reject[np.argmax(pixel_residuals)]=0
    return pixel_reject
    
def fit_mn_hght_bg(xvals,zorig,invorig,sigj,mn_new,powj=2):
#    mn_new = xc-xj
    mn_old = -100
    lp_ct = 0
    while abs(mn_new-mn_old)>0.001:
        mn_old = np.copy(mn_new)
        hght, bg = sf.best_linear_gauss(xvals,sigj,mn_old,zorig,invorig,power=powj)
        mn_new, mn_new_std = sf.best_mean(xvals,sigj,mn_old,hght,bg,zorig,invorig,power=powj)
        lp_ct+=1
        if lp_ct>1e3: break
    return mn_new, hght,bg
       

model_ccd = np.zeros((np.shape(ccd))) #build model as we go to compare to data

mn_tr = 0
mn_fit = 0
mn_diff = 0
mn_cnt = 0
for i in range(num_fibers):
#    if i != 2:
#        continue
#        plt.plot(zspec2[i-1,:])
#        plt.show()
#        plt.close()
#    slit_num = np.floor((i)/args.telescopes)
    print("extracting trace {}".format(i+1))
    for j in range(actypix):
#            print powj, sigj
#            plt.plot(xvals,zorig,np.linspace(xvals[0],xvals[-1],100),fitprecise)
#            plt.show()
#            plt.close()
#        if j != 1576:
#            continue
        yj = (yspec[j]-actypix/2)/actypix
        xcshift = -0.08*(1+np.mod(i-1,4))
        xc = trace_coeffs[2,i]*yj**2+trace_coeffs[1,i]*yj+trace_coeffs[0,i]+xcshift#np.poly1d(meansmooth[i])(yj)
        Ij = trace_intense_coeffs[2,i]*yj**2+trace_intense_coeffs[1,i]*yj+trace_intense_coeffs[0,i]
        sigj = trace_sig_coeffs[2,i]*yj**2+trace_sig_coeffs[1,i]*yj+trace_sig_coeffs[0,i]
        powj = trace_pow_coeffs[2,i]*yj**2+trace_pow_coeffs[1,i]*yj+trace_pow_coeffs[0,i]
#        sigj = np.poly1d(sigsmooth[i])(yj)
#        powj = np.poly1d(powsmooth[i])(yj)
        if np.isnan(xc):
            zspec[i,j] = 0
            zspec2[i,j] = 0
            zmask[i,j] = False
        else:
            try:
                xpad = 5
                xvals = np.arange(-xpad,xpad+1)
                xj = int(xc)
                xwindow = xj+xvals
                xvals = xvals[(xwindow>=0)*(xwindow<xpix)]
#                slitvals = np.poly1d(slit_coeffs[slit_num,j])(xj+xvals)
                zvals = ccd[xj+xvals,yspec[j]]/Ij
                invvals = 1/(abs(zvals)+readnoise)
                zorig = ccd[xj+xvals,yspec[j]]
                if len(zorig)<1:
                    zspec[i,j] = 0#gain*sum(zvals)
                    zspec2[i,j] = 0#gain*sum(zorig)
                    zmask[i,j] = False
                    continue
#                plt.figure()
#                plt.plot(xj+xvals,zorig,xj+xvals,zvals)
#                plt.show()
#                plt.close()
#                time.sleep(0.5)
                invorig = 1/(abs(zorig)+readnoise**2)
#                zinvar = 1/(abs(zorig)+readnoise)
                zspecbox[i,j] = gain*sum(zvals)
                zspecbox2[i,j] = gain*sum(zorig)
                if np.max(zorig)<20:
#                    print("zorig max less than 10 for pixel {}".format(j))
                    zspec[i,j] = 0#gain*sum(zvals)
                    zspec2[i,j] = 0#gain*sum(zorig)
                else:
#                    print "Actually doing fitting!"
#                    pguess = [sigj,xc-xj-1,np.max(orig),0,0] #std, mean, height, bgmean, bgslope, power
                    mn_new, hght, bg = fit_mn_hght_bg(xvals,zorig,invorig,sigj,xc-xj-1,powj=powj)
#                    params, errarr = sf.gauss_fit(xvals,zvals,invr=invvals,xcguess=xc-xj,fit_exp='y')#,pguess=pguess)
#                    paramsorig, errorig = sf.gauss_fit(xvals,zorig,invr=invorig,xcguess=xc-xj,fit_exp='y',verbose='y')
#                    height1 = params[2]
#                    height2 = paramsorig[2]
#                    height1 = sf.fit_height(xvals,zvals,invvals,sigj,xc-xj-1,bgmrough[i,j]/Ij,bgsrough[i,j]/Ij,powj)[0]
#                    height2 = sf.fit_height(xvals,zorig,invorig,sigj,xc-xj-1,bgmrough[i,j],bgsrough[i,j],powj)[0]
                    ### Now add first primitive cosmic ray masking
                    ### Make this recursive!
#                    fit = sf.gaussian(xvals,params[0],params[1],params[2],params[3],params[4],params[5])
#                    fitorig = sf.gaussian(xvals,paramsorig[0],paramsorig[1],paramsorig[2],power=paramsorig[5])#,paramsorig[3],paramsorig[4])
#                    fitprecise = sf.gaussian(np.linspace(xvals[0],xvals[-1],100),abs(paramsorig[0]),paramsorig[1],paramsorig[2],power=paramsorig[5])
                    fitorig = sf.gaussian(xvals,sigj,mn_new,hght,power=powj)#,paramsorig[3],paramsorig[4])
                    fitprecise = sf.gaussian(np.linspace(xvals[0],xvals[-1],100),sigj,mn_new,hght,power=powj)
#                    if j==1576:
#                        print np.linspace(xvals[0],xvals[-1],100)
#                        print paramsorig
#                        print fitorig
#                        print invorig
                    fstd = sum(fitprecise)
                    #Add following if/else to handle failure to fit
                    if fstd==0:
                        fitnorm = np.zeros(len(zorig))
                    else:
                        fitnorm = fitorig/fstd
                    invorig = 1/(readnoise**2 + abs(fstd*fitnorm)*gain)
#                    fit = sf.gaussian(xvals,sigj,xc-xj-1,height1,bgmrough[i,j]/Ij,bgsrough[i,j]/Ij,powj)
#                    fitorig = sf.gaussian(xvals,sigj,xc-xj-1,height2,bgmrough[i,j],bgsrough[i,j],powj)
                    rej_min = 0
                    loop_count=0
#                    print i, j
                    while rej_min==0:
#                        print "starting loop"
                        pixel_reject = cosmic_ray_reject(zorig,fstd,fitnorm,invorig,S=bg,threshhold=0.3*np.mean(zorig),verbose=True)
#                        print "still in loop"
                        rej_min = np.min(pixel_reject)
#                        print rej_min
                        mn_tr+=xc-xj-1
                        mn_fit+=mn_new
                        mn_diff+=(mn_new-(xc-xj-1))
                        mn_cnt+=1
#                        if i==2 and j==1921:
#                            print("Mean diff = {}".format(mn_diff/mn_cnt))
#                            print fstd
#                            plt.plot(xvals,zorig,np.linspace(xvals[0],xvals[-1],100),fitprecise)
#                            plt.show()
#                            print "This point exists!"
#                        if i==0 and j==1576:#abs(1577-j)<3:
#                            print("Mean diff = {}".format(mn_diff/mn_cnt))
#                            print sigj/3
#                            plt.plot(xvals,zorig,np.linspace(xvals[0],xvals[-1],100),fitprecise)
#                            plt.show()
#                            plt.close()
#                            print len(zorig)
#                            print rej_min
#                            print pixel_reject
#                            print xvals
#                            print fitprecise
#                            print zorig
#                        print rej_min
                        if rej_min==0:
#                            print zorig
#                            print pixel_reject
#                            plt.plot(xvals,zorig)
                            zorig = zorig[pixel_reject==1]
#                            print zorig
                            invorig = invorig[pixel_reject==1]
                            xvals = xvals[pixel_reject==1]
#                            print "mark 1"
                            mn_new, hght, bg = fit_mn_hght_bg(xvals,zorig,invorig,sigj,xc-xj-1)
                            fitorig = sf.gaussian(xvals,sigj,mn_new,hght,power=powj)#,paramsorig[3],paramsorig[4])
                            fitprecise = sf.gaussian(np.linspace(xvals[0],xvals[-1],100),sigj,mn_new,hght,power=powj)
#                            plt.plot(xvals,zorig,np.linspace(xvals[0],xvals[-1],100),fitprecise)
#                            plt.show()
#                            plt.close()
#                            paramsorig, errorig = sf.gauss_fit(xvals,zorig,invr=invorig,xcguess=xc-xj,fit_exp='y',verbose='y')
##                            print "mark 2"
#                            fitorig = sf.gaussian(xvals,paramsorig[0],paramsorig[1],paramsorig[2],power=paramsorig[5])#,paramsorig[3],paramsorig[4])
#                            fitprecise = sf.gaussian(np.linspace(xvals[0],xvals[-1],100),paramsorig[0],paramsorig[1],paramsorig[2],power=paramsorig[5])
                            ftmp = sum(fitprecise)
                            fitnorm = fitorig/ftmp
                            fstd = sum(fitnorm*zorig*invorig)/sum(fitnorm**2*invorig)
                            invorig = 1/(readnoise**2 + abs(fstd*fitnorm)*gain)
#                            print("Cosmic ray rejected on trace {} at pixel {}".format(i,j))
#                            print rej_min==0
                        if loop_count>3:
                            break
                        loop_count+=1
#                    print "Out of loop"
#                    print i, j
#                    plt.plot(xvals,zorig)
#                    plt.show()
#                    plt.close()
#                    if i == 0 and j == 1576:
#                        print "in this other loop"
#                        plt.plot(xvals,zorig,np.linspace(xvals[0],xvals[-1],100),fitprecise)
#                        plt.show()
                    zspec2[i,j] = fstd
#                    differ = fit-zvals
#                    difforig = fitorig-zorig
#                    refit = False
#                    refitorig = False
#                    for k in range(len(differ)):
#                        if differ[k] > 5*np.std(differ):# and differ[k] > 30: #primitive sigma clipping
#                            refit = True
#                            invvals[k] = 0.00000001
#                        if difforig[k] > 5*np.std(difforig): #and difforig[k] > 30: #primitive sigma clipping
#                            refitorig = True
#                            invorig[k] = 0.00000001
#                    if refitorig:
##                        print refit,i,j, invvals, height1
#                        fit = sf.gaussian(xvals,sigj,xc-xj-1,height1,0,0)
#                        fitorig = sf.gaussian(xvals,sigj,xc-xj-1,height2,0,0)
#                    print refit,height1,refitorig,height2
#                    if refitorig:
#                        print invorig
#                        print difforig
#                    plt.figure()
#                    plt.plot(xj+xvals,zvals,xj+xvals,fit)
#                    plt.figure()
#                    plt.plot(xj+xvals,zorig,xj+xvals,fitorig)
#                    plt.show()
#                    plt.close()
#                    params, errarr = sf.gauss_fit(xvals,zvals,invvals)
#                    paramsorig, errorig = sf.gauss_fit(xvals,zorig,invorig)
#                    zspec[i,j] = gain*params[2]*1.24*np.sqrt(2*np.pi)
#                    zspec2[i,j] = gain*paramsorig[2]*1.24*np.sqrt(2*np.pi)
#                        print j
#                        print fitorig
#                        print zorig
#                        fit = sf.gaussian(xvals,abs(params[0]),params[1],params[2],params[3],params[4],params[5])
#                        fitorig = sf.gaussian(xvals,abs(paramsorig[0]),paramsorig[1],paramsorig[2],paramsorig[3],paramsorig[4],paramsorig[5])
#                        plt.plot(xj+xvals,zorig,xj+xvals,fitorig)
#                        plt.show()
#                        plt.close()
#                    #Restrict fitting off-trace
#                    if abs(params[1]-xc+xj+1)>1:
#                        zspec[i,j] = 0
#                    if abs(paramsorig[1]-xc+xj+1)>1:
#                        zspec2[i,j] = 0
#                        zmask[i,j] = False
#                        model_ccd[xj+xvals,j] = 0
#                    #Block anything that has a large spread in errors, compared to max
#                    if np.max(abs(differ))/np.max(zvals) > 0.1:
#                        zspec[i,j] = 0
#                    if np.max(abs(difforig))/np.max(zorig) > 0.1:
#                        zspec2[i,j] = 0
#                        zmask[i,j] = False
#                        model_ccd[xj+xvals,j] = 0
#                    if refit:
#                        params, errarr = sf.gauss_fit(xvals,zvals,invr=invvals,fit_exp='y')
#                        params, errarr = sf.gauss_fit(xvals,zvals,invr=invvals,pguess=pguess,fit_exp='y')
#                        height1 = params[2]
#                        height1 = sf.fit_height(xvals,zvals,invvals,sigj,xc-xj-1,bgmrough[i,j]/Ij,bgsrough[i,j]/Ij,powj)[0]
#                    if refitorig:
#                        paramsorig, errorig = sf.gauss_fit(xvals,zorig,invr=invorig,fit_exp='y')
#                        paramsorig, errorig = sf.gauss_fit(xvals,zorig,invr=invorig,pguess=pguess,fit_exp='y')
#                        height2 = paramsorig[2]
#                        height2 = sf.fit_height(xvals,zorig,invorig,sigj,xc-xj-1,bgmrough[i,j],bgsrough[i,j],powj)[0]
#                    gauss_model = sf.gaussian(xvals,params[0],params[1],params[2],power=params[5])#,params[3])#,params[4],params[5])
#                    gauss_model2 = sf.gaussian(xvals,paramsorig[0],paramsorig[1],paramsorig[2],power=paramsorig[5])#,paramsorig[3])#,paramsorig[4],paramsorig[5])
#                    gauss_modelp = sf.gaussian(np.linspace(xvals[0],xvals[-1],100),paramsorig[0],paramsorig[1],paramsorig[2],power=paramsorig[5])#,paramsorig[3])#,paramsorig[4],paramsorig[5])
##                    zspec[i,j] = gain*sum(gauss_model)#height1*sigj*np.sqrt(2*np.pi)
#                    gm_norm = gauss_model2/sum(gauss_modelp)
#                    zspec2[i,j] = gain*sum(gm_norm*zorig*invorig)/(sum(gm_norm**2*invorig))
#                    zspec2[i,j] = gain*sum(gauss_model2)#height2*sigj*np.sqrt(2*np.pi)
#                    model_ccd[xj+xvals,j] += gauss_model2
#                    if i == 6 and j <= 1950   and j >= 1944:
##                        print abs(paramsorig[1]-xc+xj+1)
#                        print np.std(difforig)/np.max(zorig)
#                        print np.max(abs(difforig))/np.max(zorig)
##                        print np.max(zorig)
#                        plt.plot(xvals,zorig,xvals,fitorig)
#                        plt.show()
#                        plt.close()
#    #                    print xc-xj
#                        time.sleep(0.5)

#                    if refitorig:
###                        print refit,i,j, invvals, height1
##                        fit = sf.gaussian(xvals,sigj,xc-xj-1,height1,0,0)
##                        fitorig = sf.gaussian(xvals,sigj,xc-xj-1,height2,0,0)
##                    print refit,height1,refitorig,height2
##                    if refitorig:
##                        print invorig
##                        print difforig
##                    plt.figure()
##                    plt.plot(xj+xvals,zvals,xj+xvals,fit)
##                    plt.figure()
##                    plt.plot(xj+xvals,zorig,xj+xvals,fitorig)
##                    plt.show()
##                    plt.close()
##                    params, errarr = sf.gauss_fit(xvals,zvals,invvals)
##                    paramsorig, errorig = sf.gauss_fit(xvals,zorig,invorig)
##                    zspec[i,j] = gain*params[2]*1.24*np.sqrt(2*np.pi)
##                    zspec2[i,j] = gain*paramsorig[2]*1.24*np.sqrt(2*np.pi)
#                        print np.std(difforig)
#                        print difforig
#                        print paramsorig
#                        fit = sf.gaussian(xvals,abs(params[0]),params[1],params[2],params[3],params[4],params[5])
#                        fitorig = sf.gaussian(xvals,abs(paramsorig[0]),paramsorig[1],paramsorig[2],paramsorig[3],paramsorig[4],paramsorig[5])
#                        plt.plot(xj+xvals,zorig,xj+xvals,fitorig)
#                        plt.show()
#                        plt.close()
            except RuntimeError:
                zspec[i,j] = 0
                zspec2[i,j] = 0
                zmask[i,j] = False
        if np.isnan(zspec2[i,j]):
            zspec2[i,j] = 0
            zmask[i,j] = False

### Check quality of model
#model_ccd += bias
#res = ccd-model_ccd
#ressc = res/(np.sqrt(ccd)+readnoise)
#pltarray = np.hstack((ccd,model_ccd,ressc))
#plt.ion()
#plt.imshow(pltarray)

############################################################
######### Import wavelength calibration ####################        
############################################################
        
i2coeffs = [3.48097e-4,2.11689] #shift in pixels due to iodine cell
i2shift = np.poly1d(i2coeffs)(np.arange(actypix))
#print i2shift

arc_date = 'n20160130'
wl_hdu1 = pyfits.open(os.path.join(redux_dir,arc_date,'wavelength_soln_T1.fits'))
wl_coeffs1 =  wl_hdu1[0].data
#wl_coeffs = wl_coeffs[::-1,:] #Re-order, need to be sure this is consistent
wl_polyord1 = wl_hdu1[0].header['POLYORD']
ypx_mod = 2*(yspec-i2shift*i2-actypix/2)/actypix #includes iodine shift (if i2 is in)
wavelength_soln_T1 = np.zeros((args.num_fibers,actypix))
for i in range(args.num_fibers):
#    if i < 2:
#        continue
#    wl_fib = np.floor((i)/3)
    wavelength_soln_T1[i] = np.poly1d(wl_coeffs1[i])(ypx_mod)
    
wl_hdu2 = pyfits.open(os.path.join(redux_dir,arc_date,'wavelength_soln_T2.fits'))
wl_coeffs2 =  wl_hdu2[0].data
#wl_coeffs = wl_coeffs[::-1,:] #Re-order, need to be sure this is consistent
wl_polyord2 = wl_hdu2[0].header['POLYORD']
#ypx_mod = 2*(yspec-actypix/2)/actypix
wavelength_soln_T2 = np.zeros((args.num_fibers,actypix))
for i in range(args.num_fibers):
#    if i < 2:
#        continue
#    wl_fib = np.floor((i)/3)
    wavelength_soln_T2[i] = np.poly1d(wl_coeffs2[i])(ypx_mod)
    
wl_hdu3 = pyfits.open(os.path.join(redux_dir,arc_date,'wavelength_soln_T3.fits'))
wl_coeffs3 =  wl_hdu3[0].data
#wl_coeffs = wl_coeffs[::-1,:] #Re-order, need to be sure this is consistent
wl_polyord3 = wl_hdu3[0].header['POLYORD']
#ypx_mod = 2*(yspec-actypix/2)/actypix
wavelength_soln_T3 = np.zeros((args.num_fibers,actypix))
for i in range(args.num_fibers):
#    if i < 2:
#        continue
#    wl_fib = np.floor((i)/3)
    wavelength_soln_T3[i] = np.poly1d(wl_coeffs3[i])(ypx_mod)
    
wl_hdu4 = pyfits.open(os.path.join(redux_dir,arc_date,'wavelength_soln_T4.fits'))
wl_coeffs4 =  wl_hdu4[0].data
#wl_coeffs = wl_coeffs[::-1,:] #Re-order, need to be sure this is consistent
wl_polyord4 = wl_hdu4[0].header['POLYORD']
#ypx_mod = 2*(yspec-actypix/2)/actypix
wavelength_soln_T4 = np.zeros((args.num_fibers,actypix))
for i in range(args.num_fibers):
#    if i < 2:
#        continue
#    wl_fib = np.floor((i)/3)
    wavelength_soln_T4[i] = np.poly1d(wl_coeffs4[i])(ypx_mod)

#plt.ion()    
#if ts == 'T1':
#    plt.figure()
#    plt.plot(wavelength_soln_T1[0,:],zspec2[0,:])
#    plt.figure()
#    plt.plot(wavelength_soln_T1[14,:],zspec2[14,:])
#    plt.figure()
#    plt.plot(wavelength_soln_T1[27,:],zspec2[27,:])
#elif ts == 'T2':
#    plt.figure()
#    plt.plot(wavelength_soln_T2[0,:],zspec2[0,:])
#    plt.figure()
#    plt.plot(wavelength_soln_T2[14,:],zspec2[14,:])
#    plt.figure()
#    plt.plot(wavelength_soln_T2[27,:],zspec2[27,:])
#elif ts == 'T3':
#    plt.figure()
#    plt.plot(wavelength_soln_T3[0,:],zspec2[0,:])
#    plt.figure()
#    plt.plot(wavelength_soln_T3[14,:],zspec2[14,:])
#    plt.figure()
#    plt.plot(wavelength_soln_T3[27,:],zspec2[27,:])
#elif ts == 'T4':
#    plt.figure()
#    plt.plot(wavelength_soln_T4[0,:],zspec2[1,:])
#    plt.figure()
#    plt.plot(wavelength_soln_T4[14,:],zspec2[15,:])
#    plt.figure()
#    plt.plot(wavelength_soln_T4[27,:],zspec2[28,:])

wavelength_soln = np.zeros((args.telescopes,args.num_fibers,actypix))
wavelength_soln[0,:,:] = wavelength_soln_T1
wavelength_soln[1,:,:] = wavelength_soln_T2
wavelength_soln[2,:,:] = wavelength_soln_T3
wavelength_soln[3,:,:] = wavelength_soln_T4

### Need to automate this (how?) but right now, here's the fiber arrangement:
###    1st (0) - T4 from order "2" (by my csv accounting)
###    2nd (1) - T1 from order "3"
###    3rd (2) - T2 from order "3"
### etc.  continues T1 through T4 and ascending orders
### right now I don't have wavelength soln for order 2, so I just eliminate
### that fiber and keep moving forward (fiber "0" isn't used)

zmask_fin = np.zeros((args.telescopes,args.num_fibers,actypix))
zmask_fin[0,:,:] = zmask[np.arange(1,num_fibers,4),:]
zmask_fin[1,:,:] = zmask[np.arange(2,num_fibers,4),:]
zmask_fin[2,:,:] = zmask[np.arange(3,num_fibers,4),:]
zmask_fin[3,:,:] = np.vstack((zmask[np.arange(4,num_fibers,4),:],np.zeros(actypix)))

zspec_fin2 = np.zeros((args.telescopes,args.num_fibers,actypix))
zspec_fin2[0,:,:] = zspec2[np.arange(1,num_fibers,4),:]
zspec_fin2[1,:,:] = zspec2[np.arange(2,num_fibers,4),:]
zspec_fin2[2,:,:] = zspec2[np.arange(3,num_fibers,4),:]
zspec_fin2[3,:,:] = np.vstack((zspec2[np.arange(4,num_fibers,4),:],np.ones(actypix)))

zspec_fin = np.zeros((args.telescopes,args.num_fibers,actypix))
zspec_fin[0,:,:] = zspec[np.arange(1,num_fibers,4),:]
zspec_fin[1,:,:] = zspec[np.arange(2,num_fibers,4),:]
zspec_fin[2,:,:] = zspec[np.arange(3,num_fibers,4),:]
zspec_fin[3,:,:] = np.vstack((zspec[np.arange(4,num_fibers,4),:],np.ones(actypix)))

zspec_finbox = np.zeros((args.telescopes,args.num_fibers,actypix))
zspec_finbox[0,:,:] = zspecbox[np.arange(1,num_fibers,4),:]
zspec_finbox[1,:,:] = zspecbox[np.arange(2,num_fibers,4),:]
zspec_finbox[2,:,:] = zspecbox[np.arange(3,num_fibers,4),:]
zspec_finbox[3,:,:] = np.vstack((zspecbox[np.arange(4,num_fibers,4),:],np.ones(actypix)))

zspec_finbox2 = np.zeros((args.telescopes,args.num_fibers,actypix))
zspec_finbox2[0,:,:] = zspecbox2[np.arange(1,num_fibers,4),:]
zspec_finbox2[1,:,:] = zspecbox2[np.arange(2,num_fibers,4),:]
zspec_finbox2[2,:,:] = zspecbox2[np.arange(3,num_fibers,4),:]
zspec_finbox2[3,:,:] = np.vstack((zspecbox2[np.arange(4,num_fibers,4),:],np.ones(actypix)))

### Actual inverse variances need to be calculated later, but
### these should be close to the true values
invvar_fin = 1/(abs(zspec_fin)+readnoise**2)
invvar_fin2 = 1/(abs(zspec_fin2)+readnoise**2)
invvar_finbox = 1/(abs(zspec_finbox)+readnoise**2)
invvar_finbox2 = 1/(abs(zspec_finbox2)+readnoise**2)

###Plot orders for visualization
#for i in range(num_fibers):
#    if mod(i,3)==0:
#        ir = num_fibers-i-1
#    #    if ~np.isnan(np.max(zspec[ir,:])):
#        plt.plot(zspec[ir,:])#wavelength_soln[ir,:],zspec[ir,:])
#        plt.plot(zspec2[ir,:])#wavelength_soln[ir,:],zspec2[ir,:])
#        plt.show()
#        plt.close()

### Split everything up for saving to the right name
root, savefile = os.path.split(filename)
savefile = savefile[:-5] #remove '.fits' for now
junk, savedate = os.path.split(root)
if not os.path.isdir(root):
    os.mkdir(root)

if not args.nosave:
#    ### Save file for optimal extraction with flat fielding ########
#    hdu1 = pyfits.PrimaryHDU(zspec_fin)
#    hdu2 = pyfits.PrimaryHDU(invvar_fin)
#    hdu3 = pyfits.PrimaryHDU(wavelength_soln)
#    hdu1.header.comments['NAXIS1'] = 'Telescope axis (T1, T2, T3, T4)'
#    hdu1.header.comments['NAXIS2'] = 'Pixel axis'
#    hdu1.header.comments['NAXIS3'] = 'Fiber axis (blue to red)'
#    hdu2.header.comments['NAXIS1'] = 'Telescope axis (T1, T2, T3, T4)'
#    hdu2.header.comments['NAXIS2'] = 'Pixel axis'
#    hdu2.header.comments['NAXIS3'] = 'Fiber axis (blue to red)'
#    hdu3.header.comments['NAXIS1'] = 'Telescope axis (T1, T2, T3, T4)'
#    hdu3.header.comments['NAXIS2'] = 'Pixel axis'
#    hdu3.header.comments['NAXIS3'] = 'Fiber axis (blue to red)'
#    hdu1.header.append(('DATA','Flux','Relative counts (no flat fielding)'))
#    hdu2.header.append(('DATA','Inv. Var','Inverse variance'))
#    hdu3.header.append(('DATA','Wavelength','Wavelength solution lambda vs px'))
#    #### Include all old header values in new header for hdu1
#    ### As usual, probably a better way, but I think this will work
#    for key in hdr.keys():
#        if key not in hdu1.header and key != 'BSCALE' and key != 'BZERO':
#            hdu1.header.append((key,hdr[key],hdr.comments[key]))
#            
#    hdulist = pyfits.HDUList([hdu1])
#    hdulist.append(hdu2)
#    hdulist.append(hdu3)
#    hdulist.writeto(os.path.join(redux_dir,savedate,savefile+'.proc.opt.flat.fits'),clobber=True)
    
    ### Save file for optimal extraction without flat fielding #####
    hdu1 = pyfits.PrimaryHDU(zspec_fin2)
    hdu2 = pyfits.PrimaryHDU(invvar_fin2)
    hdu3 = pyfits.PrimaryHDU(wavelength_soln)
    hdu4 = pyfits.PrimaryHDU(zmask_fin)
    hdu1.header.comments['NAXIS1'] = 'Pixel axis'
    hdu1.header.comments['NAXIS2'] = 'Fiber axis (blue to red)'
    hdu1.header.comments['NAXIS3'] = 'Telescope axis (T1, T2, T3, T4)'
    hdu2.header.comments['NAXIS1'] = 'Pixel axis'
    hdu2.header.comments['NAXIS2'] = 'Fiber axis (blue to red)'
    hdu2.header.comments['NAXIS3'] = 'Telescope axis (T1, T2, T3, T4)'
    hdu3.header.comments['NAXIS1'] = 'Pixel axis'
    hdu3.header.comments['NAXIS2'] = 'Fiber axis (blue to red)'
    hdu3.header.comments['NAXIS3'] = 'Telescope axis (T1, T2, T3, T4)'
    hdu4.header.comments['NAXIS1'] = 'Pixel axis'
    hdu4.header.comments['NAXIS2'] = 'Fiber axis (blue to red)'
    hdu4.header.comments['NAXIS3'] = 'Telescope axis (T1, T2, T3, T4)'
    ### Additional new header values
    hdu1.header.append(('UNITS','Flux','Relative photon counts (no flat fielding)'))
    hdu2.header.append(('UNITS','Inv. Var','Inverse variance'))
    hdu3.header.append(('UNITS','Wavelength','Wavelength solution lambda (Angstroms) vs px'))
    hdu4.header.append(('UNITS','Mask','True (1) or False (0) good data point'))
    hdu1.header.append(('VERSION',software_vers,'Reduction software version'))
    #### Include all old header values in new header for hdu1
    ### As usual, probably a better way, but I think this will work
    for key in spec_hdr.keys():
        if key not in hdu1.header and key != 'BSCALE' and key != 'BZERO':
            hdu1.header.append((key,spec_hdr[key],spec_hdr.comments[key]))
                
    hdulist = pyfits.HDUList([hdu1])
    hdulist.append(hdu2)
    hdulist.append(hdu3)
    hdulist.append(hdu4)
    if not os.path.isdir(os.path.join(redux_dir,savedate)):
        os.makedirs(os.path.join(redux_dir,savedate))
    print os.path.join(redux_dir,savedate,savefile+'.proc.fits')
    hdulist.writeto(os.path.join(redux_dir,savedate,savefile+'.proc.fits'),clobber=True)
        
#    ### Save file for boxcar extraction with flat fielding ########
#    hdu1 = pyfits.PrimaryHDU(zspec_finbox)
#    hdu2 = pyfits.PrimaryHDU(invvar_finbox)
#    hdu3 = pyfits.PrimaryHDU(wavelength_soln)
#    hdu1.header.comments['NAXIS1'] = 'Telescope axis (T1, T2, T3, T4)'
#    hdu1.header.comments['NAXIS2'] = 'Pixel axis'
#    hdu1.header.comments['NAXIS3'] = 'Fiber axis (blue to red)'
#    hdu2.header.comments['NAXIS1'] = 'Telescope axis (T1, T2, T3, T4)'
#    hdu2.header.comments['NAXIS2'] = 'Pixel axis'
#    hdu2.header.comments['NAXIS3'] = 'Fiber axis (blue to red)'
#    hdu3.header.comments['NAXIS1'] = 'Telescope axis (T1, T2, T3, T4)'
#    hdu3.header.comments['NAXIS2'] = 'Pixel axis'
#    hdu3.header.comments['NAXIS3'] = 'Fiber axis (blue to red)'
#    hdu1.header.append(('DATA','Flux','Relative counts (no flat fielding)'))
#    hdu2.header.append(('DATA','Inv. Var','Inverse variance'))
#    hdu3.header.append(('DATA','Wavelength','Wavelength solution lambda vs px'))
#    #### Include all old header values in new header for hdu1
#    ### As usual, probably a better way, but I think this will work
#    for key in hdr.keys():
#        if key not in hdu1.header and key != 'BSCALE' and key != 'BZERO':
#            hdu1.header.append((key,hdr[key],hdr.comments[key]))
#            
#    hdulist = pyfits.HDUList([hdu1])
#    hdulist.append(hdu2)
#    hdulist.append(hdu3)
#    hdulist.writeto(os.path.join(redux_dir,savedate,savefile+'.proc.box.flat.fits'),clobber=True)
#    
#    ### Save file for boxcar extraction with flat fielding ########
#    hdu1 = pyfits.PrimaryHDU(zspec_finbox2)
#    hdu2 = pyfits.PrimaryHDU(invvar_finbox2)
#    hdu3 = pyfits.PrimaryHDU(wavelength_soln)
#    hdu1.header.comments['NAXIS1'] = 'Telescope axis (T1, T2, T3, T4)'
#    hdu1.header.comments['NAXIS2'] = 'Pixel axis'
#    hdu1.header.comments['NAXIS3'] = 'Fiber axis (blue to red)'
#    hdu2.header.comments['NAXIS1'] = 'Telescope axis (T1, T2, T3, T4)'
#    hdu2.header.comments['NAXIS2'] = 'Pixel axis'
#    hdu2.header.comments['NAXIS3'] = 'Fiber axis (blue to red)'
#    hdu3.header.comments['NAXIS1'] = 'Telescope axis (T1, T2, T3, T4)'
#    hdu3.header.comments['NAXIS2'] = 'Pixel axis'
#    hdu3.header.comments['NAXIS3'] = 'Fiber axis (blue to red)'
#    hdu1.header.append(('DATA','Flux','Relative counts (no flat fielding)'))
#    hdu2.header.append(('DATA','Inv. Var','Inverse variance'))
#    hdu3.header.append(('DATA','Wavelength','Wavelength solution lambda vs px'))
#    #### Include all old header values in new header for hdu1
#    ### As usual, probably a better way, but I think this will work
#    for key in hdr.keys():
#        if key not in hdu1.header and key != 'BSCALE' and key != 'BZERO':
#            hdu1.header.append((key,hdr[key],hdr.comments[key]))
#            
#    hdulist = pyfits.HDUList([hdu1])
#    hdulist.append(hdu2)
#    hdulist.append(hdu3)
#    hdulist.writeto(os.path.join(redux_dir,savedate,savefile+'.proc.box.noflat.fits'),clobber=True)

tf = time.time()
print("Total extraction time = {}s".format(tf-t0))
