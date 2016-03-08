#!/usr/bin/env python 2.7

#Designed to check quality of extraction

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
#import scipy.sparse as sparse
#import scipy.signal as signal
#import scipy.linalg as linalg
#import solar
import special as sf
import argparse

parser = argparse.ArgumentParser()
#parser.add_argument("-f","--filename",help="Name of image file (.fits) to extract",
#                    default=os.path.join(pathd,'n20160216','n20160216.HR2209.0025.fits'))
#                    default=os.path.join(pathd,'n20160115','n20160115.daytimeSky.0006.fits'))
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

######## Import environmental variables #################
data_dir = os.environ['MINERVA_DATA_DIR']
redux_dir = os.environ['MINERVA_REDUX_DIR']

### Hardcoded for now:
raw_data_file = os.path.join(data_dir,'n20160216','n20160216.HR2209.0025.fits')
#ext_spec_file = os.path.join(redux_dir,'n20160216','n20160216.HR2209.0025opt.noflat.fits')
ext_spec_file = os.path.join(redux_dir,'n20160216','n20160216.HR2209.0025.proc.opt.noflat.fits')

### Reload all the tracing stuff...

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
        params, errarr = sf.gauss_fit(xvals,zvals)
        xc = x+params[1] #offset plus center
        zc = params[2] #height (intensity)
        sig = params[0] #standard deviation
#        pxn = np.linspace(xvals[0],xvals[-1],1000)
        fit = sf.gaussian(xvals,abs(params[0]),params[1],params[2],params[3],params[4])
        chi = sum((fit-zvals)**2/zvals)
        return xc, zc, abs(sig), chi


#########################################################
########### Load Background Requirments #################
#########################################################

#hardcode in n20160115 directory
filename = raw_data_file#os.path.join(data_dir,'n20160115',args.filename)

gain = 1.3
readnoise = 3.63

spectrum = pyfits.open(filename,uint=True)
hdr = spectrum[0].header
ccd = spectrum[0].data

#####CONVERT NASTY FORMAT TO ONE THAT ACTUALLY WORKS#####
#Dimensions
ypix = hdr['NAXIS1']
xpix = hdr['NAXIS2']
### Next part checks if iodine cell is in, assumes keyword I2POSAS exists
if hdr['I2POSAS']=='in':
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
    hdr = ff[0].header
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
yspace = int(floor(ypix/(num_points+1)))
yvals = yspace*(1+np.arange(num_points))

xtrace = np.zeros((num_fibers,num_points)) #xpositions of traces
ytrace = np.zeros((num_fibers,num_points)) #ypositions of traces
sigtrace = np.zeros((num_fibers,num_points)) #standard deviation along trace
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
        xtrace[i,0], Itrace[i,0], sigtrace[i,0], chi_vals[i,0] = fit_trace(xtrace[i,0],y,trace_ccd)
    else:
        Itrace[i,0], sigtrace[i,0], chi_vals[i,0] = nan, nan, nan  


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
                xtrace[j,i], Itrace[j,i], sigtrace[j,i], chi_vals[j,i] = fit_trace(xtrace[j,i],y,trace_ccd)
            else:
                xtrace[j,i], Itrace[j,i], sigtrace[j,i], chi_vals[j,i] = nan, nan, nan, nan
        else:
            xtrace[j,i], Itrace[j,i], sigtrace[j,i], chi_vals[j,i] = nan, nan, nan, nan
            
Itrace /= np.median(Itrace) #Rescale intensities

#Finally fit x vs. y on traces.  Start with quadratic for simple + close enough
trace_coeffs = np.zeros((3,num_fibers))
trace_intense_coeffs = np.zeros((3,num_fibers))
trace_sig_coeffs = np.zeros((3,num_fibers))
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
    else:
        tmp_coeffs = nan*np.ones((3))
        tmp_coeffs2 = nan*np.ones((3))
        tmp_coeffs2 = nan*np.ones((3))
    trace_coeffs[0,i] = tmp_coeffs[0]
    trace_coeffs[1,i] = tmp_coeffs[1]
    trace_coeffs[2,i] = tmp_coeffs[2]
    trace_intense_coeffs[0,i] = tmp_coeffs2[0]
    trace_intense_coeffs[1,i] = tmp_coeffs2[1]
    trace_intense_coeffs[2,i] = tmp_coeffs2[2]
    trace_sig_coeffs[0,i] = tmp_coeffs3[0]
    trace_sig_coeffs[1,i] = tmp_coeffs3[1]
    trace_sig_coeffs[2,i] = tmp_coeffs3[2]      

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

#########################################################
########## Now build model ccd from extracted data ######
#########################################################

model_ccd = np.zeros((np.shape(ccd)))
yspec = np.arange(0,actypix)
exspec = pyfits.open(ext_spec_file)
exdat = exspec[0].data

for i in range(num_fibers):
    if i == 0:
        continue #skip the first one since I didn't extract it
#    if i > 3:
#        plt.plot(zspec2[i-1,:])
#        plt.show()
#        plt.close()
#    slit_num = np.floor((i)/args.telescopes)
    print("starting on trace {}".format(i+1))
    for j in range(actypix):
#        if j != 1000:
#            continue
        yj = (yspec[j]-actypix/2)/actypix
        xc = trace_coeffs[2,i]*yj**2+trace_coeffs[1,i]*yj+trace_coeffs[0,i]
        Ij = trace_intense_coeffs[2,i]*yj**2+trace_intense_coeffs[1,i]*yj+trace_intense_coeffs[0,i]
        sigj = trace_sig_coeffs[2,i]*yj**2+trace_sig_coeffs[1,i]*yj+trace_sig_coeffs[0,i]
        if np.isnan(xc):
            continue
        tind = np.mod(i-1,4)
        fibind = np.floor((i-1)/4)
        height = exdat[fibind,j,tind]
        height /= (1.24*np.sqrt(2*np.pi))#(sigj*np.sqrt(2*np.pi))
        xpad = 7
        xvals = np.arange(-xpad,xpad+1)
        xj = int(xc)
        xwindow = xj+xvals
        xvals = xvals[(xwindow>=0)*(xwindow<xpix)]
        zorig = ccd[xj+xvals,yspec[j]]
#                plt.figure()
#                plt.plot(xj+xvals,zorig,xj+xvals,zvals)
#                plt.show()
#                plt.close()
#                time.sleep(0.5)
        invorig = 1/(abs(zorig)+readnoise)
        if np.max(zorig)<10:
#                    print("zorig max less than 10 for pixel {}".format(j))
            cent = xc
        else:
            paramsorig, errorig = sf.gauss_fit(xvals,zorig,invorig)
            height2 = paramsorig[2]
#                    height2 = sf.fit_height(xvals,zorig,invorig,sigj,xc-xj-1)[0]
            ### Now add first primitive cosmic ray masking
            ### Make this recursive!
            fitorig = sf.gaussian(xvals,paramsorig[0],paramsorig[1]-xj-1,height2,0,0)
            difforig = fitorig-zorig
            refitorig = False
            for k in range(len(difforig)):
                if difforig[k] > 3*np.std(difforig): #primitive sigma clipping
                    refitorig = True
                    invorig[k] = 999999
            if refitorig:
                paramsorig, errorig = sf.gauss_fit(xvals,zorig,invorig)
                height2 = paramsorig[2]
#                        height2 = sf.fit_height(xvals,zorig,invorig,sigj,xc-xj-1)[0]
        sigj = paramsorig[0]
        xc = paramsorig[1]
        gauss_model = sf.gaussian(xvals,sigj,xc-xj-1,height,0,0)
        model_ccd[xj+xvals,j] += gauss_model
        
model_ccd += bias
res = ccd-model_ccd
ressc = res/(np.sqrt(ccd))
pltarray = np.hstack((ccd,model_ccd,ressc))
plt.ion()
plt.imshow(pltarray)
