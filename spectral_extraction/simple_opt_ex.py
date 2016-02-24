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

######## Import environmental variables #################
pathd = os.environ['MINERVA_DATA_DIR']
redux_dir = os.environ['MINERVA_REDUX_DIR']

#########################################################
########### Allow input arguments #######################
#########################################################
parser = argparse.ArgumentParser()
parser.add_argument("-f","--filename",help="Name of image file (.fits) to extract",
                    default=os.path.join(pathd,'n20160216','n20160216.HR2209.0025.fits'))
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
parser.add_argument("-T","--tscopes",help="T1, T2, T3, and/or T4 (remove later)",
                    type=str,default=['T1','T2','T3','T4'])
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

#hardcode in n20160115 directory
filename = args.filename#os.path.join(pathd,'n20160115',args.filename)

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

ts = args.tscopes

#for ts in args.tscopes:#['T1','T2','T3','T4']:
#    #Choose fiberflats with iodine cell in
if ts=='T1':
    flnmflat = 'n20160130.fiberflat_T1.0023.fits'
    flnmarc = 'n20160130.thar_T1_i2test.0025.fits'
    print ts
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
#    continue
#Import tungsten fiberflat
fileflat = os.path.join(pathd,'n20160130',flnmflat)
filearc = os.path.join(pathd,'n20160130',flnmarc)
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
        xtrace[i,0], Itrace[i,0], chi_vals[i,0] = fit_trace(xtrace[i,0],y,trace_ccd)
    else:
        Itrace[i,0], chi_vals[i,0] = nan, nan  


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
                xtrace[j,i], Itrace[j,i], chi_vals[j,i] = fit_trace(xtrace[j,i],y,trace_ccd)
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
date = 'n20160115'
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
#ccd = ccd[::-1,:] #Reverse order?  Not sure which is right for my code
      
#########################################################
########### Now Extract the Spectrum ####################
#########################################################

print("Starting Extraction")

yspec = np.arange(0,actypix) #"Y" spectrum is just pixel values
zspec = np.zeros((num_fibers,actypix)) #relative intensity at each point
zspec2 = np.zeros((num_fibers,actypix))
zspecbox = np.zeros((num_fibers,actypix))
zspecbox2 = np.zeros((num_fibers,actypix))

#plt.plot(ccd[:,1000])
#plt.show()


for i in range(num_fibers):
#    if i > 3:
#        plt.plot(zspec2[i-1,:])
#        plt.show()
#        plt.close()
#    if i < 82 or i >= 83: #Manually skip first two fibers since they don't work
#        continue 
    slit_num = np.floor((i)/args.telescopes)
    print("starting on trace {}".format(i+1))
    for j in range(actypix):
#        if j != 1000:
#            continue
        yj = (yspec[j]-actypix/2)/actypix
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
                if np.max(zorig<10):
                    zspec[i,j] = gain*sum(zvals)
                    zspec2[i,j] = gain*sum(zorig)
                    zspecbox[i,j] = gain*sum(zvals)
                    zspecbox2[i,j] = gain*sum(zorig)                   
                else:
    #                if (j == 1800 or j == 1816) and i == 2:
    #                    plt.plot(zorig/0.33)
    #                    print slitvals
    #                    plt.plot(zorig/slitvals)
    #                    plt.show()
    #                    plt.close()
    #                if np.max(zvals)>bg_cutoff:
                    params = sf.gauss_fit(xvals,zvals)
                    paramsorig = sf.gauss_fit(xvals,zorig)
    #                if (j == 1800 or j == 1816) and i == 2:
    
                    zspec[i,j] = gain*params[2]*1.24*np.sqrt(2*np.pi)
                    zspec2[i,j] = gain*paramsorig[2]*1.24*np.sqrt(2*np.pi)
                    zspecbox[i,j] = gain*sum(zvals)
                    zspecbox2[i,j] = gain*sum(zorig)
#                if i < 6 and j == 1000:
#                    fit = sf.gaussian(xvals,abs(params[0]),params[1],params[2],params[3],params[4])
#                    fitorig = sf.gaussian(xvals,abs(paramsorig[0]),paramsorig[1],paramsorig[2],paramsorig[3],paramsorig[4])
#                    plt.plot(xj+xvals,zorig,xj+xvals,fitorig)
#                    print i
#                    plt.show()
##                    plt.plot(xvals,zvals,xvals,fit)
#                    plt.close()
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

### And re-arrange for logistical reasons
#zspec2 = zspec2[::-1]        

############################################################
######### Import wavelength calibration ####################        
############################################################
        
i2coeffs = [3.48097e-4,2.11689] #shift in pixels due to iodine cell
i2shift = np.poly1d(i2coeffs)(np.arange(actypix))
#print i2shift

wl_hdu1 = pyfits.open(os.path.join(redux_dir,'n20160216','wavelength_soln_T1.fits'))
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
    
wl_hdu2 = pyfits.open(os.path.join(redux_dir,'n20160216','wavelength_soln_T2.fits'))
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
    
wl_hdu3 = pyfits.open(os.path.join(redux_dir,'n20160216','wavelength_soln_T3.fits'))
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
    
wl_hdu4 = pyfits.open(os.path.join(redux_dir,'n20160216','wavelength_soln_T4.fits'))
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

plt.ion()    
if ts == 'T1':
    plt.figure()
    plt.plot(wavelength_soln_T1[0,:],zspec2[0,:])
    plt.figure()
    plt.plot(wavelength_soln_T1[14,:],zspec2[14,:])
    plt.figure()
    plt.plot(wavelength_soln_T1[27,:],zspec2[27,:])
elif ts == 'T2':
    plt.figure()
    plt.plot(wavelength_soln_T2[0,:],zspec2[0,:])
    plt.figure()
    plt.plot(wavelength_soln_T2[14,:],zspec2[14,:])
    plt.figure()
    plt.plot(wavelength_soln_T2[27,:],zspec2[27,:])
elif ts == 'T3':
    plt.figure()
    plt.plot(wavelength_soln_T3[0,:],zspec2[0,:])
    plt.figure()
    plt.plot(wavelength_soln_T3[14,:],zspec2[14,:])
    plt.figure()
    plt.plot(wavelength_soln_T3[27,:],zspec2[27,:])
elif ts == 'T4':
    plt.figure()
    plt.plot(wavelength_soln_T4[0,:],zspec2[1,:])
    plt.figure()
    plt.plot(wavelength_soln_T4[14,:],zspec2[15,:])
    plt.figure()
    plt.plot(wavelength_soln_T4[27,:],zspec2[28,:])

wavelength_soln = np.zeros((args.num_fibers,actypix,args.telescopes))
wavelength_soln[:,:,0] = wavelength_soln_T1
wavelength_soln[:,:,1] = wavelength_soln_T2
wavelength_soln[:,:,2] = wavelength_soln_T3
wavelength_soln[:,:,3] = wavelength_soln_T4

### Need to automate this (how?) but right now, here's the fiber arrangement:
###    1st (0) - T4 from order "2" (by my csv accounting)
###    2nd (1) - T1 from order "3"
###    3rd (2) - T2 from order "3"
### etc.  continues T1 through T4 and ascending orders
### right now I don't have wavelength soln for order 2, so I just eliminate
### that fiber and keep moving forward (fiber "0" isn't used)


zspec_fin2 = np.zeros((args.num_fibers,actypix,args.telescopes))
zspec_fin2[:,:,0] = zspec2[np.arange(1,num_fibers,4),:]
zspec_fin2[:,:,1] = zspec2[np.arange(2,num_fibers,4),:]
zspec_fin2[:,:,2] = zspec2[np.arange(3,num_fibers,4),:]
zspec_fin2[:,:,3] = zspec2[np.arange(4,num_fibers,4),:]

zspec_fin = np.zeros((args.num_fibers,actypix,args.telescopes))
zspec_fin[:,:,0] = zspec[np.arange(1,num_fibers,4),:]
zspec_fin[:,:,1] = zspec[np.arange(2,num_fibers,4),:]
zspec_fin[:,:,2] = zspec[np.arange(3,num_fibers,4),:]
zspec_fin[:,:,3] = zspec[np.arange(4,num_fibers,4),:]

zspec_finbox = np.zeros((args.num_fibers,actypix,args.telescopes))
zspec_finbox[:,:,0] = zspecbox[np.arange(1,num_fibers,4),:]
zspec_finbox[:,:,1] = zspecbox[np.arange(2,num_fibers,4),:]
zspec_finbox[:,:,2] = zspecbox[np.arange(3,num_fibers,4),:]
zspec_finbox[:,:,3] = zspecbox[np.arange(4,num_fibers,4),:]

zspec_finbox2 = np.zeros((args.num_fibers,actypix,args.telescopes))
zspec_finbox2[:,:,0] = zspecbox2[np.arange(1,num_fibers,4),:]
zspec_finbox2[:,:,1] = zspecbox2[np.arange(2,num_fibers,4),:]
zspec_finbox2[:,:,2] = zspecbox2[np.arange(3,num_fibers,4),:]
zspec_finbox2[:,:,3] = zspecbox2[np.arange(4,num_fibers,4),:]

invvar_fin = 1/(abs(zspec_fin)+readnoise) #Super rough estimate for now
invvar_fin2 = 1/(abs(zspec_fin2)+readnoise) #Super rough estimate for now
invvar_finbox = 1/(abs(zspec_finbox)+readnoise) #Super rough estimate for now
invvar_finbox2 = 1/(abs(zspec_finbox2)+readnoise) #Super rough estimate for now

###Plot orders for visualization
#for i in range(num_fibers):
#    if mod(i,3)==0:
#        ir = num_fibers-i-1
#    #    if ~np.isnan(np.max(zspec[ir,:])):
#        plt.plot(zspec[ir,:])#wavelength_soln[ir,:],zspec[ir,:])
#        plt.plot(zspec2[ir,:])#wavelength_soln[ir,:],zspec2[ir,:])
#        plt.show()
#        plt.close()
if not args.nosave:
    ### Save file for optimal extraction with flat fielding ########
    hdu1 = pyfits.PrimaryHDU(zspec_fin)
    hdu2 = pyfits.PrimaryHDU(invvar_fin)
    hdu3 = pyfits.PrimaryHDU(wavelength_soln)
    hdu1.header.comments['NAXIS1'] = 'Telescope axis (T1, T2, T3, T4)'
    hdu1.header.comments['NAXIS2'] = 'Pixel axis'
    hdu1.header.comments['NAXIS3'] = 'Fiber axis (blue to red)'
    hdu2.header.comments['NAXIS1'] = 'Telescope axis (T1, T2, T3, T4)'
    hdu2.header.comments['NAXIS2'] = 'Pixel axis'
    hdu2.header.comments['NAXIS3'] = 'Fiber axis (blue to red)'
    hdu3.header.comments['NAXIS1'] = 'Telescope axis (T1, T2, T3, T4)'
    hdu3.header.comments['NAXIS2'] = 'Pixel axis'
    hdu3.header.comments['NAXIS3'] = 'Fiber axis (blue to red)'
    hdu1.header.append(('DATA','Flux','Relative counts (no flat fielding)'))
    hdu2.header.append(('DATA','Inv. Var','Inverse variance'))
    hdu3.header.append(('DATA','Wavelength','Wavelength solution lambda vs px'))
    #### Include all old header values in new header for hdu1
    ### As usual, probably a better way, but I think this will work
    for key in hdr.keys():
        if key not in hdu1.header and key != 'BSCALE' and key != 'BZERO':
            hdu1.header.append((key,hdr[key],hdr.comments[key]))
            
    hdulist = pyfits.HDUList([hdu1])
    hdulist.append(hdu2)
    hdulist.append(hdu3)
    hdulist.writeto(os.path.join(redux_dir,'n20160216','HR2209.0025.test.opt.flat.fits'),clobber=True)
    
    ### Save file for optimal extraction without flat fielding #####
    hdu1 = pyfits.PrimaryHDU(zspec_fin2)
    hdu2 = pyfits.PrimaryHDU(invvar_fin2)
    hdu3 = pyfits.PrimaryHDU(wavelength_soln)
    hdu1.header.comments['NAXIS1'] = 'Telescope axis (T1, T2, T3, T4)'
    hdu1.header.comments['NAXIS2'] = 'Pixel axis'
    hdu1.header.comments['NAXIS3'] = 'Fiber axis (blue to red)'
    hdu2.header.comments['NAXIS1'] = 'Telescope axis (T1, T2, T3, T4)'
    hdu2.header.comments['NAXIS2'] = 'Pixel axis'
    hdu2.header.comments['NAXIS3'] = 'Fiber axis (blue to red)'
    hdu3.header.comments['NAXIS1'] = 'Telescope axis (T1, T2, T3, T4)'
    hdu3.header.comments['NAXIS2'] = 'Pixel axis'
    hdu3.header.comments['NAXIS3'] = 'Fiber axis (blue to red)'
    hdu1.header.append(('DATA','Flux','Relative counts (no flat fielding)'))
    hdu2.header.append(('DATA','Inv. Var','Inverse variance'))
    hdu3.header.append(('DATA','Wavelength','Wavelength solution lambda vs px'))
    #### Include all old header values in new header for hdu1
    ### As usual, probably a better way, but I think this will work
    for key in hdr.keys():
        if key not in hdu1.header and key != 'BSCALE' and key != 'BZERO':
            hdu1.header.append((key,hdr[key],hdr.comments[key]))
                
    hdulist = pyfits.HDUList([hdu1])
    hdulist.append(hdu2)
    hdulist.append(hdu3)
    hdulist.writeto(os.path.join(redux_dir,'n20160216','HR2209.0025.test.opt.noflat.fits'),clobber=True)
        
    ### Save file for boxcar extraction with flat fielding ########
    hdu1 = pyfits.PrimaryHDU(zspec_finbox)
    hdu2 = pyfits.PrimaryHDU(invvar_finbox)
    hdu3 = pyfits.PrimaryHDU(wavelength_soln)
    hdu1.header.comments['NAXIS1'] = 'Telescope axis (T1, T2, T3, T4)'
    hdu1.header.comments['NAXIS2'] = 'Pixel axis'
    hdu1.header.comments['NAXIS3'] = 'Fiber axis (blue to red)'
    hdu2.header.comments['NAXIS1'] = 'Telescope axis (T1, T2, T3, T4)'
    hdu2.header.comments['NAXIS2'] = 'Pixel axis'
    hdu2.header.comments['NAXIS3'] = 'Fiber axis (blue to red)'
    hdu3.header.comments['NAXIS1'] = 'Telescope axis (T1, T2, T3, T4)'
    hdu3.header.comments['NAXIS2'] = 'Pixel axis'
    hdu3.header.comments['NAXIS3'] = 'Fiber axis (blue to red)'
    hdu1.header.append(('DATA','Flux','Relative counts (no flat fielding)'))
    hdu2.header.append(('DATA','Inv. Var','Inverse variance'))
    hdu3.header.append(('DATA','Wavelength','Wavelength solution lambda vs px'))
    #### Include all old header values in new header for hdu1
    ### As usual, probably a better way, but I think this will work
    for key in hdr.keys():
        if key not in hdu1.header and key != 'BSCALE' and key != 'BZERO':
            hdu1.header.append((key,hdr[key],hdr.comments[key]))
            
    hdulist = pyfits.HDUList([hdu1])
    hdulist.append(hdu2)
    hdulist.append(hdu3)
    hdulist.writeto(os.path.join(redux_dir,'n20160216','HR2209.0025.test.box.flat.fits'),clobber=True)
    
    ### Save file for boxcar extraction with flat fielding ########
    hdu1 = pyfits.PrimaryHDU(zspec_finbox2)
    hdu2 = pyfits.PrimaryHDU(invvar_finbox2)
    hdu3 = pyfits.PrimaryHDU(wavelength_soln)
    hdu1.header.comments['NAXIS1'] = 'Telescope axis (T1, T2, T3, T4)'
    hdu1.header.comments['NAXIS2'] = 'Pixel axis'
    hdu1.header.comments['NAXIS3'] = 'Fiber axis (blue to red)'
    hdu2.header.comments['NAXIS1'] = 'Telescope axis (T1, T2, T3, T4)'
    hdu2.header.comments['NAXIS2'] = 'Pixel axis'
    hdu2.header.comments['NAXIS3'] = 'Fiber axis (blue to red)'
    hdu3.header.comments['NAXIS1'] = 'Telescope axis (T1, T2, T3, T4)'
    hdu3.header.comments['NAXIS2'] = 'Pixel axis'
    hdu3.header.comments['NAXIS3'] = 'Fiber axis (blue to red)'
    hdu1.header.append(('DATA','Flux','Relative counts (no flat fielding)'))
    hdu2.header.append(('DATA','Inv. Var','Inverse variance'))
    hdu3.header.append(('DATA','Wavelength','Wavelength solution lambda vs px'))
    #### Include all old header values in new header for hdu1
    ### As usual, probably a better way, but I think this will work
    for key in hdr.keys():
        if key not in hdu1.header and key != 'BSCALE' and key != 'BZERO':
            hdu1.header.append((key,hdr[key],hdr.comments[key]))
            
    hdulist = pyfits.HDUList([hdu1])
    hdulist.append(hdu2)
    hdulist.append(hdu3)
    hdulist.writeto(os.path.join(redux_dir,'n20160216','HR2209.0025.test.box.noflat.fits'),clobber=True)