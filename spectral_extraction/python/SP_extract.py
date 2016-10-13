#!/usr/bin/env python

#Implementation of 2D "Spectro-perfectionism" extraction for MINERVA

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
import psf_utils as psf
import minerva_utils as utils

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
num_fibers = args.num_fibers*args.telescopes-4

### Snipped from find_ip.py
### Should roughly be able to find good peaks on arc frames
### Add lines at the end to fit a 2Dspline PSF along a trace

#redux_dir = os.environ['MINERVA_REDUX_DIR']
#data_dir = os.environ['MINERVA_DATA_DIR']
#Tarc = pyfits.open(os.path.join(redux_dir,'n20160130','n20160130.thar_T1_i2test.0025.proc.fits'))
#Tarc = pyfits.open(os.path.join(redux_dir,'n20160130','n20160130.thar_T2_i2test.0020.proc.fits'))
#Tarc = pyfits.open(os.path.join(redux_dir,'n20160130','n20160130.thar_T3_i2test.0012.proc.fits'))
Tarc = pyfits.open(os.path.join(redux_dir,'n20160130','n20160130.thar_T4_i2test.0017.proc.fits'))
data = Tarc[0].data
wvln = Tarc[2].data
invar = Tarc[1].data
mask = Tarc[3].data

Traw = pyfits.open(os.path.join(data_dir,'n20160130','n20160130.thar_T4_i2test.0017.fits'))
raw_img = Traw[0].data


#plt.plot(data[0,0,:])
#plt.show()
#plt.close()

#########################################################
######## Find location and information on arc peaks #####
#########################################################

ts=3
pos_d, wl_d, mx_it_d, stddev_d, chi_d, err_d = psf.arc_peaks(data,wvln,invar,ts=ts)

#########################################################
########### Load Background Requirments #################
#########################################################

#hardcode in n20160115 directory
filename = args.filename#os.path.join(data_dir,'n20160115',args.filename)
software_vers = 'v0.2.1' #Later grab this from somewhere else

gain = 1.3
readnoise = 3.63

spectrum = pyfits.open(filename,uint=True)
spec_hdr = spectrum[0].header
#ccd = spectrum[0].data
ccd = Traw[0].data

#####CONVERT NASTY FORMAT TO ONE THAT ACTUALLY WORKS#####
#Dimensions
ypix = spec_hdr['NAXIS1']
xpix = spec_hdr['NAXIS2']
### Next part checks if iodine cell is in, assumes keyword I2POSAS exists
try:
    if spec_hdr['I2POSAS']=='in':
        i2 = True
    else:
        i2 = False
except KeyError:
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

for tscope in ['T1','T2','T3','T4']:
    #Choose fiberflats with iodine cell in
    if tscope=='T1':
        flnmflat = 'n20160130.fiberflat_T1.0023.fits'
        flnmarc = 'n20160130.thar_T1_i2test.0025.fits'
    #        continue
    elif tscope=='T2':
        flnmflat = 'n20160130.fiberflat_T2.0022.fits'
        flnmarc = 'n20160130.thar_T2_i2test.0020.fits'
    #        continue
    elif tscope=='T3':
        flnmflat = 'n20160130.fiberflat_T3.0014.fits'
        flnmarc = 'n20160130.thar_T3_i2test.0012.fits'
    #        continue
    elif tscope=='T4':
        flnmflat = 'n20160130.fiberflat_T4.0015.fits'
        flnmarc = 'n20160130.thar_T4_i2test.0017.fits'
    else:
        print("{} is not a valid telescope".format(tscope))
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


trace_coeffs, trace_intense_coeffs, trace_sig_coeffs, trace_pow_coeffs = utils.find_trace_coeffs(trace_ccd,2,args.fiber_space,num_points=args.num_points,num_fibers=28*args.telescopes,skip_peaks=1)

###Plot to visualize traces      
#fig,ax = plt.subplots()
#ax.pcolorfast(trace_ccd)
#for i in range(num_fibers-4):
#    ys = (np.arange(ypix)-ypix/2)/ypix
#    xs = trace_coeffs[2,i]*ys**2+trace_coeffs[1,i]*ys+trace_coeffs[0,i]
#    yp = np.arange(ypix)
#    plt.plot(yp,xs)
#plt.show()


#plt.imshow(raw_img,interpolation='none')
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
raw_img = raw_img[::-1,0:actypix]
raw_img -= bias #Note, if ccd is 16bit array, this operation can cause problems
### TODO check on whether this next part is valid...
#raw_img -= readnoise
raw_img[raw_img<0] = 0 #Enforce positivity

#########################################################
################## PSF fitting ##########################
#########################################################

trace = 10
#ts_num = 3
idx = trace*4 + ts + 1#_num
hcenters = pos_d[trace]
hscale = (hcenters-actypix/2)/actypix
#print("trace coeffs")
#print idx
#print hscale[0]
vcenters = trace_coeffs[2,idx]*hscale**2+trace_coeffs[1,idx]*hscale+trace_coeffs[0,idx]
#print vcenters[0]
#vcenters = xpix - vcenters.astype(int)
heights = trace_intense_coeffs[2,idx]*hscale**2+trace_intense_coeffs[1,idx]*hscale+trace_intense_coeffs[0,idx]
sigmas = trace_sig_coeffs[2,idx]*hscale**2+trace_sig_coeffs[1,idx]*hscale+trace_sig_coeffs[0,idx]
powers = trace_pow_coeffs[2,idx]*hscale**2+trace_pow_coeffs[1,idx]*hscale+trace_pow_coeffs[0,idx]
#vcenters = np.poly1d(trace_coeffs[::-1,i])(hscale)
#print hcenters
#print vcenters


#>>> for i in range(28):
#...     low_peak += len(mx_it_d[i][mx_it_d[i]<500])

print("Running PSF Fitting")


### 1. Estimate spline amplitudes, centers, w/ circular model
#r_breakpoints = [0, 1.2, 2.5, 3.7, 5, 8, 10]
## 2.3, 3
#r_breakpoints = np.hstack(([0, 1.5, 2.4, 3],np.arange(3.5,10,0.5))) #For cpad=8
r_breakpoints = np.hstack(([0, 1.5, 2.4, 3],np.arange(3.5,7.6,1))) #For cpad=6
#r_breakpoints = np.hstack(([0, 1.2, 2.3, 3],np.arange(3.5,10,0.5)))
theta_orders = [0]
cpad = 6
bp_space = 2 #breakpoint spacing in pixels
invar = 1/(raw_img+readnoise**2)
### Initial spline coeff guess
spline_coeffs, s_scale, fit_params, new_hcenters, new_vcenters = psf.spline_coeff_fit(raw_img,hcenters,vcenters,invar,r_breakpoints,sigmas,powers,theta_orders=theta_orders,cpad=cpad,bp_space=bp_space,return_new_centers=True)

#'''
### 2. Set up and initialize while loop (other steps embedded within loop)
num_bases = spline_coeffs.shape[1]
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
coeff_matrix_min = np.zeros((3,np.shape(spline_coeffs)[1])).T
params_min = lmfit.Parameters()
dlt_chi = 1e-4 #difference between successive chi_squared values to cut off
mx_loops = 100 #eventually must cutoff
loop_cnt = 0
fit_bg = True ## True fits a constant background at each subimage
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
    bg_data = np.zeros(len(new_hscale))
    for k in range(len(new_hscale)):
        ### Slice subset of image data around each peak
        harr = np.arange(-cpad,cpad+1)+int(np.floor(new_hcenters[k]))
        varr = np.arange(-cpad,cpad+1)+int(np.floor(new_vcenters[k]))
        data_for_fitting[:,:,k] = raw_img[varr[0]:varr[-1]+1,harr[0]:harr[-1]+1]#/s_scale[k]
        invar_for_fitting[:,:,k] = invar[varr[0]:varr[-1]+1,harr[0]:harr[-1]+1]#/s_scale[k]
        d_scale[k] = np.sum(data_for_fitting[:,:,k])
        bg_data[k] = psf.poisson_bg(data_for_fitting[:,:,k])
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
    print bg_data
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
        data_for_fitting[:,:,k] -= bg_data[k] ### remove bg first
        data_for_fitting[:,:,k] /= s_scale[k]
        invar_for_fitting[:,:,k] *= s_scale[k]
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
        print bg_array*s_scale
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
#        print chi_new[i]
#        print chi_new[i]*s_scale[i]
#        print chi_new[i]*s_scale[i]**2
        ### Set new scale - drive sum of image toward unity
        s_scale[i] = s_scale[i]*np.sum(fitted_image)
    #    plt.imshow(np.hstack((img_matrix,fitted_image,(img_matrix-fitted_image)*invar_matrix)),interpolation='none')
#        plt.imshow((img_matrix-fitted_image)*invar_matrix,interpolation='none')
#        plt.plot(img_matrix[:,5])
#        plt.plot(fitted_image[:,5])
#        plt.show()
#        plt.close()
    
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
    loop_cnt += 1
    ### Record minimum values (some subsequent iterations give higher chi2)
    if np.sum(chi_new) < chi_min:
        print "Better fit on loop ", loop_cnt
        chi_min = np.sum(chi_new)
        coeff_matrix_min = np.copy(coeff_matrix)
        params_min = lmfit.Parameters(params1)

### End of loop
print("End of Loop")
### Check that q, PA, aren't driving toward unphysical answers
test_hscale = np.arange(-1,1,0.01)
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
    plt.plot(fitted_image[:,cpad]/np.max(fitted_image[:,cpad]))
    plt.plot(np.sum(fitted_image,axis=1)/np.max(np.sum(fitted_image,axis=1)))
    plt.show()
    plt.imshow(np.hstack((img_matrix,fitted_image,(img_matrix-fitted_image))),interpolation='none')
#    plt.imshow((img_matrix-fitted_image)*invar_matrix,interpolation='none')
    plt.show()
    plt.close()

np.save('fitted_spline_coeffs_{}'.format(trace),coeff_matrix_min)
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
    
centers, ellipse = params_to_array(params_min)

np.save('elliptical_fits_{}'.format(trace),ellipse)
np.save('center_fits_{}'.format(trace),centers)
#'''

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
    
ncenters = np.load('center_fits_{}.npy'.format(trace))
nellipse = np.load('elliptical_fits_{}.npy'.format(trace))
best_coeffs = np.load('fitted_spline_coeffs_{}.npy'.format(trace))
params = arrays_to_params(ncenters,nellipse)


#and for eccentricity, position angle:
#ecc_pa_polyfit, ep_errs = sf.interpolate_coeffs(fit_params.T,np.ones((np.shape(fit_params.T))),pord,new_hcenters,np.zeros(len(new_hcenters)),pos_vals=True)
#ecc_pa_polyfit = ecc_pa_polyfit.T

### Check radial spline fitting code:
#spline_fit = spline_coeff_eval(raw_img,hcenters,hcenters[0]-5,vcenters,vcenters[0]-5,invar,r_breakpoints,spline_coeffs_polyfit,s_scale,sigmas,powers,full_image=False,view_plot=True)

#####################################################################
######### Now test 2D extraction with spline PSF ####################
#####################################################################

### Old junk code, but buried within is what I've done so far on the 2D extraction
#def intensity(xsize,ysize,xc,yc,b=0.1,sigma=1.1,q=0.75,r0=36,theta=np.pi/4,
#    rc=0.0001):
#    """Calculates intensity as a function of pixel position for the I(x,y)
#       given in Bolton and Schlegel - Spectroperfectionism.
#       
#       Inputs (NOTE, all must be numbers, not arrays):
#           xsize = number of "x" pixels
#           ysize = number of "y" pixels
#           xc = pixel coordinate of "x" center
#           yc = pixel coordinate of "y" center
#           b,sigma,q,r0,theta,rc = various parameters of profile
#           The last set is hardcoded for Bolton's paper by default
#           
#       Outputs:
#           Icore+Itail = 2-D mapping of intensity as a fct of x and y
#    """
#    x = np.tile(np.arange(xsize),(ysize,1))
#    y = np.tile(np.arange(ysize),(xsize,1)).transpose()
#    r = np.sqrt((x-xc)**2 + (y-yc)**2) #Radius
#    xprime = (x-xc)*cos(theta)-(y-yc)*sin(theta)
#    yprime = (x-xc)*sin(theta)+(y-yc)*cos(theta)
#    rell = np.sqrt(q*(xprime)**2 + (1/q)*(yprime)**2) #Elliptical radius
#    Icore = (1-b)/np.sqrt(2*np.pi*sigma)*np.exp(-rell**2/(2*sigma**2))
#    Itail = b*np.exp(-r/r0)/(2*np.pi*r0*(r+rc))
#    return (Icore+Itail)
#    
#
##Initialize parameters for I(x,y)
#b = 0.1
#sigma = 1.1
#q = 0.75
#r0 = 36
#theta = pi/4
#rc = 0.001 #Fudge rcore for wings to avoid divide by 0 situations
#xc = 31/2*np.ones((11)) + 0.05*rand.randn(11) #xpixel centers
#yc = 80/11*np.arange(11) + 10 + 0.05*rand.randn(11) #ypixel centers
#t0 = time.time()
#xsize = 31
#ysize = 101
##x = np.tile(np.arange(31),(101,1))
##y = np.tile(np.arange(101),(31,1)).transpose()
##Build simulated data CCD (no convolution for now, very basic)
#ccd = np.zeros((ysize,xsize)) #weird python thing - x,y coords are swapped
##Components of Intensity - repeat for each point
#for ii in range(len(xc)):
##    r = np.sqrt((x-xc[ii])**2 + (y-yc[ii])**2)
##    R = np.array(([cos(theta),-sin(theta)],[sin(theta),cos(theta)])) #Rotation Matrix
##    xprime = (x-xc[ii])*cos(theta)-(y-yc[ii])*sin(theta)
##    yprime = (x-xc[ii])*sin(theta)+(y-yc[ii])*cos(theta)
##    rell = np.sqrt(q*(xprime)**2 + (1/q)*(yprime)**2)
##    Icore = (1-b)/np.sqrt(2*np.pi*sigma)*np.exp(-rell**2/(2*sigma**2))
##    Itail = b*np.exp(-r/r0)/(2*np.pi*r0*(r+rc))
#    ccd += 10*intensity(xsize,ysize,xc[ii],yc[ii],b,sigma,q,r0,theta,rc)
#
#intval = np.sum(intensity(xsize,ysize,xc[ii],yc[ii],b,sigma,q,r0,theta,rc))
##Can delete - this is the slow way of doing calc with nested for loops
##t1 = time.time()
##rell = np.zeros((np.shape(ccd)))
##r = np.zeros((np.shape(ccd)))
##ccd2 = np.zeros((101,31))
##for ii in range(len(xc)):
##    for jj in range(31):
##        for kk in range(101):
##            r = np.sqrt((jj-xc[ii])**2 + (kk-yc[ii])**2)
##            xprime = (jj-xc[ii])*np.cos(theta)-(kk-yc[ii])*np.sin(theta)
##            yprime = (jj-xc[ii])*np.sin(theta)+(kk-yc[ii])*np.cos(theta)
##            rell = np.sqrt(q*(xprime)**2 + (1/q)*(yprime)**2)
##            I1 = (1-b)/np.sqrt(2*np.pi*sigma)*np.exp(-rell**2/(2*sigma**2))
##            I2 = b*np.exp(-r/r0)/(2*np.pi*r0*(r+rc))
##            ccd2[kk,jj] += I1+I2
##
##t2 = time.time()
##print("Array method takes time %f0.06s" % (t1-t0))
##print("Loop method takes time %f0.06s" % (t2-t1))
#
##plt.figure(1)
##plt.imshow(np.log10(ccd))
##plt.show()
#
##Add Noise
#N = np.identity(101)
#Ninv = np.matrix(N)
#ccd += random.poisson(np.shape(ccd))

#plt.plot(ccd[:,40])
#plt.plot(np.arange(20)+93,ccd_small[:,18])
#plt.show()
#plt.close()
#plt.imshow(ccd_small,interpolation='none')
#plt.show()
#plt.close()
    
#### Refine traces, etc.
############################################################################

###special functions to use during extraction
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
    
def fit_mn_hght_bg(xvals,zorig,invorig,sigj,mn_new,spread,powj=2):
#    mn_new = xc-xj
    mn_old = -100
    lp_ct = 0
    while abs(mn_new-mn_old)>0.001:
        mn_old = np.copy(mn_new)
        hght, bg = sf.best_linear_gauss(xvals,sigj,mn_old,zorig,invorig,power=powj)
        mn_new, mn_new_std = sf.best_mean(xvals,sigj,mn_old,hght,bg,zorig,invorig,spread,power=powj)
        lp_ct+=1
        if lp_ct>1e3: break
    return mn_new, hght,bg
       

### First fit xc parameters for traces
fact = 10 #do 1/fact of the available points
rough_pts = int(np.ceil(actypix/fact))
xc_ccd = np.zeros((num_fibers,rough_pts))
yc_ccd = np.zeros((num_fibers,rough_pts))
inv_chi = np.zeros((num_fibers,rough_pts))
yspec = np.arange(2048)
#sigrough = np.zeros((num_fibers,rough_pts)) #std
#meanrough = np.zeros((num_fibers,rough_pts)) #mean
#bgmrough = np.zeros((num_fibers,rough_pts)) #background mean
#bgsrough = np.zeros((num_fibers,rough_pts)) #background slope
#powrough = np.zeros((num_fibers,rough_pts)) #exponential power
print("Refining trace centers")
for i in range(num_fibers):
    if i != idx:
        continue
    else:
        print("Running on index {}".format(i))
#    if i > 3:
#        plt.plot(zspec2[i-1,:])
#        plt.show()
#        plt.close()
#    slit_num = np.floor((i)/args.telescopes)
#    print("Refining trace number {}".format(i+1))
    for j in range(0,actypix,fact):
#        if j != 1000:
#            continue
        jadj = int(np.floor(j/fact))
        yj = (yspec[j]-actypix/2)/actypix
        yc_ccd[i,jadj] = yspec[j]
        xc = trace_coeffs[2,i]*yj**2+trace_coeffs[1,i]*yj+trace_coeffs[0,i]
        Ij = trace_intense_coeffs[2,i]*yj**2+trace_intense_coeffs[1,i]*yj+trace_intense_coeffs[0,i]
        sigj = trace_sig_coeffs[2,i]*yj**2+trace_sig_coeffs[1,i]*yj+trace_sig_coeffs[0,i]
        powj = trace_pow_coeffs[2,i]*yj**2+trace_pow_coeffs[1,i]*yj+trace_pow_coeffs[0,i]
        if np.isnan(xc):
            xc_ccd[i,jadj] = np.nan
            inv_chi[i,jadj] = 0
#            sigrough[i,jadj] = np.nan
#            meanrough[i,jadj] = np.nan
#            bgmrough[i,jadj] = np.nan
#            bgsrough[i,jadj] = np.nan
#            powrough[i,jadj] = np.nan
        else:
            xpad = 7
            xvals = np.arange(-xpad,xpad+1)
            xj = int(xc)
            xwindow = xj+xvals
            xvals = xvals[(xwindow>=0)*(xwindow<xpix)]
            zorig = ccd[xj+xvals,yspec[j]]
            if len(zorig)<1:
                xc_ccd[i,jadj] = np.nan
                inv_chi[i,jadj] = 0
                continue
#                plt.figure()
#                plt.plot(xj+xvals,zorig,xj+xvals,zvals)
#                plt.show()
#                plt.close()
#                time.sleep(0.5)
            invorig = 1/(abs(zorig)+readnoise**2)
            if np.max(zorig)<20:
                xc_ccd[i,jadj] = np.nan
                inv_chi[i,jadj] = 0
            else:
                mn_new, hght, bg = fit_mn_hght_bg(xvals,zorig,invorig,sigj,xc-xj-1,sigj,powj=powj)
                fitorig = sf.gaussian(xvals,sigj,mn_new,hght,power=powj)
                fitorig2 = sf.gaussian(xvals,sigj,xc-xj-1,hght,power=powj)
#                if i==idx and j == 60:
#                    print("Other trace coeffs")
#                    print i
#                    print xc
#                    print idx
#                    print xj+xvals
#                    print yspec[j]
#                    print ccd[xj+xvals,yspec[j]]
#                    print zorig/1.3
#                    print j
#                    plt.figure("First fit attempt")
#                    print "Mn_new:", mn_new+xj+1
#                    plt.plot(np.arange(len(zorig))+3,zorig)
#                    plt.plot(xvals,zorig,xvals,fitorig)#2,xvals,fitorig)
#                    plt.show()
#                    plt.close()
                if j == 715:
                    print mn_new
                    plt.plot(xvals,zorig,xvals,fitorig)
                    plt.show()
                    plt.close()
                inv_chi[i,jadj] = 1/sum((zorig-fitorig)**2*invorig)
                xc_ccd[i,jadj] = mn_new+xj+1
                

tmp_poly_ord = 10
trace_coeffs_ccd = np.zeros((tmp_poly_ord+1,num_fibers))
for i in range(num_fibers):
    if i != idx:
        continue
    #Given orientation makes more sense to swap x/y
    mask = ~np.isnan(xc_ccd[i,:])
    profile = np.ones((len(yc_ccd[i,:][mask]),tmp_poly_ord+1)) #Quadratic fit
    for order in range(tmp_poly_ord):
#    profile[:,1] = (yc_ccd[i,:][mask]-ypix/2)/ypix #scale data to get better fit
        profile[:,order+1] = ((yc_ccd[i,:][mask]-actypix/2)/actypix)**(order+1)
    noise = np.diag(inv_chi[i,:][mask])
    if len(xc_ccd[i,:][mask])>3:
        tmp_coeffs, junk = sf.chi_fit(xc_ccd[i,:][mask],profile,noise)
        yvals = (yc_ccd[i,:][mask]-actypix/2)/actypix
        fit = np.poly1d(tmp_coeffs[::-1])(yvals)
        fit_old = np.poly1d(trace_coeffs[::-1,i])(yvals)
#        tmp_coeffs = tmp_coeffs[:,0]
#        if i==idx:
#            print yvals
#            print xc_ccd[i,:]
#            plt.plot(yvals,xc_ccd[i,:][mask],yvals,fit)
#            plt.plot((new_hcenters-actypix/2)/actypix,new_vcenters)
#            plt.show()
#            plt.close()
    else:
        tmp_coeffs = np.nan*np.ones((tmp_poly_ord+1))
    trace_coeffs_ccd[:,i] = tmp_coeffs
     
############################################################################


def intensity(xsize,ysize,xc,yc,b=0.1,sigma=1.36,q=1,r0=3,theta=0,
    rc=0.0001):
    """Calculates intensity as a function of pixel position for the I(x,y)
       given in Bolton and Schlegel - Spectroperfectionism.
       
       Inputs (NOTE, all must be numbers, not arrays):
           xsize = number of "x" pixels
           ysize = number of "y" pixels
           xc = pixel coordinate of "x" center
           yc = pixel coordinate of "y" center
           b,sigma,q,r0,theta,rc = various parameters of profile
           The last set is hardcoded for Bolton's paper by default
           
       Outputs:
           Icore+Itail = 2-D mapping of intensity as a fct of x and y
    """
    x = np.tile(np.arange(xsize),(ysize,1))
    y = np.tile(np.arange(ysize),(xsize,1)).transpose()
    r = np.sqrt((x-xc)**2 + (y-yc)**2) #Radius
    xprime = (x-xc)*cos(theta)-(y-yc)*sin(theta)
    yprime = (x-xc)*sin(theta)+(y-yc)*cos(theta)
    rell = np.sqrt(q*(xprime)**2 + (1/q)*(yprime)**2) #Elliptical radius
    Icore = (1-b)/np.sqrt(2*np.pi*sigma)*np.exp(-rell**2/(2*sigma**2))
    Itail = b*np.exp(-r/r0)/(2*np.pi*r0*(r+rc))
    return (Icore+Itail)

######################################################################
########## Load and calibrate data ###################################
######################################################################

### Load section to extract
testfits = pyfits.open('n20160323.HR4828.0020.fits')
ccd = testfits[0].data
ccd = ccd[::-1,0:2048]
date = 'n20160115' #Fixed for now, late make this dynamic
bias_hdu = pyfits.open(os.path.join(redux_dir,date,'bias_avg.fits'),uint=True)
bias = bias_hdu[0].data

### subtract bias (slit will be handled in loop)
bias = bias[::-1,0:2048] #Remove overscan
ccd -= bias #Note, if ccd is 16bit array, this operation can cause problems
### TODO, check and see if this next part is valid
#print np.median(ccd[ccd<np.mean(ccd)])
#ccd -= np.median(ccd[ccd<np.mean(ccd)])#+readnoise
#ccd[ccd<0] = 0 #Enforce positivity
ccd *= gain #include gain
#plt.imshow(ccd,interpolation='none')
#plt.show()
#plt.close()

### Choose small section to extract
hc = hcenters[5]
vc = vcenters[5]
#print hc, vc
vsp = 6
hsp = 50
vinds = np.arange(int(vc-vsp),int(vc+vsp))
hinds = np.arange(hc-hsp,hc+hsp)
### set coordinates for opposite corners of box (use in building profile matrix)
vtl = vinds[0]
htl = hinds[0]
vbr = vinds[-1]
hbr = hinds[-1]
#print vinds
#print hinds
ccd_small = ccd[vinds[0]:vinds[-1],hinds[0]:hinds[-1]]
ccd_small_invar = 1/(ccd_small + readnoise**2)

### try removing background...
bg_mask = np.zeros(ccd_small.shape,dtype=bool)
bg_mask[int(vsp/2):3*int(vsp/2),:] = 1
#plt.imshow(bg_mask,interpolation='none')
#plt.show()
ccd_tmp = np.copy(ccd_small)
ccd_tmp[bg_mask] = 0
rm_bg = np.median(ccd_small[bg_mask==0])
#plt.imshow(ccd_tmp,interpolation='none')
#plt.show()
ccd_bg = psf.poisson_bg(ccd_small)
#print rm_bg
ccd_small -= rm_bg

        
#Attempt 2-D extraction

print("Running 2D Extraction")
tstart = time.time()
#Build profile Matrix A_ijl
wls = int(np.shape(ccd_small)[1]*1)
hcents = np.linspace(hc-hsp,hc+hsp-1,wls)
hscale = (hcents-actypix/2)/actypix
vcents = sf.eval_polynomial_coeffs(hscale,trace_coeffs_ccd[:,idx])
d0 = ccd_small.shape[0]
d1 = ccd_small.shape[1]
A = np.zeros((wls,d0,d1)) #last dimension is #of wavelengths - use 1/pixel
A2 = np.zeros((wls,d0,d1))
sigmas = trace_sig_coeffs[2,idx]*hscale**2+trace_sig_coeffs[1,idx]*hscale+trace_sig_coeffs[0,idx]
powers = trace_pow_coeffs[2,idx]*hscale**2+trace_pow_coeffs[1,idx]*hscale+trace_pow_coeffs[0,idx]

#xc = 31/2
dlth = np.mean(np.ediff1d(hcents))
wl_pad = 0#int(cpad/dlth)
A = np.zeros((wls+2*wl_pad,d0,d1)) # pad A matrix
A_bg = np.zeros((A.shape[0]))
bg_idx = 0
fitpad = 4 ### Padding for fit
final_centers = np.zeros((wls+2*wl_pad,2))
for jj in range(-wl_pad,wls+wl_pad):
#    print jj
    if jj < 0:
        hcent = hcents[0]+jj*dlth
        vcent = sf.eval_polynomial_coeffs((hcent-actypix/2)/actypix,trace_coeffs_ccd[:,idx])[0]
    elif jj >= wls:
        hcent = hcents[-1]+(jj-wls+1)*dlth
        vcent = sf.eval_polynomial_coeffs((hcent-actypix/2)/actypix,trace_coeffs_ccd[:,idx])[0]
    else:
        hcent = hcents[jj]
        vcent = vcents[jj]
    vcent -= 1  ### Something is wrong above - shouldn't need this...
    center = [np.mod(hcent,1)+fitpad,np.mod(vcent,1)+fitpad]
    final_centers[jj,:] = center
    hpoint = (hcent-actypix/2)/actypix
    sp_ft = spline.make_spline_model(params,best_coeffs,center,hpoint,[2*fitpad+1,2*fitpad+1],r_breakpoints,theta_orders,fit_bg=False)
#    plt.plot(np.sum(sp_ft,axis=0))
#    plt.show()
#    plt.close()
    bg_lvl = np.median(sp_ft[sp_ft<np.mean(sp_ft)])
    sp_ft -= bg_lvl
#    plt.imshow(sp_ft,interpolation='none')
#    plt.show()
#    plt.close()
#    plt.plot(sp_ft[:,5])
#    plt.show()   
    sp_ft /= np.sum(sp_ft) # Normalize to 1
#    A_bg[bg_idx] += (bg_lvl/np.sum(sp_ft))
#    bg_idx += 1
#    plt.imshow(sp_ft,interpolation='none')
#    plt.show()
#    plt.close()
    ### indices of sp_ft slice to use
    sp_l = max(0,fitpad+(htl-int(hcent))) #left edge
    sp_r = min(2*fitpad+1,fitpad+(hbr-int(hcent))) #right edge
    sp_t = max(0,fitpad+(vtl-int(vcent))) #top edge
    sp_b = min(2*fitpad+1,fitpad+(vbr-int(vcent))) #bottom edge
    ### indices of A slice to use
    a_l = max(0,int(hcent)-htl-fitpad) # left edge
    a_r = min(A.shape[2],int(hcent)-htl+fitpad+1) # right edge
    a_t = max(0,int(vcent)-vtl-fitpad) # top edge
    a_b = min(A.shape[1],int(vcent)-vtl+fitpad+1) # bottom edge    
#    sp_ft = psf.spline_coeff_eval(ccd_small,[hcent],hcents[0],[vcent],np.ceil(vcents[0])-vsp+1,ccd_small_invar,r_breakpoints,spline_coeffs_polyfit,[1],[sigmas[jj]],[powers[jj]],cpad=0,view_plot=False,ecc_pa_coeffs=ecc_pa_polyfit)
    A[jj+wl_pad,a_t:a_b,a_l:a_r] = sp_ft[sp_t:sp_b,sp_l:sp_r] #Normalize to integral of 1
#    plt.imshow(A[jj+wl_pad,a_t:a_b,a_l:a_r],interpolation='none')
#    plt.show()
#    plt.close()
#    A2[jj,:,:] = intensity(2*hsp-1,2*vsp-1,hcent-hcents[0],vcent-(np.ceil(vcents[0])-vsp+2),sigma=sigmas[jj])
#    if jj<=30:
#        plt.figure("Intensity")
#        plt.imshow(A[jj,:,:]/np.sum(A[jj,:,:])-(A2[jj,:,:]/np.sum(A2[jj,:,:])))
#        plt.show()
#        plt.close()
#        plt.plot(A[jj,:,5]/np.max(A[jj,:,5]))
#        plt.plot(ccd_small[:,jj]/np.max(ccd_small[:,jj]))
#        plt.plot(A2[jj,:,jj]/np.max(A2[jj,:,jj]))
#        plt.plot(sf.gaussian(np.arange(9),sigmas[jj],vcent-(np.ceil(vcents[0])-vsp+1),power=powers[jj]))
#        plt.show()
#        plt.close()
    
A = A[wl_pad:wls+wl_pad] #cut off edges
#A_bg_mn = np.mean(A_bg)/(2*fitpad+1)
#A += A_bg_mn
A_proj = np.sum(A,axis=0)
#plt.imshow(np.vstack((A_proj/np.sum(A_proj),ccd_small/np.sum(ccd_small))),interpolation='none')
#plt.imshow(A_proj/np.sum(A_proj)-ccd_small/np.sum(ccd_small),interpolation='none')
#plt.imshow(A_proj/np.sum(A_proj)-A_2/np.sum(A_2),interpolation='none')
#plt.show()
#plt.show()
#plt.plot(np.vstack((A_proj/np.sum(A_proj),ccd_small/np.sum(ccd_small)))[:,20])
#plt.show()

##Now using the full available data
B = np.matrix(np.resize(A.T,(d0*d1,wls)))
B = np.hstack((B,np.ones((d0*d1,1)))) ### add background term
p = np.matrix(np.resize(ccd_small.T,(d0*d1,1)))
n = np.diag(np.resize(ccd_small_invar.T,(d0*d1,)))
#print np.shape(B), np.shape(p), np.shape(n)
text_sp_st = time.time()
fluxtilde2 = sf.extract_2D_sparse(p,B,n)
t_betw_ext = time.time()
#fluxtilde3 = sf.extract_2D(p,B,n)
tfinish = time.time()
print "Total Time = ", tfinish-tstart
print("PSF modeling took {}s".format(text_sp_st-tstart))
print("Sparse extraction took {}s".format(t_betw_ext-text_sp_st))
#print("Regular extraction took {}s".format(tfinish-t_betw_ext))
flux2 = sf.extract_2D_sparse(p,B,n,return_no_conv=True)
#Ninv = np.matrix(np.diag(np.resize(ccd_small_invar.T,(d0*d1,))))
#Cinv = B.transpose()*Ninv*B
#U, s, Vt = linalg.svd(Cinv)
#Cpsuedo = Vt.transpose()*np.matrix(np.diag(1/s))*U.transpose();
#flux2 = Cpsuedo*(B.transpose()*Ninv*p)
#
#d, Wt = linalg.eig(Cinv)
#D = np.matrix(np.diag(np.asarray(d)))
#WtDhW = Wt*np.sqrt(D)*Wt.transpose()
#
#WtDhW = np.asarray(WtDhW)
#s = np.sum(WtDhW,axis=1)
#S = np.matrix(np.diag(s))
#Sinv = linalg.inv(S)
#WtDhW = np.matrix(WtDhW)
#R = Sinv*WtDhW
#fluxtilde2 = R*flux2
#fluxtilde2 = np.asarray(fluxtilde2)
#flux2 = np.asarray(flux2)

img_est = np.dot(B,flux2)
img_estrc = np.dot(B,fluxtilde2)
img_recon = np.real(np.resize(img_estrc,(d1,d0)).T)
plt.figure("Residuals of 2D fit")
plt.imshow(np.vstack((ccd_small,img_recon,ccd_small-img_recon)),interpolation='none')
chi_red = np.sum((ccd_small-img_recon)[:,fitpad:-fitpad]**2*ccd_small_invar[:,fitpad:-fitpad])/(np.size(ccd_small[:,fitpad:-fitpad])-jj+1)
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


waves = np.arange(wls)#*101/wls
scale = np.mean(np.ediff1d(hcents))
#fluxtilde2.reshape(wls,)
#plt.plot(waves,flux2)
plt.figure("2D vs. Optimal Extraction")
fluxtilde2 = fluxtilde2[0:-1] ### remove last point - bg
plt.plot(hcents,fluxtilde2/scale,linewidth='2')
#plt.plot(hcents,2*fluxtilde3*scale)
#plt.show()
#plt.plot(hcents,3.7*flux2*scale)

###Compare to optimal extraction:
opt_fits = pyfits.open('n20160323.HR4828.0020.proc.fits')
opt_dat = opt_fits[0].data
opt_wav = opt_fits[1].data
opt_dat_sec = opt_dat[ts,trace,int(hcents[0]):int(hcents[-1])]
plt.plot(np.linspace(hcents[0],hcents[-1],len(opt_dat_sec)),opt_dat_sec,linewidth='2')
##plt.plot(yc,max(fluxtilde2)*np.ones(len(yc)),'ko')
plt.show()

#plt.imshow(fitted_image,interpolation='none')
#plt.show()

#img_opt = np.resize(np.dot(B,opt_dat_sec),(d1,d0)).T
#plt.imshow(img_opt,interpolation='none')
#plt.show()

ln = 0
#print(ccd[np.arange(481,496),60])
#print(ccd_small[:,ln])
#plt.plot(ccd_small[:,ln]/np.max(ccd_small[:,ln]))
#plt.plot(A[ln,:,0]/np.max(A[ln,:,0]))
#plt.show()

#xaxis = np.arange(21)-9.5
#yaxis = np.arange(21)-9.5
#sigma = 2
#gaussgrid = sf.gauss2d(xaxis,yaxis,sigma,sigma*1.2)
##plt.imshow(gaussgrid)
##plt.show()
#
#for order in range(6):
#    xherm = np.arange(-3,4,.01)
#    sig = 1
#    hout = sf.hermite(order,xherm,sig)
##    plt.plot(xherm,hout)
#    
##plt.show()
#sigx = 2
#sigy = 2.4
#hermgrid = np.zeros((np.shape(gaussgrid)))
#for ii in range(len(xaxis)):
#    for jj in range(len(yaxis)):
#        #Need to figure out a way to permute all order combinations
#        for orderx in range(3):
#            for ordery in range(3):
#                hermx = sf.hermite(orderx,ii,sigx)
#                hermy = sf.hermite(ordery,jj,sigy)
#                weight = 1/(1+orderx+ordery)
#                hermgrid[jj,ii] += weight*hermx*hermy
#            
#netgrid=gaussgrid*hermgrid
##plt.imshow(netgrid.T,interpolation='none')
##plt.show()
plt.close()

###Below is code snipped from simple_opt_ex
###Probably a better choice to make new code with general functions (trace-fitting
###in particular) instead of using this duplicate, but some of it should still be useful
'''

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
software_vers = 'v0.2.1' #Later grab this from somewhere else

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
try:
    if spec_hdr['I2POSAS']=='in':
        i2 = True
    else:
        i2 = False
except KeyError:
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


### Code for testing 2D spline - DELETE when done
#xpix = 20
#ypix = 20
#xgauss = np.reshape(sf.gaussian(np.arange(xpix),3,center=xpix/2),(1,xpix))
#ygauss = np.reshape(sf.gaussian(np.arange(ypix),3,center=ypix/2),(1,ypix))
#truth = np.dot(xgauss.T,ygauss)
#signal = truth + 0.1*np.mean(truth)*np.random.randn(xpix,ypix)
#bp_space = 3 #breakpoint spacing in pixels
#xarr = np.arange(xpix)
#yarr = np.arange(ypix)
#x_breakpoints = xarr[np.mod(xarr,bp_space)==0]
#y_breakpoints = yarr[np.mod(yarr,bp_space)==0]
#
#spline_fit = spline_2D(signal,np.ones((xpix,ypix))*0.1*np.mean(truth),y_breakpoints,x_breakpoints)
#
#visualization = np.hstack((signal,spline_fit,signal-spline_fit))
#plt.imshow(visualization,interpolation='none')
#plt.show()



### Old junk code, but buried within is what I've done so far on the 2D extraction
def intensity(xsize,ysize,xc,yc,b=0.1,sigma=1.1,q=0.75,r0=36,theta=np.pi/4,
    rc=0.0001):
    """Calculates intensity as a function of pixel position for the I(x,y)
       given in Bolton and Schlegel - Spectroperfectionism.
       
       Inputs (NOTE, all must be numbers, not arrays):
           xsize = number of "x" pixels
           ysize = number of "y" pixels
           xc = pixel coordinate of "x" center
           yc = pixel coordinate of "y" center
           b,sigma,q,r0,theta,rc = various parameters of profile
           The last set is hardcoded for Bolton's paper by default
           
       Outputs:
           Icore+Itail = 2-D mapping of intensity as a fct of x and y
    """
    x = np.tile(np.arange(xsize),(ysize,1))
    y = np.tile(np.arange(ysize),(xsize,1)).transpose()
    r = np.sqrt((x-xc)**2 + (y-yc)**2) #Radius
    xprime = (x-xc)*cos(theta)-(y-yc)*sin(theta)
    yprime = (x-xc)*sin(theta)+(y-yc)*cos(theta)
    rell = np.sqrt(q*(xprime)**2 + (1/q)*(yprime)**2) #Elliptical radius
    Icore = (1-b)/np.sqrt(2*np.pi*sigma)*np.exp(-rell**2/(2*sigma**2))
    Itail = b*np.exp(-r/r0)/(2*np.pi*r0*(r+rc))
    return (Icore+Itail)
    

#Initialize parameters for I(x,y)
b = 0.1
sigma = 1.1
q = 0.75
r0 = 36
theta = pi/4
rc = 0.001 #Fudge rcore for wings to avoid divide by 0 situations
xc = 31/2*np.ones((11)) + 0.05*rand.randn(11) #xpixel centers
yc = 80/11*np.arange(11) + 10 + 0.05*rand.randn(11) #ypixel centers
t0 = time.time()
xsize = 31
ysize = 101
#x = np.tile(np.arange(31),(101,1))
#y = np.tile(np.arange(101),(31,1)).transpose()
#Build simulated data CCD (no convolution for now, very basic)
ccd = np.zeros((ysize,xsize)) #weird python thing - x,y coords are swapped
#Components of Intensity - repeat for each point
for ii in range(len(xc)):
#    r = np.sqrt((x-xc[ii])**2 + (y-yc[ii])**2)
#    R = np.array(([cos(theta),-sin(theta)],[sin(theta),cos(theta)])) #Rotation Matrix
#    xprime = (x-xc[ii])*cos(theta)-(y-yc[ii])*sin(theta)
#    yprime = (x-xc[ii])*sin(theta)+(y-yc[ii])*cos(theta)
#    rell = np.sqrt(q*(xprime)**2 + (1/q)*(yprime)**2)
#    Icore = (1-b)/np.sqrt(2*np.pi*sigma)*np.exp(-rell**2/(2*sigma**2))
#    Itail = b*np.exp(-r/r0)/(2*np.pi*r0*(r+rc))
    ccd += 10*intensity(xsize,ysize,xc[ii],yc[ii],b,sigma,q,r0,theta,rc)

intval = np.sum(intensity(xsize,ysize,xc[ii],yc[ii],b,sigma,q,r0,theta,rc))
#Can delete - this is the slow way of doing calc with nested for loops
#t1 = time.time()
#rell = np.zeros((np.shape(ccd)))
#r = np.zeros((np.shape(ccd)))
#ccd2 = np.zeros((101,31))
#for ii in range(len(xc)):
#    for jj in range(31):
#        for kk in range(101):
#            r = np.sqrt((jj-xc[ii])**2 + (kk-yc[ii])**2)
#            xprime = (jj-xc[ii])*np.cos(theta)-(kk-yc[ii])*np.sin(theta)
#            yprime = (jj-xc[ii])*np.sin(theta)+(kk-yc[ii])*np.cos(theta)
#            rell = np.sqrt(q*(xprime)**2 + (1/q)*(yprime)**2)
#            I1 = (1-b)/np.sqrt(2*np.pi*sigma)*np.exp(-rell**2/(2*sigma**2))
#            I2 = b*np.exp(-r/r0)/(2*np.pi*r0*(r+rc))
#            ccd2[kk,jj] += I1+I2
#
#t2 = time.time()
#print("Array method takes time %f0.06s" % (t1-t0))
#print("Loop method takes time %f0.06s" % (t2-t1))

#plt.figure(1)
#plt.imshow(np.log10(ccd))
#plt.show()

#Add Noise
N = np.identity(101)
Ninv = np.matrix(N)
#ccd += random.poisson(np.shape(ccd))
            
############################################################################
            
#Attempt 2-D extraction

#Build profile Matrix A_ijl
wls = 200
A = np.zeros((wls,101,31)) #last dimension is #of wavelengths - use 1/pixel
xc = 31/2
for jj in range(wls):
    A[jj,:,:] = intensity(xsize,ysize,xc,jj*(101/wls),b,sigma,q,r0,theta,rc)
    

##Now using the full available data
B = np.matrix(np.resize(A.transpose(),(101*31,wls)))
p = np.matrix(np.resize(ccd.transpose(),(101*31,1)))
Ninv = np.matrix(np.identity(101*31));
Cinv = B.transpose()*Ninv*B
U, s, Vt = linalg.svd(Cinv)
Cpsuedo = Vt.transpose()*np.matrix(np.diag(1/s))*U.transpose();
flux2 = Cpsuedo*(B.transpose()*Ninv*p)

d, Wt = linalg.eig(Cinv)
D = np.matrix(np.diag(np.asarray(d)))
WtDhW = Wt*np.sqrt(D)*Wt.transpose()

WtDhW = np.asarray(WtDhW)
s = np.sum(WtDhW,axis=1)
S = np.matrix(np.diag(s))
Sinv = linalg.inv(S)
WtDhW = np.matrix(WtDhW)
R = Sinv*WtDhW

fluxtilde2 = R*flux2
fluxtilde2 = np.asarray(fluxtilde2)
flux2 = np.asarray(flux2)

waves = np.arange(wls)*101/wls
#fluxtilde2.reshape(wls,)
plt.plot(waves,flux2)
plt.plot(waves,fluxtilde2)
plt.plot(yc,max(fluxtilde2)*np.ones(len(yc)),'ko')
plt.show()

xaxis = np.arange(21)-9.5
yaxis = np.arange(21)-9.5
sigma = 2
gaussgrid = sf.gauss2d(xaxis,yaxis,sigma,sigma*1.2)
#plt.imshow(gaussgrid)
#plt.show()

for order in range(6):
    xherm = np.arange(-3,4,.01)
    sig = 1
    hout = sf.hermite(order,xherm,sig)
#    plt.plot(xherm,hout)
    
#plt.show()
sigx = 2
sigy = 2.4
hermgrid = np.zeros((np.shape(gaussgrid)))
for ii in range(len(xaxis)):
    for jj in range(len(yaxis)):
        #Need to figure out a way to permute all order combinations
        for orderx in range(3):
            for ordery in range(3):
                hermx = sf.hermite(orderx,ii,sigx)
                hermy = sf.hermite(ordery,jj,sigy)
                weight = 1/(1+orderx+ordery)
                hermgrid[jj,ii] += weight*hermx*hermy
            
netgrid=gaussgrid*hermgrid
#plt.imshow(netgrid.T,interpolation='none')
#plt.show()
'''