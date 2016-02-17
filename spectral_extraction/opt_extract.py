#!/usr/bin/env python 2.7

#Code to run conventional extraction for comparision to "extreme"
#forward modeling method.

#Import all of the necessary packages
from __future__ import division
import pyfits
import os
#import math
import time
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from matplotlib import cm
#import scipy
#import scipy.stats as stats
import scipy.special as sp
import scipy.interpolate as si
#import scipy.optimize as opt
import scipy.sparse as sparse
#import scipy.signal as sig
import scipy.linalg as linalg

t0 = time.clock()

def amp_vs_pos(ypos,y_st,l_st,a,b,samp,res):#,A,wl,tp):
    """Takes a given y coordinate and returns the wavelength (assuming the
       provided A vs. wavelength function).
       Inputs:
       ypos = the position to be evaluated
       y_st = the starting y coordinate of the given trace
       l_st = the wavelength corresponding to y_st
       a = the first polynomial coefficient of the given trace (a*y**2)
       b = the second polynomial coefficient of the given trace (b*y)
       samp = sampling
       res = resolution
       """
    pix_st = 1/(4*a)*(sqrt(4*a**2*y_st**2 + 4*a*b*y_st + b**2 + 1)*(2*a*y_st + 
             b) + np.arcsin(2*a*y_st + b))
    pix_pos = 1/(4*a)*(sqrt(4*a**2*ypos**2 + 4*a*b*ypos + b**2 + 1)*(2*a*ypos + 
             b) + np.arcsin(2*a*ypos + b))
    pix_pos_rel = pix_pos - pix_st
    SR = samp*res
    wave_pos = l_st*np.exp(pix_pos_rel/SR)
#    amp = 0
#    for i in range(1,len(A)):
#        if wl[i] >= wave_pos:
#            if wl[i-1] <= wave_pos:
#                Aslope = (A[i] - A[i-1]) / (wl[i] - wl[i-1])
#                tpslope = (tp[i] - tp[i-1]) / (wl[i] - wl[i-1])
#                amp = (A[i-1] + Aslope*(wave_pos-wl[i-1]))*(tp[i-1] + 
#                       tpslope*(wave_pos-wl[i-1]))
    return wave_pos
    
def guess_to_gauss(axis, center, sigma_g, height, width, scale_gauss):
    """
    Fit for gaussian, assuming parameters given for Lorentian wings.
    
    Function to evaluate the chi-squared for a given guess.  The input
    "guess" must be composed of a width(Gaussian), width_ratio(Lorentz/Gauss),
    scale_gauss(Gauss/total), constant, and slope (latter two for background
    fitting).

    This implementation assumes an integrated Gaussian with integrated
    Lorentzian wings.  The argument data must be the raw data values,
    invvars is the inverse variance at each point, and axis is the index
    of each point.x_originals = open(path + 'blue_x.txt')
    y_originals = open(path + 'blue_y.txt')
    lambda_starts = open(path + 'lambda_start_b.txt')
    lambda_deltas = open(path + 'lambda_delta_b.txt')
    """  
#    center = guess[0]
#    sigma_g = guess[1]
#    height = guess[2]
#    width = abs(width)
#    if scale_gauss > 1:
#        scale_gauss = 1 - (scale_gauss-1)
#    if scale_gauss < 0:
#        scale_gauss = 0.5
    modelg = height*scale_gauss*(1/(sqrt(2*pi*sigma_g**2)))*exp(-(axis-center)**2/(2*sigma_g**2))
    return modelg

mode = input("Enter frame 'b'(blue), 'r'(red), 'sb'/'sr', or 'ab'/'ar'(arc blue/red): ")
path = os.environ['EXPRES_OUT_DIR'] + '/'
if mode == 'b':
    flat_fits = pyfits.open(path + 'solar_b_flat.fits')    
    fits_name = 'data_v_b0_0_0.fits'
    image_fits = pyfits.open(path + fits_name)
    image = image_fits[0].data
    invr = image_fits[1].data
    x_originals = open(path + 'blue_x.txt')
    y_originals = open(path + 'blue_y.txt')
    lambda_starts = open(path + 'lambda_start_b.txt')
    lambda_deltas = open(path + 'lambda_delta_b.txt')
elif mode == 'sb':
    flat_fits = pyfits.open(path + 'solar_b_flat.fits')
    fits_name = 'solar_v_b0_0.fits'
    image_fits = pyfits.open(path + fits_name)
    image = image_fits[0].data
    bg_mean = 4
    gain = 2
    invr = 1/(abs(image-bg_mean*gain) + bg_mean**2)
    x_originals = open(path + 'blue_x.txt')
    y_originals = open(path + 'blue_y.txt')
    lambda_starts = open(path + 'lambda_start_b.txt')
    lambda_deltas = open(path + 'lambda_delta_b.txt')
elif mode == 'r':
    flat_fits = pyfits.open(path + 'solar_r_flat.fits')
    fits_name = 'data_v_r0_0_0.fits'
    image_fits = pyfits.open(path + fits_name)
    image = image_fits[0].data
    invr = image_fits[1].data
    x_originals = open(path + 'red_x.txt')
    y_originals = open(path + 'red_y.txt')
    lambda_starts = open(path + 'lambda_start_r.txt')
    lambda_deltas = open(path + 'lambda_delta_r.txt')
elif mode == 'sr':
    flat_fits = pyfits.open(path + 'solar_r_flat.fits')
    fits_name = 'solar_v_r0_0.fits'
    image_fits = pyfits.open(path + fits_name)
    image = image_fits[0].data
    bg_mean = 4
    gain = 2
    invr = 1/(abs(image-bg_mean*gain) + bg_mean**2)
    x_originals = open(path + 'red_x.txt')
    y_originals = open(path + 'red_y.txt')
    lambda_starts = open(path + 'lambda_start_r.txt')
    lambda_deltas = open(path + 'lambda_delta_r.txt')
elif mode == 'ab':
    flat_fits = pyfits.open(path + 'solar_b_flat.fits')
    fits_name = 'solar_b_arc.fits'
    image_fits = pyfits.open(path + fits_name)
    image = image_fits[0].data
    invr = ones((np.shape(image)))
    x_originals = open(path + 'blue_x.txt')
    y_originals = open(path + 'blue_y.txt')
    lambda_starts = open(path + 'lambda_start_b.txt')
    lambda_deltas = open(path + 'lambda_delta_b.txt')
elif mode == 'ar':
    flat_fits = pyfits.open(path + 'solar_r_flat.fits')
    fits_name = 'solar_r_arc.fits'
    image_fits = pyfits.open(path + fits_name)
    image = image_fits[0].data
    invr = ones((np.shape(image)))
    x_originals = open(path + 'red_x.txt')
    y_originals = open(path + 'red_y.txt')
    lambda_starts = open(path + 'lambda_start_r.txt')
    lambda_deltas = open(path + 'lambda_delta_r.txt')
else:
    print ("Not a valid mode!  Enter 'b', 'r', 'ab', or 'ar'")

flat = flat_fits[0].data
#ax = plt.subplot(1,1,1)
#ax.pcolorfast(image)
#plt.show()
ypixel = 9216 ###MANUAL CORRECTION FOR PX SHIFT!
xpixel = 9232 ###MANUAL CORRECTION FOR PX SHIFT!
samp = 3.5
res = 230e3
num_traces = 36
traces = zeros((num_traces,ypixel))
yvals = np.arange(ypixel) - ypixel/2
wlvals = zeros((num_traces,ypixel))
for i in range(num_traces):
    x = zeros((5))
    y = zeros((5))
    for r in range(5):
        x[r] = float(x_originals.readline()[:-2])*100
        y[r] = float(y_originals.readline()[:-2])*100
    coeffs = np.polyfit(y, x, 2)
    a = coeffs[0]
    b = coeffs[1]
    c = coeffs[2]
    traces[i,:] = a*yvals**2 + b*yvals + c + ypixel/2 ###MANUAL CORRECTION FOR PX SHIFT!
    l_st = float(lambda_starts.readline()[:-2]) #Angstroms
    l_dlta = float(lambda_deltas.readline()[:-2])
    #l_end = l_st + l_dlta
    y_st = np.round(y[-1]) + 8 ###MANUAL CORRECTION FOR PX SHIFT!
    wlvals[i,:] = amp_vs_pos(yvals,y_st,l_st,a,b,samp,res)
#    if i == 0:
#        print a, b, y[0], y_st
#    wlvals[i,:] = amp_vs_pos(yvals[::-1],y[0],y[-1],l_st,l_dlta,a,b)
#    plt.plot(traces[i,:],yvals+xpixel/2)
#    plt.plot(wlvals[i,:],ones((ypixel)))
    
#plt.show()
    
# - (xt % 1)
#    plt.plot(xaxis[j,:],profile[j,:])
#    
#plt.show()

#pro_sort = np.sort(profile,axis=0)
#pro_sort.flatten('F')

#tck_pro = zeros((num_traces,3))
#scale = zeros((num_traces))
#for l in range(num_traces):
#    tck_pro[l,:] = si.splrep(xaxis[l,:],profile[l,:])
##tck_first = si.splrep(xaxis[0,:],profile[0,:])
##tck_last = si.splrep(xaxis[-1,:],profile[-1,:])
#    x_full = np.linspace(xaxis[0,0],xaxis[0,-1],1e5)
#    spline_full = si.splev(x_full,tck_pro)
#    dlt_x = (x_full[-1]-x_full[0])/(1e5-1)
#    scale[l] = sum(spline_full*dlt_x)

#plt.plot(x_full,spline_full)
#plt.show()

#x_test = np.linspace(0,pro_len,1e4)
#y_test = si.splev(x_test,tck_pro)
#plt.plot(x_test,y_test)
#plt.show()

#xt = traces[3,ypix]
#xmin = np.floor(xt-10)
#xmax = np.floor(xt+43)
#cross = flat[ypix,xmin:xmax]
#xaxis = np.arange(0,xmax-xmin)
#spline = si.splev(xaxis,tck_pro)
#plt.plot(xaxis,cross,xaxis,spline)
#plt.show()

ti = time.clock()

#Find amplitude as a function of pixel for each trace
#ypix = np.arange(1500,1519)
#ypix = np.arange(1342,7900)#np.append(np.arange(3000,3100),np.arange(6000,6010))
ypix = np.arange(ypixel)
amps = zeros((num_traces,ypixel))
invars = zeros((num_traces,ypixel))
model = zeros((ypixel,xpixel))
for n in range(8,len(ypix)): ###MANUAL CORRECTION FOR PX SHIFT!
    pro_array = zeros((xpixel,num_traces+1))
    pro_array[:,-1] = ones((xpixel))
    pro_len = 10 + 3*11 + 10
    profile = zeros((num_traces,pro_len))
    xaxis = zeros((num_traces,pro_len))
    for j in range(num_traces):
        xt = traces[j,ypix[n]-8] ###MANUAL CORRECTION FOR PX SHIFT!
        profile[j,:] = flat[ypix[n],np.floor(xt-10):np.floor(xt+43)]
        xaxis[j,:] = np.arange(pro_len)
#            if j == num_traces/2:
#                xshift1 = (xt-10)%(np.floor(xt-10))
#                jshift = j
    for k in range(num_traces):
        tck_pro = si.splrep(xaxis[k,:],profile[k,:])
        x_full = np.linspace(xaxis[k,0],xaxis[k,-1],1e5)
        spline_full = si.splev(x_full,tck_pro)
        dlt_x = (x_full[-1]-x_full[0])/(1e5-1)
        scale = sum(spline_full*dlt_x)
        if scale == 0:
            scale = 1
        xarr = zeros((xpixel))
        xt = traces[k,ypix[n] - 8] ###MANUAL CORRECTION FOR PX SHIFT!
        xmin = np.floor(xt-10)
        xshift2 = (xt-10)%xmin
        xmax = np.floor(xt+43)
        xvals = np.arange(0,53)#-xshift1+xshift2
        spline = si.splev(xvals,tck_pro)/scale
        for m in range(xpixel):
            if m >= xmin:
                if m < xmax:
                    xarr[m] = spline[m-xmin]
        pro_array[:,k] = xarr
#        plt.plot(pro_array[:,k])
#        
#Pull in data, then convert to integrated data (match to erf)
    data = np.transpose(np.matrix(image[ypix[n]]))
    data = sparse.csc_matrix(data)
    covar = sparse.dia_matrix(np.diag(invr[ypix[n]]))
    #sub_profile1 = profile[j]
    sub_profile = sparse.csc_matrix(pro_array)
    #Set up inputs to maximum likelihood equation.
    coeff_inv = np.transpose(sub_profile)*covar*sub_profile
    coeff_inv = coeff_inv.todense()
    coeff_data = np.transpose(sub_profile)*covar*data
    #Pre-compute pseudo inverse using singular value decomposition
    #The sparse svd (sparse.linalg.svds) lost too much accuracy
    U, s, V = np.linalg.svd(coeff_inv)
#    s_inv = zeros((len(s)))
#    for l in range(len(s)):
#        if s[l] != 0:
#            s_inv[l] = 1/s[l]
#        else:
#            s_inv[l] = s[l]
    S = np.diag(1/s)
    coeff_pseudo_inv = sparse.csc_matrix(np.dot(np.transpose(V),np.dot(S,
                       np.transpose(U))))
    #Next equation is the maximum likelihood solution for the coefficient matrix
    coeff = coeff_pseudo_inv*coeff_data
    coeff = coeff.todense()
    ceoff = np.resize(np.asarray(coeff),(num_traces+1,))
    model[ypix[n],:] = np.resize(np.asarray(np.dot(pro_array, coeff)),xpixel)
    amps[:,ypix[n]] = np.resize(coeff[0:num_traces],(36))
    invars[:,ypix[n]] = np.resize(np.diag(coeff_inv),(36))

tm = time.clock()
#Merge wavlength vs. amplitude solutions
#If you can think of a simpler way please tell me...
wl_fin = wlvals[0,:]
amp_fin = amps[0,:]
inv_fin = invars[0,:]
for p in range(num_traces-1):
    wlf = wl_fin[-1]
    wl0 = wlvals[p+1,0]
    iw = 1
    while wl0 < wl_fin[-iw]:
        iw += 1
    jw = 0
    while wlf > wlvals[p+1,jw]:
        jw+=1
    for q in range(iw):
        for r in range(jw):
            if wlvals[p+1,r+1] > wl_fin[-(iw+1-q)]:
                if wlvals[p+1,r] <= wl_fin[-(iw+1-q)]:
                    slope = (amps[p+1,r+1] - amps[p+1,r])/(wlvals[p+1,r+1] -
                             wlvals[p+1,r])
                    amp_fin[-(iw+1-q)] += amps[p+1,r] + slope*(
                                          wl_fin[-(iw+1-q)]-wlvals[p+1,r])
                    slopei = (invars[p+1,r+1] - invars[p+1,r])/(
                              wlvals[p+1,r+1] - wlvals[p+1,r])
                    inv_fin[-(iw+1-q)] += invars[p+1,r] + slopei*(
                                          wl_fin[-(iw+1-q)]-wlvals[p+1,r])
    wl_fin = np.append(wl_fin,wlvals[p+1,jw-1:])
    amp_fin = np.append(amp_fin,amps[p+1,jw-1:])
    inv_fin = np.append(inv_fin,invars[p+1,jw-1:])

    
#guess = (3.2,1,14500,5.5,0.95)
#args = (data, invvars, axis)
#answer = opt.minimize(full_profile,guess,args)
#vals, errs = opt.curve_fit(guess_to_gauss,axis,data,p0=guess)
     
#Save File
hdu1 = pyfits.PrimaryHDU(wl_fin)
hdu2 = pyfits.PrimaryHDU(amp_fin)
hdu3 = pyfits.PrimaryHDU(inv_fin)
hd1 = pyfits.PrimaryHDU(wlvals)
hd2 = pyfits.PrimaryHDU(amps)
hd3 = pyfits.PrimaryHDU(invars)
hdulist = pyfits.HDUList([hdu1])
hdulist.append(hdu2)
hdulist.append(hdu3)
hdlist = pyfits.HDUList([hd1])
hdlist.append(hd2)
hdlist.append(hd3)
if mode == 'b':
    hdulist.writeto(path + 'solar_1D_b0_0.fits')
    hdlist.writeto(path + 'solar_tr_b0_0.fits')
elif mode == 'sb':
#    hdulist.writeto(path + 'solar_sp_b0_0.fits')
    hdlist.writeto(path + 'solar_sp_b0_0.fits')
elif mode == 'r':
    hdulist.writeto(path + 'solar_1D_r0_0.fits') 
    hdlist.writeto(path + 'solar_tr_r0_0.fits')
elif mode == 'sr':
#    hdulist.writeto(path + 'solar_sp_b0_0.fits')
    hdlist.writeto(path + 'solar_sp_r0_0.fits')
elif mode == 'ab':
    hdulist.writeto(path + 'solar_1D_arc_b.fits')
    hdlist.writeto(path + 'solar_tr_arc_b.fits')
elif mode == 'ar':
    hdulist.writeto(path + 'solar_1D_arc_r.fits')   
    hdlist.writeto(path + 'solar_tr_arc_r.fits')

tf = time.clock()
#plt.plot(model)
#plt.plot(flat[ypix])
#plt.plot(flat[ypix]-model)
#plt.show()

def spec_env(wl, mag=7.5, minutes=15, tele_dia=4.3):
    """Finds the "envelope" for a stellar spectrum (perfect blackbody).
    Assumes no distortion in the star and no absorption lines.
    This is converted into the function output as photon counts vs. wavelength.
    All units are in SI (MKS).
    
    Inputs:
       wl -> array of wavelengths at which to find photon counts.
       mag -> apparent magnitude of the star (default is 7.5)
       minutes -> exposure time in MINUTES (default is 15)
       tele_dia -> telescope diameter in meters (default is 4.3 for DCT)
       Files:
           sun_reference_stis_002.fits - Calspec solar spectrum
           alpha_lyr_stis_008.fits - Calspec Vega spectrum
       
    Output:
       array (of length = len(wl)) of photon counts
    """
    #Open the solar and Vega fits files
    sun = pyfits.open(path + 'sun_reference_stis_002.fits')
    vega = pyfits.open(path + 'alpha_lyr_stis_008.fits')
    #Unpack the solar data
    sun_data = sun[1].data
    num_sun = len(sun_data)
    sun_wl = zeros((num_sun))
    sun_flux = zeros((num_sun))
    sun_err = zeros((num_sun))
    for i in range(num_sun):
        row = sun_data[i]
        sun_wl[i] = row[0]
        sun_flux[i] = row[1]
        sun_err[i] = row[2]
    #Unpack the Vega data
    vega_data = vega[1].data
    num_vega = len(vega_data)
    vega_wl = zeros((num_vega))
    vega_flux = zeros((num_vega))
    vega_err = zeros((num_vega))
    for i in range(num_vega):
        row = vega_data[i]
        vega_wl[i] = row[0]
        vega_flux[i] = row[1]
        vega_err[i] = row[2] + row[3]
    #Import V-band profile approximation
    v_filt = open(path + 'v_band.txt')
    line_count = sum(1 for line in v_filt)
    v_filt = open(path + 'v_band.txt')
    v_wl = zeros((line_count))
    v_amp = zeros((line_count))
    for i in range(line_count):
        tmp = v_filt.readline()
        v_wl[i] = tmp[0:4]
        v_amp[i] = tmp[-6:-1] 
    #Functional estimate of V-band profile (at each wavelength - very rough)
    def vband_amp(wl, v_wl=v_wl, v_amp=v_amp):
        for i in range(1,len(v_wl)):
            if wl < v_wl[0]:
                return 0
                break
            elif wl < v_wl[i]:
                slope = (v_amp[i]-v_amp[i-1])/(v_wl[i]-v_wl[i-1])
                amp = v_amp[i-1] + (wl-v_wl[i-1])*slope
                return amp
                break
            elif wl >= v_wl[len(v_wl)-1]:
                return 0
                break
    #Apply V-band filter to the solar and Vega spectra
    filt1 = zeros((len(sun_wl)))
    for k in range(len(filt1)):
        filt1[k] = vband_amp(sun_wl[k])
    filt2 = zeros((len(vega_wl)))
    for k in range(len(filt2)):
        filt2[k] = vband_amp(vega_wl[k])
    sun_filt = sun_flux*filt1
    vega_filt = vega_flux*filt2
    #Estimate the total number flux of the sun and Vega in the V-band
    Fs = sum(sun_filt * sun_wl)
    Fv = sum(vega_filt * vega_wl)
    #Calculate the scale factor to apply to the solar spectrum
    sun_scale = (10**(-mag/2.5))*Fv/Fs
    sun_spec = sun_scale*sun_flux
#    plt.plot(sun_wl,sun_spec,vega_wl,vega_flux)
#    plt.show()
    #Apply telescope area and exposure time to convert to energy/Angstrom
    obs = 0.12 #percent central obscuration of DCT
    Area = pi*(((1-obs)*tele_dia*100/2)**2) #Using DCT, 4.3m telescope
    time = 60*minutes #60 seconds/minute * 15 minute exposures 
    sun_ergs = sun_spec*Area*time
    sun_erg_err = sun_err*sun_scale*Area*time
    #Smooth the continuum spectrum
    tck_tuple = si.splrep(sun_wl,sun_ergs,w=1/sun_erg_err, s=2*len(sun_wl))
    sun_smooth = si.splev(wl,tck_tuple)
    return sun_smooth

d_wl = 0.01 #delta wavelength (Angstroms)
wl_all = np.arange(3750,6910.01,d_wl) #wavelength range (Angstroms)
#Add Doppler Shift:
wl0 = wl_all
vc = 2.99792458*10**10 #speed of light (cm/s)
hp = 6.62606957*10**(-27) #Planck's constant (erg*s)
#wl_all = wl_all*(1-vl/vc)
A_all = zeros((len(wl_all)))
#Solar Spectrum
amplitude = open(path + 'amplitude.txt')
for s in range(len(wl_all)):
    A_all[s] = int(amplitude.readline()[:-2])
#Scale up to reasonable level
A_scale = spec_env(wl0,minutes=15)
A_all = A_all/10000*A_scale
#    plt.plot(wl_all,A_all)
#    plt.plot(wl_all,A_scale)
#    plt.show()
#Convert erg/Angstrom to counts/px
sampling = 3.5
resolution = 230000    
SR = sampling*resolution
PF = wl_all/SR
A_all = 0.1*(A_all * (wl_all*10**(-8)) * PF) / (hp*vc)

aup = 1#40902/37964.9
shft = -0.04
#plt.plot(wlvals[0],amps[0],wl_all[5000:5500],A_all[5000:5500])
#plt.show()

#plt.plot(model[3000])
#plt.plot(image[3000])
#plt.plot(image[3000]-model[3000])
#plt.show()
    
total_time = tf - t0
iter_time = tm - ti
print ("Total time = %0.06fs" % total_time)
print ("Iterable time = %0.06fs" % iter_time)