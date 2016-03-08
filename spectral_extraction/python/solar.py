#!/usr/bin/env python 2.7

#Adapted from trace_model_b - calculates solar spectrum
#Right now hardcoded for v=0 (no Doppler shift) and EXPRES characteristics
#TODO: build another function to take various inputs for more modular use

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
import scipy.special as sp
import scipy.interpolate as si

t0 = time.time()

def spec_env(wl, mag=7.5, minutes=15, tele_dia=4.3, path=os.environ[
             'EXPRES_OUT_DIR']):
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
       array (of length = len(wl)) of photon energy/wl_bin
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
    #plt.plot(sun_wl,sun_spec,vega_wl,vega_flux)
    #plt.show()
    #Apply telescope area and exposure time to convert to energy/Angstrom
    obs = 0.12 #percent central obscuration of DCT
    Area = pi*(((1-obs)*tele_dia*100/2)**2) #Using DCT, 4.3m telescope
    time = 60*minutes #60 seconds/minute * 15 minute exposures 
    sun_ergs = sun_spec*Area*time
    sun_erg_err = sun_err*sun_scale*Area*time
    #Smooth the continuum spectrum
    tck_tuple = si.splrep(sun_wl,sun_ergs,w=1/sun_erg_err, s=2*len(sun_wl))
    sun_smooth = si.splev(wl,tck_tuple)
    #Re-normalize (conservation of energy)
#    total_energy=sum(sun_ergs[255:541]*ediff1d(sun_wl)[255:541])
#    print "Total Energy = ", total_energy, "; len = ", len(sun_ergs)
#    d_wlsm = np.mean(ediff1d(wl))
#    smoothed_energy=sum(sun_smooth[1:]*d_wlsm)
#    print "Smoothed Energy = ", smoothed_energy, "; len = ",len(sun_smooth)
    #Scale for dynamic wavelength range    
    #wlscale = (wl[-1]-wl[0])/(6910-3750) #Hardcoded for Expres baseline
#    print("WL scale = {:f}".format(wlscale))
    #sun_smooth = sun_smooth*(total_energy/smoothed_energy)*wlscale
#    print "Normalized Energy = ", sum(sun_smooth)
    return sun_smooth

def amp_vs_pos(ypos,y_st,l_st,a,b,samp,res,A,wl,tp):
    """Takes a given y coordinate and returns the amplitude (assuming the
       provided A vs. wavelength function).
       Inputs:
       ypos = the position(s) to be evaluated
       y_st = the starting y coordinate of the given trace
       l_st = the wavelength corresponding to y_st
       a = the first polynomial coefficient of the given trace (a*y**2)
       b = the second polynomial coefficient of the given trace (b*y)
       samp = sampling
       res = resolution
       A = Amplitude (counts) as a function of wavelength (array)
       wl = wavelength coordinates for A (array)
       tp = throughput as a function of wavelength (array)
       """
    pix_st = 1/(4*a)*(sqrt(4*a**2*y_st**2 + 4*a*b*y_st + b**2 + 1)*(2*a*y_st + 
             b) + np.arcsin(2*a*y_st + b))
    pix_pos = 1/(4*a)*(sqrt(4*a**2*ypos**2 + 4*a*b*ypos + b**2 + 1)*(2*a*ypos + 
             b) + np.arcsin(2*a*ypos + b))
    pix_pos_rel = pix_pos - pix_st
    SR = samp*res
    wave_pos = l_st*np.exp(pix_pos_rel/SR)
    amp = 0
    for i in range(1,len(A)):
        if wl[i] >= wave_pos:
            if wl[i-1] <= wave_pos:
                Aslope = (A[i] - A[i-1]) / (wl[i] - wl[i-1])
                tpslope = (tp[i] - tp[i-1]) / (wl[i] - wl[i-1])
                amp = (A[i-1] + Aslope*(wave_pos-wl[i-1]))*(tp[i-1] + 
                       tpslope*(wave_pos-wl[i-1]))
    return amp, wave_pos


def sol_spec(wl_file,A_file,vl=0,sampling=3.5,resolution=230000,d_wl=0.01,
             wl_1=3750,wl_2=6910):
    """Function to return the solar spectrum used for EXPRES simulation
       Will add a Doppler shift based on input vl
       INPUTS:
           wl_file and A_file (returned from sf.spec_amp)
           All are set to expres defaults for now
       
       OUTPUTS:
           wl_all = wavelengths range of spectrum
           A_all = amplitude of spectrum at above wavelengths
    """ 
    wl_all = np.arange(wl_1,wl_2+0.1*d_wl,d_wl) #wavelength range (Angstroms)
    #Add Doppler Shift:
    wl0 = copy(wl_all)
    vc = 2.99792458*10**10 #speed of light (cm/s)
    hp = 6.62606957*10**(-27) #Planck's constant (erg*s)
    wl_all = wl_all*(1-vl/vc)
    A_all = zeros((len(wl_all)))
    #Input Amplitude (counts) as a function of wavelength
    if wl_1 < 3750:
        print("Lower wavelength limit is 3750 Angstroms.")
        exit(0)
    if wl_2 > 6910:
        print("Upper wavelength limit is 6910 Angstroms.")
        exit(0)
    #Shorten wl_file and A_file to relevant indices
    ind_mask = (wl_file>=wl_1)*(wl_file<=wl_2)
    wl_file = wl_file[ind_mask]
    A_file = A_file[ind_mask]
    #Discrete wavelengths (eg. Laser Frequency Comb)
#    elif spectrum == 'arc':
#        for s in range(len(wl_all)):
#            if wl_all[s] % 1 < 0.0001:
#                #Pick based on calibration characteristics
#                A_all[s] = (500**2)*10000
    #Now adjust wl_all, A_all based on wl_file, A_file
    #If desired d_wl is at points already in file, just grab those
    wl_tail, junk = math.modf(wl_1) #Pull out decimal portion only
    if d_wl%0.002 == 0 and wl_tail%0.002 == 0:#BASS2000 spectrum is at 0.002A intervals
        step=int(d_wl/0.002)
        A_all = A_file[0:len(A_file):step]
    #Otherwise, linearly extrapolate between points
    else:
        bb=1
        for aa in range(len(wl0)-1):
            #First while loop may be unnecessary
            while wl0[aa]<wl_file[bb-1] and bb!=1:
                bb-=1
            while wl0[aa]>wl_file[bb]:
                bb+=1
            if bb>=len(wl_file):
                break
            slope = (A_file[bb]-A_file[bb-1])/(wl_file[bb]-wl_file[bb-1])
            A_all[aa] = A_file[bb-1]+slope*(wl0[aa]-wl_file[bb-1])
        #Special treatment for last case
        A_all[aa+1] = A_file[bb]+slope*(wl0[aa+1]-wl_file[bb])
    #Check to see if this looks right
#    print("{:d} {:d} {:d} {:d}".format(len(wl_all),len(A_all),len(wl_file),len(A_file)))
#    plt.plot(wl_all,A_all,wl_file,A_file)
#    plt.show()
    #Scale up to reasonable level
    A_scale = spec_env(wl0,minutes=15)
    A_all = A_all/10000*A_scale
    #Convert erg/Angstrom to counts/Angstrom  
#    SR = sampling*resolution
#    PF = wl_all/SR #"Plate Factor"
    A_all = (A_all * (wl_all*10**(-8))) / (hp*vc) #*PF for counts/px
    #Estimate amplitudes at the shifted points
    dlt_wl = wl0 - wl_all
    A_prime = zeros((len(A_all)))
    for t in range(len(wl_all)):
        if t == len(wl_all)-1:
            slope = A_all[-1] - A_all[-2]
            A_prime[t] = A_all[-1] - dlt_wl[-1]*slope
        else:
            slope = A_all[t+1] - A_all[t]
            A_prime[t] = A_all[t] - dlt_wl[t]*slope
    A_all = A_prime
#    print(sum(A_all*d_wl))
#    plt.plot(wl_all,A_all,wl_file,A_file)
#    plt.show()
    return wl_all, A_all
    