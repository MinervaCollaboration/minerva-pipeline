#!/usr/bin/env python 2.7

#This code is for the MINERVA project and builds a model ccd science exposure
#for a range of radial velocity shifts.

#INPUT: Files relevant to the spectrograph and calibration data:
# sun_reference_stis_002.fits
#       low-res solar profile for flux scaling
# alpha_lyr_stis_008.fits
#       low-res vega profile for flux scaling
# v_band.txt
#       low-res V-band filter response curve
# lam.txt
#       wavelength along each echelle order
# amplitude.txt
#       high-resolution solar contiuum normalized intensity spectrum
# xpos.txt
#       physical ccd x coordinates(mm) for each echelle order
# ypos.txt
#       physical ccd y coordinates(mm) for each echelle order
#Also user should select desired velocity range (in-code right now)

#OUTPUT: .fits file(s) with the model ccd image (no errors) for each
#        specified velocity input.
#        Naming convention: solar_v_b[velocity].fits
#        text file - fits_files_b.txt - for chi_compare

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
#import scipy.signal as sig
#import scipy.linalg as linalg
import solar
import special as sf
import sys

#def sim(d_wls,resolution,sampling,wl_file,A_file):
"""Right now makes a science simulation around wl 4276A 200px long
   and 50px wide.
   
   INPUTS:
       d_wls - array of discrete wavelength steps to use (in A)
       resolution - instrument resolution
       sampling - intstrument sampling
       wl_file, A_file - full solar spectrum from BASS2000 file
"""
t0 = time.clock()


#Set paths
paths = os.environ['MINERVA_SIM_DIR']
pathm = os.environ['MINERVA_DIR']
svnm = sys.argv[1]
spectrum = sys.argv[2]
#svnm = 'minerva_ccd.fits'
#svnm = 'minerva_arc.fits'
#svnm = 'minerva_flat_1.fits'
#svnm = 'minerva_ccd_7.fits'

#Import velocity file (start/stop/number of steps)
#vparts = open(pathm+'velocities.txt')
#for aa in range(6): #Six lines, data on every other line
#    line = vparts.readline()[:-1]
#    if aa==1:
#        vmin = float(line)
#    elif aa==3:
#        vmax = float(line)
#    elif aa==5:
#        vsteps = int(line)
#    
#velocities = np.arange(vmin,vmax,vsteps)
#velocities = [0,]
vl = 0
#d_wls = np.arange(0.001,0.05,0.001)
#d_wls = [0.051,0.052]

#MINERVA/KiwiSpec Characteristics (may eventually load from outside file)
resolution = 8e4
sampling = 3 #Change this later to use dispersion
resst = str(resolution)
sampst = str(sampling)
SR = sampling*resolution
#Set Spectral Type
#spectrum='arc'
#spectrum='solar'
#Background and Gain (Estimates now, need CCD info)
bg = 4
gain = 2
#Pixel configuration (load from outside file)
xpixel = 2048
ypixel = 2048
cntrx = xpixel/2
cntry = ypixel/2
um_per_px = 15 #Double check microns per pixel conversion factor
#fiber_sep = 220/um_per_px #220 micron fiber separation - converted to px units
fiber_sep = 110/um_per_px #half of above for the seven fiber configuration


#background = ones((ypixel,xpixel))*gain*bg #background matrix to add in at end

#Relative intensity in each of the four fibers
#sp_tr = [1,];
num_same_fibers = 1 #Number of "identical" fibers used
sp_tr = ones((num_same_fibers)); #4 equal intensity fibers default
#sp_tr = sp_tr/sum(sp_tr) #Normalize
#Size for psf model (one side of a square)
N = 20

#Make a fresh text file with the output names (for chi_compare)
#outfile = 'sim_fits/fits_'+resst+'_'+sampst+'.txt'
#if os.path.isfile(outfile):
#    os.remove(outfile)
#output = open(outfile,'w')    

d_wl = 0.05 #Computed from resolution
#for f in range(len(d_wls)):
#d_wl = d_wls[f]
dwlst = str(d_wl)
#See if an output has already been created for this velocity
#    vl = velocities[f]
#    vel = str(vl)
#    vel = vel.replace('.','_') #no extra decimals in file name
#Before running loop, see if file already exists
if os.path.isfile(paths+svnm):
    print ("File '{}' already exists!".format(svnm))
    #output.write(paths+'minerva_ccd.fits\n')
#Find Doppler shifted wavelength and amplitude of solar spectrum
if spectrum=='solar':
    wl_file, A_file = sf.spec_amps()
    wl_all, A_all = solar.sol_spec(wl_file,A_file,vl=vl,sampling=sampling,
                    resolution=resolution,d_wl=d_wl,wl_1=4850,wl_2=6500)
elif spectrum=='arc':
    wl_all, A_all = sf.thar_lamp()
    amp_scale = 100
    A_all*=amp_scale
elif spectrum=='flat':
    wl_all = np.arange(4874,6468,d_wl)
    A_all = 2**15*np.ones((len(wl_all)))
else:
    print("ERROR: Invalid spectral type")
#And throughput as a function of wavelength
#TODO - Update with formula from MINERVA paper
tp_0 = 0.10*ones((len(wl_all))) #90% light reduction estimate
#Open wavelength range - specific to MINERVA
lambdas = np.loadtxt(paths + 'lam.txt',delimiter="\t")
l_st_max = lambdas[-1,0]*10000
#Import x, y coordinates from Zeemax simulation 
xorders = np.loadtxt(paths + 'xpos.txt',delimiter="\t")
yorders = np.loadtxt(paths + 'ypos.txt',delimiter="\t")
#Build blank ccd model - use lil_matrix to modify cheaply
ccd = sparse.lil_matrix((ypixel,xpixel))
#    psf = build_psf(0,0,N,sampling)
#Loop to build each order
for p in range(31): #31 from SB simulation
    #Find wavelength range data
    lam_line = lambdas[p,:] #um unit
    l_st = lam_line[0]*10000 #Angstroms
    l_end = lam_line[-1]*10000
#        l_dlta = float(lambda_deltas.readline()[:-2])
#        l_end = l_st + l_dlta
    #ypx0 = 0 #set arbitrarily
    #l_end = l_st*exp((ypixel-ypx0)/SR) #Start at ypx0, cover whole px range
    wl_wid = 1 #pad wl range to ensure no unnaturally sharp cutoff
    #Take relevant portion of solar spectrum
    #Maybe a little confusing, but finds first and last wl indices
    idx1 = nonzero(floor(wl_all-(l_st-wl_wid))>=0)[0][0]
    idx2 = nonzero(floor(wl_all-(l_end+wl_wid))>=0)[0][0]
    wl = wl_all[idx1:idx2]
    A = A_all[idx1:idx2]
    tp = tp_0[idx1:idx2]
    #Find trace center line from Zeemax simulation
    x = xorders[p]*1000/um_per_px #convert mm to px units
    y = yorders[p]*1000/um_per_px #convert mm to px units
    #Fit for coefficients
    a, b, c = np.polyfit(x, y, 2)
    env = ones((len(wl)))
    #print(len(wl))
    PF = wl/SR
    for k in range(len(wl)):
        #Find x and y pixel center and decimal offset at each wavelength
        #Split into decimal and integer portions
        xcd, xci = math.modf(x[0] + SR*np.log(wl[k]/l_st)+cntrx)
        xc = xci+xcd-cntrx
        ycd, yci = math.modf(a*xc**2 + b*xc + c + cntrx)
        #yci = yci-y[-1]-cntry #Shift to start near zero
        #xci = xci-x[-1]-cntrx+10
        #print("yc={:f} and xc={:f}".format(yci+ycd,xci+xcd))
        #Start and end (in and out) indices for frame y coords
        #Build_psf at decimal center
        psf = sf.build_psf(xcd,ycd,N,sampling,offset=[0,0])
        for q in range(len(sp_tr)):
            #Add spacing of 220um at max, less on low lambda end
            #TODO - get real values
            yctmp = yci + fiber_sep*q#*l_st/l_st_max
            yin = int(yctmp-N/2)
            #yin = int(yci-N/2)
            yout = int(yin + N + 1)
            xin = int(xci-N/2)
            xout = int(xin + N + 1)
            #Trim edges if needed
            psf_trim = np.copy(psf)
            if yin < 0:
                psf_trim = psf_trim[-yin:,:]
                yin = 0
            if xin < 0:
                psf_trim = psf_trim[:,-xin:]
                xin = 0
            if yout > ypixel:
                psf_trim = psf_trim[:(-yout+ypixel),:]
                yout = (ypixel)
            if xout > (xpixel):
                psf_trim = psf_trim[:,:(-xout+xpixel)]
                xout = (xpixel)
            scale = gain*(env[k]*A[k]*d_wl*tp[k])#*PF[k])#*sp_tr[q]
            ccd[yin:yout,xin:xout] += psf_trim*scale
        
ccd = ccd.todense()
ccd = ccd[::-1,:]
#    ccd += background
#Save file
hdu1 = pyfits.PrimaryHDU(ccd)
hdulist = pyfits.HDUList([hdu1])  
prihdr = hdulist[0].header
prihdr['DLAMBDA'] = d_wl
prihdr['RES'] = resolution
prihdr['SAMP'] = sampling
hdulist.writeto(paths+svnm)
#output.write(savepath+dwlst+'.fits\n')
#    print ("Finished with velocity %scm/s!" % vel)
print ("Finished with delta wavelength {:f}cm/s!".format(d_wl))

#output.close()
tf = time.clock()
total_time = tf-t0
print ("Total time = {:0.6f}s".format(total_time))

#Plot image    
ax = plt.subplot(1,1,1)
ax.pcolorfast(ccd,cmap=cm.gray)
plt.show()

