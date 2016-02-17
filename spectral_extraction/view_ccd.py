#!/usr/bin/env python 2.7

#this code is written to visually inspect trace simulations and view
#differences between two different models or models/"data"

#Import all of the necessary packages (and maybe some unnecessary ones)
from __future__ import division
import pyfits
import os
import sys
import math
import time
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from matplotlib import cm
#import scipy
#import scipy.stats as stats
import scipy.special as sp
#import scipy.optimize as opt
#import scipy.sparse as sparse
#import scipy.signal as sig
#import scipy.linalg as linalg
import special as sf
import modify_fits as mf #For MINERVA DATA ONLY

#def chi_fit(d,P,N):
#    """Routine to find parameters c for a model using chi-squared minimization.
#       Note, all calculations are done using numpy matrix notation.
#       Inverse is calculated using the matrix SVD.
#       Inputs:
#           d = data (n-dim array)
#           P = Profile (n x c dim array) (c = number of parameters)
#           N = Noise (n x n array) (reported as inverse variance)
#           
#       Outputs:
#           c = coefficients (parameters)
#           chi_min = value of chi-squared for coefficients
#    """
#    Pm = np.matrix(P)
#    Pmt = np.transpose(P)
#    dm = np.transpose(np.matrix(d))
#    Nm = np.matrix(N)
#    PtNP = Pmt*Nm*Pm
#    U, s, V = np.linalg.svd(PtNP)
#    S = np.matrix(np.diag(1/s))
#    PtNPinv = np.transpose(V)*S*np.transpose(U)
#    PtNd = Pmt*Nm*dm
#    c = PtNPinv*PtNd
#    chi_min = np.transpose(dm - Pm*c)*Nm*(dm - Pm*c)
#    c = np.asarray(c)
#    #chi_min2 = np.transpose(dm)*Nm*dm - np.transpose(c)*(np.transpose(Pm)*Nm*Pm)*c
#    return c, chi_min

path = os.environ['EXPRES_OUT_DIR']

filename = sys.argv[1]
#filename2 = sys.argv[2]
hdulist = pyfits.open(filename,ignore_missing_end=True,uint16=True,do_not_scale_image_data=True,mode='update')
#hdulist2 = pyfits.open(filename2,ignore_missing_end=True,uint16=True,do_not_scale_image_data=True,mode='update')
hdr = hdulist[0].header
#hdr2 = hdulist2[0].header
#hdr['NAXIS'] = 3
#hdr['BITPIX'] = 16
#print repr(hdr)
image = hdulist[0].data
#image2 = hdulist2[0].data
#image = mf.unpack_minerva(filename)
#image2 = mf.unpack_minerva(filename2)
#im1_bg = (image-np.mean(image[image<2*np.mean(image)]))
#image_new = np.copy(image)
#image_new[image2>2000] = image[image2>2000]/image2[image2>2000]*np.median(image2)
#im1 = im1_bg*(im1_bg>0)
#im2_bg = (image2-np.mean(image2[image2<0.5*np.mean(image2)]))
#im2 = im2_bg*(im2_bg>0)
#image_mod = im1/im2
#image_flip = image[::-1,:]
fig, ax = plt.subplots()
ax.pcolorfast(image)
plt.ion()
plt.show()

#plt.figure()
#plt.plot(image[420:470,1000])
#
##Cladding test
#axis = np.arange(-25,25,0.01)
#wid = 10
#hw = 120
#bg = 520
#wing = sf.cladding(axis,wid,height=hw)
#sig = 1
#h1 = 2100
#cent = sf.gaussian(axis,sig,height=h1)
#tot = cent+wing+bg
#plt.plot(axis+25-1.5,tot,'k')

#User Select:
data_basis = 'b0_0_1'

bases = ['b-20_0', 'b4_0', 'b20_0']
#model1_basis = 'b-1_0'
#model2_basis = 'b0_0'
#scales = zeros((len(bases)))
#hdulist2 = pyfits.open(path+'solar_v_'+model2_basis+'.fits')
#model2_image = hdulist2[0].data
#model2_small = model2_image[2500:4000,7340:7360]
#for i in range(len(bases)):
#    hdulist1 = pyfits.open(path+'solar_v_'+bases[i]+'.fits')
#    model1_image = hdulist1[0].data
#    model1_small = model1_image[2500:4000,7340:7360]
#    model_diff = model1_image - model2_image
#    scales[i] = np.max(model_diff)/np.max(model2_image)

#Import Data
hdulist1 = pyfits.open(path+'data_v_'+data_basis+'.fits')
data_image = hdulist1[0].data
invar = hdulist1[1].data
data_small = data_image[2500:4000,7340:7360]
#Import Model 1
#hdulist2 = pyfits.open(path+'solar_v_'+bases[1]+'.fits')
#model1_image = hdulist2[0].data
#model1_small = model1_image[2500:4000,7340:7360]
##Import Model 2
#hdulist3 = pyfits.open(path+'solar_v_'+bases[0]+'.fits')
#model2_image = hdulist3[0].data
#model2_small = model2_image[2500:4000,7340:7360]
#
##Poster Fig.0 and Fig.1 - CCD image
#fig,ax0 = plt.subplots()
#dplt0 = ax0.pcolorfast(data_image,cmap=cm.hot)
#ax0.set_title("Simulated CCD Data")
#ax0.set_xlabel("Pixel")
##ax1.set_xticks([4600,4630,4660])
#ax0.set_ylabel("Pixel")
#dcbar0 = fig.colorbar(dplt0)
##ax1.text(0,0,'Counts')
#dcbar0.set_label('Counts',rotation=90,labelpad=10)
##plt.show()
##Fig1
#fig,ax1 = plt.subplots()
#xpos = np.arange(4580,4660)
#ypos = np.arange(4350,4700)
#dplt = ax1.pcolorfast(xpos,ypos,data_image[4350:4700,4580:4660],cmap=cm.hot)
#ax1.set_title("Simulated CCD Data")
#ax1.set_xlabel("Pixel")
##ax1.set_xticks([4600,4630,4660])
#ax1.set_ylabel("Pixel")
#dcbar = fig.colorbar(dplt)
##ax1.text(0,0,'Counts')
#dcbar.set_label('Counts',rotation=90,labelpad=10)
##plt.show()
#
##Poster Fig.2a - CCD vs models residuals
##fig,ax2 = plt.subplots()
##xpos = np.arange(4580,4660)
##ypos = np.arange(4350,4700)
##a = sum(data_image*model1_image*invar)/(sum((model1_image**2)*invar))
##res = data_image - a*model1_image
##m1plt = ax2.pcolorfast(xpos,ypos,res[4350:4700,4580:4660],cmap=cm.hot)
##ax2.set_title("Residuals with RV=0 cm/s model")
##ax2.set_xlabel("Pixel")
###ax1.set_xticks([4600,4630,4660])
##ax2.set_ylabel("Pixel")
##m1cbar = fig.colorbar(m1plt)
###ax1.text(0,0,'Counts')
##m1cbar.set_label('Counts',rotation=90,labelpad=10)
##plt.show()
#
##Poster Fig.2b - noise estimate scaled residuals
#fig,ax3 = plt.subplots()
#xpos = np.arange(4580,4660)
#ypos = np.arange(4350,4700)
#b = sum(data_image*model1_image*invar)/(sum((model2_image**2)*invar))
#res = data_image - b*model2_image
#res_norm = res*np.sqrt(invar)
#m2plt = ax3.pcolorfast(xpos,ypos,res_norm[4350:4700,4580:4660],cmap=cm.hot)
#ax3.set_title("Noise-Estimate Normalized Residuals")
#ax3.set_xlabel("Pixel")
##ax1.set_xticks([4600,4630,4660])
#ax3.set_ylabel("Pixel")
#m2cbar = fig.colorbar(m2plt)
##ax1.text(0,0,'Counts')
#m2cbar.set_label('Counts',rotation=90,labelpad=10)
#plt.show()
#
##Poster Fig.3 - Chi-squared plot
#hduchi = pyfits.open(path+'chi_comp_res1.fits')
#v = hduchi[0].data
#chi_dlt = hduchi[1].data
#coeffs = hduchi[2].data
#profile = np.transpose(array((v**2,v,ones(len(v)))))
#weights = np.diag(ones((len(v))))
#data = chi_dlt
#coeffs, junk = chi_fit(data,profile,weights)
#aq = coeffs[0]
#bq = coeffs[1]
#cq = coeffs[2]
#sigma = 1
#vel_1p = (-bq + np.sqrt(4*aq*(sigma**2)))/(2*aq)
#vel_1m = (-bq - np.sqrt(4*aq*(sigma**2)))/(2*aq)
#xvals = np.linspace(v[0],v[-1],100)
#yvals = aq*xvals**2 + bq*xvals + cq
#yoff = np.min(chi_dlt) - np.min(yvals)
#plt.plot(xvals,yvals + yoff,'g')
#plt.plot(v,chi_dlt + yoff,'bx')
#plt.xlabel('Radial Velocity (cm/s)')
#plt.ylabel('$\chi^2 - \chi^2_{min}$')
#plt.title('$\chi^2$ vs. Radial Velocity Model')
#yvert = [0,6]
#yhor = [1,1]
#xlow = vel_1m*ones((2))
#xhigh = vel_1p*ones((2))
#xhor = [v[0],v[-1]]
#plt.plot(xhor,yhor,'k')
#plt.plot(xlow,yvert,'k--')
#plt.plot(xhigh,yvert,'k--')
#plt.xlim(-15,15)
#plt.ylim(0,4)
#vlow = np.round(vel_1m[0],2)
#vhigh = np.round(vel_1p[0],2)
#plt.text(vel_1m[0]+0.5, 2, str(vlow)+'cm/s')
#plt.text(vel_1p[0]-3.75, 2, str(vhigh)+'cm/s')
#plt.savefig(path+"chi_plot.png", bbox_inches='tight')
#plt.savefig(path+"chi_plot.pdf", bbox_inches='tight')
#plt.show()


#ax1 = plt.subplot(3,1,1)
#ax1.pcolorfast(data_image,cmap=cm.gray)
#plt.title("Model 1 Comparison")
#ax2 = plt.subplot(3,1,2)
#ax2.pcolorfast(model1_image,cmap=cm.gray)
#ax3 = plt.subplot(3,1,3)
#ax3.pcolorfast(data_image-model1_image,cmap=cm.gray)
##ax1.colorbar()
#plt.show()
#
#ax1 = plt.subplot(3,1,1)
#ax1.pcolorfast(data_image,cmap=cm.gray)
#plt.title("Model 2 Comparison")
#ax2 = plt.subplot(3,1,2)
#ax2.pcolorfast(model2_image,cmap=cm.gray)
#ax3 = plt.subplot(3,1,3)
#ax3.pcolorfast(data_image-model2_image,cmap=cm.gray)
##ax1.colorbar()
#plt.show()

#ax1 = plt.subplot(2,1,1)
#ax1.pcolorfast(data_image-model1_image,cmap=cm.gray)
#plt.title("Residuals: Correct and -5cm/s")
#ax2 = plt.subplot(2,1,2)
#ax2.pcolorfast(data_image-model2_image,cmap=cm.gray)
#plt.show()
#
#plt.plot(scales)
##plt.plot(scales[::-1])
#plt.show()

#print "Residual Scale:", a
#model_diff = model1_small - model2_small
#a = np.max(model_diff)/np.max(model2_small)
#ax1 = plt.subplot(1,1,1)
#ax1.pcolorfast(np.hstack((model1_small,model2_small,(1/a)*model_diff))
#               ,cmap=cm.gray)
#plt.title("Residuals: Correct and -5cm/s")
#plt.show()

#xpix = 7350
#line1 = model1_image[:,xpix]
#line2 = model2_image[:,xpix]
#axis = np.arange(len(line1))
#plt.plot(axis,line1,axis,line2,axis,line1-line2)
#plt.show()
