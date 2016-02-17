#!/usr/bin/env python 2.7

#This code holds special functions that I've built that I've had a recurring
#need to use.

#Function List:
# gaussian - builds 1-D gaussian curve
# chi_fit - finds coefficients for linear chi-squared equation d = P*c + N

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
import scipy.optimize as opt
#import scipy.sparse as sparse
#import scipy.signal as sig
#import scipy.linalg as linalg

def gaussian(axis, sigma, center=0, height=1,bg_mean=0,bg_slope=0):
    """Returns gaussian output values for a given input array, sigma, center,
       and height.  Center defaults to zero and height to 1."""
    #Height scale factor*(1/(sqrt(2*pi*sigma**2)))
    gaussian = height*exp(-(axis-center)**2/(
               2*sigma**2))+bg_mean+bg_slope*(axis-center)
    return gaussian
    
def cladding(axis,width,center=0,height=1):
    """Function ONLY for rough use on preliminary MINERVA data.  This is to
       give a VERY rough estimate for the cladding profile.
    """
    zl = axis[axis<=center]
    zr = axis[axis>center]
    left = height*0.5*(1+sp.erf((zl-center+width/2)/width))
    right = height*0.5*(1+sp.erf(-(zr-center-width/2)/width))
    return np.append(left,right)

def chi_fit(d,P,N):
    """Routine to find parameters c for a model using chi-squared minimization.
       Note, all calculations are done using numpy matrix notation.
       Inverse is calculated using the matrix SVD.
       Inputs:
           d = data (n-dim array)
           P = Profile (n x c dim array) (c = number of parameters)
           N = Noise (n x n array) (reported as inverse variance)
           
       Outputs:
           c = coefficients (parameters)
           chi_min = value of chi-squared for coefficients
    """
    Pm = np.matrix(P)
    Pmt = np.transpose(P)
    dm = np.transpose(np.matrix(d))
    Nm = np.matrix(N)
    PtNP = Pmt*Nm*Pm
    try:
        U, s, V = np.linalg.svd(PtNP)
    except np.linalg.linalg.LinAlgError:
        return nan*np.ones((len(d))), nan
    else:
        S = np.matrix(np.diag(1/s))
        PtNPinv = np.transpose(V)*S*np.transpose(U)
        PtNd = Pmt*Nm*dm
        c = PtNPinv*PtNd
        chi_min = np.transpose(dm - Pm*c)*Nm*(dm - Pm*c)
        c = np.asarray(c)
        #chi_min2 = np.transpose(dm)*Nm*dm - np.transpose(c)*(np.transpose(Pm)*Nm*Pm)*c
        return c, chi_min
  
def build_psf(xcenter,ycenter,N,samp,offset=[0.5,0.5]):
    """Function to build 2D PSF for a given central point (in pixel units).
       Inputs:
              xcenter (of the trace in the horizontal direction, in px)
              ycenter (of the trace in the vertical direction, in px)
              N = number of pixels to evaluate along each axis
              samp = sampling (in pixels)
              offset = [x,y] offset to center (default is [0,0])"""
    def gaussian2d(axisx,axisy,meanx,meany,sigmax,sigmay):
        """Two dimensional normalized gaussian function.  Inputs are:
        axisx, axisy = points to evaluate (x and y data on an x, y vs z plot)
        meanx = mean/center of the Guassian on the x axis
        meany = mean/center of the Guassian on the y axis
        sigmax = standard deviation/width of the Gaussian on the x axis
        sigmay = standard deviation/width of the Gaussian on the y axis"""
        norm = (1/(sigmax*sigmay*2*pi))
        xexp = -(1/2)*(((axisx-meanx)/sigmax)**2)
        xgrid = np.tile(xexp,[len(axisy),1])
        yexp = np.reshape(-(1/2)*(((axisy-meany)/sigmay)**2),[len(axisy),1])
        ygrid = np.tile(yexp,[1,len(axisx)])
        gauss2d = norm*exp(xgrid+ygrid)
        return gauss2d 
    def erf2d(axisx,axisy,meanx,meany,sigmax,sigmay):
        """Two dimensional error function (integrated gaussian)
        """
        erfx = 0.5*(1+sp.erf((axisx-meanx)/(sqrt(2)*sigmax)))
        erfy = np.reshape(0.5*(1+sp.erf((axisy-meany)/(sqrt(2)*sigmay))),
                          [len(axisy),1])
        exgrid = np.tile(erfx,[len(axisy),1])
        eygrid = np.tile(erfy,[1,len(axisx)])
        erf2d = exgrid*eygrid
        return erf2d
    def delt_erf2d(axisx,axisy,meanx,meany,sigmax,sigmay):
        """Two dimensional error function (integrated gaussian)
        """
        erfx = 0.5*(1+sp.erf((axisx-meanx)/(sqrt(2)*sigmax)))
        erfy = 0.5*(1+sp.erf((axisy-meany)/(sqrt(2)*sigmay)))
        derfx = ediff1d(erfx)
        derfy = np.reshape(ediff1d(erfy),[len(axisy)-1,1])
        exgrid = np.tile(derfx,[len(axisy)-1,1])
        eygrid = np.tile(derfy,[1,len(axisx)-1])
        erf2d = exgrid*eygrid
        return erf2d
    def boxcar2d(axisx,axisy,xc,yc,xw,yw):
        """Basic unit height boxcar function.  Inputs are:
        axis = points to evaluate (x data on an x vs y plot)
        front = position of leading edge of the boxcar function
        width = width/length of the boxcar function"""
        grid = zeros((len(axisx),len(axisy)))
        for i in range(len(axisx)):
            for j in range(len(axisy)):
                grid[i,j] = (i>=(xc-xw/2))*(i<=(xc+xw/2-1))*(
                             j>=(yc-yw/2))*(j<=(yc+yw/2-1))
        return grid    
    axisx = np.arange(-N/2,N/2+1) + offset[0]
    axisy = np.arange(-N/2,N/2+1) + offset[1] 
    axisxd = np.arange(-N/2,N/2+2) + offset[0] - 0.5
    axisyd = np.arange(-N/2,N/2+2) + offset[1] - 0.5
#    axisxe = axisx - 0.5
#    axisye = axisy - 0.5
    sigma = samp/(2*np.sqrt(2*log(2)))
    width = 4 
    dl_erf = delt_erf2d(axisxd,axisyd,xcenter,ycenter,2*sigma,sigma)
    #Bunch of extra options, don't use for now
#    gauss2d = gaussian2d(axisx,axisy,xcenter,ycenter,sigma,sigma)
#    erfx1y1 = erf2d(axisxe,axisye,xcenter,ycenter,sigma,sigma)
#    erfx2y1 = erf2d(axisxe+1,axisye,xcenter,ycenter,sigma,sigma)
#    erfx1y2 = erf2d(axisxe,axisye+1,xcenter,ycenter,sigma,sigma)
#    erfx2y2 = erf2d(axisxe+1,axisye+1,xcenter,ycenter,sigma,sigma)
#    #Want to look at the difference in erf
#    derf1 = erfx1y2-erfx1y1#-erfx1y2#+erfx2y2)
#    derf2 = erfx2y2-erfx2y1
#    derf3 = derf2-derf1
    #Error checking options
#    plt.figure(1)
#    plt.imshow(derf3,interpolation='none')
#    plt.figure(2)
#    plt.imshow(dl_erf,interpolation='none')
#    plt.figure(3)
#    plt.imshow(derf3-dl_erf,interpolation='none')
#    print np.max(derf3), np.max(derf3-dl_erf)    
#    plt.show()
    conv2d = zeros((len(axisx),len(axisy)))
    #Convolve along x axis at each y step
    for l in range(len(axisx)):
        for m in range(len(axisy)):
            conv2d[m,l] = sum(dl_erf*boxcar2d(axisx,axisy,m,l,width,width))
    conv2d = conv2d/sum(conv2d)*sum(dl_erf)
    #    plt.pcolormesh(hstack((gauss2d,conv2d,gauss2d-conv2d)))
#    plt.show()
    return dl_erf
  
def gauss2d(xaxis, yaxis, sigma_x, sigma_y, xcenter=0, ycenter=0):
    """Returns 2-d normalized Guassian (symmetric in x and y)
    INPUTS:
        xaxis = xvalues to be evaulated
        yaxis = yvalues to be evaluated
        sigma_x, sigma_y = spread (assume sigma_x = sigma_y by default)
        x/ycenter = optional center offset
        
    OUTPUTS:
        gauss2d = 2-D array of heights at all values of x and y
    """
    #Reshuffle xaxis and yaxis inputs to be 1-D arrays
    lx = len(xaxis)
    ly = len(yaxis)
    x = np.array((xaxis-xcenter)).reshape(1,lx)
    y = np.array((yaxis-ycenter)).reshape(1,ly)
    
    #Convert x and y into a 2-D grid for 2d gaussian exponent
    sig_sq = sigma_x*sigma_y
    xygrid = sigma_y**2*np.tile(x,(ly,1))**2 + sigma_x**2*np.tile(y.T,(1,lx))**2
    gauss2d = 1/(2*np.pi*sig_sq)*np.exp(-(1/2)*(xygrid/(sig_sq**2)))

    return gauss2d

def hermite(order,axis,sigma,center=0):
    """Function to calculate probabilists Hermite function
    INPUTS:
        order = hermite order (function is likely slow for high orders)
        axis = "x" values at which to calculate Hermite polynomial
        sigma = scaling factor for axis
        center = optional axis center offset
        
    OUTPUTS:
        herm = array of len(axis)
    """
    #axis = np.array(([axis,]))
    uplim = int(np.floor(order/2)) #Upper limit for sum
    def insum(n,m,axis):
        """Formula for inner sum of explicit Hermite function.
        """
        term1 = (-1)**m/(np.math.factorial(m)*np.math.factorial(n-2*m))
        term2 = axis**(n-2*m)/(2**m)
        return term1*term2
        
    try:
        hsum = zeros(len(axis))
    except TypeError:
        hsum = 0
            
    for ii in range(uplim+1):
        hsum += insum(order,ii,axis)
        
    herm = np.math.factorial(order)*hsum
    return herm
    
def spec_amps():
    """Function to load a specific solar spectrum from solar_spectrum.txt.
       Would need to rewrite if generality is needed.
       
       INPUTS
       none
       
       OUTPUTS
       wl_file - wavelengths from file (or sim)
       A_file - solar amplitudes from file (not adjusted)
       """
    pathe = os.environ['EXPRES_OUT_DIR']
    sol = np.loadtxt(pathe+'solar_spectrum.txt')
    wl_file = sol[:,0]
    A_file = sol[:,1]
#    sol = open(pathe + 'solar_spectrum.txt')
#    linecount = sum(1 for line in sol)
#    sol = open(pathe + 'solar_spectrum.txt')
#    wl_file = zeros(linecount)
#    A_file = zeros(linecount)
#    for s in range(linecount-1):
#        line = sol.readline()
#        #Following lines based on format of solar_spectrum.txt
#        wl_file[s] = np.round(float(line[:-6]),5)
#        A_file[s] = int(line[-5:-2]) #Int on BASS2000, could be float for others
    return wl_file, A_file
                
def thar_lamp():
    """Function to load a specific thorium argon lamp profile
   
      INPUTS
      none
   
      OUTPUTS
      wl_file - wavelengths from file (or sim)
      A_file - relative intensities from file (not adjusted)
      """
    pathsim = os.environ['MINERVA_SIM_DIR']
    with open(pathsim+'table1.dat','r') as tbdat:
        rows = sum(1 for l in tbdat)
    wl_file = np.zeros((rows))
    A_file = np.zeros((rows))
    with open(pathsim+'table1.dat','r') as ll:
        ln = ll.readlines()
        for i in range(rows):
            wl_file[i] = ln[i][0:11]
            A_file[i] = ln[i][33:40]
    return wl_file, A_file
    
def plt_deltas(xarr,yarr,color='b',linewidth=1):
    """Function will plot vertical lines at each x point with y amplitude
    """
    xplt = np.vstack((xarr,xarr))
    yplt = np.vstack((np.zeros(len(yarr)),yarr))
    plt.plot(xplt,yplt,color,linewidth=linewidth)
    return
    
def gauss_fit(xarr,yarr,fit_background='y'):
    """3 parameter gaussian fit of the data and 2 parameter background fit
       returns 5 parameter array:
           [sigma, center, height, background mean, background slope]
       or 3 parameter array (if no background fit):
           [sigma, center, height]
    """
    if len(xarr)!=len(yarr):
         print('x and y dimensions don\'t agree!')      
         print('returning zeros parameter array')
         if fit_background=='y':
             return np.zeros((5))
         else:
             return np.zeros((3))
#        raise ValueError('x and y dimensions don\'t agree!')
#        exit(0)
    elif len(xarr)<10:
        print('array is too short for a good fit')
        print('returning zeros parameter array')
        if fit_background=='y':
            return np.zeros((5))
        else:
            return np.zeros((3))
#        raise ValueError('array is too short for a good fit')
#        exit(0)
    elif fit_background=='y':
        #background mean initial guess: average of first two and last two points
        bg_m0 = np.mean([yarr[0:2],yarr[-3:-1]])
        #background slope initial guess: slope between avg of first two and last two
        bg_s0 = (np.mean(yarr[0:2])-np.mean(yarr[-3:-1]))/(np.mean(xarr[0:2])-np.mean(xarr[-3:-1]))
        #height guess - highest point
        h0 = np.max(yarr)-bg_m0
        #center guess - x value at index of highest point
        c0 = xarr[np.argmax(yarr)]
        #sigma guess - difference between first point and last point below 1/2 max
        idx1 = 0
        idx2 = 1
        for i in range(len(xarr)-1):
            if yarr[i+1]>(h0/2+bg_m0) and yarr[i+1]>yarr[i]:
                idx1 = i
                break
        for i in range(idx1,len(xarr)-1):
            if yarr[i+1]<(h0/2+bg_m0) and yarr[i+1]<yarr[i]:
                idx2 = i
                break
        sig0 = (xarr[idx2]-xarr[idx1])/2.355
        p0 = np.array(([sig0,c0,h0,bg_m0,bg_s0]))
        params, errarr = opt.curve_fit(gaussian,xarr,yarr,p0=p0)
        return params
    else:
        #height guess - highest point
        h0 = np.max(yarr)
        #center guess - x value at index of highest point
        c0 = xarr[np.argmax(yarr)]
        #sigma guess - difference between first point and last point below 1/2 max
        idx1 = 0
        idx2 = 1
        for i in range(len(xarr)-1):
            if yarr[i+1]>h0/2 and yarr[i+1]>yarr[i]:
                idx1 = i
                break
        for i in range(idx1,len(xarr)-1):
            if yarr[i+1]<h0/2 and yarr[i+1]<yarr[i]:
                idx2 = i
                break
        sig0 = (xarr[idx2]-xarr[idx1])/2.355
        p0 = np.array(([sig0,c0,h0]))
        params, errarr = opt.curve_fit(gaussian,xarr,yarr,p0=p0)
        return params