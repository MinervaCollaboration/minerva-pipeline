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
import scipy.integrate as integrate
import lmfit
import scipy.sparse as sparse
#import scipy.signal as sig
#import scipy.linalg as linalg
#import astropy.stats as stats

def gaussian(axis, sigma, center=0, height=1,bg_mean=0,bg_slope=0,power=2):
    """Returns gaussian output values for a given input array, sigma, center,
       and height.  Center defaults to zero and height to 1.  Can also
       tweak the exponent parameter if desired."""
    #Height scale factor*(1/(sqrt(2*pi*sigma**2)))
#    print sigma, center, height
    gaussian = height*exp(-abs(axis-center)**power/(2*(abs(sigma)**power)))+bg_mean+bg_slope*(axis-center)
#    print sigma, center, height, bg_mean,bg_slope,power
#    print gaussian
    return gaussian
    
def gaussian_int(axis, sigma, center=0, height=1,bg_mean=0,bg_slope=0,power=2,pts_per_px=100):
    """Returns gaussian output values for a given input array, sigma, center,
       and height.  Center defaults to zero and height to 1.  Can also
       tweak the exponent parameter if desired."""
    dlt_axis = axis[1]-axis[0] ### assumes even spacing
    axis_new = np.arange(axis[0]-dlt_axis/2,axis[-1]+dlt_axis/2,dlt_axis/pts_per_px)
    gaussian_long = height*exp(-abs(axis_new-center)**power/(2*(abs(sigma)**power)))+bg_mean+bg_slope*(axis_new-center)
    gaussian_int = np.sum(gaussian_long.reshape(len(axis),pts_per_px),axis=1)/pts_per_px
    return gaussian_int
    
def gaussian_erf(axis, sigma, center=0, height=1,bg_mean=0,bg_slope=0,power=2):
    """Returns gaussian output values for a given input array, sigma, center,
       and height.  Center defaults to zero and height to 1.  Can also
       tweak the exponent parameter if desired."""
    dlt_axis = axis[1]-axis[0]
    axis_new = np.arange(axis[0]-dlt_axis/2,axis[-1]+3*dlt_axis/2,dlt_axis)
    z = ((axis_new-center)/sigma)**(power/2)
    hmax = 1#height*np.sqrt(pi)/2
    gauss_int_erf = hmax*sp.erf(z)
    gaussian_int = np.ediff1d(gauss_int_erf)
    print len(axis),len(axis_new),len(gauss_int_erf),len(gaussian_int)
    return gaussian_int
    
def poisson(x,lam):
    """ Poisson function, written for numerical stability
    """
    return np.exp(x*np.log(lam)-lam-sp.gammaln(x+1))
    
def schechter_fct(L,L_star,Phi_star,alpha):
    """ To match eqn. 1 of Guo, 2015 (CANDELS)
    """
    N_L = Phi_star*(L/L_star)**alpha*np.exp(-L/L_star)
    return N_L
    
def weibull(x,k,lam):
    """ Weibull distribution
    """
    return (k/lam)*(x/lam)**(k-1)*np.exp(-(x/lam)**k)   
    
def sersic1d(rarr,Ie,re,n):
    bn = 0.868*n-0.142
    I_r = Ie*10**(-bn*((rarr/re)**(1/n)-1))
    return I_r

def sersic2d(x,y,xc,yc,Ie,re,n,q=1,PA=0):
    """ makes a 2D image (dimx x dimy) of a Sersic profile centered at [xc,yc]
        and parameters Ie, re, and n.
        Optionally can add in ellipticity with axis ratio (q) and position
        angle (PA).
    """
    rarr = make_rarr(x,y,xc,yc,q=q,PA=PA)
    image = sersic1d(rarr,Ie,re,n)
    return image    
    
def gauss_residual(params,xvals,zvals,invals):
    """ Residual function for gaussian with constant background, for use with
        lmfit.minimize()
    """
    sigma = params['sigma'].value
    mn = params['mean'].value
    hght = params['hght'].value
    bg = params['bg'].value
    power = params['power'].value
    gauss = gaussian(xvals,sigma,center=mn,height=hght,bg_mean=bg,power=power)
    residuals = (gauss-zvals)*np.sqrt(invals)
    return residuals

def sigma_clip(residuals,sigma=3,max_iters=1):
    """ My own sigma clipping since I don't like the astropy version.
        Use the median, not the mean for comparison to outliers.
        Only clips one point per iteration
        Also only written for symmetric clipping (upper and lower bounds are the same)
        INPUTS:
            residuals - array of data minus fit, or just data is fine too
            sigma - number of stddevs to clip
            max_iters - will repeatedly clip data up to max_iters cycles
        OUTPUTS:
            residual_mask - boolean mask of clipped data (1 = keep, 0 = mask/reject)
    """
    residuals = np.ravel(residuals)
    iter_count = 0
    residual_mask = np.ones((residuals.size),dtype=bool)
    residual_mask_old = np.zeros((residuals.size),dtype=bool)
    while (sum(residual_mask) != sum(residual_mask_old)) and iter_count < max_iters:
#        print residual_mask
#        print residuals
        residual_mask_old = np.copy(residual_mask)        
        med = np.median(residuals[residual_mask])
        std = np.std(residuals[residual_mask]) #May be a better measure than this, but it's a start
        if np.max(abs(residuals[residual_mask])) >= (med + sigma*std):
            r_ind = np.argsort(abs(residuals))[::-1][iter_count]
            residual_mask[r_ind] = 0
        iter_count += 1
    return residual_mask
                
#def cladding(axis,width,center=0,height=1):
#    """Function ONLY for rough use on preliminary MINERVA data.  This is to
#       give a VERY rough estimate for the cladding profile.
#    """
#    zl = axis[axis<=center]
#    zr = axis[axis>center]
#    left = height*0.5*(1+sp.erf((zl-center+width/2)/width))
#    right = height*0.5*(1+sp.erf(-(zr-center-width/2)/width))
#    return np.append(left,right)

def chi_fit(d,P,N,return_errors=False):
    """Routine to find parameters c for a model using chi-squared minimization.
       Note, all calculations are done using numpy matrix notation.
       Inverse is calculated using the matrix SVD.
       Inputs:
           d = data (n-dim array)
           P = Profile (n x c dim array) (c = number of parameters)
           N = Noise (n x n array) (reported as inverse variance)
           return_errors = gives the errors (diagonals of covariance)
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
        s[s==0]=0.001
        S = np.matrix(np.diag(1/s))
        PtNPinv = np.transpose(V)*S*np.transpose(U)
        PtN = Pmt*Nm
        err_matrix = PtNPinv*PtN
        c = err_matrix*dm
        chi_min = np.transpose(dm - Pm*c)*Nm*(dm - Pm*c)
        c = np.asarray(c)[:,0]
        #chi_min2 = np.transpose(dm)*Nm*dm - np.transpose(c)*(np.transpose(Pm)*Nm*Pm)*c
        if return_errors:
            return c, np.sqrt(np.diag(abs(err_matrix)))
        else:
            return c, chi_min

def fit_polynomial_coeffs(xarr,yarr,invar,order,return_errors=False):
    """ Use chi-squared fitting routine to return polynomial coefficients.
        INPUTS:
            xarr - x values of data
            yarr - y values of data
            invar - inverse variance of each yarr value
            order - order of polynomial to fit (2 = quadratic)
            return_errors - returns error estimates for each coefficient
        OUTPUTS:
            poly_coeffs - array of length order+1 with coefficients
    """
    if len(xarr) != len(yarr) or len(xarr) != len(invar):
        print("x, y, and invar arrays must have the same length")
        exit(0)
    # Build profile matrix
    profile = np.power(np.tile(xarr,(order+1,1)).T,np.arange(order+1))
    noise = np.diag(invar) #assumes independent data points
    poly_coeffs, coeff_errs = chi_fit(yarr,profile,noise,return_errors=return_errors)
#    plt.plot(xarr,yarr)
#    plt.plot(xarr,np.dot(profile,poly_coeffs))
#    print np.dot(profile,poly_coeffs)
#    print np.shape(poly_coeffs)
#    plt.show()
#    plt.close()
    if return_errors:
        return poly_coeffs, coeff_errs
    else:
        return poly_coeffs
    
def eval_polynomial_coeffs(xarr,poly_coeffs):
    """ Given polynomial coefficients, evaluates f(x) at all given xarr
        INPUTS:
            xarr - points to evaluate
            poly_coeffs - array of polynomial coefficients
        OUTPUTS:
            ypoly - y values calculated from polynomial
    """
    order = len(poly_coeffs)-1 #assumes a nonzero array is entered
    profile = np.power(np.tile(xarr,(order+1,1)).T,np.arange(order+1))
    ypoly = np.dot(profile,poly_coeffs)
#    print ypoly    
#    print poly_coeffs
#    plt.plot(xarr,ypoly)
#    plt.show()
#    plt.close()
    return ypoly
    
    

def best_linear_gauss(axis,sig,mn,data,invar,power=2):
    """ Use linear chi^2 fitting to find height and background estimates
    """
#    t1 = time.time()
    noise = np.diag(invar)
#    t2 = time.time()
    profile = np.ones((len(axis),2))
    profile[:,0] = gaussian(axis,sig,mn,1,power=power)
#    profile = np.vstack((gaussian(axis,sig,mn,1,power=power),np.ones(len(axis)))).T
#    t3 = time.time()    
    coeffs, chi = chi_fit(data, profile, noise)
#    t4 = time.time()
#    print("  % time noise = {}s".format((t2-t1)/(t4-t1)))
#    print("  % time profile = {}s".format((t3-t2)/(t4-t1)))
#    print("  % time chi = {}s".format((t4-t3)/(t4-t1)))
#    time.sleep(2)
    return coeffs[0], coeffs[1]

def best_mean(axis,sig,mn,hght,bg,data,invar,spread,power=2):
    """ Finds the mean that minimizes squared sum of weighted residuals
        for a given sigma and height (guassian profile)
        Uses a grid-search, then fits around minimum chi^2 regionS
    """
    def mn_find(axis,mn,spread):
        """ Call with varying sig ranges for coarse vs. fine.
        """
        mn_rng = np.linspace(mn-spread,mn+spread,10)
        chis = np.zeros(len(mn_rng))
        ### Find chi^2 at a range of means aroung the best guess
        for i in range(len(mn_rng)):        
            gguess = gaussian(axis,sig,mn_rng[i],hght,bg,power=power)
            chis[i] = sum((data-gguess)**2*invar)
        ### Now take a subset of 5 points around the lowest chi^2 for fitting
        chi_min_inds =np.arange(-2,3)+np.argmin(chis)
        chi_min_inds = chi_min_inds[(chi_min_inds>=0)*(chi_min_inds<len(mn_rng))]
        ### If 3 or less good points remain, can't fit - return guesses
        if len(chi_min_inds)<4:
            return mn, spread
        else:
            ### Otherwise, do
            poly_coeffs = np.polyfit(mn_rng[chi_min_inds],chis[chi_min_inds],2)
            bq = poly_coeffs[1]
            aq = poly_coeffs[0]
            best_mn = -bq/(2*aq)
            sigma = 1 #1 standard dev, 4 = 2 std devs
            aq = abs(aq) ### Ensure positive
            ### Find values sqrt(sigma) std devs to each side
            mn_p = (-bq + np.sqrt(4*aq*np.sqrt(sigma)))/(2*aq)
            mn_m = (-bq - np.sqrt(4*aq*np.sqrt(sigma)))/(2*aq)
            best_mn_std = abs(mn_p-mn_m)/2
            return best_mn, best_mn_std
        
    ###Coarse find
    ###Tradeoff with coarse sig - too small and initial guess is critical,
    ### too big and more susceptible to comsic ray influence
    best_mn, best_mn_std = mn_find(axis,mn,spread)
    ###Fine find
    best_mn, best_mn_std = mn_find(axis,best_mn,best_mn_std)
    ###Finer find - doesn't seem to change final answer at all
#    best_mn, best_mn_std = mn_find(axis,best_mn,best_mn_std/100)
    return best_mn, best_mn_std
    
def fit_mn_hght_bg(xvals,zorig,invorig,sigj,mn_new,spread,powj=2):
#    mn_new = xc-xj
    mn_old = -100
    lp_ct = 0
    while abs(mn_new-mn_old)>0.001:
        mn_old = np.copy(mn_new)
        hght, bg = best_linear_gauss(xvals,sigj,mn_old,zorig,invorig,power=powj)
        mn_new, mn_new_std = best_mean(xvals,sigj,mn_old,hght,bg,zorig,invorig,spread,power=powj)
        lp_ct+=1
        if lp_ct>1e3: break
    return mn_new, hght,bg

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
  
def gauss2d(xaxis, yaxis, sigma_x, sigma_y, xcenter=0, ycenter=0, q=1, PA=0,unity_height=False):
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
#    lx = len(xaxis)
#    ly = len(yaxis)
#    x = np.array((xaxis-xcenter)).reshape(1,lx)
#    y = np.array((yaxis-ycenter)).reshape(1,ly)
    
    #Convert x and y into a 2-D grid for 2d gaussian exponent
    sig_sq = sigma_x*sigma_y
    xygrid = make_rarr(xaxis, yaxis, xcenter, ycenter, q=q, PA=PA)
#    xygrid = sigma_y**2*np.tile(x,(ly,1))**2 + sigma_x**2*np.tile(y.T,(1,lx))**2
    if unity_height:
        hght = 1
    else:
        hght = 1/(2*np.pi*sig_sq)
    gauss2d = hght*np.exp(-(1/2)*(xygrid/(sig_sq**2)))
#    plt.imshow(gauss2d,interpolation='none')
#    plt.show()
#    plt.close()
    return gauss2d
 
def gauss2d_lmfit(params, xaxis, yaxis, cnt):
    """Returns 2-d normalized Guassian (symmetric in x and y)
    INPUTS:
        xaxis = xvalues to be evaulated
        yaxis = yvalues to be evaluated
        sigma_x, sigma_y = spread (assume sigma_x = sigma_y by default)
        x/ycenter = optional center offset
        
    OUTPUTS:
        gauss2d = 2-D array of heights at all values of x and y
    """
    xc = params['xcb{}'.format(cnt)].value
    yc = params['ycb{}'.format(cnt)].value
#    sigx = params['sigx{}'.format(cnt)].value
#    sigy = params['sigy{}'.format(cnt)].value
    sig = params['sigb{}'.format(cnt)].value
    hght = params['hght{}'.format(cnt)].value
    ### Assume q, PA of host galaxy apply to all blobs...
    q = params['qb{}'.format(cnt)].value
    PA = params['PAb{}'.format(cnt)].value
    #Reshuffle xaxis and yaxis inputs to be 1-D arrays
#    lx = len(xaxis)
#    ly = len(yaxis)
#    x = np.array((xaxis-xcenter)).reshape(1,lx)
#    y = np.array((yaxis-ycenter)).reshape(1,ly)
    
    #Convert x and y into a 2-D grid for 2d gaussian exponent
#    sig_sq = sigx*sigy
    sig_sq = sig*sig
    xygrid = make_rarr(xaxis, yaxis, xc, yc, q=q, PA=PA)
#    xygrid = sigma_y**2*np.tile(x,(ly,1))**2 + sigma_x**2*np.tile(y.T,(1,lx))**2
    gauss2d = hght/(2*np.pi*sig_sq)*np.exp(-(1/2)*(xygrid/(sig_sq**2)))
#    plt.imshow(gauss2d,interpolation='none')
#    plt.show()
#    plt.close()
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
        term2 = ((axis-center)/sigma)**(n-2*m)/(2**m)
        return term1*term2
        
    try:
        hsum = zeros(len(axis))
    except TypeError:
        hsum = 0
            
    for ii in range(uplim+1):
        hsum += insum(order,ii,axis)
        
    herm = np.math.factorial(order)*hsum
    return herm
    
def gauss_herm1d(axis,sigma,center=0,hg=1,h0=0,h3=0,h4=0):
    """ Function for fitting to a 1d gauss hermite with orders 0,3, and 4
        Inputs:
            axis - points to evaluate
            sigma - spread in gaussian
            center - mean of gaussian
            h0 - weight for hermite order 0
            h3 - weight for hermite order 3
            h4 - weight for hermite order 4
        Outputs:
            gh1d - array of y values at each x axis point
    """    
#    print sigma, center, h0, h3, h4
#    time.sleep(0.2)
    gh1d = gaussian(axis,sigma,center)*(hg+h0*hermite(0,axis,sigma,center)+h3*hermite(3,axis,sigma,center)+
                    h4*hermite(4,axis,sigma,center))
    return gh1d
    
def fit_gauss_herm1d(xarr,yarr,invr=1):
    """ Fits to the gauss_herm1d function
    """
    #plen = 3+2*(fit_background=='y')+1*(fit_exp=='y')
    plen = 5 #depends on hermite orders used
    if len(xarr)!=len(yarr):
        if verbose=='y':
            print('x and y dimensions don\'t agree!')      
            print('returning zeros parameter array')
        return np.zeros((plen)), np.zeros((plen,plen))
#        raise ValueError('x and y dimensions don\'t agree!')
#        exit(0)
    elif len(xarr)<6:
        if verbose=='y':
            print('array is too short for a good fit')
            print('returning zeros parameter array')
        return np.zeros((plen)), np.zeros((plen,plen))
#        raise ValueError('array is too short for a good fit')
#        exit(0)
    else:
        ### Set sigma weighting
        if type(invr) == int:
            sigma = np.sqrt(abs(yarr))
        else:
            sigma = np.sqrt(1/abs(invr.astype(float)))
        ### Set initial guess values
        #height guess - highest point
        h0 = np.max(yarr)    
        #center guess - x value at index of highest point
        c0 = xarr[np.argmax(yarr)]
        #sigma guess - difference between first point and last point below 1/2 max
        idx1 = 0
        idx2 = 1
        for i in range(len(xarr)-1):
            if yarr[i+1]>(h0/2) and yarr[i+1]>yarr[i]:
                idx1 = i
                break
        for i in range(idx1,len(xarr)-1):
            if yarr[i+1]<(h0/2) and yarr[i+1]<yarr[i]:
                idx2 = i
                break
        sig0 = (xarr[idx2]-xarr[idx1])/2.355  
#        h0 *= (sig0*np.sqrt(2*np.pi))
        h3 = 0
        h4 = 0
        p0 = np.array(([sig0,c0,h0,h3,h4]))
#        plt.plot(xarr,yarr)
#        plt.plot(xarr,gauss_herm1d(xarr,sig0,c0,h0,h3,h4))
#        plt.plot(xarr,gaussian(xarr,sig0,c0,h0))
#        plt.show()
#        plt.close()
        params, errarr = opt.curve_fit(gauss_herm1d,xarr,yarr,p0=p0,sigma=sigma)
        return params#, errarr
        
def gauss_5(axis,sig0,c0,h0,h1,h2,h3,h4):
    """ For fit_3_gauss
    """
    return gaussian(axis,sig0,c0,h0)+gaussian(axis,sig0,c0-sig0/2,h1)+gaussian(axis,sig0,c0-sig0,h2)+gaussian(axis,sig0,c0+sig0/2,h3)+gaussian(axis,sig0,c0+sig0,h4)
        
def fit_5_gauss(xarr,yarr,invr):
    """ Fits three guassian profiles at once
    """
    sigma = np.sqrt(1/abs(invr.astype(float)))
    ### Set initial guess values
    #height guess - highest point
    h0 = np.max(yarr)    
    #center guess - x value at index of highest point
    c0 = xarr[np.argmax(yarr)]
    #sigma guess - difference between first point and last point below 1/2 max
    idx1 = 0
    idx2 = 1
    for i in range(len(xarr)-1):
        if yarr[i+1]>(h0/2) and yarr[i+1]>yarr[i]:
            idx1 = i
            break
    for i in range(idx1,len(xarr)-1):
        if yarr[i+1]<(h0/2) and yarr[i+1]<yarr[i]:
            idx2 = i
            break
    sig0 = (xarr[idx2]-xarr[idx1])/2.355  
#        h0 *= (sig0*np.sqrt(2*np.pi))
#    c1 = c0-sig0/2
#    c2 = c0+sig0/2
    h1 = 0
    h2 = 0
    h3 = 0
    h4 = 0
    p0 = np.array(([sig0,c0,h0,h1,h2,h3,h4]))
    params, errarr = opt.curve_fit(gauss_5,xarr,yarr,p0=p0,sigma=sigma)
    return params#, errarr
        
    
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
    
def gauss_fit(xarr,yarr,invr=1,xcguess=-10,pguess=0,fit_background='y',fit_exp='n',verbose='n'):
    """3 parameter gaussian fit of the data and 2 parameter background fit
       returns 5 parameter array:
           [sigma, center, height, background mean, background slope]
       or 3 parameter array (if no background fit):
           [sigma, center, height]
       or 6 parameter array if also fitting exponential power:
           [sigma, center, height, background mean, background slope, power]
       in all cases, returns the covariance matrix of parameter estimates
    """
#    print "in gauss_fit"
    plen = 3+2*(fit_background=='y')+1*(fit_exp=='y')
#    plen = 3+1*(fit_background=='y')+1*(fit_exp=='y')
    if len(xarr)!=len(yarr):
        if verbose=='y':
            print('x and y dimensions don\'t agree!')      
            print('returning zeros parameter array')
        return np.zeros((plen)), np.zeros((plen,plen))
#        raise ValueError('x and y dimensions don\'t agree!')
#        exit(0)
    elif len(xarr)<(plen+1):
        if verbose=='y':
            print('array is too short for a good fit')
            print('returning zeros parameter array')
        return np.zeros((plen)), np.zeros((plen,plen))
#        raise ValueError('array is too short for a good fit')
#        exit(0)
    else:
        ### Set sigma weighting
        if type(invr) == int:
            sigma = np.sqrt(abs(yarr))
        else:
            sigma = np.sqrt(1/abs(invr.astype(float)))
        ### Set initial guess values
        if type(pguess) == int:
            #background mean initial guess: average of first two and last two points
            bg_m0 = np.mean([yarr[0:2],yarr[-3:-1]])
            #background slope initial guess: slope between avg of first two and last two
            bg_s0 = (np.mean(yarr[0:2])-np.mean(yarr[-3:-1]))/(np.mean(xarr[0:2])-np.mean(xarr[-3:-1]))
            #height guess - highest point
            h0 = np.max(yarr)-bg_m0*(fit_background=='y')
            #center guess - x value at index of highest point
            c0 = xarr[np.argmax(yarr)]
            #Option to override xcenter guess only
            if xcguess!=-10:
                c0=xcguess
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
#            h0 *= (sig0*np.sqrt(2*np.pi))
            power0 = 2
            if fit_background == 'y' and fit_exp == 'y':
                p0 = np.array(([sig0,c0,h0,bg_m0,bg_s0,power0]))
            elif fit_background == 'y' and fit_exp == 'n':
                p0 = np.array(([sig0,c0,h0,bg_m0,bg_s0]))
#                p0 = np.array(([sig0,c0,h0,bg_m0]))
            elif fit_background == 'n' and fit_exp == 'n':
                p0 = np.array(([sig0,c0,h0]))
            else:
                raise ValueError('Invalid fit_background and/or fit_exp:')
                print("Acceptable values are 'y' or 'n'.")
                print("Also, cannot fit power without also fitting background")
                exit(0)
        else:
            if fit_background == 'y' and fit_exp == 'y':
                rightlen = len(pguess)==6
            elif fit_background == 'y' and fit_exp == 'n':
                rightlen = len(pguess)==5
#                rightlen = len(pguess)==4
            elif fit_background == 'n' and fit_exp == 'n':
                rightlen = len(pguess)==3
            else:
                raise ValueError('Invalid fit_background and/or fit_exp:')
                print("Acceptable values are 'y' or 'n'.")
                exit(0)
            if rightlen:
                p0 = pguess
            else:
                raise ValueError("Guess array is wrong length.")
                print("For given choices, guess must be length {}.".format(plen))
                exit(0)
#        if len(p0)==6:
#            print gaussian (xarr,p0[0],p0[1],p0[2],p0[3],p0[4],p0[5])
#        plt.plot(xarr,yarr)
#        plt.plot(xarr,gaussian(xarr,sig0,c0,h0))
#        plt.show()
#        plt.close()
#        print "attempting fit"
#        if verbose=='y':
#            print p0
#            print sigma
#            plt.plot(xarr,yarr)
#            plt.show()
#            plt.close()
        ###CURVE FIT METHOD
        params, errarr = opt.curve_fit(gaussian,xarr,yarr,p0=p0,sigma=sigma)
        ###LEASTSQ METHOD
#        def gauss_residual(argarray):
#            """ Residual function - depends on global variables right now... 
#            """
#            sig0 = argarray[0]
#            x0 = argarray[1]
#            hght0 = argarray[2]
#            if len(argarray)<4:
#                return (yarr-gaussian(xarr,sig0,x0,hght0))/sigma
#            else:
#                bgm0 = argarray[3]
#                bgs0 = argarray[4]
#                if len(argarray)<6:
#                    return (yarr-gaussian(xarr,sig0,x0,hght0,bgm0,bgs0))/sigma
#                else:
#                    pow0 = argarray[5]
#                    return (yarr-gaussian(xarr,sig0,x0,hght0,bgm0,bgs0,pow0))/sigma
#             
#        print "Estimating params"
#        params, cov = opt.leastsq(gauss_residual,p0)
#        print sum(params)
#        print "Params estimated"
#        ### figure out correct errarr later
#        errarr = cov*np.ones((len(params),len(params)))
        return params, errarr
        
def fit_height(xvals,zvals,invvals,sig,xc,bgm=0,bgs=0,power=2):
    """ Function to fit only the height for gaussian function
        Right now this is used only in Minerva's "simple_opt_ext"
        Output - best fit height
    """
    profile = gaussian(xvals,sig,xc,bgm,bgs,power).T
    profile = np.resize(profile,(len(profile),1))
    noise = np.diag(invvals)
    height, chi = chi_fit(zvals,profile,noise)
    return height
    
def interpolate_coeffs(coeffs,c_invars,pord,hcenters,coeff_mask,method='polynomial',view_plot=False,pos_vals=False):
    """ Routine designed to be used iteratively to smooth coefficients
        (of splines, models, etc.) across one trace of an arc image.
        INPUTS:
            coeffs - best fit coefficients for each emission line (2D array, #emission lines x #coeffs/line)
            c_invars - inverse variance estimates of each coefficient
            pord - polynomial order (2=quadratic, etc.)
            hcenters - best fit horizontal (x) centers of emission lines
            coeff_mask - a mask, same length as hcenters, to omit "bad" lines.  For now "bad" is hardcoded as 3sigma deviation from smooth estimate.
                My convention is opposite 'normal.'  That is, 1 = masked point, 0 = good point
            method - NOT IMPLEMENTED YET.  Can be 'polynomial' or 'bspline' interpolation.  Right now, only polynomial works
            view_plot - boolean, will show coefficients and fits
            pos_vals - boolean, masks lines with negative amplitude in first coeff (only useful right now in radial spline fit)
        OUTPUTS:
            For recursion only:
                coeffs_smooth - fitted (smoothed) coefficient values
                coeffs_smooth_invar - inverse variance estimate for each fitted coefficient
                coeffs_smooth_mask - updated mask of any bad lines (from sigma clipping)
            Final outputs:
                poly_coeffs - a pord+1 by #coeffs/line array of fitted coefficients
                poly_errs - error estimates (std) on each of the coefficients
    """
    if view_plot:
        colormap = iter(cm.rainbow(np.linspace(0,1,len(coeffs[0]))))
        fig, ax = plt.subplots()
#    coeffs_smooth = zeros((np.shape(coeffs)))
    coeff_mask_old = np.ones((np.shape(coeff_mask)))
    poly_coeffs = np.zeros((pord+1,len(coeffs[0])))
    poly_errs = np.zeros((pord+1,len(coeffs[0])))
    while (sum(coeff_mask_old) != sum(coeff_mask)):
        coeff_mask_old = np.copy(coeff_mask)
    ### Loop through each coefficient, fits as a function of position
        for l in range(len(coeffs[0])):
#            print coeff_mask
            if pos_vals:
                if np.min(coeffs[:,0])<0:
                    coeff_mask[coeffs[:,0]<0] += len(coeffs[0])
#            print coeff_mask
            poly_coeffs[:,l], poly_errs[:,l] = fit_polynomial_coeffs(hcenters[coeff_mask==0],coeffs[:,l][coeff_mask==0],c_invars[:,l][coeff_mask==0],pord,return_errors=True)
            coeffs_smooth = eval_polynomial_coeffs(hcenters,poly_coeffs[:,l])
            #Crude error estimates - there must be a better way...
    #        print coeff_mask
    #        print poly_coeffs
    #        print poly_errs
    #        coeffs_smooth_maxes = eval_polynomial_coeffs(hcenters,poly_coeffs[:,l]+poly_errs[:,l])
    #        coeffs_smooth_mins = eval_polynomial_coeffs(hcenters,poly_coeffs[:,l]-poly_errs[:,l])
    #        coeff_errs = coeffs_smooth_maxes-coeffs_smooth_mins
    #        coeffs_smooth_invar[:,l] = 1/(coeff_errs**2)
            ### Mask based on sigma clipping of residuals
            sclip_mask = sigma_clip(coeffs_smooth-coeffs[:,l],sigma=2,max_iters=3)
    #        print coeffs_smooth[:,l]
    #        print sclip_coeffs
            coeff_mask[sclip_mask==1] += 1 ### Allow this to be greater than 1 - will return to 1 at the end
        ### Now only mask a line if a threshhold number of coefficients reject that line
    #    mask_thresh = np.floor(0.1*np.shape(coeffs)[1]) #10% of coeffs show bad fit
        mask_thresh = 2 #For strictest masking (may falsely mask some lines)
    #    print (coeffs_smooth_mask[coeff_mask>0] >= mask_thresh)
        coeff_mask[coeff_mask<mask_thresh] = np.zeros(len(coeff_mask[coeff_mask<mask_thresh]))
        coeff_mask[coeff_mask>=mask_thresh] = np.ones(len(coeff_mask[coeff_mask>=mask_thresh]))
#        if mask_thresh > 1:
#            coeff_mask += coeff_mask_old
        ### If any extra points were masked, repeat with the new mask
    #    print sum(coeffs_smooth_mask)
    #    print sum(coeff_mask)
    if view_plot:
        for l in range(len(coeffs[0])):
            color = next(colormap)
            coeffs_smooth = eval_polynomial_coeffs(hcenters,poly_coeffs[:,l])
    #        ax.plot(hcenters,coeffs[:,l]-coeffs_smooth[:,l],color=color)
            ax.plot(hcenters,coeffs[:,l],color=color)
            ax.plot(hcenters,coeffs_smooth,color=color)
            ax.axhline(y=0, color='k')
        plt.show()
        plt.close()
    return poly_coeffs, poly_errs
    
def build_2D_action():
    """ Function to build the profile/action matrix for 2D extraction.
        INPUTS:
        
        OUTPUTS:
    """
    
def extract_2D(d,A,N,return_no_conv=False):
    """ Runs 2D extraction on a sample of data.  Requires input of a 
        profile (aka. action) matrix.
        INPUTS:
            d - data (usually converted from ccd section) to extract (mx1)
            A - profile/action matrix (can be matrix or 2D array) (mxc)
            N - Per pixel inverse variance matrix (diagonal if independent) (mxm)
            return_no_conv - Boolean T/F.  If True, does not do reconvolution (F by default)
        OUTPUTS:
            fluxtilde - reconvolved flux output (default when return_no_conv = False) (cx1)
            flux - non reconvolved flux output (only if return_no_conv = True) (cx1)
    """
    A = np.matrix(A)
    d = np.reshape(np.matrix(d),(len(d),1)) #Ensure column vector
    Ninv = np.matrix(N)
    Cinv = A.T*Ninv*A
    ### TODO - refine this, possibly with sparse matrix implementation)
    U, s, Vt = linalg.svd(Cinv)
    Cpsuedo = Vt.T*np.matrix(np.diag(1/s))*U.T
    flux = Cpsuedo*(A.T*Ninv*d)
    if return_no_conv:
        flux = np.asarray(flux)
        flux = np.reshape(flux,(len(flux),))
        return flux
    else:
        ### Now find reconvolution matrix
        f, Wt = linalg.eig(Cinv)
        F = np.matrix(np.diag(np.asarray(f)))
        F = np.abs(F) ## Hack...
        Wt = np.real(Wt)
        WtDhW = Wt*np.sqrt(F)*Wt.T
        WtDhW = np.asarray(WtDhW)
#        s = 
        S = np.matrix(np.diag(np.sum(WtDhW,axis=1)))
        Sinv = linalg.inv(S)
        WtDhW = np.matrix(WtDhW)
        R = Sinv*WtDhW
        ### Convert to final formats    
        fluxtilde = R*flux
        fluxtilde = np.asarray(fluxtilde)
        fluxtilde = np.reshape(fluxtilde,(len(fluxtilde),))
        return fluxtilde
        
def extract_2D_sparse(d,A,N,return_no_conv=False):
    """ Runs 2D extraction on a sample of data.  Requires input of a 
        profile (aka. action) matrix.  Uses sparse matrices for calculation.
        INPUTS:
            d - data (usually converted from ccd section) to extract (mx1)
            A - profile/action matrix (can be matrix or 2D array) (mxc)
            N - Per pixel inverse variance matrix (diagonal if independent) (mxm)
            return_no_conv - Boolean T/F.  If True, does not do reconvolution (F by default)
        OUTPUTS:
            fluxtilde - reconvolved flux output (default when return_no_conv = False) (cx1)
            flux - non reconvolved flux output (only if return_no_conv = True) (cx1)
    """
    A = np.matrix(A)
    d = np.reshape(np.matrix(d),(len(d),1)) #Ensure column vector
    Ninv = np.matrix(N)
    ### Convert to sparse
    A = sparse.csr_matrix(A)
    d = sparse.csr_matrix(d)
    Ninv = sparse.csr_matrix(Ninv)
    ###
    Cinv = A.T*Ninv*A
    U, s, Vt = linalg.svd(Cinv.todense())
    Cpsuedo = Vt.T*np.matrix(np.diag(1/s))*U.T
    Cpsuedo = sparse.csr_matrix(Cpsuedo)
#    U, s, Vt = sparse.linalg.svds(Cinv,k=np.min(Cinv.shape)-2)
#    U = sparse.csr_matrix(U)
#    Vt = sparse.csr_matrix(Vt)
#    U, s, Vt = linalg.svd(Cinv)
#    Cpsuedo = Vt.T*sparse.csr_matrix(np.diag(1/s))*U.T
    flux = Cpsuedo*(A.T*Ninv*d)
    flux = flux.todense()
    if return_no_conv:
        flux = np.asarray(flux)
        flux = np.reshape(flux,(len(flux),))
        return flux
    else:
        ### Now find reconvolution matrix
        Cinv = sparse.csc_matrix(Cinv)
        f, Wt = sparse.linalg.eigs(Cinv,k=int(np.min(Cinv.shape)-2))
        F = np.matrix(np.diag(np.asarray(f)))
        F = np.abs(F) ## Hack...
        ### Faster to do dense than sparse (at least in my one test session)
#        F = sparse.csr_matrix(F)
#        print np.real(Wt)
        Wt = np.real(Wt)
#        Wt = sparse.csr_matrix(Wt)
        WtDhW = Wt*np.sqrt(F)*Wt.T
#        WtDhW = WtDhW.toarray()
        WtDhW = np.asarray(WtDhW)
        s = np.sum(WtDhW,axis=1)
        Sinv = linalg.inv(np.diag(s))
#        S = sparse.csr_matrix(np.diag(s)) ### warning goes away if this is csc, but then it's slower...
#        Sinv = sparse.linalg.inv(S)  ### This line gives a warning, but still works
#        WtDhW = sparse.csr_matrix(WtDhW)
        WtDhW = np.matrix(WtDhW)
        R = Sinv*WtDhW
#        R = R.todense()
        ### Convert to final formats    
        fluxtilde = R*flux
        fluxtilde = np.asarray(fluxtilde)
        fluxtilde = np.reshape(fluxtilde,(len(fluxtilde),))
        return fluxtilde
        
def recenter_img(img,old_center,new_center):
    """ Resamples image to shift center.  Note that this assumes a sub-pixel
        shift.  Also note, my convention, "v" is row, "h" is column.
        INPUTS:
            img - 2D data array
            old_center - [vc,hc] in pixel coordinates
            new_center - [vn,hn] in pixel coordinates
        OUTPUTS:
            new_img
    """
    [vc,hc] = old_center
    [vn,hn] = new_center
    dv = vn-vc
    dh = hn-hc
    #set sign
    if dv>= 0:
        sv = 1
    else:
        sv = -1
    if dh >= 0:
        sh = 1
    else:
        sh = -1
    dv = abs(dv)
    dh = abs(dh)
    new_img = np.zeros(np.shape(img))
    for i in range(np.shape(img)[0]):
        for j in range(np.shape(img)[1]):
            ### Linear combination of nearest 4 (excluding edges)
            px1 = img[i,j]*(1-dv)*(1-dh)
            if i-sv >= 0 and i-sv < np.shape(img)[0]:
                px2 = img[i-sv,j]*dv*(1-dh)
            else:
                px2 = img[i,j]*dv*(1-dh)
            if j-sh >= 0 and j-sh < np.shape(img)[1]:
                px3 = img[i,j-sh]*(1-dv)*dh
            else:
                px3 = img[i,j]*(1-dv)*dh            
            if i-sv >= 0 and i-sv < np.shape(img)[0] and j-sh >= 0 and j-sh < np.shape(img)[1]:
                px4 = img[i-sv,j-sh]*dv*dh
            else:
                px4 = img[i,j]*dv*dh
            new_img[i,j] = px1 + px2 + px3 + px4
    return new_img
    
def radial_quad_fct(xarr,a,b,c,x0,y0):
    val = a + b*(((xarr-x0)**2+y0**2)**(1/2)) + c*((xarr-x0)**2+y0**2)
    return val

def radial_quad_res(params,data,xarr):
    """ For use with lmfit - residuals of cross section of radial quadratic
        function (built for testing purposes)
    """
    if len(data) != len(xarr):
        print("Data and xarr array must be equal length")
        exit(0)
    ### quadratic parameters
    a = params['a'].value
    b = params['b'].value
    c = params['c'].value
    ### Offset from center of circle
    x0 = params['x0'].value
    y0 = params['y0'].value
    ### residuals
    residuals = data - radial_quad_fct(xarr,a,b,c,x0,y0)
    return residuals
    
def make_rarr(x,y,xc,yc,q=1,PA=0):
    """ Makes 2D array with radius from center at each coordinate.
    """
    x_matrix = np.tile(x,(len(y),1))
    y_matrix = np.tile(y,(len(x),1)).T
    x_ell = (y_matrix-yc)*cos(PA) + (x_matrix-xc)*sin(PA)
    y_ell = (x_matrix-xc)*cos(PA) - (y_matrix-yc)*sin(PA)
    r_matrix = np.sqrt((x_ell)**2*q + (y_ell)**2/q)
    return r_matrix
  
def pdf_draw(pdf,n=1,args=None,int_lims=None,res=1e4):
    """ Returns n values drawn from function 'pdf'.  Optional input
        args passed to pdf (parameters, for example).
        int_lims - used to adjust bounds for normalization integration.
        I think this is going to be pretty slow in general...
    """
    x = np.random.rand(n)
    xarr = np.zeros((int(res)))
    if int_lims is None:
        yarr = np.linspace(-1e3,1e3,int(res))
        norm = integrate.quad(pdf,-np.inf,np.inf,args=args)
        for i in range(len(yarr)):
            result = integrate.quad(pdf,-np.inf,yarr[i],args=args)
            xarr[i] = result[0]/norm[0]
    else:
        a, b = int_lims
        yarr = np.linspace(a,b,int(res))
        if args is None:
            norm = integrate.quad(pdf,a,b)
        else:
            norm = integrate.quad(pdf,a,b,args=args)
        for i in range(len(yarr)):
            if args is None:
                result = integrate.quad(pdf,a,yarr[i])
            else:
                result = integrate.quad(pdf,a,yarr[i],args=args)
            xarr[i] = result[0]/norm[0]
    if n==1:
        xind = np.sum(xarr<x)
        y = (yarr[xind-1]+yarr[xind])/2
    else:
        y = np.zeros((n))
        for j in range(n):
            xind = int(np.sum(xarr<x[j]))
            if xind == n:
                y[j] = yarr[xind-1]
            else:
                y[j] = (yarr[xind-1]+yarr[xind])/2
    return y
    
def find_peaks2D(image,min_space=1,min_px=4,bg_thresh=None,min_amp=None,offset=[0,0]):
    """ Finds all peaks in a 2 dimensional image.  Peaks must be separated by
        at least min_space with magnitude above bg_thresh and at least min_px
        pixels lit in (2*min_space+1)**2 square.
        Allow an offset to be used in the case of systematic errors.
        Returns boolean array, True at peak locations
    """
    if bg_thresh is None:
        ### Decent approximation if majority of image is noise
        bg_thresh = np.median(image<np.mean(image))
    image[image<bg_thresh] = 0
    peaks = np.zeros((image.shape),dtype=bool)
    for i in range(min_space,image.shape[0]-min_space): ### avoid edges
        for j in range(min_space,image.shape[1]-min_space):
            if image[i,j] == np.max(image[i-min_space:i+min_space+1,j-min_space:j+min_space+1]) and image[i,j] != 0:
                if np.sum(image[i-min_space:i+min_space+1,j-min_space:j+min_space+1]>0) > min_px:
#                    print "peak found"
#                    plt.imshow(image[i-3:i+3+1,j-3:j+3+1],interpolation='none')
#                    plt.show()
#                    plt.close()
                    if min_amp is not None:
                        if np.sum(image[i-min_space:i+min_space+1,j-min_space:j+min_space+1])>min_amp:
#                            print "peak above min_amp"
                            peaks[i-offset[1],j-offset[0]] = True
                    else:
                        peaks[i-offset[1],j-offset[0]] = True
    return peaks
    
def array_to_dict(dictionary,array,arraynames=None):
    """ Puts each element of 'array' into 'dictionary'.
        Uses arraynames, if present, otherwise numbers 0:len(array)
    """
    if arraynames is None:
        arraynames = np.arange(len(array))
    if len(array) != len(arraynames):
        print("Inputs 'array' and 'arraynames' must have same length")
        exit(0)
    for i in len(array):
        dictionary[arraynames[i]] = array[i]
    return dictionary
    
def array_to_Parameters(params,array,arraynames=None,minarray=None,maxarray=None,fixed=None):
    """ Puts each element of 'array' into lmfit Parameter object
        Uses arraynames, if present, otherwise numbers 0:len(array)
        Uses min/maxarray if present
        Note, if only one minimum matters, set others to -np.inf
              if only one maximum matters, set others to np.inf
        Uses fixed if present (1 = variable, 0 = fixed)
    """
    try:
        if arraynames is None:
            arraynames = np.arange(len(array))
        if len(array) != len(arraynames):
            print("Inputs 'array' and 'arraynames' must have same length")
            exit(0)
        for i in range(len(array)):
            params.add(arraynames[i], value = array[i])
            if minarray is not None:
                params[arraynames[i]].min = minarray[i]
            if maxarray is not None:
                params[arraynames[i]].max = maxarray[i]
            if fixed is not None:
                params[arraynames[i]].vary = fixed[i]
    ### The exception allows a single value to be added to params
    ### Note, arraynames (str, not array) must be used, otherwise 0 is only key
    except TypeError:
        if arraynames is None:
            arraynames = 0
        params.add(arraynames, value = array)
        if minarray is not None:
            params[arraynames].min = minarray
        if maxarray is not None:
            params[arraynames].max = maxarray
        if fixed is not None:
            params[arraynames].vary = fixed
    return params
    
def find2dcenter(image,invar,c0,method='quadratic'):
    """ Estimates x,y position of peak in an image with initial guess c0.
        c0 = [xc, yc]
    """
    if method == 'quadratic':
        points = 5 ## may add this as an input option later...
        xmatrix = np.tile(np.arange(points)-np.floor(points/2)+c0[0],(points,1))
        ymatrix = np.tile(np.arange(points)-np.floor(points/2)+c0[1],(points,1)).T
        xarray = np.reshape(xmatrix,(points**2,1))
        yarray = np.reshape(ymatrix,(points**2,1))
        profile = np.hstack((xarray**2,xarray,yarray**2,yarray,np.ones((points**2,1))))
        data = np.ravel(image)
        noise = np.diag(np.ravel(invar))
        coeffs, chi = chi_fit(data,profile,noise)
        return [-coeffs[1]/(2*coeffs[0]),-coeffs[3]/(2*coeffs[2])]
    elif method == 'gaussian':
        sigma = 1 ## may add this as an input option later
#        print np.max(image)
#        print(image[c0[1],c0[0]])
#        hght = 4*(2*np.log(2))*image[c0[1],c0[0]]
#        hght = np.sqrt(np.sum(image))
        hght = image.shape[0]*image[c0[1],c0[0]]
        params = lmfit.Parameters()
        params.add('xcb0', value = c0[0], min = c0[0]-2, max = c0[0]+2)
        params.add('ycb0', value = c0[1], min = c0[1]-2, max = c0[1]+2)
        params.add('sigb0', value = sigma, min = 0)
        params.add('hght0', value = hght, min = 0)
        ### For now have to add these to be compatible with my function
        ### Set vary = 0 since I don't want to fit them
        params.add('qb0', value = 1, vary = 0)
        params.add('PAb0', value = 0, vary = 0)
        args = (image,invar)
        def gauss2d_for_f2c(params,image,invar):
            iargs = (np.arange(image.shape[1]),np.arange(image.shape[0]),0)
            model = gauss2d_lmfit(params,*iargs)
#            plt.imshow(np.hstack((image,model,image-model)),interpolation='none')
#            plt.show()
#            plt.close()
            return np.ravel((image-model)**2*invar)
        results = lmfit.minimize(gauss2d_for_f2c,params,args=args)
        model = gauss2d_lmfit(results.params,np.arange(image.shape[1]),np.arange(invar.shape[0]),0)
#        plt.imshow(np.hstack((image,model,image-model)),interpolation='none')
#        plt.show()
#        plt.close()
        xc = results.params['xcb0']
        yc = results.params['ycb0']
        return [xc,yc]
    else:
        print("Invalid method")
        print("Choose quadratic or gaussian")
        exit(0)
        
def merge_close_peaks(peaks,min_space):
    """ If peaks x and y are both within min_space, call them one peak, average
        Peaks must be num_peaks x 2 array, col0 = xc, col1 = yc
    """
    if peaks.shape[0] <= 1:
        return peaks
    merge = np.zeros((peaks.shape[1]))
    merge_val = 1
    for i in range(peaks.shape[0]):
        trial_peak = peaks[i]
        delt_peaks = peaks-trial_peak
        merge_plus = False
        for j in range(i+1,peaks.shape[0]):
            if abs(delt_peaks[j,0]) < min_space and abs(delt_peaks[j,1]) < min_space:
                merge[i] = merge_val
                merge[j] = merge_val
                merge_plus = True
                print("Merging!")
        merge_val += merge_plus
    close_peaks = int(np.max(merge))
    unique_peaks = np.zeros((np.sum(merge==0) + close_peaks,2))
    if close_peaks > 0:
        for k in range(close_peaks):
            print peaks[merge==k+1]
            unique_peaks[k] = np.sum(peaks[merge==k+1],axis=0)/len(peaks[merge==k+1])
        unique_peaks[k+1:,0] = peaks[:,0][merge==0]
        unique_peaks[k+1:,1] = peaks[:,1][merge==0]
        return unique_peaks
    else:
        return peaks
        
def random_quadrupole_2d(x,r,Qxx,Qyy,Qxy):
    """ Highly specialized.  Attempt to make random draws follow a given
        quadrupole distribution.
    """
    return (2*Qxy*x*np.sqrt(r**2-x**2) + Qxx*x**2 + Qyy*(r**2-x**2))/r**5