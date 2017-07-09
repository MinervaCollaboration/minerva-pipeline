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
plt.rcParams['image.cmap'] = 'gray'
from matplotlib.colors import LinearSegmentedColormap
#import scipy
import scipy.stats as stats
import scipy.special as sp
import scipy.interpolate as si
import scipy.optimize as opt
import scipy.integrate as integrate
import lmfit
import scipy.sparse as sparse
#import scipy.signal as sig
#import scipy.linalg as linalg
#import astropy.stats as stats

#######################################################################
######################## START OF PDFS ################################
#######################################################################

def gaussian(axis, sigma, center=0, height=None,bg_mean=0,bg_slope=0,power=2):
    """Returns gaussian output values for a given input array, sigma, center,
       and height.  Center defaults to zero and height to 1.  Can also
       tweak the exponent parameter if desired."""
    #Height scale factor*(1/(sqrt(2*pi*sigma**2)))
#    print sigma, center, height
    if height is None:
        height = 1/(np.sqrt(2*np.pi*sigma))
    gaussian = height*exp(-abs(axis-center)**power/(2*(abs(sigma)**power)))+bg_mean+bg_slope*(axis-center)
#    print sigma, center, height, bg_mean,bg_slope,power
#    print gaussian
    return gaussian
    
def gaussian_2d(axisx,axisy,meanx,meany,sigmax,sigmay,height=1):
    norm = (1/(sigmax*sigmay*2*pi))
    xexp = -(1/2)*(((axisx-meanx)/sigmax)**2)
    xgrid = np.tile(xexp,[len(axisy),1])
    yexp = np.reshape(-(1/2)*(((axisy-meany)/sigmay)**2),[len(axisy),1])
    ygrid = np.tile(yexp,[1,len(axisx)])
    gauss2d = norm*exp(xgrid+ygrid)
    return gauss2d
    
def gaussian_lmfit(params, x, idx=0):
    """ normalized gaussian for lmfit
    """
    if type(params) == np.ndarray:
        mn = params[0]
        sigma = params[1]
    else:
        mn = params['c{}'.format(idx)].value
        sigma = params['s{}'.format(idx)].value
    return (1/(np.sqrt(2*np.pi*sigma**2)))*np.exp(-((x-mn)**2)/(2*sigma**2))
    
def gaussian_lmfit_trunc(params, x, idx=0):
    """ normalized gaussian for lmfit
    """
    if type(params) == np.ndarray:
        mn = params[0]
        sigma = params[1]
    else:
        mn = params['c{}'.format(idx)].value
        sigma = params['s{}'.format(idx)].value
    return trunc_gaussian(x, sigma, xc=mn, low=np.min(x), high = np.max(x)) 
    
def gaussian_2d_lmfit(params, x, y, idx=0):
    """ normalized gaussian for lmfit
    """
    sigmax = params['sigx{}'.format(idx)].value
    mnx = params['mnx{}'.format(idx)].value
    sigmay = params['sigy{}'.format(idx)].value
    mny = params['mny{}'.format(idx)].value
    norm = (1/(sigmax*sigmay*2*pi))
    return norm*np.exp(-((x-mnx)**2)/(2*sigmax**2))*np.exp(-((y-mny)**2)/(2*sigmay**2))
    
def gaussian_4d_lmfit(params, x, y, z, a, idx=0):
    """ normalized gaussian for lmfit
    """
    sigmax = params['sigx{}'.format(idx)].value
    mnx = params['mnx{}'.format(idx)].value
    sigmay = params['sigy{}'.format(idx)].value
    mny = params['mny{}'.format(idx)].value
    sigmaz = params['sigz{}'.format(idx)].value
    mnz = params['mnz{}'.format(idx)].value
    sigmaa = params['siga{}'.format(idx)].value
    mna = params['mna{}'.format(idx)].value
    norm = (1/(sigmax*sigmay*2*pi))*(1/(sigmaz*sigmaa*2*pi))
    return norm*np.exp(-((x-mnx)**2)/(2*sigmax**2))*np.exp(-((y-mny)**2)/(2*sigmay**2))*np.exp(-((z-mnz)**2)/(2*sigmaz**2))*np.exp(-((a-mna)**2)/(2*sigmaa**2))
    
def trunc_gaussian(x, sig, xc=0, low=0, high=np.inf):
    err = 0.5*(sp.erf((high-xc)/(np.sqrt(2)*sig))-sp.erf((low-xc)/(np.sqrt(2)*sig)))
    return gaussian(x, sig, center=xc, height = 1/(np.sqrt(2*pi)*sig))/err
    
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
        
    
def sine(theta):
    ### sine pdf
    rtn = np.sin(theta)
    rtn[theta<0] = 0
    rtn[theta>np.pi/2] = 0
    return rtn    
    
def inv_power(x,xc,n):
    """ y = 1/(x-xc)**n
        Normalized by default on the interval [x0, xn]
    """
    if n == 1:
        A = 1/(np.log((x[-1]-xc)/(x[0]-xc)))
    elif n == 0:
        A = (x[-1]-x[0])/len(x)
    else:
        A = 1/(-(1/(n-1))*(1/(x[-1]-xc)**(n-1) - 1/(x[0]-xc)**(n-1)))
    return A/(x-xc)**n
    
def inv_power_lmfit(params, x, idx=0):
    """ Simple inverse power law (x-cp)**(-pwr), not normalized
        allow offset of zero/inf point
    """
    pwr = params['pwr{}'.format(idx)].value
    cp = params['cp{}'.format(idx)].value
    x -= cp
    x[x==0] = 0.0001 # To remove inf behavior
#    print "Params:", pwr, cp
    x = abs(x) ### make always positive...
    return x**(-pwr)
    
def inv_power_int(xc,n,low,high,low0,high0):
    """ Integral, low0 and high0 set bounds for pdf, low, high within pdf.
        Must have low0<low<high<high0 (reverse l/h signs technically okay)
    """
    if n == 1:
        A = 1/(np.log((high0-xc)/(low0-xc)))
        return A*(np.log((high-xc)/(low-xc)))
    elif n == 0:
        return (high0-low0)/(high-low)
    else:
        A = -(1/(n-1))*(1/(high0-xc)**(n-1) - 1/(low0-xc)**(n-1))
        return -(A/(n-1))*(1/(high-xc)**(n-1) - 1/(low-xc)**(n-1))
    
def cauchy(x,xc,a,lims=None):
    """ Truncated Cauchy Distribution
        Normalized by construction on interval [low, high]
    """
    if lims is None:
        denom = 1
    else:
        denom = (1/np.pi)*(np.arctan((lims[1]-xc)/a) - np.arctan((lims[0]-xc)/a))
    num = 1/(np.pi*a*(1+((x-xc)/a)**2))
    return num/denom
#    A = 1/(a*(np.arctan((high-xc)/a)-np.arctan((low-xc)/a)))
#    return A*(1/(1+((x-xc)/a)**2))
    
def cauchy_lmfit(params, x, idx=0):
    """ For use with lmfit params
        This function is not normalized!
    """
    if type(params) == np.ndarray:
        xc = params[0]
        a = params[1]
    else:
        xc = params['c{}'.format(idx)].value
        a = params['s{}'.format(idx)].value
    return 1/(np.pi*a*(1+((x-xc)/a)**2))
    
def cauchy_lmfit_trunc(params, x, idx=0):
    """ For use with lmfit params
        This function is not normalized!
    """
    if type(params) == np.ndarray:
        xc = params[0]
        a = params[1]
    else:
        xc = params['c{}'.format(idx)].value
        a = params['s{}'.format(idx)].value
    ### Hack to fit Qgal in wider range
    if idx == 999:
        xl = 0.1
    else:
        xl = np.min(x)
#    xarr = np.linspace(0.1, 0.9, 100)
#    cf = cauchy(xarr, xc, a, lims=[xl, np.max(x)])
#    plt.hist(x.T, bins=9, normed=True)
#    plt.plot(xarr,cf,'k',linewidth=2)
#    plt.show()
#    plt.close()
    return cauchy(x, xc, a, lims=[xl, np.max(x)])
    
def cauchy_2d(x,y,xc,yc,a1,a2,gam,low=0,high=np.inf):
    """ Not normalized...
    """
    return 1/(1+((x-xc)/a1)**2+((y-yc)/a2)**2)**(gam/2)
    
def cauchy_nd(X, c0, s0, c1, s1, c2, s2, gam0):
    if X.dtype == tuple:
        X = np.asarray(X)
    ndim = X.shape[0]
    ndim = 3
    denom = (1+((X[0]-c0)/s0)**2+((X[1]-c1)/s1)**2+((X[2]-c2)/s2)**2)*gam
##    print X[0].shape
#    denom = np.ones(X[0].shape)
##    print denom.shape
#    idx = range(ndim)
##    if idx is None:
##        idx = range(ndim)
##    else:
##        idx = idx*np.ones((ndim),dtype=int)
#    for i in range(ndim):
#        denom += ((X[i]-params['c{}'.format(idx[i])].value) / params['s{}'.format(idx[i])].value)**2
#    denom *= params['gam0'].value
    if np.sum(denom==0) > 0:
        print("Cannot divide by zero")
        exit(0)
    return 1/denom    
    
def cauchy_nd_lmfit(params, X, idx=None):
    """ For use in PDF fitting.
        params is an lmfit object (lables must match below)
        c#, s# (center, scale for each parameter)
        X in an n by m array (or tuple of arrays)
        This is not normalized - do analytically by integrating
        Added an envelope scale factor
    """
    if X.dtype == tuple:
        X = np.asarray(X)
    ndim = X.shape[0]
    ndim = 3
#    print X[0].shape
    denom = np.ones(X[0].shape)
#    print denom.shape
    idx = range(ndim)
#    if idx is None:
#        idx = range(ndim)
#    else:
#        idx = idx*np.ones((ndim),dtype=int)
    for i in range(ndim):
        denom += ((X[i]-params['c{}'.format(idx[i])].value) / params['s{}'.format(idx[i])].value)**2
    denom == denom**params['gam0'].value
    if np.sum(denom==0) > 0:
        print("Cannot divide by zero")
        exit(0)
    return 1/denom
    

def multi_mod_cauchy(params,X,low=0,high=np.inf):
    ''' Multi dimensional modified cauchy pdf.
        Form is P(x) = A/(1+((x-xc1)/a1)**2+((x-xc2)/a2)**2+...)**(n/2)
        A is determined numerically based upon provided upper and lower bounds
        params must be of form [xc1, a1, xc2, a2, ..., n]
        if len(params) is even, n is assumed to be 2.
        X is a 2d array, # rows = num components
        Order is arbitrary, but X row must be paired with correct params
    '''
    if len(params)%2 == 1:
        n = params[-1]
    else:
        n = 2
    components = int(np.floor(len(params)/2))
    denom = np.ones((X[0].shape))
    for i in range(components):
        denom += ((X[i] - params[2*i])/params[2*i+1])**2
    denom = denom**(n/2)
    pdf = 1/denom
    ### simple, less accurate integral
    pdf /= np.sum(pdf)
    return pdf 
    
def cauchy_int(xc, a, low, high, low0=0, high0=np.inf):
    """ Integral of inv_xsq_power of given parameters (and limits)
        optional inputs low0 and high0 set the range of the pdf.
    """
    A = 1/(a*(np.arctan((high0-xc)/a)-np.arctan((low0-xc)/a)))
    return A*(a*(np.arctan((high-xc)/a)-np.arctan((low-xc)/a)))
    
def cauchy_residual(params,x,y,inv):
    """ For use with lmfit
    """
    xc = params['xc'].value
    a = params['a'].value
    try:
        h = params['h'].value
    except:
        h = 1
    return (h*cauchy(x,xc,a)-y)*np.sqrt(inv)
    
def weibull(x,k,lam):
    """ Weibull distribution
    """
    x[x==0] = 0.0001 #Hack to prevent infinity at zero...
    factor = (x/lam)**(k-1)
    return (k/lam)*factor*np.exp(-(x/lam)**k)
    
def weibull_lmfit(params, x, idx=0):
    """ Weibull distribution with lmfit.Parameters() object as input
    """
    if type(x) == np.ndarray:
        x[x<=0] == 0.0001
    else:
        if x <= 0:
            x = 0.0001
    if type(params) == np.ndarray:
        k = params[0]
        lam = params[1]
    else:
        k = params['s{}'.format(idx)].value
        lam = params['c{}'.format(idx)].value
#        print "using lmfit for sf.weibull!"
#        exit(0)
#        k = params['k{}'.format(idx)].value
#        lam = params['lam{}'.format(idx)].value
    return weibull(x, k, lam)
    
def weibull_residual(params,x,y,inv):
    """ For use with lmfit
    """
    k = params['k'].value
    lam = params['lam'].value
    try:
        h = params['h'].value
    except:
        h = 1
    return (h*weibull(x,k,lam)-y)*np.sqrt(inv)
    
def exponential_lmfit(params, x, idx=0):
    """ with lmfit...
    """
    tau = params['tau{}'.format(idx)].value
    return tau*np.exp(-x/tau)
    
def exponential_residual(params,x,y,inv):
    """ For use with lmfit
    """
    tau = params['tau'].value
    try:
        h = params['h'].value
    except:
        h = 1
    try:
        xc = params['xc'].value
    except:
        xc = 0
    return (h*np.exp(-(x-xc)/tau)-y)*np.sqrt(inv)

def sine_draw(n=1):
    """ Draws n variables from sin(x), returns on interval [0, 1]
    """
    return np.arccos(np.random.rand(n))/(np.pi)
    
def cos_draw(n=1):
    """ Draws n variables from cos(x), returns on interval [-1, 1]
    """
    return np.arcsin(2*np.random.rand(n)-1)/(np.pi/2)

### Super specific PDFs for fitting joint parameter distributions in
### LAE source determination (master.py, source_utils.py)

def exponential_ie_lmfit(params, X, idx=0):
    if X.shape[0] != 3:
        print "You're not giving this function the right input!"
        exit(0)
#    re = X[0]
    ie = X[1]
    nsers = X[2]
    taui = params['taui'].value
    tausn = params['tausn'].value
    tau = taui + tausn*nsers
#    plt.plot(nsers,tau,'k.')
#    plt.show()
#    plt.close()
    return tau*np.exp(-ie/tau)
#    return 0.2*np.exp(-ie/0.2)
    
def gaussian_re_lmfit(params, X, idx=0):
    if X.shape[0] != 3:
        print "You're not giving this function the right input!"
        exit(0)
    re = X[0]
#    ie = X[1]
#    nsers = X[2]
    mnr = params['mnr'].value
    sigr = params['sigr'].value
    return (1/np.sqrt(2*np.pi*sigr**2))*np.exp(-(re-mnr)**2/(2*sigr**2))
    
def cauchy_re_lmfit(params, X, idx=0):
    if X.shape[0] != 3:
        print "You're not giving this function the right input!"
        exit(0)
    re = X[0]
#    ie = X[1]
#    nsers = X[2]
    xc = params['mnr'].value
    a = params['sigr'].value
    return 1/(np.pi*a*(1+((re-xc)/a)**2))
    
def gaussian_nsers_lmfit(params, X, idx=0):
    if X.shape[0] != 3:
        print "You're not giving this function the right input!"
        exit(0)
    re = X[0]
#    ie = X[1]
    nsers = X[2]
    mnn = params['mnn'].value
    mnsr = params['mnsr'].value
    sign = params['sign'].value
    sigsr = params['sigsr'].value
    mn = mnn + mnsr*re
    sig = sign + sigsr*re
    return (1/np.sqrt(2*np.pi*sig**2))*np.exp(-(nsers-mn)**2/(2*sig**2))
    
def weibull_nsers_lmfit(params, X, idx=0):
    if X.shape[0] != 3:
        print "You're not giving this function the right input!"
        exit(0)
    re = X[0]
#    ie = X[1]
    nsers = X[2]
    lamn = params['lamn'] .value
    lamsr = params['lamsr'] .value
    kn = params['kn'] .value
    ksr = params['ksr'] .value
    lam = lamn + lamsr*re
    k = kn + ksr*re
    return weibull(nsers, k, lam)
    
def gauss3_irn(params, X, idx=0):
    if X.shape[0] != 3:
        print "You're not giving this function the right input!"
        exit(0)
    re0 = X[0]
    ie0 = X[1]
    nsers0 = X[2]
    if type(params) == np.ndarray:
        mnr = params[0]
        mni = params[1]
        mnn = params[2]
        scr = params[3]
        sci = params[4]
        scn = params[5]
        th1 = params[6]
        th2 = params[7]
        th3 = params[8]
    else:
        mnr = params['mnr'].value
        mni = params['mni'].value
        mnn = params['mnn'].value
        scr = params['scr'].value
        sci = params['sci'].value
        scn = params['scn'].value
        th1 = params['th1'].value
        th2 = params['th2'].value
        th3 = params['th3'].value
        #gam = params['gam'].value
    ### my best guess at 3d rotations...
    ### Zero-mean everything
    re = re0 - mnr
    ie = ie0 - mni
    nsers = nsers0 - mnn
    ### two rotations
#    re_pr = re*np.cos(th2) + ie*np.sin(th1)*np.sin(th2) + nsers*np.cos(th1)*np.sin(th2)
#    ie_pr = ie*np.cos(th1) - nsers*np.sin(th1)
#    ns_pr = -re*np.sin(th2) + ie*np.cos(th2)*np.sin(th1) + nsers*np.cos(th1)*np.cos(th2)
    ### Three rotations    
    re_pr = re*np.cos(th2)*np.cos(th3) + ie*(np.cos(th3)*np.sin(th1)*np.sin(th2) - np.cos(th1)*np.sin(th3)) + nsers*(np.cos(th3)*np.cos(th1)*np.sin(th2) + np.sin(th3)*np.sin(th1))
    ie_pr = re*np.sin(th3)*np.cos(th2) + ie*(np.cos(th1)*np.cos(th3) + np.sin(th1)*np.sin(th2)*np.sin(th3)) + nsers*(np.cos(th1)*np.sin(th2)*np.sin(th3)-np.cos(th3)*np.sin(th1))
    ns_pr = -re*np.sin(th2) + ie*np.cos(th2)*np.sin(th1) + nsers*np.cos(th1)*np.cos(th2)
#    re_pr = Xn[0]
#    ie_pr = Xn[0]
#    ns_pr = Xn[0]
#    Nr = gaussian(re_pr, scr, center=mnr)
#    Ni = gaussian(ie_pr, sci, center=mni)
#    Nn = gaussian(ns_pr, scn, center=mnn)
    Nr = cauchy(re_pr, 0, scr, lims=[np.min(re_pr), np.max(re_pr)])
    Ni = cauchy(ie_pr, 0, sci, lims=[np.min(ie_pr), np.max(ie_pr)])
    Nn = cauchy(ns_pr, 0, scn, lims=[np.min(ns_pr), np.max(ns_pr)])
    return (Nr*Ni*Nn)
    
def cauchy3_irq(params, X, idx=0, lims=None, normalized=True):
    if X.shape[0] != 3:
        print "You're not giving this function the right input!"
        exit(0)
    re0 = X[0]
    ie0 = X[1]
    q0 = X[2]
    if type(params) == np.ndarray:
        mnr = params[0]
        mni = params[1]
        mnq = params[2]
        scr = params[3]
        sci = params[4]
        scq = params[5]
        th1 = params[6]
        th2 = params[7]
        th3 = params[8]
    else:
        mnr = params['mnr'].value
        mni = params['mni'].value
        mnq = params['mnq'].value
        scr = params['scr'].value
        sci = params['sci'].value
        scq = params['scq'].value
        th1 = params['th1'].value
        th2 = params['th2'].value
        th3 = params['th3'].value
        #gam = params['gam'].value
    ### my best guess at 3d rotations...
    ### Zero-mean everything
    re = re0 - mnr
    ie = ie0 - mni
    qs = q0 - mnq
    ### two rotations
#    re_pr = re*np.cos(th2) + ie*np.sin(th1)*np.sin(th2) + nsers*np.cos(th1)*np.sin(th2)
#    ie_pr = ie*np.cos(th1) - nsers*np.sin(th1)
#    ns_pr = -re*np.sin(th2) + ie*np.cos(th2)*np.sin(th1) + nsers*np.cos(th1)*np.cos(th2)
    ### Three rotations    
    re_pr = re*np.cos(th2)*np.cos(th3) + ie*(np.cos(th3)*np.sin(th1)*np.sin(th2) - np.cos(th1)*np.sin(th3)) + qs*(np.cos(th3)*np.cos(th1)*np.sin(th2) + np.sin(th3)*np.sin(th1))
    ie_pr = re*np.sin(th3)*np.cos(th2) + ie*(np.cos(th1)*np.cos(th3) + np.sin(th1)*np.sin(th2)*np.sin(th3)) + qs*(np.cos(th1)*np.sin(th2)*np.sin(th3)-np.cos(th3)*np.sin(th1))
    qs_pr = -re*np.sin(th2) + ie*np.cos(th2)*np.sin(th1) + qs*np.cos(th1)*np.cos(th2)
#    re_pr = Xn[0]
#    ie_pr = Xn[0]
#    ns_pr = Xn[0]
#    Nr = gaussian(re_pr, scr, center=mnr)
#    Ni = gaussian(ie_pr, sci, center=mni)
#    Nn = gaussian(ns_pr, scn, center=mnn)
    if lims is None:
        limsr = [np.min(re_pr), np.max(re_pr)]
        limsi = [np.min(ie_pr), np.max(ie_pr)]
        limsq = [np.min(qs_pr), np.max(qs_pr)]
    else:
        limsr = lims[0]
        limsi = lims[1]
        limsq = lims[2]
    if normalized:
#        Nr = trunc_gaussian(re_pr, scr, xc=0, low=limsr[0], high=limsr[1])
        Nr = cauchy(re_pr, 0, scr, lims=limsr)
        Ni = cauchy(ie_pr, 0, sci, lims=limsi)
        Nq = cauchy(qs_pr, 0, scq, lims=limsq)
    else:
#        Nr = gaussian(re_pr, scr)
        Nr = cauchy(re_pr, 0, scr)
        Ni = cauchy(ie_pr, 0, sci)
        Nq = cauchy(qs_pr, 0, scq)
    return (Nr*Ni*Nq)
    
def gen_central2d(params, X, idx=0, lims=None, rots=False, normalized=False, dists=['gaussian','cauchy']):#dists=['gaussian', 'gaussian']):
    """ For any central 2d profile, where both params have a center and scale
        Fits for a rotation, centers and scales.
        Each param can have a different distribution.
        Choices are gaussian and cauchy (might add voigt - convolution of
        the two, but needs one extra parameter)
    """
    if X.shape[0] != 2:
        print "You're not giving gen_central2d the right input!"
        exit(0)
    if idx == 0:
        idx = [0, 1, 0]
    else:
        try:
            if len(idx) != 3:
                raise TypeError
        except:
            print "idx must be a len=3 list"
            exit(0)
    x = X[0]
    y = X[1]
    if type(params) == np.ndarray:
        if len(params) == 5:
            xo = params[0]
            yo = params[1]
            sx = params[2]
            sy = params[3]
            th = params[4]
        elif len(params) == 2:
            xo = 0
            yo = 0
            sx = params[0]
            sy = params[1]
            th = 0
    else:
        idx1, idx2, idx3 = idx
        xo = params['x{}'.format(idx1)].value
        yo = params['x{}'.format(idx2)].value
        sx = params['s{}'.format(idx1)].value
        sy = params['s{}'.format(idx2)].value
        th = params['th{}'.format(idx3)].value
#    if rots:
    ## boolean to rotate around centers (gaussian, cauchy) or 0 (weibull)
    if len(dists) == 1 and dists[0] == 'lorentz':
        return lorentz(X[0], X[1], params)
    else:
        xc, yc, q, PA, sig = params ### assumes this form
        cts = [dists[0] != 'weibull', dists[1] != 'weibull']
        xpr = (x-xo*cts[0])*np.cos(th) + (y-yo*cts[1])*np.sin(th)
        ypr = (y-yo*cts[1])*np.cos(th) - (x-xo*cts[0])*np.sin(th)
        if dists[0] == 'gaussian':
            xdist = np.exp(-(xpr**2/(2*sx**2)))
        elif dists[0] == 'cauchy':
            xdist = 1/(1+(xpr/sx)**2)
        elif dists[0] == 'weibull':
            if np.min(xpr) < 0:
                print "Weibull distribution only good for positive inputs!"
                exit(0)
            if xo == 0:
                xo = 1
            ### set k = sx, lamda = xo
            xdist = (sx/xo)*(xpr/xo)**(sx-1)*np.exp(-(xpr/xo)**(sx))
        if dists[1] == 'gaussian':
            ydist = np.exp(-(ypr**2/(2*sy**2)))
        elif dists[1] == 'cauchy':
            ydist = 1/(1+(ypr/sy)**2)
        elif dists[1] == 'weibull':
            if np.min(ypr) < 0:
                print "Weibull distribution only good for positive inputs!"
                exit(0)
            if yo == 0:
                yo = 1
            ### set k = sx, lamda = xo
            ydist = (sy/yo)*(ypr/yo)**(sy-1)*np.exp(-(ypr/yo)**(sy))
        return xdist*ydist
#    return np.exp(-(xpr**2/(2*sx**2) + ypr**2/(2*sy**2)))
#    else:
#        a = np.cos(th)**2/(2*sx**2) + np.sin(th)**2/(2*sy**2)
#        b = -np.sin(2*th)/(4*sx**2) + np.sin(2*th)/(4*sy**2)
#        c = np.sin(th)**2/(2*sx**2) + np.cos(th)**2/(2*sy**2)
#        return np.exp(-(a*(x-xo)**2-2*b*(x-xo)*(y-yo)+c*(y-yo)**2))

def gen_central3d(params, X, idx=0, lims=None, normalized=False, dists=['cauchy','gaussian','gaussian']):    
    if X.shape[0] != 3:
        print "You're not giving this function the right input!"
        exit(0)
    re0 = X[0]
    ie0 = X[1]
    q0 = X[2]
    if type(params) == np.ndarray:
        mnr = params[0]
        mni = params[1]
        mnq = params[2]
        scr = params[3]
        sci = params[4]
        scq = params[5]
        th1 = params[6]
        th2 = params[7]
        th3 = params[8]
    else:
        mnr = params['mnr'].value
        mni = params['mni'].value
        mnq = params['mnq'].value
        scr = params['scr'].value
        sci = params['sci'].value
        scq = params['scq'].value
        th1 = params['th1'].value
        th2 = params['th2'].value
        th3 = params['th3'].value
        #gam = params['gam'].value
    ### Zero-mean everything
    re = re0 - mnr
    ie = ie0 - mni
    qs = q0 - mnq
    ### Three rotations    
    re_pr = re*np.cos(th2)*np.cos(th3) + ie*(np.cos(th3)*np.sin(th1)*np.sin(th2) - np.cos(th1)*np.sin(th3)) + qs*(np.cos(th3)*np.cos(th1)*np.sin(th2) + np.sin(th3)*np.sin(th1))
    ie_pr = re*np.sin(th3)*np.cos(th2) + ie*(np.cos(th1)*np.cos(th3) + np.sin(th1)*np.sin(th2)*np.sin(th3)) + qs*(np.cos(th1)*np.sin(th2)*np.sin(th3)-np.cos(th3)*np.sin(th1))
    qs_pr = -re*np.sin(th2) + ie*np.cos(th2)*np.sin(th1) + qs*np.cos(th1)*np.cos(th2)
    if lims is None:
        limsr = [np.min(re_pr), np.max(re_pr)]
        limsi = [np.min(ie_pr), np.max(ie_pr)]
        limsq = [np.min(qs_pr), np.max(qs_pr)]
    else:
        limsr = lims[0]
        limsi = lims[1]
        limsq = lims[2]
    if normalized:
        print "Haven't programmed for this yet"
        exit(0)
#        Nr = trunc_gaussian(re_pr, scr, xc=0, low=limsr[0], high=limsr[1])
        Nr = cauchy(re_pr, 0, scr, lims=limsr)
        Ni = cauchy(ie_pr, 0, sci, lims=limsi)
        Nq = cauchy(qs_pr, 0, scq, lims=limsq)
    else:
        if dists[0] == 'gaussian':
            Nr = gaussian(re_pr, scr)
        elif dists[0] == 'cauchy':
            Nr = cauchy(re_pr, 0, scr)
        if dists[1] == 'gaussian':
            Ni = gaussian(ie_pr, sci)
        elif dists[1] == 'cauchy':
            Ni = cauchy(ie_pr, 0, sci)
        if dists[2] == 'gaussian':
            Nq = gaussian(qs_pr, scq)
        elif dists[2] == 'cauchy':
            Nq = cauchy(qs_pr, 0, scq)  
    return (Nr*Ni*Nq)
#######################################################################
######################### END OF PDFS #################################
#######################################################################


#######################################################################
################# Special (mostly astro) functions ####################
#######################################################################
    
def moffat_lmfit(params, xarr):
    alpha = params['alpha'].value
    beta = params['beta'].value
    try:
        xc = params['xc'].value
    except:
        xc = 0
    try:
        bg = params['bg'].value
    except:
        bg = 0
    try:
        hght = params['hght'].value
    except:
        hght = np.sqrt((beta-1)/(np.pi*alpha**2))
    try:
        power = params['power'].value
    except:
        power = 2
    return hght*(1+abs(xarr-xc)**power/alpha**2)**(-beta) + bg
    
def schechter_fct(L,L_star,Phi_star,alpha):
    """ To match eqn. 1 of Guo, 2015 (CANDELS)
    """
    N_L = Phi_star*(L/L_star)**alpha*np.exp(-L/L_star)
    return N_L   
   
def sersic1d_old(rarr,Ie,re,n):
    bn = 0.868*n-0.142
    I_r = Ie*10**(-bn*((rarr/re)**(1/n)-1))
    return I_r
    
def sersic1d(rarr,A,sig,n):
    	if n >= 0.36: # from Ciotti & Bertin 1999, truncated to n^-3
		k=2.0*n-1./3+4./(405.*n)+46./(25515.*n**2.)+131./(1148175.*n**3.)
	else: # from MacArthur et al. 2003
		k=0.01945-0.8902*n+10.95*n**2.-19.67*n**3.+13.43*n**4.
	return A*np.exp(-k*(rarr/sig)**(1./n))

def sersic2d_old(x,y,xc,yc,Ie,re,n,q=1,PA=0):
    """ makes a 2D image (dimx x dimy) of a Sersic profile centered at [xc,yc]
        and parameters Ie, re, and n.
        Optionally can add in ellipticity with axis ratio (q) and position
        angle (PA).
    """
    rarr = make_rarr(x,y,xc,yc,q=q,PA=PA)
    image = sersic1d(rarr,Ie,re,n)
    return image
    
def sersic2d(x,y,xc,yc,A,sig,n,q=1,PA=0):
    rarr = make_rarr(x,y,xc,yc,q=q,PA=PA)
    return sersic1d(rarr,A,sig,n)
    
def sersic2d_lmfit(params,x,y,i,ab=False):
    """ makes a 2D image (dimx x dimy) of a Sersic profile with parameters
        in lmfit object params (xc, yc, Ie, re, n, q, PA)
    """
    xc = params['xcb{}'.format(i)].value
    yc = params['ycb{}'.format(i)].value
    PA = params['PAb{}'.format(i)].value
    Ie = params['Ieb{}'.format(i)].value
    n = params['nb{}'.format(i)].value
    if not ab:
        q = params['qb{}'.format(i)].value    
        re = params['reb{}'.format(i)].value
    else:
        aa = params['ab{}'.format(i)].value
        bb = params['bb{}'.format(i)].value
        q = bb/aa
        re = np.sqrt(aa*bb)
    rarr = make_rarr(x,y,xc,yc,q=q,PA=PA)
    image = sersic1d(rarr,Ie,re,n)
    return image

def eff(rarr, Io, a, gam=3):
    """ Elson, Fall, and Freeman, 1987 (see, ex. Schweitzer, Luminosity 
        Profiles of Resolved Young Massive Clusters)
    """
    return Io*(1+(rarr/a)**2)**(-gam/2)
    
def eff2d(x, y, xc, yc, Io, scl, gam=3, q=1, PA=0):
    """ 2D EFF image with input parameters (gamma = 3 default)
    """
    rarr = make_rarr(x,y,xc,yc,q=q,PA=PA)
    image = eff(rarr,Io,scl,gam)
    return image    
    
def eff2d_lmfit(params, x, y, i, ab=False):
    """ Makes a 2D image using the EFF profile and specified parameters
    """
    xc = params['xcb{}'.format(i)].value
    yc = params['ycb{}'.format(i)].value
    PA = params['PAb{}'.format(i)].value
    Io = params['Ieb{}'.format(i)].value
    gam = params['nb{}'.format(i)].value
    if not ab:
        q = params['qb{}'.format(i)].value    
        scl = params['reb{}'.format(i)].value
    else:
        aa = params['ab{}'.format(i)].value
        bb = params['bb{}'.format(i)].value
        q = bb/aa
        scl = np.sqrt(aa*bb)
    rarr = make_rarr(x,y,xc,yc,q=q,PA=PA)
    image = eff(rarr,Io,scl,gam)
    return image
    
  
##############################################################################
################### Various other functions... ###############################
##############################################################################
  
def gauss_residual(params,xvals,zvals,invals):
    """ Residual function for gaussian with constant background, for use with
        lmfit.minimize()
    """
    sigma = params['sigma'].value
    mn = params['mean'].value
    try:
        hght = params['hght'].value
    except:
        hght = 1
    try:
        bg = params['bg'].value
    except:
        bg = 0
    try:
        power = params['power'].value
    except:
        power = 2
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
    if min(N.shape) == 1 or N.ndim == 1:
        N = np.diag(N)
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
        chi_min = np.asarray(chi_min)[0]
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

def best_mean(axis,sig,mn,hght,bg,data,invar,spread,power=2,beta=2,profile='gaussian'):
    """ Finds the mean that minimizes squared sum of weighted residuals
        for a given sigma and height (guassian profile)
        Uses a grid-search, then fits around minimum chi^2 regionS
    """
    def mn_find(axis,mn,spread, params=None, profile='gaussian'):
        """ Call with varying sig ranges for coarse vs. fine.
        """
        mn_rng = np.linspace(mn-spread,mn+spread,10)
        chis = np.zeros(len(mn_rng))
        ### Find chi^2 at a range of means aroung the best guess
        for i in range(len(mn_rng)):
            if profile == 'gaussian':
                gguess = gaussian(axis,sig,mn_rng[i],hght,bg,power=power)
            elif profile == 'moffat':
                gguess = moffat_lmfit(params, axis)
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
        
    if profile == 'gaussian':
        ###Coarse find
        ###Tradeoff with coarse sig - too small and initial guess is critical,
        ### too big and more susceptible to comsic ray influence
        best_mn, best_mn_std = mn_find(axis,mn,spread)
        ###Fine find
        best_mn, best_mn_std = mn_find(axis,best_mn,best_mn_std)
        ###Finer find - doesn't seem to change final answer at all
    #    best_mn, best_mn_std = mn_find(axis,best_mn,best_mn_std/100)
    elif profile == 'moffat':
        params0 = lmfit.Parameters()
        params0.add('xc', value = mn)
        params0.add('alpha', value = sig)
        params0.add('beta', value = beta)
        params0.add('bg', value = bg)
        params0.add('power', value = power)
        params0.add('hght', value = hght)
        ###Coarse find
        ###Tradeoff with coarse sig - too small and initial guess is critical,
        ### too big and more susceptible to comsic ray influence
        best_mn, best_mn_std = mn_find(axis,mn,spread,params=params0,profile=profile)
        ###Fine find
        best_mn, best_mn_std = mn_find(axis,best_mn,best_mn_std,params=params0,profile=profile)
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
    gauss2d = hght*np.exp(-(1.0/2)*(xygrid/(sig_sq)))
#    plt.imshow(gauss2d,interpolation='none')
#    plt.show()
#    plt.close()
    return gauss2d
 
def gauss2d_lmfit(params, xaxis, yaxis, cnt, ab=False):
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
    hght = params['hght{}'.format(cnt)].value
    PA = params['PAb{}'.format(cnt)].value
    if not ab:
        q = params['qb{}'.format(i)].value    
        sig = params['sigb{}'.format(i)].value
    else:
        aa = params['ab{}'.format(i)].value
        bb = params['bb{}'.format(i)].value
        q = bb/aa
        sig = np.sqrt(aa*bb)
    sig_sq = sig*sig
    xygrid = make_rarr(xaxis, yaxis, xc, yc, q=q, PA=PA)
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
        hsum = np.zeros(axis.shape)
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
    
def gauss_herm2d(xarr, yarr, params, weights, mx_ord=4, skip_ords=[(0,1),(1,0),(0,2),(2,0)], ord2diff=True, return_profile=False, empirical_norm=True):
    """ 2D gauss-hermite polynomials using lmfit.Parameters() object or array
        Maximum hermite order can be adjusted (m+n <= mx_ord)
        Also allow order to be skipped - by default skip corrections to center, std
        Can include std x/y asymmetry with ord2diff=True
    """
    if type(params) == np.ndarray:
        ### Order must match below
        xc, yc = params[0], params[1]
        sig = params[2]
        try:
            power = params[3]
        except:
            power = 2
    elif type(params) == lmfit.parameter.Parameters:
        ### Naming must match below
        xc, yc = params['xc'].value, params['yc'].value
        sig = params['sig'].value
        try:
            power = params['power'].value
        except:
            power = 2
    else:
        print "Invalid parameter input format to 'gauss_herm2d'"
        print "Must be numpy array or lmfit.Parameters() object"
        exit(0)
    xgrid, ygrid = np.meshgrid(xarr-xc, yarr-yc)
    num_ords = int(0.5*(mx_ord+1)*(mx_ord+2) - len(skip_ords) + ord2diff)
    gh_matrix = np.zeros((num_ords,)+xgrid.shape)
    ord_idx = 0
    G = np.exp(-0.5*(abs(xgrid)**power + abs(ygrid)**power)/sig**power)
    if empirical_norm:
        norm = np.sum(G)*np.ediff1d(xarr)[0]*np.ediff1d(yarr)[0]
    else:
        norm = (2*np.pi*sig**2)
#    if power != 2:
#        norm = 1/((2/power)**2*(sig)**(1/power)*(2*np.pi)**(2/power))
#    print norm
    G /= norm
    for m in range(mx_ord+1):
        for n in range(mx_ord+1):
            if m+n > mx_ord:
                continue
            if (m,n) in skip_ords:
                continue
            Hx = hermite(m,xgrid,sig)
            Hy = hermite(n,ygrid,sig)
            gh_matrix[ord_idx] = Hx*Hy*G
            ord_idx += 1
    if ord2diff and (0,2) in skip_ords and (2,0) in skip_ords:
        ### Move (2,0) and (0,2) difference to 2nd position
        gh_matrix[2:] = gh_matrix[1:-1]
        Hx20 = hermite(2,xgrid,sig)
        Hx02 = hermite(0,xgrid,sig)
        Hy20 = hermite(0,ygrid,sig)
        Hy02 = hermite(2,ygrid,sig)
        gh20 = (Hx20*Hy20-Hx02*Hy02)*G
        gh_matrix[1] = gh20
    if return_profile:
        profile = np.zeros((xgrid.size, num_ords))
        for i in range(num_ords):
            profile[:,i] = np.ravel(gh_matrix[i])
        return profile
    else:
        if weights.ndim == 1:
            weights = weights.reshape((len(weights),1))
        if weights.size != num_ords:
            print "Weights matrix is wrong size {} for number of orders {}".format(weights.size,num_ords)
            exit(0)
        return np.dot(gh_matrix.T,weights).T[0]

def lorentz(xarr, yarr, params):
    xc, yc, q, PA, sig = params ### assumes this form
    xnew = (xarr-xc)*np.cos(PA) - (yarr-yc)*np.sin(PA)
    ynew = (yarr-yc)*np.cos(PA) + (xarr-xc)*np.sin(PA)
    rarr = np.sqrt(q*xnew**2 + ynew**2/q)
#    rarr = make_rarr(xarr, yarr, xc, yc, q, PA)
    return 1/(2*np.pi*sig**2)/(1+rarr**2/sig**2)**(1.5)

def lorentz_int(params, lims):
    xc, yc, q, PA, sig = params

def lorentz_for_ghl(data, params, norm, cpad):
    xc, yc = params['xc'].value, params['yc'].value
    sigl, ratio = abs(params['sigl'].value), params['ratio'].value
    hc = int(np.round(xc))
    vc = int(np.round(yc))
    xarr = np.arange(hc-cpad,hc+cpad+1)
    yarr = np.arange(vc-cpad,vc+cpad+1)
    xgrid, ygrid = np.meshgrid(xarr-xc, yarr-yc)
    return (1-ratio)*norm/(2*np.pi*sigl**2)/(1+(xgrid/sigl)**2 + (ygrid/sigl)**2)**(1.5)

def gh_lorentz(xarr, yarr, params, weights):
    """ Saves 2D Gauss-Hermite with Lorentz envelope
    """
    gh = gauss_herm2d(xarr, yarr, params, weights)
    xc, yc = params['xc'].value, params['yc'].value
    sigl, ratio = params['sigl'].value, params['ratio'].value
    xgrid, ygrid = np.meshgrid(xarr-xc, yarr-yc)
#    lorentz1 = sigl**2/(np.pi)*1/(1+(xgrid/sigl)**2)/(1+(ygrid/sigl)**2)*weights[0]
    lorentz = sigl**2/(2*np.pi)/(1+(xgrid/sigl)**2 + (ygrid/sigl)**2)**(1.5)*weights[0]
    return (ratio*gh + (1-ratio)*lorentz) + params['bg'].value

def ghl_nonlin_fit(data, invar, params, weights):
    def ghl_res(params, data, invar, weights):
        ghl = gh_lorentz(np.arange(data.shape[1]), np.arange(data.shape[0]), params, weights)
        return np.ravel((data-ghl)*np.sqrt(invar))
    args = (data, invar, weights)
    results = lmfit.minimize(ghl_res, params, args=args)
    return results.params
    
def ghl_linear_fit(data, invar, params, weights):
    ### find lorentz component and subtract from data
    xc, yc = params['xc'].value, params['yc'].value
    sigl, ratio = params['sigl'].value, params['ratio'].value
    xarr, yarr = np.arange(data.shape[1]), np.arange(data.shape[0])
    xgrid, ygrid = np.meshgrid(xarr-xc, yarr-yc)
#    lorentz = sigl**2/(np.pi)*1/(1+(xgrid/sigl)**2)/(1+(ygrid/sigl)**2)*weights[0]
    lorentz = sigl**2/(2*np.pi)/(1+(xgrid/sigl)**2 + (ygrid/sigl)**2)**(1.5)*weights[0]
    lin_data = data-(1-ratio)*lorentz - params['bg'].value
    profile = ratio*gauss_herm2d(xarr, yarr, params, weights, return_profile=True)
    init_model = np.dot(profile,weights)
    init_model = np.reshape(init_model,xgrid.shape)
    new_weights, chi = chi_fit(np.ravel(lin_data), profile, np.diag(np.ravel(invar)))
    return new_weights
        
    
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
    
def downsample(image,pxfact,extra='exclude'):
    """ Takes an image and downsamples it by pxfact (must be an int)
        For example, pxfact=2 means that a 2x2 square in image will be
        summed to form a single pixel in the downsampled image.
        extra='include' includes pixels on the edge, and assumes average
        pixel value for missing.  'exclude' just truncates shape
    """
    vdim, hdim = image.shape
    if extra=='exclude':
        vnew, hnew = int(np.floor(vdim/pxfact)), int(np.floor(hdim/pxfact))
    elif extra == 'include':
        vnew, hnew = int(np.ceil(vdim/pxfact)), int(np.ceil(hdim/pxfact))
    else:
        print("Invalid value for extra, must be 'include' or 'exclude'")
        exit(0)
    new_image = np.zeros((vnew, hnew))
    for i in range(vnew):
        for j in range(hnew):
            io = i*pxfact
            jo = j*pxfact
            ih = io+pxfact
            jh = jo+pxfact
            if extra == 'include' and (ih > vdim or jh > hdim):
                ip = min(ih, vdim)
                jp = min(jh, hdim)
                exi, exj = ih-ip, jh-jp
                imsub = np.sum(image[io:ip,jo:jp])*(pxfact**2/((pxfact-exi)*(pxfact-exj)))
            else:
                imsub = np.sum(image[io:ih,jo:jh])           
            new_image[i,j] = imsub
    return new_image
    
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
    if len(x.shape) == 2 and len(y.shape) == 2:
        x_matrix = x
        y_matrix = y
    else:
        x_matrix = np.tile(x,(len(y),1))
        y_matrix = np.tile(y,(len(x),1)).T
    x_ell = (y_matrix-yc)*np.cos(PA) + (x_matrix-xc)*np.sin(PA)
    y_ell = (x_matrix-xc)*np.cos(PA) - (y_matrix-yc)*np.sin(PA)
    r_matrix = np.sqrt((x_ell)**2*q + (y_ell)**2/q)
#    r_matrix = np.sqrt((x_ell)**2 + (y_ell)**2/q**2)
    return r_matrix
  
def cauchy_draw(n=1,params=None,lims=None):
    y = np.zeros(n)
    if params is None:
        yc = 0
        ys = 1
    elif type(params) == np.ndarray:
        yc = params[0]
        ys = params[1]
    else:
        yc = params['c0'].value
        ys = params['s0'].value
    for i in range(n):
        x = np.random.rand()
        if lims is None:
            y[i] = ys*np.tan(np.pi*(x-1/2)) + yc
        else:
            xl = 1/np.pi*np.arctan((lims[0]-yc)/ys)
            ytmp = ys*np.tan(np.pi*(x-xl)) + yc
            while ytmp < lims[0] or ytmp > lims[1]:
                x = np.random.rand()
                ytmp = ys*np.tan(np.pi*(x-xl)) + yc
            y[i] = ytmp
    return y

def gauss_draw(n=1, params=None, lims=None):
    y = np.zeros(n)
    if params is None:
        yc = 0
        ys = 1
    elif type(params) == np.ndarray:
        yc = params[0]
        ys = params[1]
    else:
        yc = params['c0'].value
        ys = params['s0'].value
    for i in range(n):
        ytmp = np.random.randn()*ys + yc
        if lims is None:
            y[i] = ytmp
        else:
            while ytmp < lims[0] or ytmp > lims[1]:
                ytmp = np.random.randn()*ys + yc
            y[i] = ytmp
    return y
    
def weibull_draw(n=1, params=None, lims=None):
    y = np.zeros(n)
    if params is None:
        lam = 1
        k = 1
    elif type(params) == np.ndarray:
        lam = params[0]
        k = params[1]
    else:
        lam = params['c0'].value
        k = params['s0'].value
    for i in range(n):
        ytmp = np.random.weibull(k)*lam
        if lims is None:
            y[i] = ytmp
        else:
            while ytmp < lims[0] or ytmp > lims[1]:
                ytmp = np.random.weibull(k)*lam
            y[i] = ytmp
    return y

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
    
def pdf_nd_grid(pdf, params, lims, mx_pnts = 10**6, bpd=1, save=False):
    """ Makes a grid of pdf values for pdf with given params between lims.
        Splits into extra bins per dimention (bpd) and sets the max of
        each to 1, helps speed up draws when a lot of the pdf space has
        a low probability of being drawn.
        lims is an n x 2 array with n = ndim and then the min/max for each
    """
    if type(lims) == np.ndarray:
        ndim = lims.shape[0]
    else:
        print("Input 'lims' must be an ndim x 2 array")
        exit(0)
    shrt_pnts = 100
    pnts_per_dim = min(int((mx_pnts)**(1/(ndim))),shrt_pnts)
    axis_arrays = []
    spacing = np.zeros((ndim))
    xaxes = np.zeros((ndim,pnts_per_dim))
    for i in range(ndim):
        ### The following assumes zero truncation...
        xaxes[i] = np.linspace(lims[i,0], lims[i,1], pnts_per_dim)
        axis_arrays.append(xaxes[i])
        spacing[i] = xaxes[i][1] - xaxes[i][0]
    mesh = np.meshgrid(*tuple(axis_arrays),indexing='ij')
    mesh = np.asarray(mesh)
#    if func[0:3].__name__ == 'sf.':
#        import special as sf
    grid = pdf(params, mesh)
    ### Zero any unphysical values
    grid[grid==np.inf] = 0
    grid[grid==np.nan] = 0
    grid /= np.max(grid)
    if bpd == 1:
        return grid, spacing
    else:
        print("I still need to add support for the bpd function")
#    nppd = int(pnts_per_dim/bpd)
#    grid_final = np.zeros((ndim*(ndim,) + ndim*(nppd,)))
#    for i in range(ndim):
#
    
def pdf_draw_nd(grid,lims,spacing,ndim,n=1,args=None):
    """ pdf draw from a grid search, split out by bins
    """
    accept = False
    while not accept:
        draw = np.random.rand(ndim+1)
        for i in range(ndim):
            draw[i] *= grid.shape[i]
            draw[i] = int(np.floor(draw[i]))
#            draw[i] *= lims[i,1]-lims[i,0]
#            draw[i] += lims[i,0]
#            draw[i] = int(np.floor(draw[i]/spacing[i]))
#            if draw[i] > grid.shape[i]:
#                draw[i] = grid.shape[i]-1
        inds = tuple(draw[0:ndim])
        val = grid[inds]
        if val >= draw[-1]:
            accept = True
    return draw[0:ndim]*spacing + lims[:,0]
    
def find_peaks2D(image, min_space=1, min_px=4, bg_thresh=None, min_amp=None, offset=[0,0]):
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
    
def array_to_Parameters(params , array, arraynames=None, minarray=None, maxarray=None, fixed=None):
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
    
def Parameters_to_array(params,arraynames=None):
    """ Converts lmfit.Parameters object to an array.
    """
    if arraynames is None:
        arraynames = params.keys()
    out_array = np.empty((len(arraynames)))
    idx = 0
    for nm in arraynames:
        out_array[idx] = params[nm].value
        idx += 1
    return out_array

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
    
def centroid(image,view_plot=False):
    """ calculates the pixel value weighted x and y centroid.
        returns xc, yc
    """ 
    A = np.sum(image)
    Mx = np.sum(np.arange(image.shape[1])*np.sum(image,axis=0))
    My = np.sum(np.arange(image.shape[0])*np.sum(image,axis=1))
    if view_plot:
        plt.figure("Evaluation of Centroid")
        plt.imshow(image,interpolation='none')
        plt.plot(Mx/A,My/A,'bo')
        plt.show()
        plt.close()
    return [Mx/A, My/A]
    
def kolmogorov(y1,y2):
    """ Runs a KS test on the two data sets (must be equal length).
        Pretty sure I'm not quite doing this right and will eventually try
        using R or something.
    """
    if len(y1) == len(y2):
        return np.max(abs(y1-y2))
    else:
        print("input arrays must be same length")
        exit(0)
        
def plot_circle(center,radius,q=1,PA=0):
    """ Plots a circle with defined center and radius (option to turn into
        an ellipse).  Center format is [xc,yc]
    """
    xc = center[0]
    yc = center[1]
    xvals = np.linspace(xc-radius,xc+radius,300)
    arc = np.sqrt(radius**2-(xvals-xc)**2)
    plt.plot(xvals,yc+arc,'r',linewidth=2)
    plt.plot(xvals,yc-arc,'r',linewidth=2)
    return
    
def plot_3D(image,x=None,y=None):
    """ Uses matplotlib plot_surface function.  Always generates its own figure.
        Option to include custom x, y axes, otherwise goes 0, 1, ..., xmax-1
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    if x is None:
        x = np.arange(image.shape[1])
    if y is None:
        y = np.arange(image.shape[0])
        y = y[::-1]
    X, Y = np.meshgrid(x,y)
    ### Try custom colormap, grayscale, but never goes fully white
    colors = [(0, 0, 0), (0.8, 0.8, 0.8)]
    graymap = LinearSegmentedColormap.from_list('graymap', colors, N=100)
    surf = ax.plot_surface(X,Y,image,rstride=1,cstride=1,cmap=graymap,linewidth=0)
    ax.set_zlim(np.min(image)-0.1*abs(np.min(image)),np.max(image)+0.1*abs(np.max(image)))
    return
    
def comoving_dist(z, z0=0, Om=0.3, Ol=0.7, Or=0, h=0.7, k=0):  
    """ Cosmological comovig distance.  Units are in pc
    """
    za = np.linspace(z0,z,1000)
    E = np.sqrt(Om*(1+za)**3 + Or*(1+za)**4 + Ol)
    Dh = (3*10**9)*h**(-1) #pc
    Dc = Dh * np.sum(1/E)*(za[1]-za[0])
    return Dc
    
    
def angular_dia_dist(z, z0=0, Om=0.3, Ol=0.7, h=0.7, k=0):
    """ Calculates angular diameter distance for a given cosmology.
        Units are in pc
    """
    Da = comoving_dist(z, z0=z0, Om=Om, Ol=Ol, h=h)*(1+z0)/(1+z)
    return Da
    
def luminosity_dist(z, z0=0, Om=0.3, Ol=0.7, h=0.7, k=0):
    """ Calculates Luminosity Distance for a given cosmology (in pc)
    """
    Dl = angular_dia_dist(z, z0=z0, Om=Om, Ol=Ol, h=h)*(1+z)**2/(1+z0)**2
    return Dl
    
def combine(images,method='median'):
    """ Input is an array or list of 2D images to median combine.
        All images must be the same size...
        Methods can be:
            - median
            - mean or average
            - sum
    """
    if type(images) is list:
        images_tmp = np.array((len(images),images.shape[0],images.shape[1]))
        try:
            for i in range(len(images)):
                images_tmp[i] = images[i]
        except:
            print("All images must be the same dimension")
            exit(0)
        images = images_tmp
    if method == 'median':
        comb_image = np.median(images,axis=0)
    elif method == 'mean' or method == 'average':
        comb_image = np.mean(images,axis=0)
    elif method == 'sum':
        comb_image = np.sum(images,axis=0)
    else:
        print("Invalid input method.  Must be one of:")
        print(" median\n mean (or average)\n sum")
        exit(0)
    return comb_image
    
def correlation_coeff(x,y,mode='pearson',high_moments=None):
    """ Calculates the sample Pearson correlation coefficient, as described
        on Wikipedia and similar references.
        Can optionally include higher moments - put in as a list/array,
        starts at [3,]
        mode can be 'pearson' or 'spearman'
    """
    try:
        n = x.size
    except:
        print("x, y must be arrays with at least 2 data points")
    if mode == 'pearson':
        num = n*np.sum(x*y) - np.sum(x)*np.sum(y)
        denom = np.sqrt(n*np.sum(x*x)-(np.sum(x))**2) * np.sqrt(n*np.sum(y*y)-(np.sum(y)**2))
        if high_moments is not None:
            mnx, mny = np.mean(x), np.mean(y)
            sx = np.sqrt((1/(n-1))*np.sum((x-mnx)**2))
            sy = np.sqrt((1/(n-1))*np.sum((y-mny)**2))
            rhigh = len(high_moments)*[0]
            for i in range(len(high_moments)):
                ndim = high_moments[i] - 1
                dim_arr = np.zeros((ndim))
                for j in range(ndim):
                    dim_arr[j] = (1/(n-1))*np.sum(((x-mnx)/sx)**(high_moments[i]-j-1)*((y-mny)/sy)**(j+1))
                rhigh[i] = dim_arr
                r2 = (1/(n-1))*np.sum(((x-mnx)/sx)*((y-mny)/sy))
            return r2, rhigh
        else:
            return num/denom
    elif mode == 'spearman':
        xr = stats.rankdata(x)
        yr = stats.rankdata(y)
        return correlation_coeff(xr,yr,mode='pearson',high_moments=high_moments)
    else:
        print("Invalid mode.  Choose one of the following:\n  'pearson'\n  'spearman'")
        exit(0)
        