# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 14:32:46 2016

@author: johnjohn
"""

import numpy as np
import matplotlib.pyplot as pl
from scipy.signal import fftconvolve
from scipy.signal import correlate
from scipy.interpolate import interp1d
from scipy.io.idl import readsav
from scipy.signal import medfilt
from scipy.special import erf
from scipy.optimize import minimize
import urllib2
import ipdb
import datetime

# query OSU page for barycentric correction
def barycorr(jdutc,ra,dec,pmra=0.0,pmdec=0.0,parallax=0.0,rv=0.0,zmeas=0.0,
             epoch=2451545.0,tbase=0.0,
             lon=-110.878977,lat=31.680407,elevation=2316.0):

    if type(jdutc) is list: jds = jdutc
    else: jds = [jdutc]

    url="http://astroutils.astronomy.ohio-state.edu/exofast/barycorr.php?" +\
        "JDS=" + ','.join(map(str,jds)) + "&RA=" + str(ra) + "&DEC=" + str(dec) +\
        "&LAT=" + str(lat) + "&LON=" + str(lon) + "&ELEVATION=" + str(elevation) +\
        "&PMRA=" + str(pmra) + "&PMDEC=" + str(pmdec) + "&RV=" + str(rv) +\
        "&PARALLAX=" + str(parallax) + "&ZMEAS=" + str(zmeas) +\
        "&EPOCH=" + str(epoch) + "&TBASE=" + str(tbase)

    request = urllib2.Request(url)
    response = urllib2.urlopen(request)

    data = response.read().split()

    if len(data) == 1: return float(data[0])
    return map(float,data)

def xcorl(spec, model, r):
    n = len(model)
    mod = model[r:-r]
    nlag = len(model)
    lag = np.arange(-n/2., n/2.)
    ccf = correlate(spec-np.mean(spec), mod-np.mean(mod), mode='same')
    maxind = np.argmax(ccf)
    return lag[maxind], ccf

def jjgauss(x, *a):
    """ For free parameters create a Gaussian defined at x """
    return a[0] * np.exp(-0.5*((x - a[1])/a[2])**2) 

def get_ip(wmin, wmax, orderin):
    """ Search for synthetic IPs in requested wavelength range and specified order"""
    ipfile = 'synthetic_IPs/ipdict.npy'
    order = orderin+1 #Stuart's relative orders are offset from orders in Matt C.'s reduced spectra
    ipdict = np.load(ipfile) # Array of dictionaries created in read_synth_IPs.py
    nel = len(ipdict)
    warr = np.zeros(nel)
    oarr = np.zeros(nel)
    for i, elem in enumerate(ipdict):
        warr[i] = elem['wav']
        oarr[i] = elem['order']
    ind = np.arange(nel)
    barr = ind[(warr > wmin) & (warr < wmax) & (oarr == order)]
    for i, b in enumerate(barr):
        if i == 0:
            ip = ipdict[b]['ip']
            xip = ipdict[b]['xip']
        else:
            ip += ipdict[b]['ip']
    ip /= len(barr)
    return xip, ip
            
def airtovac(wav):
    sigma2 = (1e4/wav)**2
    fact = 1.0 + 6.4328e-5 + 2.94981e-2/(146.0 - sigma2) +  2.554e-4/(41.0 - sigma2)
    return wav * fact

def get_template(templatename, wmin, wmax):
    sav = readsav(templatename)

    if 'nso.sav' in templatename:
        wavall = sav.w
        templateall = sav.s
    else: 
        wavall = sav.sdstwav
        templateall = sav.sdst

    use = np.where((wavall >= wmin) & (wavall <= wmax))
    wav = wavall[use]
    template = templateall[use]
    return wav, template
    
def get_iodine(wmin, wmax):
    sav = np.load('templates/MINERVA_I2_0.1_nm.npy')
    wav = sav[0,:] * 10 # Convert wavelenghts from nm to Angstroms
    iod = sav[1,:]
    use = np.where((wav >= wmin) & (wav <= wmax))
    w = wav[use]
    s = iod[use]
    nwin = 16
    window = np.linspace(wmin,wmax,num=nwin)
    dw = window[1] - window[0]
    ww = window
    ss = np.zeros(nwin)
    for i in range(nwin-1):
        lw = window[i]
        ssub = s[(w > lw) & (w < (lw+dw))]
        wsub = w[(w > lw) & (w < (lw+dw))]
        npix = len(ssub)
        inds = np.argsort(ssub)
        ww[i] = wsub[inds[npix*0.95]]
        ss[i] = ssub[inds[npix*0.95]]
    par = np.polyfit(ww[:nwin-1], ss[:nwin-1], 1)
    polyf = np.poly1d(par)
    #pl.plot(w, s)
    #pl.plot(ww, ss, 'o')
    #pl.plot(w, polyf(w)-s, '.-')
    #pl.show()
    return  w, s/polyf(w)
    
def extend(arr, nel, undo=False):
    new = arr.copy()
    if undo:
        return new[nel:len(new)-nel]
    else:
        new = np.append(np.append(np.zeros(nel)+arr[0], new), np.zeros(nel)+arr[-1])
        return new

def numconv(y, kern):
    ynew = y.copy()
    lenex = 10
    ynew = extend(ynew, lenex)
    new = fftconvolve(ynew, kern, mode='full')
    new /= kern.sum()
    nel = len(kern)+lenex*2
    return new[nel/2.:len(new)-nel/2.+1]
    
def rebin(wold, sold, wnew, z=0):
    """ Define the left and right limits of the first Wnew pixel. Keep in mind
    that pixels are labled at their centers. """    
    dw_new = wnew[1]-wnew[0]
    w_left = wnew.min() - 0.5 * dw_new
    w_right = w_left + dw_new
    Nsub = 10. # use 10 sub pixels for the template across new pixel 
    """ Create a finely sampled 'sub-grid' across a pixel. We'll later integrate
    the stellar spectrum on this fine grid for each pixel  """
    wfine_sub = np.linspace(w_left, w_right, Nsub, endpoint=False)
    Npixels = len(wnew) #Number of pixels in the new wavelength scale
    """ Copy an individual pixel subgrid Npixels times, aka 'tiling' """
    wsub_tile = np.tile(wfine_sub, Npixels)
    """ Create a separate array that ensures that each pixel subgrid is 
    properly incremented, pixel-by-pixel following the Wnew wavelength scale"""
    step = np.repeat(np.arange(Npixels), Nsub) * dw_new
    wfine = wsub_tile + step #Finely sampled new wavelength scale
    dsub = wfine[1] - wfine[0]
    wfine += dsub/2. # Center each subgrid on the pixel
    ifunction = interp1d(wold*(1+z), sold) #Create spline-interpolation function
    sfine = ifunction(wfine) #Calculate interpolated spectrum 
    sfine_blocks = sfine.reshape([Npixels,Nsub]) #Prepare for integration
    snew = np.sum(sfine_blocks, axis=1)/Nsub #Efficient, vector-based integration! 
    return snew

def crclean(spec):
    ind = np.arange(len(spec))
    wzero = ind[(spec == 0)]    
    if len(wzero) > 0:
        spec[wzero] = np.median(spec)
    medspec = medfilt(spec)
    mresid = spec - medspec
    notzero = ind[(mresid != 0)]
    sigma = np.std(mresid[notzero])
    bad = ind[(np.abs(mresid) > 3*sigma) | (spec == 0)]
    newspec = np.copy(spec)
    if len(bad) > 0:
        newspec[bad] = medspec[bad]
    return newspec

def pdf(x):
    return 1/np.sqrt(2*np.pi) * np.exp(-x**2/2)

def cdf(x):
    return (1 + erf(x/np.sqrt(2))) / 2

def skewnorm(x,xo=0,sigma=1,alpha=0):
    t = (x - xo) / sigma
    return 2 / sigma * pdf(t) * cdf(alpha*t)

def fo_loglike(p, *args):
    resid = args[0]
    #resid = p[0] - y
    sigma = np.abs(p[0])
    f = np.abs(p[1])
    s = np.abs(p[2])
    norm = 1./(sigma*np.sqrt(2*np.pi))
    part1 = f/s * np.exp(-0.5 * (resid/(s*sigma))**2)
    part2 = (1-f) * np.exp(-0.5 * (resid/sigma)**2)
    loglike = np.log(norm * (part1 + part2))
    ll = -np.sum(loglike)
    #print p, ll
    return ll
    
def find_outliers(resid, plot=False):
    x = np.arange(len(resid))
    bnds = ((1e-8, None),
            (0.,1.),
            (1., None))
    sig0 = np.std(medfilt(resid))
    sig0 = np.std(resid)
    p0 = [sig0/5., 0.1, sig0*100]
    result = minimize(fo_loglike, p0, (resid), method='SLSQP', bounds=bnds, options={'maxiter':100})
    bp = result.x
    bad = np.where(np.abs(resid) > (4.*bp[0]))
    nbad = len(bad[0])
    #print bad
    if plot:
        pl.plot(resid,'bo')
        pl.plot(x[bad], resid[bad], 'ro')
        pl.show()
    return bad[0], nbad

def robust_mean(d, cut):
    data = d.flatten()
    Ndata = len(data)
    ind = np.arange(Ndata)
    dmed = np.median(data)
    AbsDev = np.abs(data - dmed)
    MedAbsDev = np.median(AbsDev)
    if MedAbsDev < 1.0E-24:
        MedAbsDev = np.mean(AbsDev)/.8 
    cutoff = cut * MedAbsDev
    GoodInd = ind[AbsDev <= cutoff]
    NumGood = len(GoodInd)
    GoodData = data[GoodInd]
    Mean = np.mean(GoodData)
    Sigma = np.sqrt(np.sum((GoodData - Mean)**2) / NumGood)
    NumRej = Ndata - NumGood
    if cut > 1.0:
        SC = cut
    else:
        SC = 1.0
    if SC <= 4.5:
        Sigma = Sigma / (-0.15405 + 0.90723*SC - 0.23584*SC**2 + 0.020142*SC**3)
    Sigma = Sigma / np.sqrt(Ndata - 1)
    return Mean

def robust_sigma(data):
    d = data.flatten()
    eps = 1e-30
    d0 = np.median(d)
    absdev =  np.abs(d - d0) 
    mad = np.median(absdev) / 0.6745
    if mad < eps:
        mad = np.mean(absdev) / 0.8
    if mad < eps:
        return 0
    u = (d-d0)/(6.0 * mad)
    uu = u**2
    q = np.where(uu <= 1.)
    count = len(q[0])
    if count < 3:
        return -1
    numerator = np.sum( (d[q] - d0)**2 * (1-uu[q])**4 )
    n = len(d)
    den1 = np.sum( (1-uu[q]) * (1-5*uu[q]) )
    sigma = n * numerator / (den1 * (den1 - 1))
    if sigma > 0:
        return np.sqrt(sigma)
    else:
        return 0
