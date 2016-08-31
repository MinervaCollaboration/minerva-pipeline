#-*- coding: utf-8 -*-
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
import time
import glob
from photutils import aperture_photometry, CircularAperture, CircularAnnulus
import pyfits

# creates a wavelength solution in the format expected for grind
# given a chunk record file of solved chunks
# wls = utils.mkwls('../../data/n20160612/n20160612.HR5511.0018.proc.chrec14.npy')
def mkwls(chrecfile):
    chrec = np.load(chrecfile)

    dpix = chrec['pixel'][1] - chrec['pixel'][0]
    xchunk = np.arange(dpix) - dpix/2

    wls= np.ndarray(shape=(2048,28),dtype=float) + np.nan
    for chunk in range(len(chrec)):
        
        Ord = chrec['order'][chunk]
        Pix = chrec['pixel'][chunk]

        w0 = chrec['w04'][chunk]
        dwdx = chrec['dwdx4'][chunk]

        try: dw2dx2 = chrec['dw2dx24'][chunk]
        except: dw2dx2 = 0.0

        wls[Pix:Pix+dpix,Ord] = w0 + dwdx*xchunk + dw2dx2*xchunk**2

    return wls


def addzb(fitsname, redo=False):
    telescopes = ['1','2','3','4']
    night = (fitsname.split('/'))[3]

    print 'beginning ' + fitsname


    # 2D spectrum
    h = pyfits.open(fitsname,mode='update')

    if 'BARYSRC4' in h[0].header.keys() and not redo:
        print fitsname + " already done"
        h.flush()
        h.close()
        return

    specstart = datetime.datetime.strptime(h[0].header['DATE-OBS'],"%Y-%m-%dT%H:%M:%S.%f")
    specmid = specstart + datetime.timedelta(seconds = h[0].header['EXPTIME']/2.0)
    specend = specstart + datetime.timedelta(seconds = h[0].header['EXPTIME'])

    t0 = datetime.datetime(2000,1,1)
    t0jd = 2451544.5

    aperture_radius = 3.398 # fiber radius in pixels
    annulus_inner = 2.0*aperture_radius
    annulus_outer = 3.0*aperture_radius

    for telescope in telescopes:
        print 'beginning telescope ' + telescope + ' on ' + fitsname

        # get the barycentric redshift for each time
        ra = h[0].header['TARGRA' + telescope]
        dec = h[0].header['TARGDEC' + telescope]
        try: pmra = h[0].header['PMRA' + telescope]
        except: pmra = 0.0
        try: pmdec = h[0].header['PMDEC' + telescope]
        except: pmdec = 0.0
        try: parallax = h[0].header['PARLAX' + telescope]
        except: parallax = 0.0
        try: rv = h[0].header['RV' + telescope]
        except: rv = 0.0

        objname = h[0].header['OBJECT' + telescope]
        faupath = '/Data/t' + telescope + '/' + night + '/' + night + '.T' + telescope + '.FAU.' + objname + '.????.fits'
        guideimages = glob.glob(faupath)

#        if telescope == '2' and "HD62613" in fitsname: ipdb.set_trace()

        times = []
        fluxes = np.array([])

        for guideimage in guideimages:
            try:
                fauimage = pyfits.open(guideimage)
            except:
                print "corrupt file for " + guideimage
                continue

            # midtime of the guide image (UTC)
            midtime = datetime.datetime.strptime(fauimage[0].header['DATE-OBS'],"%Y-%m-%dT%H:%M:%S") +\
                datetime.timedelta(seconds=fauimage[0].header['EXPTIME']/2.0)
            
            # convert to Julian date
            midjd = t0jd + (midtime-t0).total_seconds()/86400.0

            # only look at images during the spectrum
            if midtime < specstart or midtime > specend: continue

            # find the fiber position
            try:
                fiber_x = fauimage[0].header['XFIBER' + telescope]
                fiber_y = fauimage[0].header['YFIBER' + telescope]
            except:
                print "keywords missing for " + guideimage
                continue

            # do aperture photometry
            positions = [(fiber_x,fiber_y)]
            apertures = CircularAperture(positions,r=aperture_radius)
            annulus_apertures = CircularAnnulus(positions, r_in=annulus_inner, r_out=annulus_outer)

            # calculate the background-subtracted flux at the fiber position
            rawflux_table = aperture_photometry(fauimage[0].data, apertures)
            bkgflux_table = aperture_photometry(fauimage[0].data, annulus_apertures)
            bkg_mean = bkgflux_table['aperture_sum'].sum() / annulus_apertures.area()
            bkg_sum = bkg_mean * apertures.area()
            flux = rawflux_table['aperture_sum'].sum() - bkg_sum
            
            # append to the time and flux arrays
            times.append(midjd)
            fluxes = np.append(fluxes,flux)

        if len(times) == 0:
            print "No guider images for " + fitsname + " on Telescope " + telescope + "; assuming mid time"
            if 'daytimeSky' in fitsname: 
                h[0].header['BARYCOR' + telescope] = ('UNKNOWN','Barycentric redshift')
                h[0].header['BARYSRC' + telescope] = ('UNKNOWN','Source for the barycentric redshift')
                continue
            
            # convert specmid to Julian date
            midjd = t0jd + (specmid-t0).total_seconds()/86400.0

            print midjd
            zb = barycorr(midjd, ra, dec, pmra=pmra, pmdec=pmdec, parallax=parallax, rv=rv)/2.99792458e8
            h[0].header['BARYCOR' + telescope] = (zb,'Barycentric redshift')
            h[0].header['BARYSRC' + telescope] = ('MIDTIME','Source for the barycentric redshift')
            continue
            

        zb = np.asarray(barycorr(times, ra, dec, pmra=pmra, pmdec=pmdec, parallax=parallax, rv=rv))/2.99792458e8

        # weight the barycentric correction by the flux
        #*******************!*!*!*!*!*!*
        #this assumes guider images were taken ~uniformly throughout the spectroscopic exposure!
        #****************!*!*!*!*!**!*!*!*!**!*!*!*!*
        wzb = np.sum(zb*fluxes)/np.sum(fluxes)
        
        # update the header to include aperture photometry and barycentric redshift
        h[0].header['BARYCOR' + telescope] = (wzb,'Barycentric redshift')
        h[0].header['BARYSRC' + telescope] = ('FAU Flux Weighted','Source for the barycentric redshift')
        hdu = pyfits.PrimaryHDU(zip(times,fluxes))
        hdu.header['TELESCOP'] = ('T' + telescope,'Telescope')
        h.append(hdu)

    # write updates to the disk
    h.flush()
    h.close()

def addallzb(path='/Data/kiwispec-proc/n*/*.fits'):
    filenames = glob.glob(path)
    for filename in filenames:
        try: addzb(filename)
        except: print filename + ' failed'

# query OSU page for barycentric correction
def barycorr(jdutc,ra,dec,pmra=0.0,pmdec=0.0,parallax=0.0,rv=0.0,zmeas=0.0,
             epoch=2451545.0,tbase=0.0,
             lon=-110.878977,lat=31.680407,elevation=2316.0):

    if type(jdutc) is list: jds = jdutc
    else: jds = [jdutc]

#    ipdb.set_trace()


    url="http://astroutils.astronomy.ohio-state.edu/exofast/barycorr.php?" +\
        "JDS=" + ','.join(map(str,jds)) + "&RA=" + str(ra) + "&DEC=" + str(dec) +\
        "&LAT=" + str(lat) + "&LON=" + str(lon) + "&ELEVATION=" + str(elevation) +\
        "&PMRA=" + str(pmra) + "&PMDEC=" + str(pmdec) + "&RV=" + str(rv) +\
        "&PARALLAX=" + str(parallax) + "&ZMEAS=" + str(zmeas) +\
        "&EPOCH=" + str(epoch) + "&TBASE=" + str(tbase)

    request = urllib2.Request(url)
    while True:
        try: 
            response = urllib2.urlopen(request)
            break
        except: 
            print "Error contacting the barycorr website"
            time.sleep(1)

    data = response.read().split()

    # there was probably an error with the call
    if len(data) != len(jds):
        time.sleep(1)
        print data
        print jds
        print 'unexpected output -- website down?'
        ipdb.set_trace()
        return barycorr(jdutc,ra,dec,pmra=pmra,pmdec=pmdec,parallax=parallax,rv=rv,zmeas=zmeas,epoch=epoch,tbase=tbase,lon=lon,lat=lat,elevation=elevation)

    if len(data) == 1: return float(data[0])
    return map(float,data)

def xcorl(spec, model, r):
    n = len(model)
    mod = model[r:-r]
    nlag = len(model)
    lag = np.arange(-n/2.0, n/2.0)
    ccf = correlate(spec-np.mean(spec), mod-np.mean(mod), mode='same')
    maxind = np.argmax(ccf)
    return lag[maxind], ccf

def jjgauss(x, *a):
    """ For free parameters create a Gaussian defined at x """
    return a[0] * np.exp(-0.5*((x - a[1])/a[2])**2) 

def get_ip(ipdict, wmin, wmax, orderin, oversamp=4.0):
    """
    Search for synthetic IPs in requested wavelength range and specified order
    Created by John Johnson, ~March 2016
    Modified by Sharon X. Wang, July 2016: enable other oversampling options (default was 4)
    """
    order = orderin+1 #Stuart's relative orders are offset from orders in Matt C.'s reduced spectra
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

    # Now resample IP
    if (oversamp != 4):
        xip_half = np.arange(0, max(xip), 1/oversamp)
        xip_new = np.append(-xip_half[1:][::-1], xip_half)

        print 'interpolating IP'
        interp_ip = interp1d(xip, ip)
        ip_new = interp_ip(xip_new)
        return xip_new, ip_new

    return xip, ip
            
def airtovac(wav):
    sigma2 = (1e4/wav)**2
    fact = 1.0 + 6.4328e-5 + 2.94981e-2/(146.0 - sigma2) +  2.554e-4/(41.0 - sigma2)
    return wav * fact

def get_template(template, wmin, wmax):
#    sav = readsav(templatename)

    try:
#    if 'nso.sav' in templatename:
        wavall = template.w
        templateall = template.s
#    else: 
    except:
        wavall = template.sdstwav
        templateall = template.sdst

    use = np.where((wavall >= wmin) & (wavall <= wmax))
    return wavall[use], templateall[use]
    
def get_iodine(iodine, wmin, wmax):

    wav = iodine[0,:] * 10.0 # Convert wavelenghts from nm to Angstroms
    iod = iodine[1,:]
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
        ww[i] = wsub[inds[int(npix*0.95)]]
        ss[i] = ssub[inds[int(npix*0.95)]]
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

def infinite_loop():
    while True:
        time.sleep(1)

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    import signal

    class TimeoutError(Exception):
        pass

    def handler(signum, frame):
        raise TimeoutError()

    # set the timeout handler
    signal.signal(signal.SIGALRM, handler) 
    signal.alarm(timeout_duration)

    try:
        result = func(*args, **kwargs)
    except TimeoutError as exc:
        result = default
    finally:
        signal.alarm(0)

    return result

def numconv(y, kern):
    ynew = y.copy()
    lenex = 10
    ynew = extend(ynew, lenex)
    
    kwargs = {'mode':'full'}
    new = timeout(fftconvolve,(ynew,kern,),kwargs=kwargs,timeout_duration=300)
#    new = fftconvolve(ynew, kern, mode='full')
    if new == None: ipdb.set_trace()

    new /= kern.sum()
    nel = len(kern)+lenex*2
    return new[int(nel/2.0):int(len(new)-nel/2.0+1.0)]
    
def rebin(wold, sold, wnew, z=0):
    """ Define the left and right limits of the first Wnew pixel. Keep in mind
    that pixels are labled at their centers. """    
    dw_new = wnew[1]-wnew[0]
    w_left = wnew.min() - 0.5 * dw_new
    w_right = w_left + dw_new
    Nsub = 10.0 # use 10 sub pixels for the template across new pixel 
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
    wfine += dsub/2.0 # Center each subgrid on the pixel
    ifunction = interp1d(wold*(1+z), sold) #Create spline-interpolation function
    sfine = ifunction(wfine) #Calculate interpolated spectrum 
    sfine_blocks = sfine.reshape([Npixels,int(Nsub)]) #Prepare for integration
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
    return 1.0/np.sqrt(2.0*np.pi) * np.exp(-x**2/2.0)

def cdf(x):
    return (1.0 + erf(x/np.sqrt(2.0))) / 2.0

def skewnorm(x,xo=0,sigma=1,alpha=0):
    t = (x - xo) / sigma
    return 2.0 / sigma * pdf(t) * cdf(alpha*t)

def fo_loglike(p, *args):
    resid = args[0]
    #resid = p[0] - y
    sigma = np.abs(p[0])
    f = np.abs(p[1])
    s = np.abs(p[2])
    norm = 1.0/(sigma*np.sqrt(2.0*np.pi))
    part1 = f/s * np.exp(-0.5 * (resid/(s*sigma))**2)
    part2 = (1.0-f) * np.exp(-0.5 * (resid/sigma)**2)
    loglike = np.log(norm * (part1 + part2))
    ll = -np.sum(loglike)
    #print p, ll
    return ll
    
def find_outliers(resid, plot=False):
    x = np.arange(len(resid))
    bnds = ((1e-8, None),
            (0.0,1.0),
            (1.0, None))
    sig0 = np.std(medfilt(resid))
    sig0 = np.std(resid)
    p0 = [sig0/5.0, 0.1, sig0*100.0]
    result = minimize(fo_loglike, p0, (resid), method='SLSQP', bounds=bnds, options={'maxiter':100})
    bp = result.x
    bad = np.where(np.abs(resid) > (4.0*bp[0]))
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
    Sigma = Sigma / np.sqrt(Ndata - 1.0)
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
        return 0.0
    u = (d-d0)/(6.0 * mad)
    uu = u**2
    q = np.where(uu <= 1.0)
    count = len(q[0])
    if count < 3:
        return -1
    numerator = np.sum( (d[q] - d0)**2 * (1.0-uu[q])**4 )
    n = len(d)
    den1 = np.sum( (1.0-uu[q]) * (1.0-5.0*uu[q]) )
    sigma = n * numerator / (den1 * (den1 - 1.0))
    if sigma > 0:
        return np.sqrt(sigma)
    else:
        return 0
