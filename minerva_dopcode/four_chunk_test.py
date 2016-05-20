# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 09:33:31 2016

@author: johnjohn
"""

import numpy as np
import matplotlib.pyplot as pl
from scipy.interpolate import interp1d
from astropy.io import fits
import numpy.random as rand
from scipy.optimize import curve_fit
import mpyfit #This requires a special installation
import emcee
from timeit import default_timer as timer
from utils import *
from astropy.time import Time
from glob import glob
import ipdb
import os

class FourChunk(object):
    def __init__(self, obsname, order, lpix, dpix, fixip=False, bstar=False, juststar=False):
        """ NOTE TO SELF: the GRIND code will have to use the BC to make an informed
        guess about where in the spectrum, pixel-wise, to start the analysis so that
        everything is performed in the star's reference frame. This will result in a
        variable lpix from obs to obs
        """
        self.c = 2.99792458e8 # speed of light
        self.order = order
        self.pixel = lpix
        ffile = obsname
        self.fits = ffile
        h = fits.open(ffile)
        t = Time(h[0].header['DATE-OBS'], format='isot', scale='utc')
        self.jd = t.jd

        bc = barycorr(self.jd+h[0].header['EXPTIME']/2.0/86400.0,h[0].header['TARGRA1'],h[0].header['TARGDEC1'],
                      pmra = h[0].header['PMRA1'], pmdec = h[0].header['PMDEC1'], 
                      parallax = h[0].header['PARLAX1'], rv = h[0].header['RV1'])
        zbc = bc/self.c


        rpix = lpix + dpix
        obspec = h[0].data[:, order, lpix:rpix]
        h.close()
        shape = obspec.shape
        ntel = shape[0]
        npix = shape[1]
        self.ntel = ntel
        self.xchunk = np.arange(npix)
        self.oversamp = 4.
        pad = 120
        self.xover  = np.arange(-pad,npix+pad,1./self.oversamp)
        
        wls = np.load(self.basedir + 'MC_bstar_wls.air.npy')
        w0g = wls[lpix, order] # Guess at wavelength zero point (w0)
        dwg = wls[lpix+1, order] - wls[lpix, order] # Guess at dispersion (dw)
        wmin = w0g - 4 * pad * dwg # Set min and max ranges of iodine template
        wmax = w0g + (npix + 4*pad) * dwg
        self.wiod, self.siod  = get_iodine(wmin, wmax)
        self.iodinterp = interp1d(self.wiod, self.siod) # Set up iodine interpolation function
        if not bstar:
            self.wtemp, self.stemp = get_template(wmin, wmax)
            self.tempinterp = interp1d(self.wtemp, self.stemp)
            zguess = 2.8e-4 # this needs to be generalized to be based on the barycentric RV of the observation
        else:
            zguess = 0
        self.bstar = bstar
        self.juststar = juststar
                
        obchunk = np.copy(obspec)
        for i in range(4):
            obchunk[i,:] = crclean(obspec[i,:]) # remove hot/dead pixels
        self.origchunk = obspec
        self.obchunk = obchunk # Bookkeeping for what was "cleaned" by crclean

        self.xip, self.ip = get_ip(wmin,wmax,order)
        """ List of Parameters:
        par[0] = z
        par[1] = w0
        par[2] = dw/dx
        par[3] = dw/dtrace, shift in w0 from one telescope trace to the next; assumed to be linear
        par[4:8] = continuum shape offset per telescope, python right index not inclusive
        par[8:12] = continuum shape slope per telescope
        par[12:] = IP parameters, sigma at minimum
        """

        self.initpar = np.append(zguess, [w0g, dwg]) # Start building parameter guesses
        coef = np.array([0.00059673, 0.06555841]) # wavelength offset per trace...per order
        dwdtrace = np.polyval(coef, order) # find wavelength offset per trace for this order
        self.initpar = np.append(self.initpar, dwdtrace)
        clevel = np.array([np.amax(self.obchunk, axis=1), np.zeros(ntel)])

        morepars = np.append(clevel.flatten(), [0.2, 0.]) #pars[4:7]
        self.initpar = np.append(self.initpar, morepars)
        npar = len(self.initpar)
        """ What follows are the parameter properties for MPYFIT """
        parinfo = [{'fixed':False, 'limited':(False,False), 'limits':(None, None), 'relstep':0.} for i in range(npar)]        
        if bstar:
            parinfo[0]['fixed'] = True # No stellar Doppler shift for Bstar-iodine
            self.initpar[0] = 0.
        #parinfo[0]['fixed'] = True
        parinfo[0]['relstep'] = 0.01
        parinfo[1]['relstep'] = 1e-6
        parinfo[2]['relstep'] = 0.005
        #parinfo[3]['fixed'] = True
        parinfo[12]['limited'] = (True,True)
        parinfo[12]['limits'] = (0.1, 4.0)
        parinfo[13]['limited'] = (True,True)
        parinfo[13]['limits'] = (-1.0, 1.0)
        if fixip:
            parinfo[12]['fixed'] = True
            parinfo[13]['fixed'] = True
        if juststar:
            parinfo[1]['fixed'] = True
            parinfo[2]['fixed'] = True
            #parinfo[3]['fixed'] = True
            
        self.parinfo = parinfo
        xstep = 1/self.oversamp
        self.dxip = xstep   
        self.initmod = self.model(self.initpar)
       
    def __call__(self, par): # For EMCEE
        model = self.model(par)
        lnprob = (self.obchunk-model)**2
        return -lnprob[3:len(lnprob)-1].sum()
    
    def model(self, par): # For other solvers such as MPYFIT
        z = par[0]
        w0 = par[1]
        dwdx = par[2]
        wobs  = w0 + dwdx*self.xchunk 
        wover = w0 + dwdx*self.xover 
        dwdtrace = par[3]
        trace = np.arange(self.ntel)
        offset = trace[::-1] * dwdtrace
        contcoef = par[4:12]
        sigma = par[12]
        alpha = par[13]
        sknorm = skewnorm(self.xip, 0., sigma, alpha)

        ip = numconv(sknorm, self.ip) # Convolve Zeemax IP with local broadening
        ip = ip / ip.sum() / self.oversamp # Normalize convolved IP
        #ip = sknorm # use a skew-normal function as IP
        self.newip = ip # Assume the same IP for each telescope. Imperfect scrambling invalidates this!

        try:
            for i in range(self.ntel):
                contf = contcoef[i] + contcoef[i+self.ntel]*self.xover # create continuum shape
                if self.bstar:
                    tempover = 1.
                else:
                    tempover = self.tempinterp((wover + offset[i])*(1-z)) # Doppler-shift stellar spectrum, put onto same oversampled scale as iodine
                if self.juststar:
                    iodover = contf * tempover
                else:
                    iodover = self.iodinterp(wover + offset[i]) * contf * tempover
                sover = numconv(iodover, ip) #oversampled model spectrum
                if i == 0:
                    model = rebin(wover, sover, wobs)
                else:
                    model = np.vstack((model, rebin(wover, sover, wobs)))
        except ValueError:
            print "iodine: ", self.wiod.min(), self.wiod.max()
            print "wover:  ", wover.min(), wover.max()
            print "WLS pars: ", par[1:4]

        return model
        
    def lm_model(self, x, *par):
        model = self.model(par)
        #print par[0], par[1]
        return model
    
    def mpfit_model(self, par, fjac=None, x=None, y=None, err=None):
        model = self.model(par).flatten()
        obs = self.obchunk.flatten()
        #err = np.sqrt(obs)
        err = np.sqrt(obs)
        return (obs-model)/err

    def fprof(self, par):
        oversamp = 1.0/self.dxip
        xip = self.xip
        cenwid = par[0]
        cengau = jjgauss(xip, 1., 0, cenwid)
        sh_sep = par[1]
        sh_amp = par[2]
        lgau = jjgauss(xip, sh_amp, -sh_sep, cenwid)
        rgau = jjgauss(xip, sh_amp,  sh_sep, cenwid)
        f = par[3]
        shoulders = f*lgau + (1-f)*rgau
        ip = cengau + shoulders
        return ip/(ip.sum()/oversamp)
        
    def gpfunc(self, pars):
        ip = jjgauss(self.xip, 1.0, 0.0, pars[0])
        for i, sig in enumerate(self.ipsig):
            gau = jjgauss(self.xip, pars[i+1], self.ippix[i], sig)
            ip += gau 
        return ip / ip.sum() / self.dxip

    def emcee_fitter(self, nwalkers, niter, nburn):
        #p0 = self.initpar
        s, p0 = self.mpfitter() # initialize parameters using least-sq fit
        ndims = len(p0)
        p0arr = np.array([])
        for i, p in enumerate(p0):
            if p != 0:
                amp = self.dstep[i] * p
            else:
                amp = self.dstep[i]
            randp = p + amp*(rand.random(nwalkers) - 0.5)
            p0arr = np.append(p0arr, randp)
        p0arr.shape = (ndims, nwalkers)
        p0arr = p0arr.transpose()
        
        sampler = emcee.EnsembleSampler(nwalkers,ndims,self)
        print "Starting burn-in with "+str(nburn)+" links"  
        pos, prob, state = sampler.run_mcmc(p0arr, nburn)
        sampler.reset()
        print "Starting main MCMC run with "+str(nwalkers)+" walkers and "+str(niter)+ " links."
        foo = sampler.run_mcmc(pos, niter, rstate0=state)
        m = np.argmax(sampler.flatlnprobability)
        bpar = sampler.flatchain[m,:]
        return sampler, bpar
        
    def lm_fitter(self):
        bestpar, cov = curve_fit(self.lm_model, self.xchunk, self.obchunk, p0=self.initpar)
        return self.model(bestpar), bestpar

    def mpfitter(self):
        #m = mpfit(self.mpfit_model, self.initpar, parinfo=self.parinfo, xtol=1e-8, quiet=True)
        pfit, results = mpyfit.fit(self.mpfit_model, self.initpar, parinfo=self.parinfo, maxiter=50, xtol=1e-8, ftol=1e-8)
        model = self.model(pfit)
        self.par = pfit
        self.mod = model
        self.resid = self.obchunk - model
        npar = len(self.par)
        dim = self.obchunk.shape
        ntel = dim[0]
        npix = dim[1]
        dof = ntel*npix - npar
        dofpertrace = npix - npar
        self.chiarr = np.sqrt(np.sum((self.resid)**2/self.mod, axis=1) / dofpertrace)
        bad = np.where(self.chiarr > 3)
        """
        if bad[0]:
            nbad = len(bad[0])
            for i in range(nbad):
                print bad[0][i], 'is bad'
        """
        self.chi = np.sqrt(np.sum(self.resid.flatten()**2/self.obchunk.flatten())/dof) # sqrt(chi^2) =~ number of sigmas 
        return model, pfit

def grind(obsname, plot=False, printit=False, bstar=False, juststar=False):
    print 'Fitting ', obsname
    start = timer()
    order = np.arange(2, 20)
    nord = len(order)
    dpix = 128
    pixel = np.arange(64, 1900, dpix)
    npix = len(pixel)
    charr = np.empty((npix,nord), dtype=object)
    for i, Ord in enumerate(order):
        for j, Pix in enumerate(pixel):
            chstart = timer()
            if printit:
                print '------'
                print 'Order:', Ord, 'Pixel:', Pix

            ch = FourChunk(obsname, Ord, Pix, dpix, bstar=bstar, juststar=juststar)
            mod, bp = ch.mpfitter() # Full fit
            if plot:
                col = ['b', 'g', 'y', 'r']
                wav = bp[1] + bp[2]*np.arange(mod.shape[1])
                for i in range(4): 
                    woff = bp[3] * i
                    pl.plot(wav-woff, ch.obchunk[i, :], col[i]+'.')
                    pl.plot(wav-woff, mod[i, :], col[i])
                    pl.plot(wav-woff, ch.resid[i, :]+0.6*mod.flatten().min(), col[i]+'.')
                pl.xlabel('Wavelength [Ang]')
                pl.ylabel('Flux [cts/pixel]')
                pl.xlim((wav.min(), wav.max()))
                pl.show()
            charr[j, i] = ch
            if printit:
                print 'chi^2 = ', ch.chi
                end = timer()
                print 'Chunk time: ', (end-chstart), 'seconds'
        end = timer()
    print 'Total time: ', (end-start)/60., 'minutes'
    return charr

def getparr(charr, plot=False):
    sh = charr.shape
    nc = sh[0]
    nord = sh[1]
    npar = len(charr[0,0].par)
    pararr = np.empty((nc,nord,npar+4), dtype=float)
    nxip = len(charr[0,0].newip)
    iparr = np.zeros((nxip, nc, nord))
    for i in range(nc):
        for j in range(nord):
            iparr[:,i,j] = charr[i,j].newip
            if plot: pl.plot(charr[i,j].newip)
            pararr[i,j,:] = np.append(charr[i,j].par, charr[i,j].chiarr)
    if plot: pl.show()       
    return pararr, iparr

def getpararray(obsname):

    night = obsname.split('.')[0]
    ofile = '/Data/kiwispec-proc/'+night+'/'+obsname
    test = np.load(ofile[0])
    sh = test.shape
    parr = np.zeros((len(ofile), sh[0], sh[1], sh[2]))
    for i, ffile in enumerate(ofile):
        p = np.load(ffile)
        parr[i, :, :, :] = p
    return parr
    
def globgrind(night, objname='*', bstar=False, returnfile=False):

    filepath = '/Data/kiwispec-proc/'+night+'/'+night+'.'+objname+'*.proc.fits'
    files = glob(filepath)
    ofarr = [] # object f? array?

    for i, ffile in enumerate(files):
        h = fits.open(ffile)

        # if the iodine cell is not in, just fit the star without iodine
        juststar = h[0].header['I2POSAS'] != 'in'
        
        ofile = os.path.splitext(ffile)[0] + '.parr.npy'
        ofarr.append(ofile)

        if not returnfile:
            charr = grind(ffile, bstar=bstar, juststar=juststar)
            parr, iparr = getparr(charr)
            np.save(ofile, parr)
            print ofile, h[0].header['I2POSAS']
    return ofarr

globobs = 'n20160305/n20160305.daytimeSky.008*.proc.fits'
obsname = 'n20160305/n20160305.daytimeSky.0065.proc.fits'
#charr = grind(obsname, plot=True, printit=True)
#parr, iparr = getparr(charr)

ofarr = globgrind('n20160305', objname="daytimeSky.008*",bstar=False, returnfile=False)

ipdb.set_trace()
parray = getpararray(ofarr)
#######################
"""
#gfile = '/Users/johnjohn/Dropbox/research/dopcode_new/MINERVA_data/n20160323.HR4828.00*.proc.fits'
#globobs = 'n20160323.HR4828.00*.proc.fits'
globobs = 'n20160410.HR5511.00*.proc.fits'

ofarr = globgrind(globobs, bstar=True, returnfile=False)
parray = getpararray(ofarr)

wr = (parray[:,:,:,1]- np.median(parray[:,:,:,1], axis=0))
sh = wr.shape
wr = wr.reshape((sh[0], sh[1]*sh[2]))
medwr = np.median(wr, axis=1)
#pl.plot(medwr, 'o')
g = glob(globobs)

for i, ff in enumerate(g):
    h = fits.open(ff)
    this = Time(h[0].header['DATE-OBS'])
    if i == 0:
        t = this.jd
        temp = float(h[0].header['TEMPD2'])
    else:
        t = np.append(t, this.jd)
        temp = np.append(temp, float(h[0].header['TEMPD2']))
    h.close()

coef = np.polyfit(t-t.min(), medwr/5500.*3e8, 1)
fit = np.polyval(coef, t-t.min())
pl.plot((t-t.min())*24, medwr/5500.*3e8, 'o')
pl.plot((t-t.min())*24, fit)
pl.ylabel('Wavelength Shift [m/s]')
pl.xlabel('Time [hrs]')
pl.show()

#pl.plot(temp, medwr/5500.*3e8, 'o')
pl.plot(t-t.min(), temp, 'o')
#pl.xlabel('Temperature [degrees C]')
#pl.ylabel('Wavelength Shift [m/s]')
pl.xlabel('Time')
pl.ylabel('Temperature [degrees C]')

pl.show()

#charr = grind(ffile, bstar=True, plot=True, printit=True)
#parr, iparr = getparr(charr)
#pl.plot(parr[:,:,1].T.flatten(), parr[:,:,14].T.flatten(), '.')
#pl.ylim((1,3))
#pl.show()


#ffile = '/Users/johnjohn/Dropbox/research/dopcode_new/MINERVA_data/n20160323.HR4828.0020.proc.fits'
obsname = 'n20160305.daytimeSky.0065.proc.fits' #iodine
#ffile = '/Users/johnjohn/Dropbox/research/dopcode_new/MINERVA_data/n20160305.daytimeSky.0069.proc.fits'
wls = np.load('/Users/johnjohn/Dropbox/research/dopcode_new/MC_bstar_wls.air.npy')

ffile = '/Users/johnjohn/Dropbox/research/dopcode_new/MINERVA_data/'+obsname
h = fits.open(ffile)
lpix = 1400
dpix = 100
rpix = lpix + dpix
Ord = 3
w0 = wls[lpix, Ord]
dw = wls[lpix+1, Ord] - w0
wguess = np.array([w0, dw])
#ch = FourChunk(ffile, Ord, lpix, dpix, bstar=False, juststar=True, fixip=False)
ch = FourChunk(obsname, Ord, lpix, dpix, fixip=False)

#for i in range(len(ch.initpar)):
#    print 'Par', i, 'value', ch.initpar[i]

col = ['b', 'g', 'y', 'r']
#mod = ch.model(ch.initpar)
mod, bp = ch.mpfitter()
wav = bp[1] + bp[2]*np.arange(mod.shape[1])

for i in range(4): 
    woff = bp[3]*i
    pl.plot(wav-woff, ch.obchunk[i, :], col[i]+'.')
    pl.plot(wav-woff, mod[i, :], col[i])
    pl.plot(wav-woff, ch.resid[i, :]+0.6*mod.flatten().min(), col[i]+'.')
pl.xlabel('Wavelength [Ang]')
pl.ylabel('Flux [cts/pixel]')
pl.show()

#print h[0].data[:, 13, 500:600].shape
#h.close()

"""
