# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 14:41:59 2016

@author: johnjohn
"""
import numpy as np
import matplotlib.pyplot as pl
from scipy.interpolate import interp1d
from astropy.io import fits
import numpy.random as rand
from scipy.optimize import curve_fit
import mpyfit
import emcee
from timeit import default_timer as timer
from minerva_dopcode.utils import *
from glob import glob

class Chunk(object):
    def __init__(self, obspec, wcof_guess, fixip=False, bstar=False, order=-99, pixel=-99):
        """ NOTE TO SELF: the GRIND code will have to use the BC to make an informed
        guess about where in the spectrum, pixel-wise, to start the analysis so that
        everything is performed in the star's reference frame. This can be accomlished
        by also supplying the pixel keyword, which is unused as of 3/9/2016
        """
        npix = len(obspec)
        self.xchunk = np.arange(npix)
        self.oversamp = 4.
        pad = 120
        self.xover  = np.arange(-pad,npix+pad,1./self.oversamp)
        
        dwg = wcof_guess[1]
        wmin = wcof_guess[0] - 2 * pad * dwg
        wmax = wcof_guess[0] + (npix + 2*pad) * dwg
        self.wiod, self.siod  = get_iodine(wmin, wmax)
        self.iodinterp = interp1d(self.wiod, self.siod)
        if not bstar:
            self.wtemp, self.stemp = get_template(wmin, wmax)
            self.tempinterp = interp1d(self.wtemp, self.stemp)
            self.bstar = False
        else:
            self.bstar = True
        
        obchunk = crclean(obspec)
        self.origchunk = obspec
        self.obchunk = obchunk
        if order > 0:
            self.order = order
        if pixel > 0:
            self.pixel = pixel
        xip, ip = get_ip(wmin,wmax,order)
        self.ip = ip
        self.xip = xip
        """ List of Parameters:
        par[0] = z
        par[1] = w0
        par[2] = dw/dx
        par[3] = quadratic wavelength term (optional)
        par[4:6] = continuum shape linear, python right index not inclusive
        par[6:] = IP parameters, sigma at minimum
        """
        zguess = -3e-4
        self.initpar = np.append(zguess, wcof_guess)
        if len(wcof_guess) == 1: # only offset guess supplied
            self.initpar = np.append(self.initpar, [0.025, 0.]) 
        if len(wcof_guess) == 2: # linear guess supplied
            self.initpar = np.append(self.initpar, 0.) 
        clevel = obspec.max() / self.siod.max()
        morepars = np.array([clevel, 0., 0.2, 0.7, 0, 0.5]) #pars[4:7]
        self.initpar = np.append(self.initpar, morepars)
        npar = len(self.initpar)
        parinfo = [{'fixed':False, 'limited':(False,False), 'limits':(None, None), 'relstep':0.} for i in range(npar)]        
        #parinfo = [{'value':0., 'fixed':0, 'limited':[0,0], 'limits':[0.,0.], 'step':0.} for i in range(npar)]        
        if bstar:
            parinfo[0]['fixed'] = True # No stellar Doppler shift for Bstar-iodine
            self.initpar[0] = 0.
        """ Adjust these parameters to 5e-6, 2e-6, 0.005"""
        parinfo[0]['relstep'] = 0.01
        parinfo[1]['relstep'] = 1e-6
        parinfo[2]['relstep'] = 0.005
        parinfo[3]['fixed'] = True
        parinfo[6]['limited'] = (True,True)
        parinfo[6]['limits'] = (0.1, 1.0)
        parinfo[7]['limited'] = (True,True)
        parinfo[7]['limits'] = (0.1, 1.0)
        parinfo[8]['limited'] = (True,True)
        parinfo[8]['limits'] = (0., 10.0)
        parinfo[9]['limited'] = (True,True)
        parinfo[9]['limits'] = (0.2, 0.8)
        if fixip:
            parinfo[6]['fixed'] = True
        parinfo[7]['fixed'] = True
        parinfo[8]['fixed'] = True
        parinfo[9]['fixed'] = True
        
        self.c = 2.99792458e8 # speed of light
        self.dstep = np.zeros(npar)  # DSTEP used For emcee
        self.dstep[0] = 3/self.c
        self.dstep[1] = 0.00001
        self.dstep[2] = 0.0005
        self.dstep[3] = 0.0001
        self.dstep[4:6] = 0.005
        self.dstep[6:] = 0.01
        
        self.parinfo = parinfo
        xstep = 1/self.oversamp
        self.dxip = xstep        
        self.initmod = self.model(self.initpar)
       
    def __call__(self, par):
        model = self.model(par)
        lnprob = (self.obchunk-model)**2
        return -lnprob[3:len(lnprob)-1].sum()
    
    def model(self, par):
        z = par[0]
        wobs  = par[1] + par[2]*self.xchunk + par[3]*self.xchunk**2
        wover = par[1] + par[2]*self.xover  + par[3]*self.xover**2
        try:
            iodover = self.iodinterp(wover)
        except ValueError:
            print "iodine: ", self.wiod.min(), self.wiod.max()
            print "wover:  ", wover.min(), wover.max()
            print "WLS pars: ", par[1:4]
        polycoef = par[4:6]
        self.cont = np.polyval(polycoef[::-1], self.xover) # create continuum shape, reverse order of coefs for Python's poly evaluator
        if self.bstar:
            product = iodover * self.cont
        else:
            tempover = self.tempinterp(wover*(1+z)) # Doppler-shift stellar spectrum, put onto same oversampled scale as iodine
            product = (iodover * self.cont)* tempover

        #g = jjgauss(self.xip, 1.0, 0.0, par[6])
        ipar = par[6:]
        g = self.fprof(ipar)

        ip = numconv(g, self.ip)
        self.newip = ip / ip.sum() / self.oversamp
        model = rebin(wover, numconv(product, ip), wobs)
        return model
        
    def lm_model(self, x, *par):
        model = self.model(par)
        #print par[0], par[1]
        return model
    
    def mpfit_model(self, par, fjac=None, x=None, y=None, err=None):
        model = self.model(par)
        status = 0
        err = np.sqrt(self.obchunk)
        return (self.obchunk-model)/err

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
        dof = len(self.obchunk) - len(self.par)
        self.chi = np.sqrt(np.sum(self.resid**2/self.obchunk)/dof)
        return model, pfit

def grind(ffile, plot=False, printit=False, bstar=False):
    print 'Fitting ', ffile
    start = timer()
    h = fits.open(ffile, do_not_scale_image_data=True)
    wls = np.load('/Users/johnjohn/Dropbox/research/dopcode_new/MC_bstar_wls.air.npy')
    wls += 0.0
    tel = 3
    order = np.arange(2, 20)
    nord = len(order)
    dpix = 100
    pixel = np.arange(100, 1900, dpix)
    npix = len(pixel)
    charr = np.empty((npix,nord), dtype=object)
    for i, Ord in enumerate(order):
        for j, Pix in enumerate(pixel):
            chstart = timer()
            if printit:
                print '------'
                print 'Order:', Ord, 'Pixel:', Pix
            lpix = Pix
            rpix = Pix + dpix
            obchunk = h[0].data[tel,Ord,lpix:rpix] # Extract observed chunk
            wguess = np.zeros(2)
            wguess[0] = wls[lpix, Ord]
            wguess[1] = wls[lpix+1, Ord] - wls[lpix, Ord]
            ch = Chunk(obchunk, wguess, order=Ord, pixel=lpix, bstar=bstar)
            mod, bp = ch.mpfitter() # Full fit
            if plot:
                pl.plot(ch.obchunk, 'g')
                pl.plot(mod, 'b')
                resid = ch.obchunk - mod
                pl.plot(resid, '.-r')
                pl.show()
            charr[j, i] = ch
            #res = ch.obchunk - mod
            #s = np.sum((res/np.sqrt(ch.obchunk))**2)/(len(ch.obchunk) - 9)
            if printit:
                print 'chi^2 = ', ch.chi
                end = timer()
                print 'Chunk time: ', (end-chstart), 'seconds'
        end = timer()
    print 'Total time: ', (end-start)/60., 'minutes'
    return charr
                
def plot_ips(charr):
    sh = charr.shape
    nc = sh[0]
    nord = sh[1]
    npar = len(charr[0,0].par)
    pararr = np.empty((nc,nord,npar+1), dtype=float)
    nxip = len(charr[0,0].newip)
    iparr = np.zeros((nxip, nc, nord))
    for i in range(nc):
        for j in range(nord):
            iparr[:,i,j] = charr[i,j].newip
            pl.plot(charr[i,j].newip)
            pararr[i,j,:] = np.append(charr[i,j].par, charr[i,j].chi)
    pl.show()       
    return pararr, iparr
    
def globgrind(exp, bstar=False):
    files = glob(exp)
    ofarr = []
    pre = '/Users/johnjohn/Dropbox/research/dopcode_new/MINERVA_output/'
    for i, ffile in enumerate(files):
        charr = grind(ffile, bstar=bstar)
        parr = plot_ips(charr)
        thing = ffile.split('.')[:-1]        
        ofile = pre+'.'.join(thing[1:])+'.parr.npy'
        np.save(ofile, parr)
        ofarr.append(ofile)
        print ofile
    parray = getpararray(ofarr)
    return ofarr, parray

def getpararray(ofile):
    test, foo = np.load(ofile[0])
    sh = test.shape
    parr = np.zeros((len(ofile), sh[0], sh[1], sh[2]))
    for i, ffile in enumerate(ofile):
        p, ip = np.load(ffile)
        parr[i, :, :, :] = p
    return parr