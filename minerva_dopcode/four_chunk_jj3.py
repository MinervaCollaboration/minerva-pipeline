# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 09:33:31 2016

@author: johnjohn
"""

import numpy as np
import matplotlib.pyplot as pl
from scipy.interpolate import interp1d
from astropy.io import fits
import pyfits
import numpy.random as rand
from scipy.optimize import curve_fit
import mpyfit #This requires a special installation
import emcee
from timeit import default_timer as timer
from utils import *
from astropy.time import Time
from glob import glob
import socket, ipdb, os
import csv
import math
import random

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
        h = pyfits.open(ffile, mode='update')
        t = Time(h[0].header['DATE-OBS'], format='isot', scale='utc')
        self.jd = t.jd
        rpix = lpix + dpix
        obspec = h[0].data[:, order, lpix:rpix]
        shape = obspec.shape
        ntel = shape[0]
        npix = shape[1]
        self.ntel = ntel
        self.xchunk = np.arange(npix)
        self.oversamp = 4.
        pad = 120
        self.xover  = np.arange(-pad,npix+pad,1./self.oversamp)
        wls = np.load('templates/MC_bstar_wls.air.npy')
        w0g = wls[lpix, order] # Guess at wavelength zero point (w0)
        dwg = wls[lpix+1, order] - wls[lpix, order] # Guess at dispersion (dw)
        wmin = w0g - 4 * pad * dwg # Set min and max ranges of iodine template
        wmax = w0g + (npix + 4*pad) * dwg
        self.wiod, self.siod  = get_iodine(wmin, wmax)
        self.iodinterp = interp1d(self.wiod, self.siod) # Set up iodine interpolation function

        # don't assume the same object or same redshift for each trace
        objname = np.array([])
        midflux = np.array([])
        ra = np.array([])
        dec = np.array([])
        pmra = np.array([])
        pmdec = np.array([])
        parallax = np.array([])
        rv = np.array([])
        bc = np.array([])
        active = np.array([])
        zguess = np.array([])
        self.tempinterp = []
        self.weight = np.zeros(self.ntel)

        for i in range(ntel):
            objname = np.append(objname,h[0].header['OBJECT' + str(i+1)])
            ra = np.append(ra, h[0].header['TARGRA' + str(i+1)])
            dec = np.append(dec, h[0].header['TARGDEC' + str(i+1)])

            # if keyword not populated, bu
            if h[0].header['FLUXMID' + str(i+1)] == 'UNKNOWN':
                midflux = np.append(midflux, self.jd+h[0].header['EXPTIME']/2.0/86400.0)
                h[0].header['FLUXMID' + str(i+1)] = midflux[-1]
            else:
                midflux = np.append(midflux, h[0].header['FLUXMID' + str(i+1)])

            # try to get the barycentric correction redshift from the header                                                               
            try:
                zbc = float(h[0].header['BARYCOR' + str(i)])
            except:
                try: pmra = np.append(pmra, h[0].header['PMRA' + str(i+1)])
                except: pmra = np.append(pmra, 0.0)
                try: pmdec = np.append(pmdec, h[0].header['PMDEC' + str(i+1)])
                except: pmdec = np.append(pmdec, 0.0)
                try: parallax = np.append(parallax,h[0].header['PARLAX' + str(i+1)])
                except: parallax = np.append(parallax, 0.0)
                try: rv = np.append(rv, h[0].header['RV' + str(i+1)])
                except: rv = np.append(rv, 0.0)
                zbc = barycorr(midflux[i], ra[i], dec[i], pmra=pmra[i], pmdec=pmdec[i], parallax=parallax[i], rv=rv[i])/self.c
                h[0].header['BARYCOR' + str(i)] = zbc

            active = np.append(active, h[0].header['FAUSTAT' + str(i+1)] == 'GUIDING')

            if not bstar:
            
                if 'HD' in objname[i] and 'A' in objname[i]:
                    objname[i] = objname[i].split('A')[0]

                templatename = None
                # get the barycentric correction of the template from kbcvel.ascii
                if 'daytimeSky' in objname[i]: 
                    # starting point for solar spectrum template
                    wtemp, stemp = get_template('templates/nso.sav',wmin, wmax)
                    self.tempinterp.append(interp1d(wtemp, stemp))
                    zguess = np.append(zguess,2.8e-4)
                elif 'HD' in objname[i]:
                    with open('templates/kbcvel.ascii','rb') as fh:
                        fh.seek(1) # skip the header
                        for line in fh:
                            entries = line.split()
                            if len(entries) >= 6:
                                obstype = entries[5]
                                tempobjname = entries[1]
                                # **this assumes all stars are designated by their HD names!**
                                if obstype == 't' and tempobjname == objname[i].split('D')[1]:
                                    ztemp = float(entries[2])/self.c # template redshift
                                    templatename = 'templates/dsst' + tempobjname + 'ad_' + entries[0].split('.')[0] + '.dat'
                                    if os.path.exists(templatename): 
                                        wtemp, stemp = get_template(templatename,wmin, wmax)
                                        self.tempinterp.append(interp1d(wtemp, stemp))
                                        break
                    if templatename == None:
                        print 'ERROR: no template for ' + obsname
                        return
                    if not os.path.exists(templatename):
                        print 'ERROR: no template for ' + obsname
                        return

                    # the starting guess for the redshift will be the difference 
                    # between the barycentric correction of the observation and 
                    # the template
                    zguess = np.append(zguess,(1.0+ztemp)/(1.0+zbc) - 1.0)


            else:
                zguess = np.append(zguess,0.0)
            self.bstar = bstar
            self.juststar = juststar


        h.flush()
        h.close()
                
        obchunk = np.copy(obspec)

#       JDE 2016-05-24: remove CR cleaning; gave DOF <= 0 error (and fixed in extraction?)
#        for i in range(4):
#            obchunk[i,:] = crclean(obspec[i,:]) # remove hot/dead pixels

        self.origchunk = obspec
        self.obchunk = obchunk # Bookkeeping for what was "cleaned" by crclean

        self.xip, self.ip = get_ip(wmin,wmax,order)
        """ List of Parameters:
        par[0 + N*parspertrace] = zguess for telescope N
        par[1 + N*parspertrace] = w0 for telescope N
        par[2 + N*parspertrace] = dw/dx for telescope N
        par[3 + N*parspertrace] = continuum shape offset for telescope N
        par[4 + N*parspertrace] = continuum shape slope for telescope N
        par[5 + N*parspertrace] = IP broadening for telescope N
        par[6 + N*parspertrace] = IP skew for telescope N
        """
        self.parspertrace = 7

        npar = self.parspertrace*self.ntel
        self.initpar = np.zeros(npar)

        # the fibers are tilted, which changes the wavelength offset per telescope
        # this has been previously measured and ~static (unless the fiber assembly is moved)
        coef = np.array([0.00059673, 0.06555841]) # wavelength offset per trace...per order
        dwdtrace = np.polyval(coef, order) # find wavelength offset per trace for this order
        offset = np.arange(self.ntel)[::-1] * dwdtrace

        # parameter properties for MPYFIT
        self.parinfo = [{'fixed':False, 'limited':(False,False), 'limits':(None, None), 'relstep':0.} for i in range(npar)]
        
        for i in range(self.ntel):

            # redshift guess (may tie these together for each trace, if same object)
            self.initpar[0+i*self.parspertrace] = zguess[i]
            self.parinfo[0+i*self.parspertrace]['relstep'] = 0.01        
            if bstar: self.parinfo[0+i*self.parspertrace]['fixed'] = True

            # wavelength zeropoint guess
            self.initpar[1+i*self.parspertrace] = w0g + offset[i]
            self.parinfo[1+i*self.parspertrace]['relstep'] = 1e-6
            if juststar: self.parinfo[1+i*self.parspertrace]['fixed'] = True

            # wavelength dispersion guess (may tie these together for each trace)
            self.initpar[2+i*self.parspertrace] = dwg
            self.parinfo[2+i*self.parspertrace]['relstep'] = 0.005
            if juststar: self.parinfo[2+i*self.parspertrace]['fixed'] = True

            # continuum offset guess
            self.initpar[3+i*self.parspertrace] = np.amax(self.obchunk[i,:])
            
            # continuum slope guess
            self.initpar[4+i*self.parspertrace] = 0.0

            # IP broadening guess
            self.initpar[5+i*self.parspertrace] = 0.2           
            self.parinfo[5+i*self.parspertrace]['limited'] = (True,True)
            self.parinfo[5+i*self.parspertrace]['limits'] = (0.1, 4.0)
            if fixip: self.parinfo[5+i*self.parspertrace]['fixed'] = True

            # IP skew guess
            self.initpar[6+i*self.parspertrace] = 0.0
            self.parinfo[6+i*self.parspertrace]['limited'] = (True,True)
            self.parinfo[6+i*self.parspertrace]['limits'] = (-1.0, 1.0)
            self.parinfo[6+i*self.parspertrace]['fixed'] = True
                
        xstep = 1/self.oversamp
        self.dxip = xstep   
        self.initmod = self.model(self.initpar)
       
    def __call__(self, par): # For EMCEE
        model = self.model(par)
        lnprob = (self.obchunk-model)**2
        return -lnprob[3:len(lnprob)-1].sum()
    
    def model(self, par): # For other solvers such as MPYFIT

        for i in range(self.ntel):

            # redshift
            z = par[0+i*self.parspertrace]

            # observed wavelength scale
            wobs = par[1+i*self.parspertrace] + par[2+i*self.parspertrace]*self.xchunk

            # oversampled wavelength scale
            wover = par[1+i*self.parspertrace] + par[2+i*self.parspertrace]*self.xover

            # create continuum shape
            contf = par[3+i*self.parspertrace] + par[4+i*self.parspertrace]*self.xover

            # instrumental profile
            sigma = par[5+i*self.parspertrace]
            alpha = 0.0# par[6+i*self.parspertrace]
            sknorm = skewnorm(self.xip, 0.0, sigma, alpha)
            ip = numconv(sknorm, self.ip) # Convolve Zemax IP with local broadening
            ip = ip / ip.sum() / self.oversamp # Normalize convolved IP
            
            if self.bstar:
                tempover = 1.0
            else:
                # Doppler-shift stellar spectrum, put onto same oversampled scale as iodine
                try: tempover = self.tempinterp[i](wover*(1.0-z))
                except ValueError: tempover = wover*np.inf 
            if self.juststar:
                iodover = contf * tempover
            else:
                try: iodover = self.iodinterp(wover)*contf*tempover
                except ValueError: iodover = wover*np.inf

            # oversampled model spectrum
            sover = numconv(iodover, ip) 
            if i == 0:
                try: model = rebin(wover, sover, wobs)
                except: model = wobs*np.inf
            else:
                try: model = np.vstack((model, rebin(wover, sover, wobs)))
                except: model = np.vstack((model,wobs*np.inf))

#        except ValueError:
#            print "iodine: ", self.wiod.min(), self.wiod.max()
#            print "wover:  ", wover.min(), wover.max()
#            print "WLS pars: ", par[1:4]
            
        return model

    def lm_model(self, x, *par):
        model = self.model(par)
        #print par[0], par[1]
        return model
    
    def mpfit_model(self, par, fjac=None, x=None, y=None, err=None):
        model = self.model(par).flatten()
        obs = self.obchunk.flatten()
        err = np.sqrt(obs)

        if len((np.where(np.logical_not(np.isfinite(obs))))[0]) != 0: ipdb.set_trace()

        bad = np.where(obs == 0)
        err[bad] = float('inf')

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

#        pfit, results = mpyfit.fit(self.mpfit_model, self.initpar, parinfo=self.parinfo, maxiter=50, xtol=1e-8, ftol=1e-8)
        kwargs = {'parinfo':self.parinfo,'maxiter':50,'xtol':1e-8,'ftol':1e-8}
        pfit, results = timeout(mpyfit.fit,(self.mpfit_model, self.initpar,), kwargs=kwargs,timeout_duration=300,default=(None,None))
        if pfit == None: ipdb.set_trace()

        model = self.model(pfit)
        self.par = pfit
        self.mod = model
        self.resid = self.obchunk - model
        npar = len(self.par)
        dim = self.obchunk.shape
        ntel = dim[0]
        npix = dim[1]
        dof = npix - self.parspertrace # neglects any fixed parameters!
#        dofpertrace = npix - npar
#        self.chiarr = np.sqrt(np.sum((self.resid)**2/self.mod, axis=1) / dofpertrace)
#        bad = np.where(self.chiarr > 3)
        """
        if bad[0]:
            nbad = len(bad[0])
            for i in range(nbad):
                print bad[0][i], 'is bad'
        """
        sigmasq = self.obchunk # sigma = sqrt(Ncts), sigma^2 = Ncts

        # mask bad values
        bad = np.where(sigmasq == 0.0)
        sigmasq[bad] = float('inf')

        chisq = np.sum(self.resid**2/sigmasq,axis=1)

        if len(np.where(np.isnan(chisq))[0]) != 0: ipdb.set_trace()

        redchi = chisq/dof
        self.chisq = chisq
        self.redchi = redchi
        self.chi = np.sqrt(redchi) # sqrt(chi^2) =~ number of sigmas 
        wav = np.append(np.append(-1, self.xchunk), 1) * pfit[2] + pfit[1]
        dV = np.median(wav - np.roll(wav,-1))/wav * self.c
        if self.bstar:
            self.weight = np.zeros(self.ntel)
        else:
            for i in range(ntel):
                I = self.tempinterp[i](wav)
                dI = I - np.roll(I, -1)
                dIdV = dI[1:len(wav)-1]/dV[1:len(wav)-1]
                sigmaV = 1.0 / np.sqrt(np.sum(dIdV**2 / I[1:len(wav)-1]))
                self.weight[i] = 1.0/sigmaV**2
        return model, pfit

def grind(obsname, plot=False, printit=False, bstar=False, juststar=False):
    print 'Fitting ', obsname
    start = timer()
    order = np.arange(2, 20)
    nord = len(order)
    dpix = 128
    pixel = np.arange(64, 1900, dpix)
    npix = len(pixel)
    #charr = np.empty((npix,nord), dtype=object)
    chrec = np.zeros(npix*nord, dtype=[('pixel',int),('order',int),
                                       ('z1',float),('z2',float),('z3',float),('z4',float),
                                       ('w01',float),('w02',float),('w03',float),('w04',float),
                                       ('dwdx1',float),('dwdx2',float),('dwdx3',float),('dwdx4',float),
                                       ('cts1',float),('cts2',float),('cts3',float),('cts4',float),
                                       ('slope1',float),('slope2',float),('slope3',float),('slope4',float),
                                       ('sigma1',float),('sigma2',float),('sigma3',float),('sigma4',float),
                                       ('alpha1',float),('alpha2',float),('alpha3',float),('alpha4',float),
                                       ('wt1',float), ('wt2',float), ('wt3',float), ('wt4',float), 
                                       ('chi1',float),('chi2',float),('chi3',float),('chi4',float)])
    chrec = np.rec.array(chrec, dtype=chrec.dtype)
    for i, Ord in enumerate(order):
        for j, Pix in enumerate(pixel):
            chstart = timer()
            if printit:
                print '------'
                print 'Order:', Ord, 'Pixel:', Pix
            ch = FourChunk(obsname, Ord, Pix, dpix, bstar=bstar)


#            mod, bp = ch.mpfitter() # Full fit
            mod,bp = timeout(ch.mpfitter,timeout_duration=300,default=(None,None))
            if mod == None: ipdb.set_trace()

#            try: mod, bp = ch.mpfitter() # Full fit
#            except: return # no template for observation

            chind = j+i*npix
            chrec[chind].pixel = Pix
            chrec[chind].order = Ord
#            chrec[chind].bp = bp

            parspertrace = 7

            # there must be a better way...
            chrec[chind].z1 = bp[0 + 0*parspertrace]
            chrec[chind].z2 = bp[0 + 1*parspertrace]
            chrec[chind].z3 = bp[0 + 2*parspertrace]
            chrec[chind].z4 = bp[0 + 3*parspertrace]

            chrec[chind].w01 = bp[1 + 0*parspertrace]
            chrec[chind].w02 = bp[1 + 1*parspertrace]
            chrec[chind].w03 = bp[1 + 2*parspertrace]
            chrec[chind].w04 = bp[1 + 3*parspertrace]

            chrec[chind].dwdx1 = bp[2 + 0*parspertrace]
            chrec[chind].dwdx2 = bp[2 + 1*parspertrace]
            chrec[chind].dwdx3 = bp[2 + 2*parspertrace]
            chrec[chind].dwdx4 = bp[2 + 3*parspertrace]

            chrec[chind].cts1 = bp[3 + 0*parspertrace]
            chrec[chind].cts2 = bp[3 + 1*parspertrace]
            chrec[chind].cts3 = bp[3 + 2*parspertrace]
            chrec[chind].cts4 = bp[3 + 3*parspertrace]

            chrec[chind].slope1 = bp[4 + 0*parspertrace]
            chrec[chind].slope2 = bp[4 + 1*parspertrace]
            chrec[chind].slope3 = bp[4 + 2*parspertrace]
            chrec[chind].slope4 = bp[4 + 3*parspertrace]

            chrec[chind].sigma1 = bp[5 + 0*parspertrace]
            chrec[chind].sigma2 = bp[5 + 1*parspertrace]
            chrec[chind].sigma3 = bp[5 + 2*parspertrace]
            chrec[chind].sigma4 = bp[5 + 3*parspertrace]

            chrec[chind].alpha1 = bp[6 + 0*parspertrace]
            chrec[chind].alpha2 = bp[6 + 1*parspertrace]
            chrec[chind].alpha3 = bp[6 + 2*parspertrace]
            chrec[chind].alpha4 = bp[6 + 3*parspertrace]
            
            chrec[chind].wt1 = ch.weight[0]
            chrec[chind].wt2 = ch.weight[1]
            chrec[chind].wt3 = ch.weight[2]
            chrec[chind].wt4 = ch.weight[3]

            chrec[chind].chi1 = ch.chi[0]
            chrec[chind].chi2 = ch.chi[1]
            chrec[chind].chi3 = ch.chi[2]
            chrec[chind].chi4 = ch.chi[3]
            if plot:
                col = ['b', 'g', 'y', 'r']
                for i in range(4): 
                    wav = bp[1+i*parspertrace] + bp[2+i*parspertrace]*np.arange(mod.shape[1])
                    pl.plot(wav, ch.obchunk[i, :], col[i]+'.')
                    pl.plot(wav, mod[i, :], col[i])
                    pl.plot(wav, ch.resid[i, :]+0.6*mod.flatten().min(), col[i]+'.')
                pl.xlabel('Wavelength [Ang]')
                pl.ylabel('Flux [cts/pixel]')
                pl.xlim((wav.min(), wav.max()))
                pl.show()
            #charr[j, i] = ch
            if printit:
                print 'chi^2 = ', ch.chi
                if len(np.where(np.isnan(ch.chi))[0]) != 0: ipdb.set_trace()
                end = timer()
                print 'Chunk time: ', (end-chstart), 'seconds'
        end = timer()
    print 'Total time: ', (end-start)/60., 'minutes'
    return chrec

'''
def getparr(charr, plot=False):
    sh = charr.shape
    if len(sh) < 2: ipdb.set_trace()
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
    ofile = getpath(night=night,data=True) + '/' + obsname
    test = np.load(ofile[0])
    sh = test.shape
    parr = np.zeros((len(ofile), sh[0], sh[1], sh[2]))
    for i, ffile in enumerate(ofile):
        p = np.load(ffile)
        parr[i, :, :, :] = p
    return parr
'''

def getpath(night=''):
    hostname = socket.gethostname()
    if hostname == 'Main':
        return '/Data/kiwispec-proc/' + night
    elif hostname == 'jjohnson-mac':
        path = '/Users/johnjohn/Dropbox/research/dopcode_new/MINERVA_data/'
    else:
        print 'hostname ' + hostname + ') not recognized; exiting'
        sys.exit()

def globgrind(globobs, bstar=False, returnfile=False, printit=False,plot=False,redo=False):
    files = glob(globobs)
    ofarr = [] # object f? array?

    for i, ffile in enumerate(files):
        h = fits.open(ffile)

        # if the iodine cell is not in, just fit the star without iodine
        juststar = h[0].header['I2POSAS'] != 'in'

        ofile = os.path.splitext(ffile)[0] + '.chrec3.npy'
        ofarr.append(ofile)

        if not returnfile:
            if redo or not os.path.exists(ofile):
                chrec = grind(ffile, bstar=bstar, juststar=juststar, printit=printit, plot=plot)
                if chrec != None: 
                    np.save(ofile,chrec)
                    print ofile, h[0].header['I2POSAS']
    return ofarr

def globgrindall(shuffle=False):
    datadirs = glob(getpath() + '*')
    if shuffle: random.shuffle(datadirs)
    for datadir in datadirs:
#        ofarr = globgrind(datadir + '/*daytimeSky*.proc.fits', bstar=False, returnfile=False, printit=True, plot=False)
#        ofarr = globgrind(datadir + '/*HD*.proc.fits', bstar=False, returnfile=False, printit=True, plot=False)
#        ofarr = globgrind(datadir + '/*HR*.proc.fits', bstar=True, returnfile=False, printit=True, plot=False)
        ofarr = globgrind(datadir + '/*HD122064*.proc.fits', bstar=False, returnfile=False, printit=True, plot=False)
        ofarr = globgrind(datadir + '/*HD185144*.proc.fits', bstar=False, returnfile=False, printit=True, plot=False)
#        ofarr = globgrind(datadir + '/*daytimeSky*.proc.fits', bstar=False, returnfile=False, printit=True, plot=False)

#ofarr = globgrind('/Data/kiwispec-proc/n20160212/n20160212.HD191408A.0017.proc.fits',bstar=False,returnfile=False,printit=True,plot=False)

#globgrind('/Data/kiwispec-proc/n20160520/n20160520.HD62613.0030.proc.fits',bstar=False, returnfile=False, printit=True, plot=False)
#ipdb.set_trace()


globgrindall(shuffle=True)
ipdb.set_trace()



'''
# a quality set of daytime sky spectra
globobs = getpath(night='n20160305') + '/n20160305.daytimeSky.008*.proc.fits'
ofarr = globgrind(globobs, bstar=False, returnfile=False, printit=True, plot=True)
ipdb.set_trace()

# a quality spectrum of HD9407
globobs = getpath(night='n20160505') + '/n20160505.HD*.proc.fits'
ofarr = globgrind(globobs, bstar=False, returnfile=False, printit=True, plot=False)
ipdb.set_trace()

# poor quality spectra (missing telescopes, etc)
globobs = getpath(night='n20160504') + '/n20160504.HD*.proc.fits'
ofarr = globgrind(globobs, bstar=False, returnfile=False, printit=True, plot=True)
ipdb.set_trace()

# mixed quality B star spectra
globobs = getpath(night='n20160504') + '/n20160504.HR*.proc.fits'
ofarr = globgrind(globobs, bstar=True, returnfile=False, printit=True, plot=True)
ipdb.set_trace()
'''

#obsname = getpath(night='n20160305') + '/n20160305.daytimeSky.0065.proc.fits'
#chrec = grind(obsname, plot=False, printit=True)
#parr, iparr = getparr(charr) #This no longer needed with CHREC variable




ipdb.set_trace()
parray = getpararray(ofarr)
#######################

ipdb.set_trace()

#gfile = getpath(night='n20160323') + '/n20160323.HR4828.00*.proc.fits'
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


#ffile = getpath(night='n20160323') + '/n20160323.HR4828.0020.proc.fits'
obsname = 'n20160305.daytimeSky.0065.proc.fits' #iodine
#ffile = getpath(night='n20160305') + '/n20160305.daytimeSky.0069.proc.fits'
wls = np.load('templates/MC_bstar_wls.air.npy')

ffile = getpath(night='n20160305') + '/'+obsname
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

