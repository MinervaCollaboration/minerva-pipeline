import matplotlib
#matplotlib.use('Agg',warn=False)
import numpy as np
import os
import glob
import ipdb
import utils
from astropy.io import fits
from astropy.time import Time
import os, sys
import matplotlib.pyplot as plt
import targetlist
import math

def vank(objname, weightthresh=1.0,chithresh=99.0, sigmaithresh=1.0):

    c = 299792458.0

    #filenames = glob.glob('/Data/kiwispec-proc/n20160[4,5,6]*/*' + objname + '*.chrec.npy')
    filenames = glob.glob('/Data/kiwispec-proc/n2017*/*' + objname + '*.chrec6.npy')

    nobs = len(filenames)
    if nobs <= 3: return

    vji = []
    wji = []
    chiji = []
    ctsji = []
    jdutcs = np.array([])

    for filename in filenames:

        chrec = np.load(filename)
        fitsname = os.path.splitext(os.path.splitext(filename)[0])[0] + '.fits'
        h = fits.open(fitsname)
        t = Time(h[0].header['DATE-OBS'], format='isot',scale='utc')
        midflux = t.jd + h[0].header['EXPTIME']/2.0/86400.0
        jdutcs = np.append(jdutcs,midflux)

        # reject chunks with bad DSST (Step 1)
        ipdb.set_trace()
        bad = np.where((chrec['wt1'] == 0) | (chrec['wt2'] == 0) | (chrec['wt3'] == 0) | (chrec['wt4'] == 0))
        chrec['z1'][bad] = chrec['z2'][bad] = chrec['z3'][bad] = chrec['z4'][bad] = np.nan

        # reject chunks where fitter failed to converge (rchi2 == 0 or 100)
        # (Step 2)
        bad = np.where(chrec['chi'] == 0.0)
        bad2 = np.where(chrec['chi'] == 100.0)
        chrec['z1'][bad] = chrec['z2'][bad] = chrec['z3'][bad] = chrec['z4'][bad] = np.nan
        if len(bad2[0]) != 0: ipdb.set_trace()

        # reject chunks with the lowest DSST weight (Step 3)
        lowweight = np.percentile(chrec['wt'],weightthresh)
        bad = np.where(chrec['wt'] <= lowweight)
        chrec['z'][bad] = np.nan

#        rverr = utils.robust_sigma(chrec['z'])*c
        ra = h[0].header['TARGRA1']
        dec = h[0].header['TARGDEC1']
        try: pmra = h[0].header['PMRA1']
        except: pmra = 0.0
        try: pmdec = h[0].header['PMDEC1']
        except: pmdec = 0.0
        try: parallax = h[0].header['PARLAX1']
        except: parallax = 0.0
        try: rv = h[0].header['RV1']
        except: rv = 0.0
        
        result = utils.barycorr(midflux, ra, dec, pmra=pmra, pmdec=pmdec, parallax=parallax, rv=rv, zmeas=0.0)
        zb = result/c
        rvs = c*((1.0+chrec['z'])*(1.0+zb)-1.0)

        if len(vji) == 0: vji = rvs
        else: vji = np.vstack((vji,rvs))

        if len(wji) == 0: wji = chrec['wt']
        else: wji = np.vstack((wji,chrec['wt']))
            
        if len(chiji) == 0: chiji = chrec['chi']
        else: chiji = np.vstack((chiji,chrec['wt']))

        if len(ctsji) == 0: ctsji = chrec['cts']
        else: ctsji = np.vstack((ctsji,chrec['cts']))

    nchunks = len(chrec)
    vij = np.transpose(vji)
    wij = np.transpose(wji)
    chiij = np.transpose(chiji)
    ctsij = np.transpose(ctsji)
    

    snr = np.sqrt(np.sum(ctsij,axis=0))

    # reject chunks with the worst fits (step 4)
    chimed = np.nanmedian(chiij,axis=1)
    hichi = np.percentile(chimed,chithresh)
    bad = np.where(chimed >= hichi)
    vij[bad,:] = np.nan  

    # adjust all chunks to have the same RV zero points (step 5)
    # subtract the mean velocity of all observations from each chunk
    vij -= np.transpose(np.tile(np.nanmean(vij,axis=1),(nobs,1)))

    # compute chunk weights (step 6):
    # median velocities for each observation
    vjmed = np.nanmedian(vij,axis=0) # eq 2.9

    # compute the matrix of velocity differences 
    Deltaij = vij - np.tile(vjmed,(nchunks,1)) # eq 2.8
    sigmai = np.nanstd(Deltaij,axis=1) # eq 2.6

    # reject the highest sigmai (step 7)
    bad = np.where(sigmai == 0.0)
    sigmai[bad] = np.inf
    hisigma = np.percentile(sigmai,sigmaithresh)
    bad = np.where(sigmai >= hisigma)
    sigmai[bad] = np.inf

    # compute rj (eq 2.7)
    rj = np.nanmedian(np.abs(Deltaij)*np.transpose(np.tile(sigmai,(nobs,1))),axis=0) # eq 2.7
    sigmaij = np.transpose(np.tile(sigmai,(nobs,1)))*np.tile(rj,(nchunks,1)) # eq 2.5

    # prevent nans
    bad = np.where(sigmaij == 0.0)
    sigmaij[bad] = np.inf

    wij = 1.0/sigmaij**2

    vj = np.nansum(vij*wij,axis=0)/np.nansum(wij,axis=0) # eq 2.10
    sigmavj = 1.0/np.sqrt(np.nansum(wij,axis=0)) # eq 2.12
    
    print objname + " RMS: " + str(np.nanstd(vj))
    
    # plot scatter vs time
    plt.plot(jdutcs-2457389,vj,'bo')
    plt.title(objname)
    plt.xlabel('Days since UT 2016-01-01')
    plt.ylabel('RV (m/s)')
    plt.savefig(objname + '.png')
    plt.close()

    # create a histogram of scatter within a night
    mindate = int(math.floor(min(jdutcs)))
    maxdate = int(math.ceil(max(jdutcs)))

    jdbin = []
    rvbin = []
    errbin = []
    sigmabin = []
    for i in range(mindate,maxdate):
        match = np.where((jdutcs >= i) & (jdutcs < (i+1)))
        if len(match[0]) > 1:
            jdbin.append(mindate)
            rvbin.append(np.mean(vj[match]))
            errbin.append(np.std(vj[match]))
            sigmabin.append(np.mean(sigmavj[match]))

#    print jdbin, rvbin, errbin, sigmabin

    hist, bins = np.histogram(errbin, bins=20)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.xlabel('Intranight RMS (m/s)')
    plt.ylabel('# of Nights')
    plt.savefig(objname + '.hist.png')
    plt.close()


    # plot scatter within a night vs SNR
    plt.plot(sigmabin, errbin,'bo')
    plt.title(objname)
    plt.xlabel('Sigma_v_j')
    plt.ylabel('Intranight RMS (m/s)')
    plt.savefig(objname + '.SNR.png')
    plt.close()

    # subtract a linear trend from vj
    good = np.where(np.isfinite(vj))
    t0 = np.nanmedian(jdutcs[good])
    coeffs = np.polyfit(jdutcs[good]-t0,vj[good],1)
    vjtrend = coeffs[1] + (jdutcs[good]-t0)*coeffs[0] 

    # plot scatter minus trend vs time
    plt.plot(jdutcs[good]-2457389,vj[good]-vjtrend,'bo')
    plt.title(objname)
    plt.xlabel('Days since UT 2016-01-01')
    plt.ylabel('RV (m/s)')
    plt.savefig(objname + '.detrended.png')
    plt.close()
    print objname + " Detrended RMS: " + str(np.nanstd(vj[good]-vjtrend))

if __name__ == "__main__":
                       
#    filename = '/Data/kiwispec-proc/n20160524/n20160524.HD185144.0023.proc.chrec.npy'
#    vank(filename)
    #objnames = ['HD10700','HD9407','HD62613','HD122064','HD191408A','HD185144','HD217107','daytimeSky']

    
    objnames =  ['HD217107']#['HD122064','HD185144']
    for objname in objnames:
        vank(objname)
    
   # sys.exit()


    #targets = targetlist.rdlist()

#    for objname in targets['name']:
#        vank(objname)
