import numpy as np
import os
import glob
import ipdb
import utils
from astropy.io import fits
import pyfits
from astropy.time import Time
import os, sys
import matplotlib.pyplot as plt
import targetlist
import math
import allantools 

def vank(objname, weightthresh=10.0,chithresh=90.0, sigmaithresh=10.0):

    c = 299792458.0

    filenames = glob.glob('/Data/kiwispec-proc/n20160[5,6]*/*' + objname + '*.chrec2.npy')

    ntel = 4
    nobs = len(filenames)*ntel
    if nobs <= 3: return

    vji = []
    wji = []
    chiji = []
    ctsji = []
    jdutcs = np.array([])
    telescope = np.array([])

    for filename in filenames:


        chrec = np.load(filename)
        fitsname = os.path.splitext(os.path.splitext(filename)[0])[0] + '.fits'
        h = pyfits.open(fitsname,mode='update')
#        h = fits.open(fitsname)
        


#        # reject chunks with bad DSST (Step 1)
#        bad = np.where(chrec['wt' + str(i)] == 0)
#        chrec['z' + str(i)][bad] = np.nan

        # reject chunks where fitter failed to converge (rchi2 == 0 or 100)
        # (Step 2)
        for i in range(1,ntel+1):

            if h[0].header['FLUXMID' + str(i)] == 'UNKNOWN':
                t = Time(h[0].header['DATE-OBS'], format='isot',scale='utc')
                midflux = t.jd+h[0].header['EXPTIME']/2.0/86400.0
                h[0].header['FLUXMID' + str(i)] = midflux
            else:
                midflux = h[0].header['FLUXMID' + str(i)]

            # very noisy data can find great fits
            bad = np.where(chrec['chi' + str(i)] <= 0.3)
            chrec['z' + str(i)][bad] = np.nan

            # very noisy data can find great fits
            bad = np.where((chrec['alpha' + str(i)] <= -0.95) | (chrec['alpha' + str(i)] >= 0.95))
            chrec['z' + str(i)][bad] = np.nan

            # very noisy data can find great fits
            bad = np.where((chrec['sigma' + str(i)] <= 0.25) | (chrec['sigma' + str(i)] >= 1.25))
            chrec['z' + str(i)][bad] = np.nan

            # reject chunks with the lowest DSST weight (Step 3) or bad weights (Step 1)
            lowweight = np.percentile(chrec['wt' + str(i)],weightthresh)
            bad = np.where((chrec['wt' + str(i)] <= lowweight) | (chrec['wt' + str(i)] == 0))
            chrec['z' + str(i)][bad] = np.nan

            if 'BARYCOR' + str(i) not in h[0].header.keys():
                # do the barycentric correction
                ra = h[0].header['TARGRA' + str(i)]
                dec = h[0].header['TARGDEC' + str(i)]
                try: pmra = h[0].header['PMRA' + str(i)]
                except: pmra = 0.0
                try: pmdec = h[0].header['PMDEC' + str(i)]
                except: pmdec = 0.0
                try: parallax = h[0].header['PARLAX' + str(i)]
                except: parallax = 0.0
                try: rv = h[0].header['RV' + str(i)]
                except: rv = 0.0
            
                result = utils.barycorr(midflux, ra, dec, pmra=pmra, pmdec=pmdec, parallax=parallax, rv=rv, zmeas=0.0)
                zb = result/c
                h[0].header['BARYCOR' + str(i)] = zb
            elif h[0].header['BARYCOR' + str(i)] == 'UNKNOWN':
                # do the barycentric correction
                ra = h[0].header['TARGRA' + str(i)]
                dec = h[0].header['TARGDEC' + str(i)]
                try: pmra = h[0].header['PMRA' + str(i)]
                except: pmra = 0.0
                try: pmdec = h[0].header['PMDEC' + str(i)]
                except: pmdec = 0.0
                try: parallax = h[0].header['PARLAX' + str(i)]
                except: parallax = 0.0
                try: rv = h[0].header['RV' + str(i)]
                except: rv = 0.0
            
                result = utils.barycorr(midflux, ra, dec, pmra=pmra, pmdec=pmdec, parallax=parallax, rv=rv, zmeas=0.0)
                zb = result/c
                h[0].header['BARYCOR' + str(i)] = zb
            else:
                zb = h[0].header['BARYCOR' + str(i)]

#            active = np.append(active ,h[0].header['FAUSTAT' + str(i)] == 'GUIDING')
            if h[0].header['FAUSTAT' + str(i)] == 'GUIDING':

                jdutcs = np.append(jdutcs,midflux)
                telescope = np.append(telescope,i)

                rvs = c*((1.0+chrec['z' + str(i)])*(1.0+zb)-1.0)

                if len(vji) == 0: vji = rvs
                else: vji = np.vstack((vji,rvs))

                if len(wji) == 0: wji = chrec['wt' + str(i)]
                else: wji = np.vstack((wji,chrec['wt' + str(i)]))
            
                if len(chiji) == 0: chiji = chrec['chi' + str(i)]
                else: chiji = np.vstack((chiji,chrec['wt' + str(i)]))

                if len(ctsji) == 0: ctsji = chrec['cts' + str(i)]
                else: ctsji = np.vstack((ctsji,chrec['cts' + str(i)]))
            else: nobs-=1

        h.flush()
        h.close()

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
    sigmai[ bad] = np.inf

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
#    plt.plot(jdutcs-2457389,vj,'bo')
    colors = ['','r','g','b','orange']
    for i in range(1,ntel+1):
        match = np.where(telescope == i)
        plt.plot(jdutcs[match]-2457389,vj[match],'o',color=colors[i])

    plt.title(objname)
    plt.xlabel('Days since UT 2016-01-01')
    plt.ylabel('RV (m/s)')
    plt.savefig(objname + '.png')
    plt.close()
    
    nightlyrvs = {'all':np.array([]),
                  'T1':np.array([]),
                  'T2':np.array([]),
                  'T3':np.array([]),
                  'T4':np.array([])}

    # bin per telescope per night
    for jd in range(2457480,2457571):
        for i in range(1,ntel+1):
            match = np.where((jdutcs >= jd) & (jdutcs < (jd+1)) & np.isfinite(vj) & (telescope==i))
            night = np.mean(jdutcs[match])-2457389
            nightlyrv = np.sum(vj[match]/sigmavj[match]**2)/np.sum(1.0/sigmavj[match]**2)
            plt.plot([night],[nightlyrv],'o',color=colors[i])
            nightlyrvs['T' + str(i)] = np.append(nightlyrvs['T' + str(i)],nightlyrv)
            
        match = np.where((jdutcs >= jd) & (jdutcs < (jd+1)) & np.isfinite(vj))
        night = np.mean(jdutcs[match])-2457389
        nightlyrv = np.sum(vj[match]/sigmavj[match]**2)/np.sum(1.0/sigmavj[match]**2)
        plt.plot([night],[nightlyrv],'ko')
        nightlyrvs['all'] = np.append(nightlyrvs['all'],nightlyrv)

    plt.title(objname)
    plt.xlabel('Days since UT 2016-01-01')
    plt.ylabel('Nightly binned RV (m/s)')
    plt.savefig(objname + '.binned.png')
    plt.close()
    print objname + ' nightly binned RMS:'
    print "All: " + str(np.nanstd(nightlyrvs['all']))
    print "T1: " + str(np.nanstd(nightlyrvs['T1']))
    print "T2: " + str(np.nanstd(nightlyrvs['T2']))
    print "T3: " + str(np.nanstd(nightlyrvs['T3']))
    print "T4: " + str(np.nanstd(nightlyrvs['T4']))
        
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

    # output text files of JD, rv, rverr
    f = open(objname + '.dat','w')
    for i in range(len(vj)):
        if np.isfinite(vj[i]):
            f.write('{0:f} {1:f} {2:f} T{3:d} {4:s}'.format(jdutcs[i],vj[i],sigmavj[i],int(telescope[i]),'\n'))
    f.close()

    # plot scatter minus trend vs time
    plt.plot(jdutcs[good]-2457389,vj[good]-vjtrend,'bo')
    plt.title(objname)
    plt.xlabel('Days since UT 2016-01-01')
    plt.ylabel('RV (m/s)')
    plt.savefig(objname + '.detrended.png')
    plt.close()
    print objname + " Detrended RMS: " + str(np.nanstd(vj[good]-vjtrend))

    for i in range(1,ntel+1):
        match = np.where((telescope == i) & np.isfinite(vj))
#        rate = 1.0/(jdutcs[match]-np.roll(jdutcs[match],1))
#        taus = np.logspace(0, 3, 50)
        taus = np.arange(1,len(vj[match])/3)
        (t2, ad, ade, adn) = allantools.oadev(vj[match], rate=1, data_type="freq", taus=taus)
        plt.loglog(t2,ad,'o',color=colors[i])

        # overplot the line
        y = np.asarray([pow(tt,-0.5) for tt in taus])*ad[0]
        plt.loglog(taus,y,'-',color=colors[i])
        

    match = np.where(np.isfinite(vj))
    taus = np.arange(1,len(vj[match])/3)
    (t2, ad, ade, adn) = allantools.oadev(vj[match], rate=1, data_type="freq", taus=taus)
    plt.loglog(t2,ad,'ko')
    # overplot the line
    y = np.asarray([pow(tt,-0.5) for tt in taus])*ad[0]
    plt.loglog(taus,y,'k-')

    plt.xlabel('N bin')
    plt.ylabel('Precision (m/s)')
        

    plt.savefig(objname + '.allan.png')
    plt.close()

#    rate = 1.0/float(data_interval_in_s)
#    taus = [1,2,3,4,8,16]
#    allantools.adev(frequency=fract_freqdata,rate=rate,taus=taus)



if __name__ == "__main__":
                       
#    filename = '/Data/kiwispec-proc/n20160524/n20160524.HD185144.0023.proc.chrec.npy'
#    vank(filename)
    objnames = ['HD10700','HD9407','HD62613','HD122064','HD191408A','HD185144','HD217107','daytimeSky']

    
    objnames = ['daytimeSky','HD122064','HD185144']
    for objname in objnames:
        vank(objname)
    
    sys.exit()


    targets = targetlist.rdlist()

    for objname in targets['name']:
        vank(objname)
