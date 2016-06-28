import numpy as np
import glob
import ipdb
import utils
from astropy.io import fits
from astropy.time import Time
import os
import matplotlib.pyplot as plt

c = 299792458.0
objnames = ['HD10700','HD9407','HD62613','HD122064','HD191408A','HD185144','HD217107','daytimeSky']
#objnames = ['HD185144']

for objname in objnames:
    filenames = glob.glob('/Data/kiwispec-proc/n2016*/*' + objname + '*.chrec.npy')
    nfiles = len(filenames)
    if nfiles == 0:
        print "No files for " + objname
        next

    jdutcs = np.zeros(nfiles)
    rvs = np.zeros(nfiles)

    i=0
    for filename in filenames:
        chrec = np.load(filename)
    
        fitsname = os.path.splitext(os.path.splitext(filename)[0])[0] + '.fits'
        h = fits.open(fitsname)
        t = Time(h[0].header['DATE-OBS'], format='isot',scale='utc')
 
        midflux = t.jd + h[0].header['EXPTIME']/2.0/86400.0
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
    
        jdutcs[i] = midflux
        zmeas = np.nanmedian(chrec['z'])#utils.robust_mean(chrec['z'],3)
        rverr = utils.robust_sigma(chrec['z'])*c
        if zmeas == 0.0 or midflux < 2457469:
            rvs[i] = np.nan
        else:
            if objname == 'daytimeSky':
                rvs[i] = zmeas*c
            else:
                rvs[i] = utils.barycorr(midflux, ra, dec, pmra=pmra, pmdec=pmdec, parallax=parallax, rv=rv, zmeas=zmeas)

        print fitsname, midflux, zmeas, rvs[i], rverr
        i += 1


    print objname, np.nanstd(rvs)
    plt.plot(jdutcs-2457389,rvs-np.nanmean(rvs),'bo')
    plt.title(objname)
    plt.xlabel('Days since UT 2016-01-01')
    plt.ylabel('RV (m/s)')
#    plt.xlim([90,160])
    plt.savefig(objname + '.png')
    plt.close()
    #plt.show()

ipdb.set_trace()
