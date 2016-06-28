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
import datetime

def vank(objname):

    c = 299792458.0
    filenames = glob.glob('/Data/kiwispec-proc/n20160[4,5,6]*/*' + objname + '*.chrec2.npy')

    allalpha1 = np.array([])
    allalpha2 = np.array([])
    allalpha3 = np.array([])
    allalpha4 = np.array([])
    allsigma1 = np.array([])
    allsigma2 = np.array([])
    allsigma3 = np.array([])
    allsigma4 = np.array([])
    alldwdx1 = np.array([])
    alldwdx2 = np.array([])
    alldwdx3 = np.array([])
    alldwdx4 = np.array([])
    allw01 = np.array([])
    allw02 = np.array([])
    allw03 = np.array([])
    allw04 = np.array([])
    allz1 = np.array([])
    allz2 = np.array([])
    allz3 = np.array([])
    allz4 = np.array([])
    allcts1 = np.array([])
    allcts2 = np.array([])
    allcts3 = np.array([])
    allcts4 = np.array([])
    allchi1 = np.array([])
    allchi2 = np.array([])
    allchi3 = np.array([])
    allchi4 = np.array([])
    alldate = np.array([])
    
    wls = np.load('templates/MC_bstar_wls.air.npy')
    coef = np.array([0.00059673, 0.06555841]) # wavelength offset per trace...per order              
    nbin = 100

    for filename in filenames:

        chrec = np.load(filename)

        allalpha1 = np.append(allalpha1,chrec['alpha1'])
        allalpha2 = np.append(allalpha2,chrec['alpha2'])
        allalpha3 = np.append(allalpha3,chrec['alpha3'])
        allalpha4 = np.append(allalpha4,chrec['alpha4'])

        allsigma1 = np.append(allsigma1,chrec['sigma1'])
        allsigma2 = np.append(allsigma2,chrec['sigma2'])
        allsigma3 = np.append(allsigma3,chrec['sigma3'])
        allsigma4 = np.append(allsigma4,chrec['sigma4'])

        alldwdx1 = np.append(alldwdx1,chrec['dwdx1'])
        alldwdx2 = np.append(alldwdx2,chrec['dwdx2'])
        alldwdx3 = np.append(alldwdx3,chrec['dwdx3'])
        alldwdx4 = np.append(alldwdx4,chrec['dwdx4'])

        w0g = wls[chrec['pixel'],chrec['order']]
        dwdtrace = np.polyval(coef, chrec['order']) # find wavelength offset per trace for this order             
        # wavelength zero point (subtract the best guess for each chunk)
        allw01 = np.append(allw01,chrec['w01'] - (w0g + 3.0*dwdtrace))
        allw02 = np.append(allw02,chrec['w02'] - (w0g + 2.0*dwdtrace))
        allw03 = np.append(allw03,chrec['w03'] - (w0g + 1.0*dwdtrace))
        allw04 = np.append(allw04,chrec['w04'] - (w0g + 0.0*dwdtrace))

        allz1 = np.append(allz1,chrec['z1'])
        allz2 = np.append(allz2,chrec['z2'])
        allz3 = np.append(allz3,chrec['z3'])
        allz4 = np.append(allz4,chrec['z4'])

        allcts1 = np.append(allcts1,chrec['cts1'])
        allcts2 = np.append(allcts2,chrec['cts2'])
        allcts3 = np.append(allcts3,chrec['cts3'])
        allcts4 = np.append(allcts4,chrec['cts4'])

        allchi1 = np.append(allchi1,chrec['chi1'])
        allchi2 = np.append(allchi2,chrec['chi2'])
        allchi3 = np.append(allchi3,chrec['chi3'])
        allchi4 = np.append(allchi4,chrec['chi4'])

        date = (datetime.datetime.strptime(filename.split('/')[3], 'n%Y%m%d') - datetime.datetime(2016,1,1)).total_seconds()/86400.0
        alldate = np.append(alldate,np.repeat(date,len(chrec['chi1'])))

#    ipdb.set_trace()

    chithresh = np.percentile(allchi1,95)
    good1 = np.where( (allchi1 < chithresh) & (allalpha1 != 1.0) & (allalpha1 != -1.0) & (allsigma1 < 3) & (allsigma1 > 0.25) & (allchi1 > 0.1) )
    chithresh = np.percentile(allchi2,95)
    good2 = np.where( (allchi2 < chithresh) & (allalpha2 != 1.0) & (allalpha2 != -1.0) & (allsigma2 < 3) & (allsigma2 > 0.25) & (allchi2 > 0.1) )
    chithresh = np.percentile(allchi3,95)
    good3 = np.where( (allchi3 < chithresh) & (allalpha3 != 1.0) & (allalpha3 != -1.0) & (allsigma3 < 3) & (allsigma3 > 0.25) & (allchi3 > 0.1) )
    chithresh = np.percentile(allchi4,95)
    good4 = np.where( (allchi4 < chithresh) & (allalpha4 != 1.0) & (allalpha4 != -1.0) & (allsigma4 < 3) & (allsigma4 > 0.25) & (allchi4 > 0.1) )


    hist, bins = np.histogram(allalpha1[good1], bins=nbin)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.step(center, hist, color='r', where='mid')

    hist, bins = np.histogram(allalpha2[good2], bins=nbin)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.step(center, hist, color='g', where='mid')

    hist, bins = np.histogram(allalpha3[good3], bins=nbin)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.step(center, hist, color='b', where='mid')

    hist, bins = np.histogram(allalpha4[good4], bins=nbin)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.step(center, hist, color='orange', where='mid')

    plt.xlabel('Alpha')
    plt.ylabel('# of chunks')
    plt.savefig(objname + '.alpha.png')
    plt.close()

    hist, bins = np.histogram(allsigma1[good1], bins=nbin)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.step(center, hist, color='r', where='mid')

    hist, bins = np.histogram(allsigma2[good2], bins=nbin)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.step(center, hist, color='g', where='mid')

    hist, bins = np.histogram(allsigma3[good3], bins=nbin)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.step(center, hist, color='b', where='mid')

    hist, bins = np.histogram(allsigma4[good4], bins=nbin)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.step(center, hist, color='orange', where='mid')

    ipdb.set_trace()

    plt.xlabel('Sigma')
    plt.ylabel('# of chunks')
    plt.savefig(objname + '.sigma.png')
    plt.close()


    plt.scatter(allchi1[good1],allsigma1[good1],color='r')
    plt.scatter(allchi2[good2],allsigma2[good2],color='g')
    plt.scatter(allchi3[good3],allsigma3[good3],color='b')
    plt.scatter(allchi4[good4],allsigma4[good4],color='orange')
    plt.xlabel('Chi^2')
    plt.ylabel('Sigma')
    plt.savefig(objname + '.sigmavchi.png')
    plt.close()

    plt.scatter(allchi1[good1],allalpha1[good1],color='r')
    plt.scatter(allchi2[good2],allalpha2[good2],color='g')
    plt.scatter(allchi3[good3],allalpha3[good3],color='b')
    plt.scatter(allchi4[good4],allalpha4[good4],color='orange')
    plt.xlabel('Chi^2')
    plt.ylabel('Alpha')
    plt.savefig(objname + '.alphavchi.png')
    plt.close()

    plt.scatter(alldate[good1],allalpha1[good1],color='r')
    plt.scatter(alldate[good2],allalpha2[good2],color='g')
    plt.scatter(alldate[good3],allalpha3[good3],color='b')
    plt.scatter(alldate[good4],allalpha4[good4],color='orange')
    plt.xlabel('Days since 2016-01-01')
    plt.ylabel('Alpha')
    plt.savefig(objname + '.alphavtime.png')
    plt.close()

    plt.scatter(alldate[good1],allsigma1[good1],color='r')
    plt.scatter(alldate[good2],allsigma2[good2],color='g')
    plt.scatter(alldate[good3],allsigma3[good3],color='b')
    plt.scatter(alldate[good4],allsigma4[good4],color='orange')
    plt.xlabel('Days since 2016-01-01')
    plt.ylabel('Sigma')
    plt.savefig(objname + '.sigmavtime.png')
    plt.close()

    plt.scatter(alldate[good1],alldwdx1[good1],color='r')
    plt.scatter(alldate[good2],alldwdx2[good2],color='g')
    plt.scatter(alldate[good3],alldwdx3[good3],color='b')
    plt.scatter(alldate[good4],alldwdx4[good4],color='orange')
    plt.xlabel('Days since 2016-01-01')
    plt.ylabel('Dwdx')
    plt.savefig(objname + '.dwdxvtime.png')
    plt.close()

    plt.scatter(alldate[good1],allw01[good1],color='r')
    plt.scatter(alldate[good2],allw02[good2],color='g')
    plt.scatter(alldate[good3],allw03[good3],color='b')
    plt.scatter(alldate[good4],allw04[good4],color='orange')
    plt.xlabel('Days since 2016-01-01')
    plt.ylabel('W0')
    plt.savefig(objname + '.w0vtime.png')
    plt.close()

    plt.scatter(alldate[good1],allchi1[good1],color='r')
    plt.scatter(alldate[good2],allchi2[good2],color='g')
    plt.scatter(alldate[good3],allchi3[good3],color='b')
    plt.scatter(alldate[good4],allchi4[good4],color='orange')
    plt.xlabel('Days since 2016-01-01')
    plt.ylabel('Chi^2')
    plt.savefig(objname + '.chivtime.png')
    plt.close()

    plt.scatter(alldate[good1],allcts1[good1],color='r')
    plt.scatter(alldate[good2],allcts2[good2],color='g')
    plt.scatter(alldate[good3],allcts3[good3],color='b')
    plt.scatter(alldate[good4],allcts4[good4],color='orange')
    plt.xlabel('Days since 2016-01-01')
    plt.ylabel('Cts')
    plt.savefig(objname + '.ctsvtime.png')
    plt.close()

    plt.scatter(allcts1[good1],allsigma1[good1],color='r')
    plt.scatter(allcts2[good2],allsigma2[good2],color='g')
    plt.scatter(allcts3[good3],allsigma3[good3],color='b')
    plt.scatter(allcts4[good4],allsigma4[good4],color='orange')
    plt.xlabel('Cts')
    plt.ylabel('Sigma')
    plt.savefig(objname + '.sigmavcts.png')
    plt.close()



#    ipdb.set_trace()
    

        


if __name__ == "__main__":

    objnames = ['HD122064','HD185144']
#    objnames = ['HD185144']
    for objname in objnames:
        vank(objname)
