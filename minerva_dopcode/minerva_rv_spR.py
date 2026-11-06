#0 6 * * * cd minerva-pipeline/minerva_dopcode; python four_chunk_jj16.py

#This script runs globgrindall function from four_chunk_jj6.py, vank function from vank.py for spectra taken for HD217107

import matplotlib
matplotlib.use('Agg',warn=False)
#import four_chunk_jj6 as chunk # MW: commented.
#import four_chunk_skew_mw as chunk # MW: use this for the fixed IP as with SP files.
import four_chunk_gauss_mw_spR as chunk #MC - test of spectro-perfectionism
skewpar = False #True #False # MW
#import four_chunk_gauss_mw as chunk # MW
#import vankA as vank
#import vankindp as vank # MW: commented
#import vank2 as vank 
#import vankindp_mw as vank # MW
import vankindp_spR as vank #MC - test of spectro-perfectionism
import targetlist

import numpy as np
import os
import glob
import ipdb
import utils
from astropy.io import fits
import pyfits
from astropy.time import Time
import os, sys

import targetlist
import math
import allantools
import socket

if __name__ == "__main__":

    # get all objects in the MINERVA spreadsheet
    targetlist = targetlist.rdlist(update=False)
    objnames = []
    for i in range(len(targetlist['name'])):
        objnames.append(targetlist['name'][i])

#    objnames = ['HD93148']
    objnames =  ['HD122064']#['HD62613'] #['HD62613'] #['HD217014'] #['HD122064'] #['HD97601']#['HD75732'] #['HD217107']  
    for objname in objnames:
        chunk.globgrindall(shuffle=True, plot=True, objname=objname, date='n20160[56]??')  #date='n201?????') #date='n20171[01]??') #date='n20160[56]??') #date=n201[78]????
        #vank.vank(objname)
        vank.vank(objname, skewpar=skewpar, cbinning=5) 
    sys.exit()



    '''
    parray = chunk.getpararray(ofarr)
   
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


    pl.plot(t-t.min(), temp, 'o')
    pl.xlabel('Time')
    pl.ylabel('Temperature [degrees C]')

    pl.show()
    '''
                                                                                                     

    targets = targetlist.rdlist()

    for objname in targets['name']:
        vank.vank(objname)

#def globgrindvank(shuffle=False):
#	objname = ['HD217107']	
#	datadirs = glob(getpath() + '*')
#	if shuffle: random.shuffle(datadirs)
#	for datadir in datadirs:
		#If new spectra was taken for HD217107, call globgrindall, vank functions		
#		ofarr = four_chunk_jj6.globgrindall(shuffle=True, plot=False)
#		vank.vank(objname)

