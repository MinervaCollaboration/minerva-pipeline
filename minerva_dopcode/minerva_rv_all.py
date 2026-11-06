#0 6 * * * cd minerva-pipeline/minerva_dopcode; python four_chunk_jj16.py

#This script runs globgrindall function from four_chunk_jj6.py, vankA function from vank.py for all spectra taken
import matplotlib
matplotlib.use('Agg',warn=False)
#import four_chunk_sw1 as chunk # MW: commented.
import four_chunk_gauss_mw as chunk # MW
#import vankA as vank
import vankindp_mw as vank #Finn McLaughlin Apr 2022
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

    #objname = "HD75732"
    objname = "HD93148"
    objname = "HD16160"
    objname = "HD201091"
    chunk.globgrindall(shuffle=False, plot=False, objname=objname, date='n20*')
    vank.vank(objname)

    ipdb.set_trace()

    obshistoryfiles = glob.glob('/home/minerva/minerva-control/schedule/rvobshistory/*.txt')
    for obshistoryfile in obshistoryfiles:
        objname = os.path.basename(os.path.splitext(obshistoryfile)[0])
        if "HD" in objname:
            chunk.globgrindall(shuffle=False, plot=False, objname=objname, date='n20*')
            vank.vank(objname)
    sys.exit()
                                                                                   
    targets = targetlist.rdlist()

    for objname in targets['name']:
        vank.vank(objname)




