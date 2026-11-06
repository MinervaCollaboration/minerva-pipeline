#0 6 * * * cd minerva-pipeline/minerva_dopcode; python four_chunk_jj16.py

#This script runs globgrindall function from four_chunk_jj6.py, vankA function from vank.py for all spectra taken
# Adopted from minerva_rv_all.py by Sharon Xuesong Wang, Jan 2018
# Test for four_chunk_swN.py for new IP models.

import matplotlib
matplotlib.use('Agg',warn=False)
#import four_chunk_sw1 as chunk_sw1
#import four_chunk_sw2 as chunk_sw2
#import four_chunk_sw3 as chunk_sw3
import four_chunk_sw4_headerboo as chunk_sw4
#import four_chunk_sw5 as chunk_sw5
#import four_chunk_sw6 as chunk_sw6
#import four_chunk_sw7 as chunk_sw7
import four_chunk_jj6_sw as chunk_jj6
import vankA_sw_headerboo as vank

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

    objname = 'HD122064'
    #objname = 'HD10700'
    date = 'n201703??'
    objname = 'HD*'
    date = 'n201?????'

    # HD73344 for Ian and co, Sharon X Wang May 2018
    objname = 'HD73344'
    date = 'n2018????'

    # Default eso50 atlas, four gaussian wings
    #print "Cranking with sw4, fixed super gaussian with wings"
    #chunk_sw4.globgrindall(shuffle=False, plot=False, objname=objname, date=date)
    chunk_jj6.globgrindall(shuffle=False, plot=False, objname=objname, date=date)
    vank.vank(objname, date=date, exten='chrec6')
    sys.exit()

    # Many options! These scripts are in swtest/old_sw_py/.
    if len(sys.argv) > 1:
        if sys.argv[1] == 'sw1':
            print "Cranking with sw1, sum of Gaussians IP..."
            chunk_sw1.globgrindall(shuffle=False, plot=False, objname=objname, date=date)
        elif sys.argv[1] == 'sw2':
            print "Cranking with sw2, super Gaussian IP..."
            chunk_sw2.globgrindall(shuffle=False, plot=False, objname=objname, date=date)
        elif sys.argv[1] == 'sw3':
            print "Cranking with sw3, single super Gaussian IP..."
            chunk_sw3.globgrindall(shuffle=False, plot=False, objname=objname, date=date)
        elif sys.argv[1] == 'sw4':
            print "Cranking with sw4, fixed super gaussian with wings"
            chunk_sw4.globgrindall(shuffle=False, plot=False, objname=objname, date=date)
        elif sys.argv[1] == 'sw5':
            print "Cranking with sw5, Moffat IP"
            chunk_sw5.globgrindall(shuffle=False, plot=False, objname=objname, date=date)
        elif sys.argv[1] == 'sw6':
            print "Cranking with sw6, fixed width super Gaussian IP"
            chunk_sw6.globgrindall(shuffle=False, plot=False, objname=objname, date=date)
        elif sys.argv[1] == 'sw7':
            print "Cranking with sw7, same as jj6 but fix alpha=0"
            chunk_sw7.globgrindall(shuffle=False, plot=False, objname=objname, date=date)
        else:
            print "No such option."
        vank.vank(objname, exten=sys.argv[1], date=date) 
    else:
        print "Cranking with default jj6..."
        chunk_jj6.globgrindall(shuffle=False, plot=False, objname=objname, date=date)
        vank.vank(objname, date=date)  # default extension is chrec6 
    sys.exit()
 
    # This folder has obs history of every star
    #obshistoryfiles = glob.glob('/home/minerva/minerva-control/schedule/rvobshistory/*.txt')



