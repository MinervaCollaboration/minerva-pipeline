#!/usr/bin/env python 2.7

#Trying to re-write headers, etc with pyfits
import pyfits
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import sys

def unpack_minerva(filename):
    #Load in "raw" data
    #py1 = pyfits.open(os.path.join(path,flnm),ignore_missing_end=True,uint=True)
    py1 = pyfits.open(filename,uint=True)
    data = py1[0].data
    hdr = py1[0].header
    
    #Dimensions
    d1 = hdr['NAXIS1']
    d2 = hdr['NAXIS2']
    
    #Test to make sure this logic is robust enough for varying inputs
    if np.shape(data)[0] > d1:
        data_new = np.resize(data,[d2,d1,2])
            
        #Data is split into two 8 bit sections (totalling 16 bit).  Need to join
        #these together to get 16bit number.  Probably a faster way than this.
        data_16bit = np.zeros((d2,d1),dtype=np.uint16)
        for row in range(d2):
            for col in range(d1):
                #Join binary strings
                binstr = "{:08d}{:08d}".format(int(bin(data_new[row,col,0])[2:]),
                          int(bin(data_new[row,col,1])[2:]))
                data_16bit[row,col] = int(binstr,base=2)
            
        return data_16bit
    else:
        return data

#plt.ion()
#path = os.environ['MINERVA_DATA_DIR']
#path = os.path.join(path,'initial_spectro_runs')
#path = os.getcwd()
#flnm = 'n20160115.daytimeSky.0006.fits'
#filename = sys.argv[1]
#
##Load in "raw" data
##py1 = pyfits.open(os.path.join(path,flnm),ignore_missing_end=True,uint=True)
#py1 = pyfits.open(filename,uint=True)
#data = py1[0].data
#hdr = py1[0].header
#
##Dimensions
#d1 = hdr['NAXIS1']
#d2 = hdr['NAXIS2']
#
##Test to make sure this logic is robust enough for varying inputs
#if np.shape(data)[0] > d1:
#    data_new = np.resize(data,[d2,d1,2])
#    
##Data is split into two 8 bit sections (totalling 16 bit).  Need to join
##these together to get 16bit number.  Probably a faster way than this.
#t0 = time.time()
#data_16bit = np.zeros((d2,d1),dtype=np.uint16)
#for row in range(d2):
#    for col in range(d1):
#        #Join binary strings
#        binstr = "{:08d}{:08d}".format(int(bin(data_new[row,col,0])[2:]),
#                  int(bin(data_new[row,col,1])[2:]))
#        data_16bit[row,col] = int(binstr,base=2)
#
#t1 = time.time()
#print("tconvert = {:0.4f}s".format(t1-t0))
#plt.imshow(data_16bit[::-1,0:2048])
#plt.show()

'''
#make new hdu, hdulist, then write to file
hdu_new = pyfits.PrimaryHDU(data_16bit,uint=True)
hdu_new.header.append(('LIGHT',203.78,'Dummy line - delete when done'))
print repr(hdu_new.header)
print hdu_new.data
hdulist_out = pyfits.HDUList([hdu_new])
hdulist_out.writeto(os.path.join(path,'test.fits'),clobber=True)

flnm = 'test.fits'
py1 = pyfits.open(os.path.join(path,flnm),uint=True)
data = py1[0].data
hdr = py1[0].header
print repr(hdr)
print data
plt.imshow(data[::-1,0:2048])
plt.show()
'''