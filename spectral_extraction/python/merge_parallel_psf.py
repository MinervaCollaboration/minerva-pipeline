#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 12:08:22 2016

@author: matt

Mini script to merge all fitted psfs from find_2d_psf run with -par into a
single .fits file
"""

#Import all of the necessary packages
from __future__ import division
import pyfits
import os
import glob
import numpy as np

redux_dir = os.environ['MINERVA_REDUX_DIR']
psf_dir = os.path.join(redux_dir,'psf')

### Aggregate coefficients from each trace
filenames = glob.glob(os.path.join(psf_dir),'psf_coeffs_???')
filenames.sort()
cntr = 0
num_fibers = len(filenames)
for filename in filenames:
    psf_hdu = pyfits.open(filename)
    psf_coeffs = psf_hdu[0].data
    if cntr==0:
        all_psf_coeffs = np.zeros((num_fibers,psf_coeffs.size))
    all_psf_coeffs[cntr,:] = psf_coeffs
    cntr += 1

### Copy over header info (assumes all input files have the same headers
### except for keyword FIBERNUM), then save
hdu1 = pyfits.PrimaryHDU(all_psf_coeffs)
psf_hdr = psf_hdu[0].header
for key in psf_hdr.keys():
    if key not in hdu1.header and key != 'FIBERNUM':
        hdu1.header.append((key,psf_hdr[key],psf_hdr.comments[key]))    
hdulist = pyfits.HDUList([hdu1])
redux_dir = os.environ['MINERVA_REDUX_DIR']
psf_dir = 'psf'
filename = 'psf_coeffs_{}'.format(psf_hdr['PSFTYPE'])
if not os.path.isdir(os.path.join(redux_dir,psf_dir)):
    os.makedirs(os.path.join(redux_dir,psf_dir))
#    print os.path.join(redux_dir,savedate,savefile+'.proc.fits')
hdulist.writeto(os.path.join(redux_dir,psf_dir,filename+'.fits'),clobber=True)

### Remove old files
for filename in filenames:
    os.remove(filename)