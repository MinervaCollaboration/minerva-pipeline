#!/usr/bin/env python

'''
# Makes stacks of bias/dark/slitflats for a given day
# Run this code before extraction (should take less than a minute)
# This needs to be re-run daily

INPUTS:
    Defaults should be fine
    Can choose which calibration frames to use
    Can also set a date other than today if needed for custom fitting
    
OUTPUTS:
    Saves stacked calibration frames for Bias, Dark, and Slit_Flat
        to MINERVA_REDUX_DIR/{date}/
'''

#Import all of the necessary packages
from __future__ import division
import os
import datetime
import argparse
import minerva_utils as m_utils

### Set image plotting keywords
myargs = {'interpolation':'none'}

######## Import environmental variables #################

try:
    data_dir = os.environ['MINERVA_DATA_DIR']
except KeyError:
    print("Must set MINERVA_DATA_DIR")
    exit(0)
#    data_dir = "/uufs/chpc.utah.edu/common/home/bolton_data0/minerva/data"

try:
    redux_dir = os.environ['MINERVA_REDUX_DIR']
except KeyError:
    print("Must set MINERVA_REDUX_DIR")
    exit(0)
#    redux_dir = "/uufs/chpc.utah.edu/common/home/bolton_data0/minerva/redux"
    
    
#########################################################
########### Allow input arguments #######################
#########################################################
parser = argparse.ArgumentParser()
parser.add_argument("-d","--date",help="Date of calibration exposures, format nYYYYMMDD",default=None)
parser.add_argument("-D","--dark",help="Use dark frames in calibration (default: False)",
                    action='store_true')
parser.add_argument("-B","--bias",help="Use bias frames in calibration (default: True)",
                    action='store_false')
parser.add_argument("-F","--slit_flat",help="Use slit flat frames in calibration (default: True)", action='store_false')
parser.add_argument("-S","--scatter",help="Perform scattered light subtraction (default: True)",
                    action='store_false')
args = parser.parse_args()


#########################################################
############## Calibrate CCD Exposure ###################
#########################################################

if args.date is None:
    dnow = datetime.datetime.today()
    date = dnow.strftime('n%Y%m%d')
    
### Stack Bias
bias = m_utils.stack_calib(redux_dir, data_dir, date)

### Stack Dark
### Opting not to do dark subtraction by default, but still prep stacks
dark, dhdr = m_utils.stack_calib(redux_dir, data_dir, date, frame='dark')

## Stack Slit Flats (if present)
sflat = m_utils.stack_flat(redux_dir, data_dir, date)