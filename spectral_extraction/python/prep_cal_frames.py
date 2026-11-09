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

data_dir = "/Data/kiwispec"
#environ['MINERVA_DATA_DIR'] = "/Data/kiwispec"
#try: data_dir = environ['MINERVA_DATA_DIR']
#except:
#    print "Set MINERVA_DATA_DIR"
#    exit()

redux_dir = "/Data/kiwispec-proc"
#environ['MINERVA_REDUX_DIR'] = "/Data/kiwispec-proc"
#try: redux_dir = environ['MINERVA_REDUX_DIR']
#except:
#    print "Set MINERVA_REDUX_DIR"
#    exit()

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
else:
    date = args.date
    
### Stack Bias
bias = m_utils.stack_calib(redux_dir, data_dir, date)

### Stack Dark
### Opting not to do dark subtraction by default, but still prep stacks
dark, dhdr = m_utils.stack_calib(redux_dir, data_dir, date, frame='dark')

# slit flats disabled due to lamp breaking!
## Stack Slit Flats (if present)
#sflat = m_utils.stack_flat(redux_dir, data_dir, date)
#if np.max(sflat) - np.min(sflat) == 0:
#    pass
#else:
#    norm_sflat, sflat_mask = m_utils.make_norm_sflat(sflat, redux_dir, date, spline_smooth=True, keep_blaze=True, plot_results=False, include_edge=False)

