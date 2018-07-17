#!/usr/bin/env python

'''
# MINERVA's optimal extraction code.  This does the primary extraction work
# and draws heavily on minerva_utils.py

INPUTS:
    Several options.  Most defaults are fine.  Only need:
    -f (--filename): full path to fits file to extract
    
OUTPUTS:
    Saves a fits file in $MINERVA_REDUX_DIR/date/*.proc.fits of extracted spectra
    - extension 0: signal (3D array, [Tscope#, trace#, pixel column])
    - extension 1: inverse variance of array
    - extension 2: wavelength solution (drawn from arc_calibrate, same for all input files)
    - extension 3: cosmic ray mask (1.0 = keep, 0.0 = masked)
    .log text file with information about relative throughput per telescope
'''

#Import all of the necessary packages
from __future__ import division
import pyfits
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import special as sf
import argparse
import minerva_utils as m_utils

### Set image plotting keywords
myargs = {'interpolation':'none'}

t0 = time.time()

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
    
try:
    sim_dir = os.environ['MINERVA_SIM_DIR']
except KeyError:
    print("Must set MINERVA_SIM_DIR")
    exit(0)
#    sim_dir = "/uufs/chpc.utah.edu/common/home/bolton_data0/minerva/sim"
    
    
#########################################################
########### Allow input arguments #######################
#########################################################
parser = argparse.ArgumentParser()
parser.add_argument("-f","--filename",help="Name of image file (.fits) to extract",
                    default=os.path.join(data_dir,'n20160216','n20160216.HR2209.0025.fits'))
#                    default=os.path.join(data_dir,'n20160115','n20160115.daytimeSky.0006.fits'))
parser.add_argument("-i","--num_fibers",help="Number of fibers to extract",
                    type=int,default=29)
parser.add_argument("-b","--bundle_space",help="Minimum spacing (in pixels) between fiber bundles",
                    type=int,default=40)
parser.add_argument("-s","--fiber_space",help="Minimum spacing (in pixels) between fibers within a bundle",
                    type=int,default=13)
parser.add_argument("-t","--telescopes",help="Number of telescopes feeding spectrograph",
                    type=int,default=4)
parser.add_argument("-p","--profile",help="Cross-dispersion profile to use for optimal extraction.  Options: gaussian, gauss_lor, bspline", type=str,default='gaussian')
parser.add_argument("-a","--fast",help="Fast fitting for extraction (slightly less precise)",
                    action='store_true')
parser.add_argument("-n","--num_points",help="Number of trace points to fit on each fiberflat",
                    type=int,default=256)#2048)
parser.add_argument("-x","--nosave",help="Don't save results",
                    action='store_true')
parser.add_argument("-z","--testing",help="Runs in test mode (overwrites files, etc.)",
                    action='store_true')
parser.add_argument("-o","--boxcar",help="Do simple boxcar extraction",
                    action='store_true')
parser.add_argument("-d","--date",help="Date of arc exposure, format nYYYYMMDD",default=None)
### The next four (capitalized) arguments are to include/exclude various calibration frames
### By default will use bias, slit flat, and scattered light subtraction (but not dark)
parser.add_argument("-D","--dark",help="Use dark frames in calibration (default: False)",
                    action='store_true')
parser.add_argument("-B","--bias",help="Use bias frames in calibration (default: True)",
                    action='store_false')
parser.add_argument("-F","--slit_flat",help="Use slit flat frames in calibration (default: True)", action='store_false')
parser.add_argument("-S","--scatter",help="Perform scattered light subtraction (default: True)",
                    action='store_false')
args = parser.parse_args()
num_fibers = args.num_fibers*args.telescopes
num_points = args.num_points
fiber_space = args.fiber_space
boxcar = args.boxcar
fast = args.fast

#########################################################
########### Load Background Requirments #################
#########################################################

filename = args.filename
software_vers = 'v1.0.0' #Must manually update for each new release

gain = 1.3 ### Based on MINERVA_CCD Datasheet
readnoise = 3.63 ### Based on MINERVA_CCD Datasheet
rn_eff = 1.0*readnoise*gain ### Effective readnoise (will find empirically if calibration data is available)

ccd, overscan, spec_hdr = m_utils.open_minerva_fits(filename,return_hdr=True)
actypix = ccd.shape[1]

### Next part checks if iodine cell is in, assumes keyword I2POSAS exists
try:
    if spec_hdr['I2POSAS']=='in':
        i2 = True
    else:
        i2 = False
except KeyError:
    i2 = False

#########################################################
############## Calibrate CCD Exposure ###################
#########################################################

date = os.path.split(os.path.split(filename)[0])[1]
### Load Bias
if args.bias:
    bias = m_utils.stack_calib(redux_dir, data_dir, date)
    bias = bias[::-1,0:actypix] #Remove overscan
    ### Analyze exposure overscan (essentially flat, very minimal correction)
    overscan_fit = m_utils.overscan_fit(overscan)
    bias -= np.median(bias) ### give zero mean overall - readjust by overscan
    bias = m_utils.bias_fit(bias, overscan_fit)
    if np.mean(bias) < 10: ### If no bias is present, median is a close estimate of background level
        ccd -= np.median(ccd)
    else:
        ccd -= bias #Note, if ccd is 16bit array, this operation can cause problems

### Load Dark
### Opting not to do dark subtraction by default.  Can lead to artifacts and
### does not show appreciable improvement in overall fit (adds noise)
if args.dark:
    dark, dhdr = m_utils.stack_calib(redux_dir, data_dir, date, frame='dark')
    dark = dark[::-1,0:actypix]
    try:
        dark *= spec_hdr['EXPTIME']/dhdr['EXPTIME'] ### Scales linearly with exposure time
    except:
        ### if EXPTIMES are unavailable, can't reliably subtract dark
        dark = np.zeros(ccd.shape)
    ccd -= dark

## Apply Slit Flats (if present)
if args.slit_flat:
    sflat = m_utils.stack_flat(redux_dir, data_dir, date)
    ### If no slit flat, sflat returns all ones, don't do any flat fielding
    if np.max(sflat) - np.min(sflat) == 0:
        norm_sflat = np.ones(ccd.shape)
        sflat_mask = 0.0*norm_sflat
    else:
        norm_sflat, sflat_mask = m_utils.make_norm_sflat(sflat, redux_dir, date, spline_smooth=True, keep_blaze=True, plot_results=False, include_edge=False)
    sflat_mask = sflat_mask.astype(bool)    
    ## Find new background level (Will overwrite this in scattered light)
    if (np.max(norm_sflat) == np.min(norm_sflat)):
        cut = int(10*readnoise)
        junk, bstd = m_utils.remove_ccd_background(ccd,cut=cut)
        rn_eff = bstd*gain
    if not args.scatter:
        ccd /= norm_sflat

### Remove scattered light background
### must have slit flats to do reliable scattered light subtraction
if args.scatter and args.slit_flat:
    if np.max(sflat_mask) == 0: ### Cues that there is no slit flat available
        ccd, bstd = m_utils.remove_ccd_background(ccd)
    else:
        ccd, bstd = m_utils.remove_scattered_light(ccd, sflat_mask, redux_dir, date, overwrite=True)
        ccd /= norm_sflat
else:
    if np.max(sflat_mask) == 0: ### Cues that there is no slit flat available
        ccd, bstd = m_utils.remove_ccd_background(ccd)
    else:
        junk, bstd = m_utils.remove_scattered_light(ccd, sflat_mask, redux_dir, date, overwrite=False)
rn_eff = bstd*gain

### Apply gain
ccd *= gain

######################################
### Find or load trace information ###
######################################

### Dynamically search for most recent arc frames (unless a date is supplied)
### Fiber flats are taken with arcs (so far...)
if args.date is None:
    arc_date = m_utils.find_most_recent_frame_date('arc', data_dir)
else:
    arc_date = args.date

### Fiber flat traces from Nov 23, 2016 - use as a backup if other fitting fails
#trace_dir = None
#arc_date = 'n20161123'
#trace_fits = pyfits.open(os.path.join(redux_dir,arc_date,'trace_{}.fits'.format(arc_date)))
##hdr = trace_fits[0].header
##profile = hdr['PROFILE']
#multi_coeffs = trace_fits[0].data

### New Fiber flats
trace_dir = os.path.join(redux_dir,'flat_trace')
trace_fits = pyfits.open(os.path.join(trace_dir,'trace_{}.fits'.format(args.profile)))
### These are packaged as full arrays by extension.  Reload into multi_coeffs
info = {}
numexts = len(trace_fits.info(info))
multi_coeffs = np.zeros((numexts,)+trace_fits[0].data.shape)
for i in range(numexts):
    multi_coeffs[i] = trace_fits[i].data
    
##############################
#### Do optimal extraction ###
##############################
profile = args.profile
px_shift = 0
spec, spec_invar, spec_mask, image_model, image_mask, chi2_array, med_pix_shift = m_utils.extract_1D(ccd, norm_sflat, multi_coeffs, profile, date=date, readnoise=rn_eff, gain=gain, px_shift=px_shift, return_model=True, verbose=True, boxcar=boxcar, fast=fast, trace_dir=trace_dir)
### Evaluate fit
invar = (1/(abs(ccd)*gain + (rn_eff)**2))
resid = (ccd-image_model)*np.sqrt(invar)
chi2tot = np.sum((ccd-image_model)**2*invar)/(np.size(ccd-actypix*num_fibers*3))
chi2total = np.sum((image_model[image_model != 0]-ccd[image_model != 0])**2*(1/(ccd[image_model != 0]+readnoise**2)))/(np.size(ccd[image_model != 0])-actypix*num_fibers*3)
print("Reduced chi^2 of ccd vs. model is {}".format(chi2total))

bg_cr_mask = 1.0*image_mask
chi2tot = np.sum(((ccd-image_model)**2*invar)*bg_cr_mask)
dof = ccd.size - np.sum(bg_cr_mask==0) - (num_fibers + 12)*ccd.shape[1]
print "Chi2/dof = {}/{}".format(chi2tot, dof)


save_model = False # Can add as an input option later if desired
if save_model:
    np.save(os.path.join(redux_dir,'n20170307','n20170307.HR3799.0015.1dproc.npy'), image_model)

nnresid = (ccd-image_model)
rsec = nnresid[1650:1667,1000]
xsec = np.arange(1650, 1667)
insec = 1/(abs(ccd[1650:1667,1000])+rn_eff**2)
np.save('test_res.npy', np.hstack((rsec, xsec, insec)))
new_pars = sf.fit_damped_cos(xsec, rsec, insec)

############################################################
######### Import wavelength calibration ####################        
############################################################
        
i2coeffs = [3.48097e-4,2.11689] #shift in pixels due to iodine cell
i2shift = np.poly1d(i2coeffs)(np.arange(actypix))

ypx_mod = 2*(np.arange(0,actypix)-i2shift*i2-actypix/2)/actypix #includes iodine shift (if i2 is in)
wavelength_soln = np.zeros((args.telescopes,args.num_fibers,actypix))
for j in range(4):
    wl_hdu = pyfits.open(os.path.join(redux_dir,arc_date,'wavelength_soln_T{}.fits'.format(j+1)))
    wl_coeffs =  wl_hdu[0].data
    for i in range(args.num_fibers):
        wavelength_soln[j,i,:] = np.poly1d(wl_coeffs[i])(ypx_mod)

###############################################################
########### Stack extracted values into 3D arrays #############
###############################################################
## Would like to automate this (how?) right now, here's the fiber arrangement:
##    1st (0) - T4 from order "2" (by my minerva csv accounting)
##    2nd (1) - T1 from order "3"
##    3rd (2) - T2 from order "3"
## etc.  continues T1 through T4 and ascending orders
## right now I don't have wavelength soln for order 2, so I just eliminate
## that fiber and keep moving forward (fiber "0" isn't used)
spec3D = np.zeros((args.telescopes,args.num_fibers,actypix))
spec3D[0,:,:] = spec[np.arange(1,num_fibers,4),:]
spec3D[1,:,:] = spec[np.arange(2,num_fibers,4),:]
spec3D[2,:,:] = spec[np.arange(3,num_fibers,4),:]
spec3D[3,:,:] = np.vstack((spec[np.arange(4,num_fibers,4),:],np.ones(actypix)))

spec_invar3D = np.zeros((args.telescopes,args.num_fibers,actypix))
spec_invar3D[0,:,:] = spec_invar[np.arange(1,num_fibers,4),:]
spec_invar3D[1,:,:] = spec_invar[np.arange(2,num_fibers,4),:]
spec_invar3D[2,:,:] = spec_invar[np.arange(3,num_fibers,4),:]
spec_invar3D[3,:,:] = np.vstack((spec_invar[np.arange(4,num_fibers,4),:],np.zeros(actypix)))

spec_mask3D = np.zeros((args.telescopes,args.num_fibers,actypix))
spec_mask3D[0,:,:] = spec_mask[np.arange(1,num_fibers,4),:]
spec_mask3D[1,:,:] = spec_mask[np.arange(2,num_fibers,4),:]
spec_mask3D[2,:,:] = spec_mask[np.arange(3,num_fibers,4),:]
spec_mask3D[3,:,:] = np.vstack((spec_mask[np.arange(4,num_fibers,4),:],np.zeros(actypix)))

#############################################################
########### And finally, save spectrum ######################
#############################################################
root, savefile = os.path.split(filename)
savefile = savefile[:-5] #remove '.fits' for now
junk, savedate = os.path.split(root)
if not os.path.isdir(root):
    os.mkdir(root)

active = np.ones((4), dtype=int) ### Array for whether telescopes are active or inactive
if not args.nosave:
    hdu1 = pyfits.PrimaryHDU(spec3D)
    hdu2 = pyfits.PrimaryHDU(spec_invar3D)
    hdu3 = pyfits.PrimaryHDU(wavelength_soln)
    hdu4 = pyfits.PrimaryHDU(spec_mask3D)
    hdu1.header.comments['NAXIS1'] = 'Pixel axis'
    hdu1.header.comments['NAXIS2'] = 'Fiber axis (blue to red)'
    hdu1.header.comments['NAXIS3'] = 'Telescope axis (T1, T2, T3, T4)'
    hdu2.header.comments['NAXIS1'] = 'Pixel axis'
    hdu2.header.comments['NAXIS2'] = 'Fiber axis (blue to red)'
    hdu2.header.comments['NAXIS3'] = 'Telescope axis (T1, T2, T3, T4)'
    hdu3.header.comments['NAXIS1'] = 'Pixel axis'
    hdu3.header.comments['NAXIS2'] = 'Fiber axis (blue to red)'
    hdu3.header.comments['NAXIS3'] = 'Telescope axis (T1, T2, T3, T4)'
    hdu4.header.comments['NAXIS1'] = 'Pixel axis'
    hdu4.header.comments['NAXIS2'] = 'Fiber axis (blue to red)'
    hdu4.header.comments['NAXIS3'] = 'Telescope axis (T1, T2, T3, T4)'
    ### Additional new header values
    hdu1.header.append(('UNITS','Counts','Relative photon counts (no flat fielding)'))
    hdu2.header.append(('UNITS','Inv. Var','Inverse variance'))
    hdu3.header.append(('UNITS','Wavelength','Wavelength solution lambda (Angstroms) vs px'))
    hdu4.header.append(('UNITS','Mask','True (1) or False (0) good data point'))
    hdu1.header.append(('VERSION',software_vers,'Reduction software version'))
    ### Check if telescopes are inactive for photometry
    for i in range(4):
        try:
            if spec_hdr['FAUSTAT{}'.format(i+1)] != 'GUIDING':
                active[i] = 0
        except:
            active[i] = 0 #Inactive if header is not present
    #### Include all old header values in new header for hdu1
    ### As usual, probably a better way, but I think this will work
    for key in spec_hdr.keys():
        if key not in hdu1.header and key != 'BSCALE' and key != 'BZERO' and key != 'COMMENT':
            try:
                spec_hdr.comments[key].decode('ascii')
            except:
                new_key = ''.join([i if ord(i) < 128 else ' ' for i in spec_hdr.comments[key]])
                hdu1.header.append((key,spec_hdr[key],new_key))
            else:
                hdu1.header.append((key,spec_hdr[key],spec_hdr.comments[key]))        
    hdulist = pyfits.HDUList([hdu1])
    hdulist.append(hdu2)
    hdulist.append(hdu3)
    hdulist.append(hdu4)
    if not os.path.isdir(os.path.join(redux_dir,savedate)):
        os.makedirs(os.path.join(redux_dir,savedate))
#    print os.path.join(redux_dir,savedate,savefile+'.proc.fits')
    hdulist.writeto(os.path.join(redux_dir,savedate,savefile+'.proc.fits'),clobber=True)

### Now add Jason Eastman's custom headers for barycentric corrections
try: 
    import utils as dop_utils
    dop_utils.addzb(os.path.join(redux_dir,savedate,savefile+'.proc.fits'),fau_dir=data_dir)
    barycentric = True
except:
    barycentric = False


#############################################################
###### Log files to allow email alerts for low counts #######
#############################################################

T_cnts = dict()
good = 2*np.ones((4), dtype=int)
low_cut = 100
moderate_cut = 1000
log_file = os.path.join(redux_dir, savedate, savefile+'.log')
lf = open(log_file, "w")
lf.write("Extraction summary for exposure {}\n\n".format(savefile+'.fits'))
if barycentric:
    lf.write("Barycentric corrections applied")
else:
    lf.write("WARNING: Barycentric corrections not applied")
lf.write("\n\n")
lf.write("Pixel shift relative to trace frame is {}\n".format(med_pix_shift))
lf.write("{}\n\n".format(med_pix_shift>1))
lf.write("Telescopes active (1=active, 0=inactive)\n")
lf.write("{} {} {} {}\n\n".format(active[0], active[1], active[2], active[3]))
lf.write("Photon count check:\n")
for i in range(4):
    Tnum = 'T{}'.format(i+1)
    if active[i] == 0:
        T_cnts[i] = 0
        good[i] = 0
        lf.write("  {}: Not Guiding".format(Tnum))
        lf.write("\n")
        continue
    T_cnts[i] = np.nanmean(spec3D[i,:,:][spec_mask3D[i,:,:]==1])
    if np.mean(T_cnts[i]) < low_cut:
        good[i] = 0 #Bad exposure on this telescope
        lf.write("  {}: WARNING - very low counts".format(Tnum))
    elif np.mean(T_cnts[i]) < moderate_cut:
        good[i] = 1 #Mediocre exposure on this telescope
        lf.write("  {}: Notice - moderately low counts".format(Tnum))
    else:
        lf.write("  {}: Good exposure".format(Tnum))
    lf.write("\n")
lf.write("\n")
lf.write("Mean photon count > {} on {}/4 telescopes\n".format(moderate_cut,np.sum(good==2)))
lf.write("Mean photon count > {} on {}/4 telescopes\n".format(low_cut,np.sum(good>=1)))
lf.write("Median counts per telescope (across all orders):\n")
lf.write("{} {} {} {}\n".format(np.median(T_cnts[0]), np.median(T_cnts[1]), np.median(T_cnts[2]), np.median(T_cnts[3])))
lf.write("Signal Count Status (2=good, 1=low, 0=negligible):\n")
lf.write("{} {} {} {}".format(good[0], good[1], good[2], good[3]))
lf.close()

tf = time.time()
print("Total extraction time = {}s".format(tf-t0))