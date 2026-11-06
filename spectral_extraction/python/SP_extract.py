#!/usr/bin/env python

'''
# Implementation of 2D "Spectro-perfectionism" extraction for MINERVA
# This relies on first running fit_2D_psf.py to get PSF shape
# This implementation lacks some of the recent features in the
# latest optimal extraction (notably scattered light subtraction)
# but otherwise is similar in function

# Some save files, etc. are hard coded for testing, so 
# be aware some work is needed to put this into a production-ready product

# Also note, cosmic ray masking is not implemented, but SP extraction is
# inherently less sensitive to cosmic ray hits

INPUTS:
    Several options.  Most defaults are fine.  Only need:
    -f (--filename): full path to fits file to extract
    
OUTPUTS:
    Saves a fits file in $MINERVA_REDUX_DIR/date/*.proc.fits of extracted spectra
    - extension 0: signal (3D array, [Tscope#, trace#, pixel column])
    - extension 1: inverse variance of array
    - extension 2: wavelength solution (drawn from arc_calibrate, same for all input files)
'''

#Import all of the necessary packages
from __future__ import division
import pyfits
import os
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
import special as sf
import argparse
import minerva_utils as m_utils

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
########### Setup input arguments #######################
#########################################################
parser = argparse.ArgumentParser()
parser.add_argument("-f","--filename",help="Name of image file (.fits) to extract", default=os.path.join(data_dir, 'n20170307','n20170307.HR3799.0015.fits'))
#                    default=os.path.join(data_dir,'n20160216','n20160216.HR2209.0025.fits'))
#                    default=os.path.join(data_dir,'n20160115','n20160115.daytimeSky.0006.fits'))
parser.add_argument("-fib","--num_fibers",help="Number of fibers to extract",
                    type=int,default=29)
parser.add_argument("-bs","--bundle_space",help="Minimum spacing (in pixels) between fiber bundles",
                    type=int,default=40)
parser.add_argument("-p","--psf",help="PSF type used", type=str,default='ghl')
parser.add_argument("-fs","--fiber_space",help="Minimum spacing (in pixels) between fibers within a bundle",
                    type=int,default=13)
parser.add_argument("-ts","--telescopes",help="Number of telescopes feeding spectrograph",
                    type=int,default=4) 
parser.add_argument("-n","--num_points",help="Number of trace points to fit on each fiber",
                    type=int,default=20)
parser.add_argument("-ns","--nosave",help="Don't save results",
                    action='store_true')
parser.add_argument("-d","--date",help="Date of arc exposure, format nYYYYMMDD",default=None)
#parser.add_argument("-T","--tscopes",help="T1, T2, T3, and/or T4 (remove later)",
#                    type=str,default=['T1','T2','T3','T4'])
args = parser.parse_args()
num_fibers = args.num_fibers*args.telescopes-4
fiber_space = args.fiber_space
num_points = args.num_points

#########################################################
########### Load Background Requirments #################
#########################################################

filename = args.filename
software_vers = 'v0.5.0' #Later grab this from somewhere else

gain = 1.3
readnoise = 3.63

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

if 'n20161123' not in filename:
    #########################################################
    ########### Load Flat and Bias Frames ###################
    #########################################################
    #date = 'n20160115' #Fixed for now, later make this dynamic
    date = os.path.split(os.path.split(filename)[0])[1]
    ### Load Bias
    bias = m_utils.stack_calib(redux_dir, data_dir, date)
    bias = bias[::-1,0:actypix] #Remove overscan
    #### Load Dark
    #dark, dhdr = m_utils.stack_calib(redux_dir, data_dir, date, frame='dark')
    #dark = dark[::-1,0:actypix]
    #try:
    #    dark *= spec_hdr['EXPTIME']/dhdr['EXPTIME'] ### Scales linearly with exposure time
    #except:
    #    ### if EXPTIMES are unavailable, can't reliably subtract dark, just turn it
    #    ### into zeros
    #    dark = np.zeros(ccd.shape)
    #### Analyze overscan (essentially flat, very minimal correction)
    overscan_fit = m_utils.overscan_fit(overscan)
    
    #### Making up this method, so not sure if it's good, but if it works it should reduce ccd noise
    bias = m_utils.bias_fit(bias, overscan_fit)
    
    ### Make master slitFlats
    sflat = m_utils.stack_flat(redux_dir, data_dir, date)
    ### If no slit flat, sflat returns all ones, don't do any flat fielding
    if np.max(sflat) - np.min(sflat) == 0:
        norm_sflat = np.ones(ccd.shape)
    else:
        norm_sflat = m_utils.make_norm_sflat(sflat, redux_dir, date, spline_smooth=True, plot_results=False)
    norm_sflat = np.ones(ccd.shape)
    
    ### Calibrate ccd
    ccd -= bias #Note, if ccd is 16bit array, this operation can cause problems
    #ccd -= dark
    
    ### Find new background level (now more than readnoise because of bias/dark)
    ### use bstd instead of readnoise in optimal extraction
    if (np.max(norm_sflat) == np.min(norm_sflat)):
        cut = int(10*readnoise)
        junk, bstd = m_utils.remove_ccd_background(ccd,cut=cut)
        rn_eff = bstd*gain
    else:
        bgonly = ccd[norm_sflat==1]
        cut = np.median(bgonly)
        if cut < 15:
            cut = 15 ### enforce minimum
        junk, bstd = m_utils.remove_ccd_background(bgonly,cut=cut)

    ### flatten ccd, and inverse variance
    ccd /= norm_sflat
    
    ### Apply gain
    ccd *= gain
else:
    norm_sflat = np.ones(ccd.shape)
    ccd -= np.median(ccd)
    rn_eff = 3.63
    gain = 1

######################################
### Find or load trace information ###
######################################

### Dynamically search for most recent arc frames (unless a date is supplied)
if args.date is None:
    arc_date = m_utils.find_most_recent_frame_date('arc', data_dir)
else:
    arc_date = args.date

### Assumes fiber flats are taken on same date as arcs
fiber_flat_files = glob.glob(os.path.join(data_dir,'*'+arc_date,'*[fF]iber*[fF]lat*'))

if os.path.isfile(os.path.join(redux_dir,arc_date,'trace_{}_2.fits'.format(arc_date))):
    print "Loading Trace Frames" 
    trace_fits = pyfits.open(os.path.join(redux_dir,arc_date,'trace_{}_2.fits'.format(arc_date)))
    hdr = trace_fits[0].header
    profile = hdr['PROFILE']
    multi_coeffs = trace_fits[0].data
else:
    ### Combine four fiber flats into one image
    profile = args.profile
    if profile == 'moffat' or profile == 'gaussian':
        pass
    else:
        print("Invalid profile choice ({})".format(profile))
        print("Available choices are:\n  moffat\n  gaussian")
        exit(0)
    trace_ccd = np.zeros((np.shape(ccd)))
    for ts in ['T1','T2','T3','T4']:
        flatfits = pyfits.open(os.path.join(redux_dir, arc_date, 'combined_flat_{}.fits'.format(ts)))
        flat = flatfits[0].data
        fhdr = flatfits[0].header
        #Choose fiberflats with iodine cell in
        norm = 10000 #Arbitrary norm to match flats
        tmmx = np.median(np.sort(np.ravel(flat))[-100:])
        trace_ccd += flat[:,0:actypix].astype(float)*norm/tmmx
    ### Find traces and label for the rest of the code
    trace_ccd -= bias
    print("Searching for Traces")
    multi_coeffs = m_utils.find_trace_coeffs(trace_ccd,2,fiber_space,num_points=num_points,num_fibers=num_fibers,skip_peaks=1, profile=profile)
    hdu1 = pyfits.PrimaryHDU(multi_coeffs)
    hdulist = pyfits.HDUList([hdu1])
    hdu1.header.append(('PROFILE',profile,'Cross-dispersion profile used for trace fitting'))
    hdulist.writeto(os.path.join(redux_dir,arc_date,'trace_{}.fits'.format(arc_date)),clobber=True)
    
### Set components of multi_coeffs
t_coeffs = multi_coeffs[0]
i_coeffs = multi_coeffs[1]
s_coeffs = multi_coeffs[2]
p_coeffs = multi_coeffs[3]

if os.path.isfile(os.path.join(redux_dir,'n20170307','n20170307.HR3799.0015.2dproc.npy')): #os.path.isfile(os.path.join(redux_dir,'n20161123','m_sim2d.npy1')):
    spectrum_2D, spec_covar, model_2D = np.load(os.path.join(redux_dir,'n20170307','n20170307.HR3799.0015.2dproc.npy'))# os.path.join(redux_dir,'n20161123','m_sim2d.npy'))
else:
    ### And then extract!
    spectrum_2D, spec_covar, model_2D = m_utils.extract_2D(ccd, norm_sflat, args.psf, t_coeffs, s_coeffs, p_coeffs, redux_dir, readnoise=rn_eff, gain=gain, return_model=True, verbose=True)

if 'n20170307.HR3799' in filename:
    np.save(os.path.join(redux_dir,'n20170307','n20170307.HR3799.0015.2dproc.npy'),(spectrum_2D, spec_covar, model_2D))

### Examine a small section to comapare 1D and 2d on n20170307.HR3799.0015.fits
'''
model_1D = np.load(os.path.join(redux_dir,'n20170307','n20170307.HR3799.0015.1dproc.npy'))
model_1D *= 1.3
vmn, vmx = 1275, 1290
hmn, hmx = 1804, 1896
#vmn, vmx = 5, 20
#hmn, hmx = 60, 100
ccd_sec = ccd[vmn:vmx,hmn:hmx]
invar = 1/(abs(ccd)+rn_eff**2)
m1d_sec = model_1D[vmn:vmx,hmn:hmx]
m2d_sec = model_2D[vmn:vmx,hmn:hmx]
inv_sec = 1/(abs(ccd_sec)+rn_eff**2)
chigrid1 = (ccd_sec-m1d_sec)**2*inv_sec
chigrid2 = (ccd_sec-m2d_sec)**2*inv_sec
chi1r = np.sum(chigrid1)/(ccd_sec.size-3*ccd_sec.shape[1])
chi2r = np.sum(chigrid2)/(ccd_sec.size-ccd_sec.shape[1])

def make_invar_mask(invar, t_coeffs, ts=[True, True, False, False], pad=7, nf=106, stf=5, data=None, model=None):
    hcents = np.arange(2048)
    hscale = (hcents-1024)/2048
    inv_mask = np.zeros(invar.shape)
    for i in range(stf,stf+nf):
        if data is not None:
            chi = np.zeros((data.shape))
        tsi = (i-1)%4
        if not ts[tsi]:
            continue
        vcents = np.poly1d(t_coeffs[:,i])(hscale)
        for j in range(2048):
            vj = int(vcents[j])
            inv_mask[vj-pad:vj+pad+1, j] = 1
            if model is not None and data is not None:
                chi[vj-pad:vj+pad+1, j] = (data[vj-pad:vj+pad+1, j]-model[vj-pad:vj+pad+1, j])**2*invar[vj-pad:vj+pad+1, j]
        if model is not None and data is not None:
            chir = np.sum(chi)/(np.sum(chi!=0)-3*2028)
            print "Chi_r for fiber {} = {}".format(i,chir)
    return invar*inv_mask
        
nf = 106
stf = 5 #37-2D better than 1D
invar = make_invar_mask(invar, t_coeffs, nf=nf, stf=stf, data=ccd, model=model_2D)
pad = 10
invar[0:pad,:] = 0
invar[-pad:,:] = 0
invar[:,0:pad] = 0
invar[:,-pad:] = 0

chigrid1A = (ccd-model_1D)**2*invar
chigrid2A = (ccd-model_2D)**2*invar

cr_msk = np.sqrt(chigrid1A) > 10
invar[cr_msk] = 0

chigrid1A = (ccd-model_1D)**2*invar
chigrid2A = (ccd-model_2D)**2*invar

chi1rA = np.sum(chigrid1A)/(np.sum(invar!=0)-3*(2048-2*pad)*nf)
chi2rA = np.sum(chigrid2A)/(np.sum(invar!=0)-(2048-2*pad)*nf)

print "1D:", chi1rA#, np.mean(ccd_sec-m1d_sec)
print "2D:", chi2rA#, np.mean(ccd_sec-m2d_sec)

plt.figure("res1d, res2d")
plt.imshow((np.vstack((ccd_sec-m1d_sec,ccd_sec-m2d_sec))), interpolation='none')
plt.plot([0,92],[14,14],'w',linewidth=2)
ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.set_xlim(0,91)
ax.set_ylim(29,0)
savedir = os.environ['THESIS']
plt.savefig(os.path.join(savedir,'op_sp_res.pdf'), bbox_inches='tight')
plt.show()
plt.close()


waves = np.arange(spectrum_2D.shape[1])#*101/wls
sp2D = spectrum_2D[52]
sperrs = np.sqrt(spec_covar[52])
plt.figure("2D vs. Optimal Extraction")
plt.errorbar(waves,sp2D,np.sqrt(spec_covar[52]),color='b',linestyle='--',linewidth='2', label='2D PSF')

###Compare to optimal extraction:
opt_fits = pyfits.open(os.path.join(redux_dir, 'n20170307','n20170307.HR3799.0015.proc.fits'))
opt_dat = opt_fits[0].data
opt_inv = opt_fits[1].data
opt_wav = opt_fits[2].data
opt_dat_sec = opt_dat[0,13,:]
opt_inv_sec = opt_inv[0,13,:]
opt_errs = np.sqrt(1/opt_inv_sec)
scl = 1#np.mean(opt_dat_sec)/np.mean(spectrum_2D)
plt.errorbar(waves,opt_dat_sec*gain,opt_errs/np.sqrt(gain),color='k',linewidth='2', label='Optimal')
i1, i2 = 800, 1200
print np.std((opt_dat_sec[i1:i2]*gain-sp2D[i1:i2])*np.sqrt(opt_inv_sec[i1:i2]))
ax = plt.gca()
plt.legend(loc=4,fontsize = 24)
ax.set_xlabel('Pixel', fontsize = 24)
ax.set_ylabel('Counts', fontsize = 24)
plt.show()
plt.close()
#'''