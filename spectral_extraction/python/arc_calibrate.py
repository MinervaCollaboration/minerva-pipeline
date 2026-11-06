#!/usr/bin/env python 2.7
'''
# Finds wavelength solution from arc frames for MINERVA
# Must have file table1.dat in MINERVA_SIM_DIR
# This holds approximate wavelength solutions at five points per trace
# table1.dat is manually generated

INPUTS:
    Many input argument options (see below)
    Defaults should be appropriate in most cases
    This will automatically search for the most recent arc frames
    
OUTPUTS:
    4 fits files (one for each telescope): wavelength_soln_T#.fits
    These are saved in MINERVA_REDUX_DIR/date/
    
'''
#Import all of the necessary packages
from __future__ import division
import pyfits
import os
import csv
import glob
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import special as sf
import minerva_utils as m_utils
import argparse

#### Input arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d","--date",help="Date of arc exposure, format nYYYYMMDD",default=None)
parser.add_argument("-fib","--num_fibers",help="Number of fibers to extract",
                    type=int,default=29)
parser.add_argument("-bs","--bundle_space",help="Minimum spacing (in pixels) between fiber bundles",
                    type=int,default=40)
parser.add_argument("-fs","--fiber_space",help="Minimum spacing (in pixels) between fibers within a bundle",
                    type=int,default=13)
parser.add_argument("-ts","--telescopes",help="Number of telescopes feeding spectrograph",
                    type=int,default=4)
parser.add_argument("-v","--verbose",help="Will display status messages",action="store_true")
parser.add_argument("-o","--overwrite",help="If enabled will overwrite saved arc/flat stacks",action="store_true")
parser.add_argument("-c","--check_results",help="Plots 2 arcs, wl vs intensity, for each telescope",action="store_true")
parser.add_argument("-u","--use_base",help="Use base date of n20160130 as foundation for wl estimates",action="store_true")
inargs = parser.parse_args()
num_fibers = inargs.num_fibers*inargs.telescopes
verbose = inargs.verbose
no_overwrite = not inargs.overwrite
use_base = inargs.use_base
fiber_space = inargs.fiber_space

#### Set Paths
try:
    data_dir = os.environ['MINERVA_DATA_DIR']
except KeyError:
    print("Must set environmental variable MINERVA_DATA_DIR")
    exit(0)
try:
    redux_dir = os.environ['MINERVA_REDUX_DIR']
except KeyError:
    print("Must set environmental variable MINERVA_REDUX_DIR")
    exit(0)    
try:
    sim_dir = os.environ['MINERVA_SIM_DIR']
except KeyError:
    print("Must set environmental variable MINERVA_SIM_DIR")
    exit(0)
    
### Dynamically search for most recent arc frames (unless a date is supplied)
if inargs.date is None:
    date = m_utils.find_most_recent_frame_date('arc', data_dir)
else:
    date = inargs.date

if use_base:
    base_date = 'n20160130'

arc_files = glob.glob(os.path.join(data_dir,'*'+date,'*[tT][hH][aA][rR]*'))
### Assumes fiber flats are taken on same date as arcs
fiber_flat_files = glob.glob(os.path.join(data_dir,'*'+date,'*[fF]iber*[fF]lat*'))
if len(fiber_flat_files) == 0:
    print "ERROR: No fiber flats found on {}".format(date)
    exit(0)

telescopes = ['T1','T2','T3','T4']
for ts in telescopes:
    if verbose:
        print("Starting on telescope {}".format(ts))
    method = 'median'
    for frm in ['arc', 'flat']:
        if frm == 'arc':
            m_utils.save_comb_arc_flat(arc_files, frm, ts, redux_dir, date, no_overwrite=no_overwrite, verbose=verbose, method=method)
        elif frm == 'flat':
            m_utils.save_comb_arc_flat(fiber_flat_files, frm, ts, redux_dir, date, no_overwrite=no_overwrite, verbose=verbose, method=method)
    flat = pyfits.open(os.path.join(redux_dir,date,'combined_flat_{}.fits'.format(ts)))[0].data
    arc = pyfits.open(os.path.join(redux_dir,date,'combined_arc_{}.fits'.format(ts)))[0].data
    
    ### Invert ccd top/bottom orientation - this is just my convention
    flat = flat[::-1,:]
    arc = arc[::-1,:]
    
    #Hardcoded characteristics of the trace
    num_fibers=29
    fiberspace = np.linspace(60,90,num_fibers)/2 #approximately half the fiber spacing
    fiberspace = fiberspace.astype(int)
    
    #Now to get traces
    ypix = 2048 #Manual, overscan is included in header value
    xpix = 2052
    num_points = 20 #2048 = all pixels
    yspace = int(floor(ypix/(num_points+1)))
    yvals = yspace*(1+np.arange(num_points))
    
    multi_coeffs = m_utils.find_trace_coeffs(flat, 2, fiber_space, num_points=num_points, num_fibers=num_fibers, skip_peaks=1)
    trace_coeffs = multi_coeffs[0]
    trace_intense_coeffs = multi_coeffs[1]
    trace_sig_coeffs = multi_coeffs[2]
    trace_pow_coeffs = multi_coeffs[3]

    ############################################################
    ############ Below start arc calibration code ##############
    ############################################################
             
    ############################################################
    ### First use trace to find cross sectional intensity of 
    ### each arc (only those above ~1.5*background) as a
    ### function of pixel position.
    ############################################################        
             
    ys = (np.arange(ypix)-ypix/2)/ypix
    xs = np.zeros((num_fibers,len(ys)))
    Is = np.zeros((num_fibers,len(ys)))
    arc_bg = np.median(arc) #approximate background estimate
    line_amps = np.zeros((num_fibers,ypix))
    nonzero_count = 0
    zero_count = 0
    line_gauss = np.zeros((num_fibers,ypix)) #intensity from gaussian fit
    if os.path.isfile(os.path.join(redux_dir,date,'line_gauss_{}.npy'.format(ts))):
        line_gauss = np.load(os.path.join(redux_dir,date,'line_gauss_{}.npy'.format(ts)))
    else:
        for j in range(num_fibers):
            pad = 4
            for k in range(ypix):
                xs[j,k] = trace_coeffs[2,j]*ys[k]**2+trace_coeffs[1,j]*ys[k]+trace_coeffs[0,j]
                if np.isnan(xs[j,k]) or xs[j,k]<0:
                    line_amps[j,k] = 0
                    line_gauss[j,k] = 0
                    continue
                xs[j,k] = int(xs[j,k])
                Is[j,k] = trace_intense_coeffs[2,j]*ys[k]**2+trace_intense_coeffs[1,j]*ys[k]+trace_intense_coeffs[0,j]
                xsub1 = np.arange(-pad,pad+1)+int(xs[j,k])
                xsub1 = xsub1[(xsub1>=0)*(xsub1<xpix)]
                zvals = arc[xsub1,k]
                if np.max(zvals)>(arc_bg+15): #Hack on setting background cutoff
                    #Lines to find more precise zmax point
                    zpeak_ind = np.argmax(zvals)
                    zsub = zvals[zpeak_ind-2:zpeak_ind+3]
                    xsub = np.arange(len(zsub))-2
                    profile = np.ones((len(xsub),3)) #Quadratic fit
                    profile[:,1] = xsub #scale data to get better fit
                    profile[:,2] = xsub**2 
                    noise = np.diag(1/zsub)
                    coeffsn, junk = sf.chi_fit(zsub-arc_bg,profile,noise)
                    xmx = -coeffsn[1]/(2*coeffsn[2])
                    zmx = coeffsn[2]*xmx**2+coeffsn[1]*xmx+coeffsn[0]
                    line_amps[j,k] = zmx
                    pad2 = 7
                    xsub = np.arange(-pad2,pad2+1)+int(xs[j,k])
                    xsub = xsub[(xsub>=0)*(xsub<xpix)]
                    zvals = arc[xsub,k]
                    try:
                        params, errarr = sf.gauss_fit(xsub,zvals)
                    except RuntimeError:
                        if verbose:
                            print("No fit for fiber {fib} at pixel {px}".format(fib=j,px=k))
                        line_gauss[j,k] = 0
                    else:
                        line_gauss[j,k] = params[2] #-background?
                    nonzero_count+=1 
                else:
                    line_amps[j,k] = 0
                    line_gauss[j,k] = 0
                    zero_count+=1
                                
        #xtraces start at red, move toward blue.  Want to reverse from proper ordering
        line_gauss = line_gauss[::-1,:]
        #Hack to shift T3 lines to match T1 and T2
        if ts != 'T1':
            line_gauss = np.vstack((line_gauss[1:29,:],np.zeros((1,np.shape(line_gauss)[1]))))
        if not os.path.isdir(os.path.join(redux_dir,date)):
            os.makedirs(os.path.join(redux_dir,date))
        ### save dictionaries of pixel peaks
        np.save(os.path.join(redux_dir,date,'line_gauss_{}.npy'.format(ts)),line_gauss)
    
    ############################################################
    ############ Open line list (hardcoded for now) ############
    ############################################################ 
    
    wl_min = 4874
    wl_max = 6466
    rows = 8452
    cols = 2
    lamp_lines = np.zeros((rows,cols))
    line_names = np.chararray((rows,1),itemsize=6)
    with open(os.path.join(sim_dir,'table1.dat'),'r') as ll:
        ln = ll.readlines()
        for i in range(rows):
            lamp_lines[i,0] = ln[i][0:11]
            lamp_lines[i,1] = ln[i][33:40]
            line_names[i] = ln[i][54:-1]
        
    wls = lamp_lines[:,0] #wavelengths
    its = lamp_lines[:,1] #relative intensities (may not need)
    itmax = max(its)
    itssc = its/itmax
    
    ############################################################
    ###Make cuts to wls, its based on proximity and intensity###
    ############################################################
    
    wlref = wls
    itref = itssc
         
    ############################################################
    ### Now make cuts to "extracted" spectrum from arc frame.
    ### These can include overlapping lines, anomalous FWHM, low 
    ### intensity, and large slope in the background.
    ### Get good results with only "overlapping line" cut
    ############################################################
    
    if os.path.isfile(os.path.join(redux_dir,date,'high_pix_{}.npy'.format(ts))):
        [pos_d, mx_it_d] = np.load(os.path.join(redux_dir,date,'high_pix_{}.npy'.format(ts)))  
    else:
        sampling_est = 3 #Whole number estimate of sampling
        #use dictionaries since number of peaks per fiber varies
        mx_it_d = dict() #max intensities of each peak
        stddev_d = dict() #FWHM of each peak
        pos_d = dict() #position of each peak
        slp_d = dict() #background slope around each peak
        for i in range(num_fibers):
            ### Find peak position estimates
            pos_est = np.zeros((len(line_gauss[i,:])),dtype=int)
            for j in range(2*sampling_est,len(line_gauss[i,:])-2*sampling_est):
                #if point is above both its neighbors, call it a peak
                if line_gauss[i,j]>line_gauss[i,j-1] and line_gauss[i,j]>line_gauss[i,j+1]:
                    pos_est[j] = j
            #Then remove extra elements from pos_est
            pos_est = pos_est[np.nonzero(pos_est)[0]]
            #Cut out any that are within 2*sampling of each other (won't be able to fit well)
            #Optional - add fitting algorithm for overlapping gaussians to recover these
            #blended lines
            pos_diff = ediff1d(pos_est)
            if np.count_nonzero(pos_diff<(2*sampling_est))>0:
                close_inds = np.nonzero(pos_diff<(2*sampling_est))[0]
                close_inds = np.concatenate((close_inds,close_inds+1))
                close_inds = np.unique(close_inds)
                close_inds = np.sort(close_inds)
                pos_est = np.delete(pos_est,close_inds)
            #variable length arrays to dump into dictionary
            num_pks = len(pos_est)
            mx_it = zeros((num_pks))
            stddev = zeros((num_pks))
            pos = zeros((num_pks))
            slp = zeros((num_pks))
            #Now fit gaussian with background to each (can improve function later)
            for j in range(num_pks):
                xarr = pos_est[j] + np.arange(-(2*sampling_est),(2*sampling_est),1)
                xarr = xarr[(xarr>0)*(xarr<2048)]
                yarr = line_gauss[i,:][xarr]
                try:
                    params, errarr = sf.gauss_fit(xarr,yarr,invr = 1/(abs(yarr)+10))
                except RuntimeError:
                    params = np.zeros(5)
                mx_it[j] = params[2] #height
                stddev[j] = params[0]#*2*sqrt(2*log(2)) #converted from std dev
                pos[j] = params[1] #center
                slp[j] = params[4] #bg_slope
            mx_it_d[i] = mx_it[np.nonzero(pos)[0]] #Remove zero value points
            stddev_d[i] = stddev[np.nonzero(pos)[0]]
            pos_d[i] = pos[np.nonzero(pos)[0]] 
            slp_d[i] = slp[np.nonzero(pos)[0]]
                
        stddev_arr = np.array(())
        slp_arr = np.array(())
        for i in range(num_fibers):
            stddevs = stddev_d[i][np.nonzero(stddev_d[i])[0]]
            stddev_arr = np.concatenate((stddev_arr,stddevs))
            slps = slp_d[i][np.nonzero(slp_d[i])[0]]
            slp_arr = np.concatenate((slp_arr,slps))
            
        #Can use more advanced statistics if needed
        stddev_mean = np.mean(stddev_arr)
        stddev_std = np.std(stddev_arr)
        slp_mean = np.mean(slp_arr)
        slp_std = np.std(slp_arr)
        
        if not os.path.isdir(os.path.join(redux_dir,date)):
            os.makedirs(os.path.join(redux_dir,date))
        ### save dictionaries of pixel peaks
        np.save(os.path.join(redux_dir,date,'high_pix_{}.npy'.format(ts)),[pos_d,mx_it_d])

    ############################################################
    ### Next, for each non-cut peak, search for nearby line from
    ### line list (must use inputs, lamba_0, lambda_f, px_0, px_f)
    ### Apply wavelength to pixel for all matches, then fit the
    ### result to a polynomial. If fit is poor (or maybe even if
    ### not), run iterative sigma clipping and re-fit.  Idea is
    ### that outliers come from unidentified lines and I can't
    ### yet think of a better way to remove these from consideration
    ############################################################

    ### If previous results are available, use those as initial estimates
    if use_base:
        base_date = 'n20160130'
        [base_pos, base_it] = np.load(os.path.join(redux_dir,base_date,'high_pix_{}.npy'.format(ts)))
        lambdas = dict()
        pixels = dict()
        ### Manual peak shift, not the best, but functional...
        ### Put in shift at orders 2, 14, 27, will linearly extrapolate
        ord1 = 2
        ordm = 14
        ord2 = 27
        ords = [ord1, ordm, ord2]
#        pix_shift = {'T1':[0, 0, 0], 'T2':[0, 0, 0], 'T3':[0, 0, 0], 'T4':[0, 0, 0]}
        pix_shift = {'T1':[-83.67, -73.71, -60.16], 'T2':[-83.021, -72.98, -58.53], 'T3':[-80.18, -69.75, -55.64], 'T4':[-77.28, -66.76, -53]}
        for order in range(len(base_pos)):
            ### Load new and old pixels and intensities
            o_pix = np.copy(base_pos[order])
            o_ints = np.copy(base_it[order])
            mx_mask = np.ones((len(o_pix)), dtype=bool)
            if len(o_pix) == 0:
                lambdas[order] = np.zeros((1))
                pixels[order] = np.zeros((1))
                continue
            if np.sort(o_ints)[-1]/np.sort(o_ints)[-2] > 5:
                mx_mask[np.argmax(o_ints)] = 0
                o_pix = o_pix[mx_mask]
                o_ints = o_ints[mx_mask]
            o_ints /= np.max(o_ints) ### change to % of max
            n_pix = np.copy(pos_d[order])
            n_ints = np.copy(mx_it_d[order])
            mx_mask_n = np.ones((len(n_pix)), dtype=bool)
            if len(n_pix) == 0:
                lambdas[order] = np.zeros((1))
                pixels[order] = np.zeros((1))
                continue
            if np.sort(n_ints)[-1]/np.sort(n_ints)[-2] > 3:
                mx_mask_n[np.argmax(n_ints)] = 0
                n_pix = n_pix[mx_mask_n]
                n_ints = n_ints[mx_mask_n]
            n_ints /= np.max(n_ints) ### change to % of max
            shft_coeffs = np.polyfit(ords, pix_shift[ts],2)
            n_shift = np.poly1d(shft_coeffs)(order)
            n_pix += n_shift
            if (order == ord2 or order == ord1 or order == ordm) and inargs.check_results:
                sf.plt_deltas(o_pix,o_ints)
                sf.plt_deltas(n_pix,n_ints,'g')
                plt.show()
                plt.close()
            ### Iterate to find new peaks within dlt_px of new_peaks
            dlt_px = 5
            int_err = 0.3
            wl_hdu =   pyfits.open(os.path.join(redux_dir,base_date,'wavelength_soln_{}.fits'.format(ts)))
            wl_coeffs =  wl_hdu[0].data
            lambda_tmp = np.zeros(len(n_ints))
            pixels_tmp = -1*np.ones(len(n_ints))
            for pk in range(len(n_pix)):
                good_peaks = o_pix[(o_pix>n_pix[pk]-dlt_px) * (o_pix<n_pix[pk]+dlt_px)]
                good_ints = o_ints[(o_pix>n_pix[pk]-dlt_px) * (o_pix<n_pix[pk]+dlt_px)]
                if len(good_peaks) == 1:
                    pixels_tmp[pk] = n_pix[pk] - n_shift
                    lambda_tmp[pk] = np.poly1d(wl_coeffs[order])(2*(good_peaks[0]-1024)/2048)
                elif len(good_peaks) > 1:
                    best_int = (good_ints < (1+int_err)*n_ints[pk]) * (good_ints > (1-int_err)*n_ints[pk])
                    if sum(best_int) > 1:
                        continue
                    elif sum(best_int) == 1:
                        pixels_tmp[pk] = n_pix[pk] - n_shift
                        tpx = 2*(good_peaks[best_int]-1024)/2048
                        lambda_tmp[pk] = np.poly1d(wl_coeffs[order])(tpx)
            lambda_tmp = lambda_tmp[lambda_tmp > 0]
            pixels_tmp = pixels_tmp[pixels_tmp >= 0]
            lambdas[order] = lambda_tmp
            pixels[order] = pixels_tmp
    else:
        lambdas = np.zeros((num_fibers,5)) #input 5 points by eye
        pixels = np.zeros((num_fibers,5))
        skipsets = 2 #skip the first two sets of 5 - these don't show on CCD
        ### Import estimates from file
        with open(os.path.join(sim_dir,'arc_order_estimates.csv')) as arccsv:
            orders = csv.reader(arccsv,delimiter=',')
            ct = 0
            fib = 0
            pt = 0
            for row in orders:
                if ct/8 < skipsets:
                    ct += 1
                    continue
                elif ct%8 == 0 or (ct-1)%8 == 0:
                    ct += 1
                    continue            
                elif (ct+1)%8 == 0:
                    pt = 0
                    fib += 1
                    ct += 1
                else:
                    lambdas[fib,pt] = row[0]
                    pixels[fib,pt] = row[1]
                    ct += 1
                    pt += 1
        #approximate correction to my initial estimates
        if ts == 'T1':
            coeffs = np.array(([8.9617e-3,9.9478e1]))
            pixels+=np.poly1d(coeffs)(pixels)
        elif ts == 'T2':
            coeffs = np.array(([9.412566e-3,1.0294e2]))
            pixels+=np.poly1d(coeffs)(pixels)
        elif ts == 'T3':
            coeffs = np.array(([9.68858e-3,1.062277e2]))
            pixels+=np.poly1d(coeffs)(pixels)
        elif ts == 'T4':
            coeffs = np.array(([1.0007e-2,1.093839e2]))
            pixels+=np.poly1d(coeffs)(pixels)
        
    ### Fit initial guesses to polynomial   
    init_poly_order = 2
    lam_vs_px_coeffs = np.zeros((num_fibers,init_poly_order+1))
    for i in range(num_fibers):
        if lambdas[i][0] != 0:
            tmp_pix = 2*(pixels[i]-ypix/2)/ypix #change interval to [-1,1]
            lam_vs_px_coeffs[i] = np.polyfit(tmp_pix,lambdas[i],init_poly_order)
            
    ### Now dial in a more precise wavelength solution
    fin_poly_order = 6
    blnd_wid = 0.06 #Angstroms, allowable error from initial guess to reference, 0.06 is conservative, resistant to blends
    lam_vs_px_final = np.zeros((num_fibers,fin_poly_order+1))
    for i in range(num_fibers):
        line_px = 2*(pos_d[i]-ypix/2)/ypix
        line_lams = np.poly1d(lam_vs_px_coeffs[i])(line_px)
        ref_lams = np.zeros((len(line_lams)))
        for j in range(len(line_lams)):
            ref_wavelength = wlref[(wlref>(line_lams[j]-blnd_wid))*
                                   (wlref<(line_lams[j]+blnd_wid))]
            if len(ref_wavelength)==1:
                ref_lams[j] = ref_wavelength
        lam_msk = np.nonzero(ref_lams)[0]
        ref_lams = ref_lams[lam_msk] #cut out any zero points
        line_px = line_px[lam_msk]
        if len(line_px) < (fin_poly_order+1):
            print("Fiber {} has too few lines to fit polynomial order {}".format(i,fin_poly_order))
            ### Just use the initial fit
            lam_vs_px_final[i] = np.pad(lam_vs_px_coeffs[i],(fin_poly_order-init_poly_order,0),'constant',constant_values=(0,0))
            if inargs.check_results:
                if i == 2 or i == num_fibers-2:
                    print("Order = {}".format(i))
                    llf = np.poly1d(lam_vs_px_final[i])((np.arange(2048)-1024)/1024)
                    plt.plot(llf, line_gauss[i])
                    plt.show()
                    plt.close()
            continue
        else:
            lam_vs_px_final[i] = np.polyfit(line_px,ref_lams,fin_poly_order)
            if inargs.check_results:
                if i == 2 or i == 14 or i == num_fibers-2:
                    llf = np.poly1d(lam_vs_px_final[i])((np.arange(2048)-1024)/1024)
                    plt.plot(llf, line_gauss[i])
                    plt.show()
                    plt.close()
        
    ###################### Save values to file ####################################   
    hdu = pyfits.PrimaryHDU(lam_vs_px_final)
    hdu.header.append(('POLYORD',fin_poly_order,'Polynomial order used for fitting'))
    hdulist = pyfits.HDUList([hdu])
    hdulist.writeto(os.path.join(redux_dir,date,'wavelength_soln_'+ts+'.fits'),clobber=True)