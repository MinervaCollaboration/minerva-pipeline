#!/usr/bin/env python 2.7

#Start of a generic tracefit program.  Geared now toward MINERVA initial data

#Import all of the necessary packages
from __future__ import division
import pyfits
import os
import math
import time
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from matplotlib import cm
#import scipy
#import scipy.stats as stats
#import scipy.special as sp
#import scipy.interpolate as si
#import scipy.optimize as opt
import scipy.sparse as sparse
import scipy.signal as sig
#import scipy.linalg as linalg
import solar
import special as sf

def fit_trace(x,y,ccd,form='gaussian'):
    """quadratic fit (in x) to trace around x,y in ccd
       x,y are integer pixel values
       input "form" can be set to quadratic or gaussian
    """
    x = int(x)
    y = int(y)
    if form=='quadratic':
        xpad = 2
        xvals = np.arange(-xpad,xpad+1)
        def make_chi_profile(x,y,ccd):
            xpad = 2
            xvals = np.arange(-xpad,xpad+1)
            zvals = ccd[x+xvals,y]
            profile = np.ones((2*xpad+1,3)) #Quadratic fit
            profile[:,1] = xvals
            profile[:,2] = xvals**2
            noise = np.diag((1/zvals))
            return zvals, profile, noise
        zvals, profile, noise = make_chi_profile(x,y,ccd)
        coeffs, chi = sf.chi_fit(zvals,profile,noise)
    #    print x
    #    print xvals
    #    print x+xvals
    #    print zvals
    #    plt.errorbar(x+xvals,zvals,yerr=sqrt(zvals))
    #    plt.plot(x+xvals,coeffs[2]*xvals**2+coeffs[1]*xvals+coeffs[0])
    #    plt.show()
        chi_max = 100
        if chi>chi_max:
            #print("bad fit, chi^2 = {}".format(chi))
            #try adacent x
            xl = x-1
            xr = x+1
            zl, pl, nl = make_chi_profile(xl,y,ccd)
            zr, pr, nr = make_chi_profile(xr,y,ccd)
            cl, chil = sf.chi_fit(zl,pl,nl)
            cr, chir = sf.chi_fit(zr,pr,nr)
            if chil<chi and chil<chir:
    #            plt.errorbar(xvals-1,zl,yerr=sqrt(zl))
    #            plt.plot(xvals-1,cl[2]*(xvals-1)**2+cl[1]*(xvals-1)+cl[0])
    #            plt.show()
                xnl = -cl[1]/(2*cl[2])
                znl = cl[2]*xnl**2+cl[1]*xnl+cl[0]
                return xl+xnl, znl, chil
            elif chir<chi and chir<chil:
                xnr = -cr[1]/(2*cr[2])
                znr = cr[2]*xnr**2+cr[1]*xnr+cr[0]
    #            plt.errorbar(xvals+1,zr,yerr=sqrt(zr))
    #            plt.plot(xvals+1,cr[2]*(xvals+1)**2+cr[1]*(xvals+1)+cr[0])
    #            plt.show()
                return xr+xnr, znr, chir
            else:
                ca = coeffs[2]
                cb = coeffs[1]
                xc = -cb/(2*ca)
                zc = ca*xc**2+cb*xc+coeffs[0]
                return x+xc, zc, chi
        else:
            ca = coeffs[2]
            cb = coeffs[1]
            xc = -cb/(2*ca)
            zc = ca*xc**2+cb*xc+coeffs[0]
            return x+xc, zc, chi
    elif form=='gaussian':
        xpad = 7
        xvals = np.arange(-xpad,xpad+1)
        zvals = ccd[x+xvals,y]
        params = sf.gauss_fit(xvals,zvals)
        xc = x+params[1] #offset plus center
        zc = params[2] #height (intensity)
#        pxn = np.linspace(xvals[0],xvals[-1],1000)
        fit = sf.gaussian(xvals,abs(params[0]),params[1],params[2],params[3],params[4])
        chi = sum((fit-zvals)**2/zvals)
        return xc, zc, chi


#set paths
pathd = os.environ['MINERVA_DATA_DIR']
paths = os.environ['MINERVA_SIM_DIR']

#Import tungsten fiberflat
fileflat = os.path.join(pathd,'initial_spectro_runs','test4.FIT')
#fileflat = os.path.join(paths,'minerva_flat.fits')
ff = pyfits.open(fileflat,ignore_missing_end=True,uint=True)
hdr = ff[0].header
ccd = ff[0].data

#Import thorium argon arc
filearc = os.path.join(pathd,'initial_spectro_runs','test11.FIT')
#filearc = os.path.join(paths,'minerva_arc.fits')
af = pyfits.open(filearc,ignore_missing_end=True,uint=True)
hdr_a = af[0].header
arc = af[0].data

#Hardcoded characteristics of the trace
num_fibers=29
fiberspace = np.linspace(60,90,num_fibers)/2 #approximately half the fiber spacing
fiberspace = fiberspace.astype(int)

#Now to get traces
ypix = hdr['NAXIS1']  #Number of pixels
xpix = hdr['NAXIS2']
num_points = 20 #2048= all pixels, probably overkill.  Maybe ~100 is good?
yspace = int(floor(ypix/(num_points+1)))
yvals = yspace*(1+np.arange(num_points))

#Check lines to plot cross sections
#for y in yvals:
#    plt.plot(ccd[:,y])
#    
#plt.show()

xtrace = np.zeros((num_fibers,num_points)) #xpositions of traces
ytrace = np.zeros((num_fibers,num_points)) #ypositions of traces
Itrace = np.zeros((num_fibers,num_points)) #relative intensity of flat at trace
chi_vals = np.zeros((num_fibers,num_points)) #returned from fit_trace
bg_cutoff = 0.1*np.max(ccd) #won't work if have cosmic rays or similar

#find initial peaks (center is best in general, but edge is okay here)
px = 1;
trct = 0;
while px<xpix:
    y = yvals[0]
    if ccd[px,y]>bg_cutoff and ccd[px,y]<ccd[px-1,y]: #not good for noisy
        xtrace[trct,0] = px-1
        ytrace[trct,0] = y
        px += 5 #jump past peak
        trct+=1
    else:
        px+=1
if trct<num_fibers:
    xtrace[trct:,0] = np.nan*np.zeros((num_fibers-trct))

for i in range(num_fibers):
    y = yvals[0]
    if not np.isnan(xtrace[i,0]):
        xtrace[i,0], Itrace[i,0], chi_vals[i,0] = fit_trace(xtrace[i,0],y,ccd)
    else:
        Itrace[i,0], chi_vals[i,0] = nan, nan  


#Right now I miss one trace that starts off the edge, but okay enough for testing
for i in range(1,len(yvals)):
    y = yvals[i]
    crsxn = ccd[:,y]
    ytrace[:,i] = y
    for j in range(num_fibers):
        if not np.isnan(xtrace[j,i-1]):
            #set boundaries
            lb = int(xtrace[j,i-1]-fiberspace[j])
            ub = int(xtrace[j,i-1]+fiberspace[j])
            #cutoff at edges
            if lb<0:
                lb = 0
            if ub > xpix:
                ub = xpix
            #set subregion
            xregion = crsxn[lb:ub]
            #only look at if max is reasonably high (don't try to fit background)
            if np.max(xregion)>bg_cutoff:
                #estimate of trace position based on tallest peak
                xtrace[j,i] = np.argmax(xregion)+lb
                #quadratic fit for sub-pixel precision
                xtrace[j,i], Itrace[j,i], chi_vals[j,i] = fit_trace(xtrace[j,i],y,ccd)
            else:
                xtrace[j,i], Itrace[j,i], chi_vals[j,i] = nan, nan, nan
        else:
            xtrace[j,i], Itrace[j,i], chi_vals[j,i] = nan, nan, nan
            
#Finally fit x vs. y on traces.  Start with quadratic for simple + close enough
trace_coeffs = np.zeros((3,num_fibers))
trace_intense_coeffs = np.zeros((3,num_fibers))
for i in range(num_fibers):
    #Given orientation makes more sense to swap x/y
    mask = ~np.isnan(xtrace[i,:])
    profile = np.ones((len(ytrace[i,:][mask]),3)) #Quadratic fit
    profile[:,1] = (ytrace[i,:][mask]-ypix/2)/ypix #scale data to get better fit
    profile[:,2] = ((ytrace[i,:][mask]-ypix/2)/ypix)**2
    noise = np.diag(chi_vals[i,:][mask])
    tmp_coeffs, junk = sf.chi_fit(xtrace[i,:][mask],profile,noise)
    trace_coeffs[0,i] = tmp_coeffs[0]
    trace_coeffs[1,i] = tmp_coeffs[1]
    trace_coeffs[2,i] = tmp_coeffs[2]
    tmp_coeffs2, junk = sf.chi_fit(Itrace[i,:][mask],profile,noise)
    trace_intense_coeffs[0,i] = tmp_coeffs2[0]
    trace_intense_coeffs[1,i] = tmp_coeffs2[1]
    trace_intense_coeffs[2,i] = tmp_coeffs2[2]
        
#Plot to check traces
#fig,ax = plt.subplots()
#ax.pcolorfast(ccd)
#for i in range(num_fibers):
#    ys = (np.arange(ypix)-ypix/2)/ypix
#    xs = trace_coeffs[2,i]*ys**2+trace_coeffs[1,i]*ys+trace_coeffs[0,i]
#    yp = np.arange(ypix)
#    plt.plot(yp,xs)
#plt.show()
#Check trace cross section
#plt.plot(ccd[:,1000])
#ys = (1000-ypix/2)/ypix
#xs = zeros((num_fibers))
#for i in range(num_fibers):
#    xs[i] = trace_coeffs[2,i]*ys**2+trace_coeffs[1,i]*ys+trace_coeffs[0,i]
#plt.plot(xs,np.max(ccd[:,1000])*ones((len(xs))),'r*')
#plt.show()
 


###############################################################
############ END OF TRACE FITTING #############################
###############################################################


'''
           
#Next  - need to assign a wavelength estimate to each point, then fit
#legendre polynomials.  Then I should be able to input to specex (after
#saving as a .fits file of course)
ys = (np.arange(ypix)-ypix/2)/ypix
xs = np.zeros((num_fibers,len(ys)))
Is = np.zeros((num_fibers,len(ys)))
arc_bg = np.median(arc) #approximate background estimate
line_amps = np.zeros((num_fibers,ypix))
nonzero_count = 0
zero_count = 0
line_gauss = np.zeros((num_fibers,ypix)) #intensity from gaussian fit
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
        if np.max(zvals)>(arc_bg*1.5):
            #Lines to find more precise zmax point
            zpeak_ind = np.argmax(zvals)
            zsub = zvals[zpeak_ind-2:zpeak_ind+3]
            xsub = np.arange(len(zsub))-2
            profile = np.ones((len(xsub),3)) #Quadratic fit
            profile[:,1] = xsub #scale data to get better fit
            profile[:,2] = xsub**2 
            noise = np.diag(1/zsub)
            coeffsn, junk = sf.chi_fit(zsub-arc_bg,profile,noise)
#            if k==65:
#                fit = coeffsn[2]*xsub**2 + coeffsn[1]*xsub + coeffsn[0] + arc_bg/1.5
#                plt.plot(xsub,fit)
#                plt.plot(xsub,zsub,'k')
#                plt.show()
            xmx = -coeffsn[1]/(2*coeffsn[2])
            zmx = coeffsn[2]*xmx**2+coeffsn[1]*xmx+coeffsn[0]
            line_amps[j,k] = zmx[0]#/Is[j,k]
            #Lines to find sum of 10px square around the peak (total intensity)
            intensity = sum(arc[-5+xs[j,k]:6+xs[j,k],-5+k:6+k])
            line_amps[j,k] = intensity/Is[j,k]
            #print zmx[0]/Is
            pad2 = 7
            xsub = np.arange(-pad2,pad2+1)+int(xs[j,k])
            xsub = xsub[(xsub>=0)*(xsub<xpix)]
            zvals = arc[xsub,k]
            try:
                params = sf.gauss_fit(xsub,zvals)
            except RuntimeError:
                print("No fit for fiber {fib} at pixel {px}".format(fib=j,px=k))
                line_gauss[j,k] = 0
            else:
                line_gauss[j,k] = params[2]
#            if j==0 and k==332:
#                fit = sf.gaussian(xsub,abs(params[0]),params[1],params[2],params[3],params[4])
#                plt.plot(xsub,zvals,xsub,fit)
#                print xs[j,k]
#                plt.show()
            nonzero_count+=1 
        else:
            line_amps[j,k] = 0
            line_gauss[j,k] = 0
            zero_count+=1
#    if j==0:
#        plt.plot(line_gauss[j,:])
#        plt.show()
#    if j==12:
##        plt.plot(line_gauss[j,:])
##        plt.show()
#        for k in range(643,646):
#            xregion = np.arange(-10,10+1)
#            zvals = arc[xregion+int(xs[j,k]),k]
#            plt.plot(xregion,zvals)
#            pxn = np.linspace(xregion[0],xregion[-1],1000)
#            fit = sf.gaussian(pxn,abs(params[0]),params[1],params[2],params[3],params[4])
#            print(params)                  
#            plt.plot(pxn,fit)
#            plt.show()
                    
        

#############################################
###INTENSITY AS SUM OF REGION AROUND PEAKS###
#############################################

intensity = np.zeros(np.shape(line_amps))
for j in range(num_fibers):
    pk_width = 6
    peaks = sig.find_peaks_cwt(line_amps[j,:],np.arange(4,5))
    pk_ind = 0
    for k in range(ypix):
        if pk_ind >= len(peaks):
            break
        elif peaks[pk_ind]==k:
            #if there is a peak here, sum in a box around it
            intensity[j,k] = sum(arc[-pk_width+xs[j,k]:pk_width+1+xs[j,k],
                                 -pk_width+k:pk_width+k])
            if j==0 and k==0:
                print xs[j,k]
                print k
            #subtract off background
            intensity[j,k]-=arc_bg*(2*pk_width)**2
            #scale by flat response (assume roughly constant over peak)
            intensity[j,k]/=Is[j,k]
            pk_ind+=1
        else:
            intensity[j,k] = 0
    

#xtraces start at red, move toward blue.  Want to reverse from proper ordering
#line_amps = line_amps[::-1,:]
line_gauss = line_gauss[::-1,:]
intensity = intensity[::-1,:]/np.max(intensity)
all_lines = np.resize(line_amps,[1,np.size(line_amps)])
#print len(all_lines[all_lines>0.1*np.max(all_lines)])
#plt.plot(all_lines[all_lines>0.1*np.max(all_lines)])
#plt.show()
#all_lines = np.resize(line_amps,np.shape(line_amps)[0]*np.shape(line_amps)[1])            
wl_min = 4874
wl_max = 6466
trc = 27
amps1 = intensity[trc,:]

#wls = np.linspace(40*2,40*3,len(amps1))
#plt.ion()
#plt.plot(wls+4874,amps1/2.8)

pathm = os.environ['MINERVA_SIM_DIR']
rows = 8442
cols = 2
lamp_lines = np.zeros((rows,cols))
line_names = np.chararray((rows,1),itemsize=6)
with open(pathm+'table1.dat','r') as ll:
    ln = ll.readlines()
    for i in range(rows):
        lamp_lines[i,0] = ln[i][0:11]
        lamp_lines[i,1] = ln[i][33:40]
        line_names[i] = ln[i][54:-1]
    
lamp_lines[0:1,:]

wls = lamp_lines[:,0]
its = lamp_lines[:,1]
itmax = max(its)
itssc = its/itmax
#Bin the ThAr lines for comparison to arc frame
bin_wid = 0.06 #Angstroms
wlrng = np.arange(4874,6466,bin_wid)
wlrng = np.reshape(wlrng,(1,len(wlrng)))
itrng = np.zeros((np.shape(wlrng)))
for i in range(len(wlrng.T)):
    itrng[0,i] = sum(itssc[(wls>(wlrng[0,i]-bin_wid/2))*(wls<(wlrng[0,i]+bin_wid/2))])

wlst = wlrng[0,0]

#Make cuts to wlrng, itrng based on proximity and intensity
wlref = wls
itref = itssc

itmask = itref>(0.30*np.max(itref)) #Mask bottom 30%
wlref = wlref[itmask]
itref = itref[itmask]

#blnd_wid = 0.35 #Angstrom seperation that is considered minimum separation
#idx = 0
#while idx < len(wlref)-1:
#    if abs(wlref[idx]-wlref[idx+1])<blnd_wid:
#        wlref = np.delete(wlref,idx)
#        wlref = np.delete(wlref,idx)
#        itref = np.delete(itref,idx)
#        itref = np.delete(itref,idx)
#    else:
#        idx+=1
#        

#wl_wid = 40 #Angstroms - estimated
bin_wid = 0.06 #Angstroms
wl_wids = np.arange(40,60,.1)
min_chi = np.zeros((len(wl_wids)))


for ww in range(len(wl_wids)):
    wl_wid = wl_wids[ww]
    px_val = wl_wid/len(amps1)
    px_st = 0
    bin_px = bin_wid/px_val #bin width in pixel units
    num_vals = int(wl_wid/bin_wid)
    wlend = wlst+wl_wid
    #Bin arc lines
    wl1 = np.arange(wlst,wlend,bin_wid)
    it_px = np.zeros((num_vals))
    cur_px = 0
    for j in range(num_vals):
        px_end = px_st+bin_px
        num_px = int(ceil(px_end-px_st))
        if num_px == 1:
            it_px[j] = (px_end-px_st)*amps1[cur_px]
            if px_end%1 == 0:
                cur_px+=1
        else:
            for k in range(num_px):
                if k==0:
                    it_px[j] += (1-px_st%1)*amps1[cur_px]
                    cur_px+=1
                elif k==(num_px-1):
                    it_px[j] += (px_end%1)*amps1[cur_px]
                else:
                    it_px[j] += amps1[cur_px]
                    cur_px+=1
        px_st = px_end
        
    #wl_org = np.linspace(wlst,wlend-px_val,ypix)
    #plt.plot(wl_org,amps1,wl1,it_px)
    #plt.show()
    
    #Mask short peaks, below 50% of max
    it_px = it_px*(it_px>0.5*np.max(it_px))
    it_msk = itrng*(itrng>0.5*np.max(itrng))
    
    checks = int((wlrng[-1]-wlrng[0]-wl_wid)/bin_wid)
    chi_vs_arc = np.zeros((checks-1))
    for ct in range(checks-1):
        chi_vs_arc[ct] = sum((it_msk[ct:ct+len(it_px)]-it_px)**2)#Don't bother normalizing
        
#    plt.plot(chi_vs_arc)
#    plt.show()
    min_chi[ww] = np.min(chi_vs_arc)
    
plt.plot(wl_wids,min_chi)
plt.show()


plt.ion()
#sdss = np.loadtxt(pathm+'sdsslines.txt',delimiter='\t')
#swl = sdss.T[0]
#sit = sdss.T[1]
#swlplt = np.vstack((swl,swl))
#sitplt = np.vstack((np.zeros((len(sit))),sit))
#plt.plot(swlplt,sitplt/np.max(sitplt))
#plt.plot(wlrng[0],itrng[0])
noao = pyfits.open(pathm+'noaolamplines.fits')
nit = noao[0].data
wit = np.linspace(4320,6965,len(nit))
plt.plot(wit,nit/np.max(nit))

#
#
#wl_wid = 55
#wl_st = 4874
#cut = 0.3
#itrng = nit[(wit>4874)*(wit<6466)]
#wlrng = wit[(wit>4874)*(wit<6466)]
#itshrt = itrng[itrng>cut*np.max(itrng)]
#wlshrt = wlrng[itrng>cut*np.max(itrng)]
#itshrt = np.reshape(itshrt,(1,len(itshrt)))
#wlshrt = np.reshape(wlshrt,(1,len(wlshrt)))
#wlplt = np.hstack((wlshrt.T,wlshrt.T))
#itplt = np.hstack((np.zeros((len(itshrt.T),1)),itshrt.T))
#plt.plot(wlplt.T,itplt.T/np.max(itplt)*-1.0)
#for i in range(num_fibers):
#    amps1 = line_amps[i,:]
#    ampwl = np.arange(len(amps1))/len(amps1)*wl_wid+wl_st+i*wl_wid
#    if len(amps1)>0:
#        ampshrt = amps1[amps1>cut*np.max(intensity)]
#        ampwlshrt = ampwl[amps1>cut*np.max(intensity)]
#    ampshrt = np.reshape(ampshrt,(len(ampshrt),1))
#    ampwlshrt = np.reshape(ampwlshrt,(len(ampwlshrt),1))
#    ampplt = np.hstack((np.zeros((len(ampshrt),1)),ampshrt))
#    ampwlplt = np.hstack((ampwlshrt,ampwlshrt))
##    print ampshrt
##    print i
#    plt.plot(ampwlplt.T,ampplt.T)
#    plt.plot(ampwl,line_amps[i,:])

j_wls = np.loadtxt(pathm+'thar_angstroms')
j_wls = j_wls[(j_wls>4874)*(j_wls<6466)]
#sf.plt_deltas(j_wls,np.ones(len(j_wls)))
#To remove close blends
blnd_wid = 0.35 #Angstrom seperation that is considered minimum separation
idx = 0
while idx < len(j_wls)-1:
    if abs(j_wls[idx]-j_wls[idx+1])<blnd_wid:
        j_wls = np.delete(j_wls,idx)
        j_wls = np.delete(j_wls,idx)
    else:
        idx+=1
        
#sf.plt_deltas(j_wls,0.5*np.ones(len(j_wls)),color='k')
   
#Now reject overlapping lines, unusual FWHM, low intensity (within 4x bg), lg slope
px_wid = 13 #reject peaks closer than this
#use dictionaries since number of peaks per fiber varies
mx_it_d = dict() #max intensities of each peak
fwhm_d = dict() #FWHM of each peak
pos_d = dict() #position of each peak
slp_d = dict() #background slope around each peak
for i in range(num_fibers):
    #use scipy to get initial guesses of peak locations
    pos_est = np.array(sig.find_peaks_cwt(line_gauss[i,:],np.arange(3,4)))
    #Cut out any that are within px_wid of each other
#    pos_diff = ediff1d(pos_est)
#    if np.count_nonzero(pos_diff<px_wid)>0:
#        close_inds = np.nonzero(pos_diff<px_wid)[0]
#        close_inds = np.concatenate((close_inds,close_inds+1))
#        close_inds = np.unique(close_inds)
#        close_inds = np.sort(close_inds)
#        pos_est = np.delete(pos_est,close_inds)
    #variable length arrays to dump into dictionary
    num_pks = len(pos_est)
    mx_it = zeros((num_pks))
    fwhm = zeros((num_pks))
    pos = zeros((num_pks))
    slp = zeros((num_pks))
    #Now fit gaussian with background to each (can improve function later)
    for j in range(num_pks):
        xarr = pos_est[j] + np.arange(-7,7+1)
        xarr = xarr[(xarr>0)*(xarr<2048)]
        yarr = line_gauss[i,:][xarr]
        try:
            params = sf.gauss_fit(xarr,yarr)
        except RuntimeError:
            params = np.zeros(5)
        mx_it[j] = params[2] #height
        fwhm[j] = params[0]*2*sqrt(2*log(2)) #converted from std dev
        pos[j] = params[1] #center
        slp[j] = params[4] #bg_slope
    mx_it_d[i] = mx_it
    fwhm_d[i] = fwhm
    pos_d[i] = pos
    slp_d[i] = slp
        
  
#Find mean and std FWHM and slope
fwhm_arr = np.array(())
slp_arr = np.array(())
for i in range(num_fibers):
    fwhms = fwhm_d[i][np.nonzero(fwhm_d[i])[0]]
    fwhm_arr = np.concatenate((fwhm_arr,fwhms))
    slps = slp_d[i][np.nonzero(slp_d[i])[0]]
    slp_arr = np.concatenate((slp_arr,slps))
    
#Can use more advanced statistics if needed
fwhm_mean = np.mean(fwhm_arr)
fwhm_std = np.std(fwhm_arr)
slp_mean = np.mean(slp_arr)
slp_std = np.std(slp_arr)
#plt.plot(slp_arr,'k.')

  
#Make cuts
for i in range(num_fibers):
    pos = pos_d[i]
    if len(pos)>0:
        fwhm = fwhm_d[i]
        mx_it = mx_it_d[i]
        slp = slp_d[i]
        msk = np.ones((len(pos)),dtype=np.int)
        for j in range(len(pos)):
            #FWHM cut
#            if fwhm[j]>(fwhm_mean+fwhm_std) or fwhm[j]<(fwhm_mean-fwhm_std):
#                msk[j] = 0
            #Weak lines cut
            if mx_it[j] < 0.3*np.max(line_gauss):#6*arc_bg:
                msk[j] = 0
            #Slope cut
#            if abs(slp[j]) > 10: #semi-arbitrarily set
#                msk[j] = 0
        #update postion array with only unmasked values
        pos_d[i] = pos[msk==1]
        mx_it_d[i] = mx_it[msk==1]
    
SR = 3*80e3#fwhm_mean*80e3
wsc = 39.571
A = wsc/(np.exp((ypix-1)/SR)-1)
ws = 4874.4
wlg1 = A*(np.exp(np.arange(ypix)/SR)-1)+ws
wlg2 = np.linspace(ws,ws+wsc,ypix)
tr = 0
sf.plt_deltas(wlref,itref,'r',2)
#plt.plot(wlg1,line_gauss[tr,:]/np.max(line_gauss),'r')
#plt.plot(wlg2,5*line_gauss[tr,:]/np.max(line_gauss),'k')
#sf.plt_deltas(pos_d[1],mx_it_d[1],'k')
#plt.ioff()

#Coarse guesses for position
bin_wid = 0.03 #Angstroms
#line_gauss=line_gauss[:,::-1]
fib_wl_wid = np.zeros((num_fibers))
fib_wl_st = np.zeros((num_fibers))
#Run through all fibers and see possible fit/match to arc frame
for i in range(1,num_fibers):
    empirical_px = pos_d[i] #Reverse trace order with (ypix-1-__)
#    wl_wids = np.arange(35,65,bin_wid)
#    l0 = np.arange(0,1600,bin_wid) + 4874
    if i==1: #Initial guess (for now 0th fiber is unused)
        wl_wids = np.arange(35,42,bin_wid)
        l0 = np.arange(0,80,bin_wid) + 4874
    else: #range of next guesses based on previous findings
        wl_wids = np.arange(-2,4,bin_wid) + fib_wl_wid[i-1]
        l0 = np.arange(-5,15,bin_wid) + fib_wl_st[i-1] + fib_wl_wid[i-1]
    SR = fwhm_mean*80e3
    A = wl_wids/(np.exp((ypix-1)/SR)-1)
    min_chi = np.zeros((len(wl_wids)))
    res = np.zeros((len(wl_wids),len(l0)))
    min_res_ind = np.zeros((len(wl_wids)))
    min_ress = np.zeros((len(wl_wids)))
    
    for ww in range(len(wl_wids)):
        for lm in range(len(l0)):
            empirical_wl = A[ww]*(np.exp(empirical_px/SR)-1)+l0[lm]
            res[ww,lm] = 0
            for em in empirical_wl:
                res[ww,lm]+=np.min(abs(em-wlref))
        min_res_ind[ww] = np.argmin(res[ww,:])
        min_ress[ww] = np.min(res[ww,:])
    
    min_res_ind1 = np.argmin(min_ress)
    min_res_ind2 = min_res_ind[min_res_ind1]
    print("for fiber {:d}:".format(i+1))
    print " ", res[min_res_ind1,min_res_ind2]/(len(empirical_px)+0.01)/bin_wid
    print " ", wl_wids[min_res_ind1]
    print " ", l0[min_res_ind2]
    fib_wl_wid[i] = wl_wids[min_res_ind1]
    fib_wl_st[i] = l0[min_res_ind2]
    Aest = wl_wids[min_res_ind1]/(np.exp((ypix-1)/SR)-1)
    wlg1 = Aest*(np.exp(np.arange(ypix)/SR)-1)+l0[min_res_ind2]
    sf.plt_deltas(wlref,itref,'b',2)
#    sf.plt_deltas(j_wls,np.ones((len(j_wls))),'r')
    plt.plot(wlg1,line_gauss[i,:]/np.max(line_gauss),'k')
    plt.show()

#    emp_wl = A[min_res_ind1]*(np.exp(empirical_px/SR)-1)+l0[min_res_ind2]
#    sf.plt_deltas(emp_wl,np.ones((len(emp_wl))))
#    sf.plt_deltas(j_wls,np.ones((len(j_wls))),'r')
#px = np.arange(1965,1980)
#val = ccd[:,1000][px]
#params = sf.gauss_fit(px,val,fit_background='n')
#pxn = np.linspace(px[0],px[-1],1000)
#fit = sf.gaussian(pxn,abs(params[0]),params[1],params[2],params[3],params[4])
#plt.plot(px,val,pxn,fit)

#for i in range(num_fibers):
#    plt.figure(i)
#    plt.plot(line_amps[i,:])
#    plt.plot(line_gauss[i,:])
#    plt.show()
'''