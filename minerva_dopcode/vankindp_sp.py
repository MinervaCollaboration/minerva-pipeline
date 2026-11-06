import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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


exten = '2'

def vank(objname, weightthresh=1.0, chithresh=1.0, sigmaithresh=1.0, nobslim=0.5, cbinning=4, skewpar=False, plot=True):  # MW: original setting was 10%, 1% and 0% for weight, chi, and sigma thresholds 
    # MW: It seems that we could just put all of the thresholds at 10%. Nothing really changes from 1% to 10% for any single threshold.  However, the threshold that changes the RMS  the most amongst the 3 is the chithresh.  If the weight and sigmai thresholds are at 1.0% (or 10%) and the chithresh is at 10% that lowers the RMS of T2 by (only) 0.2 m/s when compared to a chithresh at 1%. That's pretty insignificant at the 2.5-3.0 m/s level but might as well take it.  The chithresh is the threshold that is most directly related to bad line-fits due to tellurics, in comparison to the other 2 thresholds.
    # MW: Changing nobslim from 20% to 50% seems to do nothing to the overall RMS, so might as well just keep it at 50%. This is shown in the investigate_chithresh_* plots.
    # MW: None of the above takes the Allan variance plots into consideration though.  After looking at the allan variance plot when making the chithresh 1% to 20% it seems to mimic the white noise line better as the percentage increases.  Based on this, we have now confirmed that the chithresh of ~10% is fine for HD122064.  Checking the allan variance plots (from 1% to 20%) and the chronologically binned RMS's was an exercise to just make sure that we were truly choosing a threshold that would not stray the allan variance binning curve away from the white noise curve. In theory we could promote the RMS from the c'binned RVs but it's fine to keep the nightly binned RMS instead, especially since that RV curve is binned in a more intuitive fashion: by night.
    # MW: Seems that the observation fraction limit "nobslim" is a good fail-safe for weird data sets regardless of rejection rates. The new iodine scan seems to favor no rejection rates according to the always-increasing RMS values produced for an ever increasing "chithresh" rejection rate. For the new iodine scan, use 1% rejection rates for now.  For other scans, use a 16% chithresh.
    # MW: After running every combination amongst IPs and iodine scans it's confirmed that the RMS increases immediately from 0 to 99% except for when the fixed IP is used with the old iodine scan.  Therefore, until this problem is solved, the default chithresh should be 1%.


    c = 299792458.0

    if socket.gethostname() == 'Main':
      	#filenames = glob.glob('/Data/kiwispec-proc/n20160[5,6]*/*' + objname + '*.chrec' + exten + '.npy') # MW
	#filenames = glob.glob('/Data/kiwispec-proc/*/*' + objname + '*.chrec6.npy')
	#filenames = glob.glob('/Data/kiwispec-proc/n20170409/n20170409.' + objname + '.*.chrec6.npy')

       	#filenames = '/Data/kiwispec-proc/n20170409/n20170409.'+ objname + '.0022.proc.chrec6.npy'

        #filenames = glob.glob('/Data/kiwispec-proc/n20171[01]*/*' + objname + '*.fixed_mw.npy') # MW: make sure to change the nightly-binning plot section (line ~300)
        #filenames = glob.glob('/Data/kiwispec-proc/n20171[01]*/*' + objname + '*.skew_mw.npy') # MW: make sure to change the nightly-binning plot section (line ~300)
        #filenames = glob.glob('/Data/kiwispec-proc/n20171[01]*/*' + objname + '*.gauss_mw.npy') # MW: make sure to change the nightly-binning plot section (line ~300)

        #filenames = glob.glob('/Data/kiwispec-proc/n20160[5,6]*/*' + objname + '*SP*.gauss_mw.npy') # MC
        filenames = glob.glob('/Data/kiwispec-proc/n20160[5,6]*/*' + objname + '*SP*.fixed_sp.npy') # MC

	#'''
        #filenames = glob.glob('/Data/kiwispec-proc/n201605*/*' + objname + '*.fixed_mw.npy') # MW
        #filenames = np.concatenate(( filenames,  glob.glob('/Data/kiwispec-proc/n201606[01]*/*' + objname + '*.fixed_mw.npy') )) # MW
        #filenames = np.concatenate(( filenames,  glob.glob('/Data/kiwispec-proc/n2016062[01]*/*' + objname + '*.fixed_mw.npy') )) # MW: Daytime sky spectra with high S/N are very scarce in the June 2016.  The only spectra used to make the June IP was on June 1st. The RV RMS for the RV standard star sky rockets from 2.8 to 88.5 m/s when including data from june 22nd and beyond. 
       	#'''
        '''
        filenames = glob.glob('/Data/kiwispec-proc/n201605*/*' + objname + '*.skew_mw.npy') # MW
        filenames = np.concatenate(( filenames,  glob.glob('/Data/kiwispec-proc/n201606[01]*/*' + objname + '*.skew_mw.npy') )) # MW
        filenames = np.concatenate(( filenames,  glob.glob('/Data/kiwispec-proc/n2016062[01]*/*' + objname + '*.skew_mw.npy') )) # MW
        '''
       	'''
        filenames = glob.glob('/Data/kiwispec-proc/n201605*/*' + objname + '*.gauss_mw.npy') # MW
        filenames = np.concatenate(( filenames,  glob.glob('/Data/kiwispec-proc/n201606[01]*/*' + objname + '*.gauss_mw.npy') )) # MW
        filenames = np.concatenate(( filenames,  glob.glob('/Data/kiwispec-proc/n2016062[01]*/*' + objname + '*.gauss_mw.npy') )) # MW
        '''
        #filenames = glob.glob('/Data/kiwispec-proc/n201*/*' + objname + '*.gauss_mw.npy') # MW

        #
        # MW: for some strange reason, the glob command does not sort the files like the "ls" terminal command.
        filenames = np.sort( filenames ) # MW
	print filenames


    else:
        filenames = glob.glob('/n/home12/jeastman/minerva/data/n2016051[4-9]/*' + objname + '*.chrec' + exten + '.npy') +\
            glob.glob('/n/home12/jeastman/minerva/data/n201605[2-3]?/*' + objname + '*.chrec' + exten + '.npy') +\
            glob.glob('/n/home12/jeastman/minerva/data/n2016060?/*' + objname + '*.chrec' + exten + '.npy') +\
            glob.glob('/n/home12/jeastman/minerva/data/n2016061[0-2]/*' + objname + '*.chrec' + exten + '.npy')

    #filenames = '/Data/kiwispec-proc/n20170409/n20170409.HD97601.0017.proc.chrec6.npy'
    ntel = 4
    nobs = len(filenames)*ntel
#    if nobs <= 3: return

    vji = []
    wji = []
    chiji = []
    ctsji = []
    alphaji = []
    ipsigmaji = []
    slopeji = []
    jdutcs = np.array([])
    telescope = np.array([])
    nspectra = 0

    chunkweights_return = { 'T1': [], 'T2': [], 'T3': [], 'T4': [] } # MW
    for filename in filenames:

        st = os.stat(filename)
#        if (st.st_size != 121120): continue
        if (st.st_size == 0.0): 
            nobs -= 4
            continue

        nspectra += 1
 
        chrec = np.load(filename)
     #   fitsname = '/Data/kiwispec-proc/n20170409/n20170409.HD97601.0017.proc.fits'
        fitsname = os.path.splitext(os.path.splitext(filename)[0])[0] + '.fits'
        h = pyfits.open(fitsname,mode='update')
        
#        # reject chunks with bad DSST (Step 1)
#        bad = np.where(chrec['wt' + str(i)] == 0)
#        chrec['z' + str(i)][bad] = np.nan
	
        # reject chunks where fitter failed to converge (rchi2 == 0 or 100)
        # (Step 2)
        for i in range(1,ntel+1):

            ''' On Nov. 18th 2018, T3 was offline due to the cable wrapping so there is no header keyword for T3 on that date to Dec. 20th, which is when T3 was fixed and re-enabled.  T1 is missing some calibrations for a different reason.
            if h[0].header['FLUXMID' + str(i)] == 'UNKNOWN':
                t = Time(h[0].header['DATE-OBS'], format='isot',scale='utc')
                midflux = t.jd+h[0].header['EXPTIME']/2.0/86400.0
                h[0].header['FLUXMID' + str(i)] = midflux
            else:
                midflux = h[0].header['FLUXMID' + str(i)]
            '''
            try:
		testing = h[0].header['FLUXMID' + str(i)]
            	fluxmid_exists = True
            except KeyError:
            	fluxmid_exists = False
	
		
            if fluxmid_exists is False or h[0].header['FLUXMID' + str(i)] == 'UNKNOWN':
                t = Time(h[0].header['DATE-OBS'], format='isot',scale='utc')
                midflux = t.jd+h[0].header['EXPTIME']/2.0/86400.0
                h[0].header['FLUXMID' + str(i)] = midflux
            else:
                midflux = h[0].header['FLUXMID' + str(i)]
            

            # very noisy data can find great fits
            bad = np.where(chrec['chi' + str(i)] <= 0.3)
            chrec['z' + str(i)][bad] = np.nan    # MW: uncommented
            #print("nbad low chi",i,len(bad[0])) # MW: commented

            # bad fits will have bad chi^2
            bad = np.where(chrec['chi' + str(i)] >= 5) 
            chrec['z' + str(i)][bad] = np.nan     # MW: uncommented
            #print("nbad high chi",i,len(bad[0])) # MW: commented

            if skewpar:
		    # very noisy data can find great fits
		    bad = np.where((chrec['alpha' + str(i)] <= -0.95) | (chrec['alpha' + str(i)] >= 0.95)) # MW: uncommented.
		    chrec['z' + str(i)][bad] = np.nan  # MW: uncommented.
		    #print("nbad extreme alpha",i,len(bad[0]))

		    # very noisy data can find great fits
		    bad = np.where((chrec['sigma' + str(i)] <= 0.25) | (chrec['sigma' + str(i)] >= 1.25)) # MW: uncommented. When using my fixed IP, I set the initial guess (in the four_chunk code) to 0.75 so that this wouldn't throw away literally every chunk and observation. When using a varying IP, I set guess back to its original setting of 0.2.
		    chrec['z' + str(i)][bad] = np.nan
		    #print("nbad extreme sigma",i,len(bad[0]))

            #print("nbad z=nan",i,len(np.where(np.isnan(chrec['z' + str(i)]))[0])) # MW: commented

            # reject chunks with the lowest DSST weight (Step 3) or bad weights (Step 1)
            lowweight = np.nanpercentile(chrec['wt' + str(i)],weightthresh)
            bad = np.where((chrec['wt' + str(i)] <= lowweight) | (chrec['wt' + str(i)] == 0))
            chrec['z' + str(i)][bad] = np.nan
            wtbad = bad # MW

            if 'BARYCOR' + str(i) not in h[0].header.keys():
                # do the barycentric correction
                ra = h[0].header['TARGRA' + str(i)]
                dec = h[0].header['TARGDEC' + str(i)]
                try: pmra = h[0].header['PMRA' + str(i)]
                except: pmra = 0.0
                try: pmdec = h[0].header['PMDEC' + str(i)]
                except: pmdec = 0.0
                try: parallax = h[0].header['PARLAX' + str(i)]
                except: parallax = 0.0
                try: rv = h[0].header['RV' + str(i)]
                except: rv = 0.0
            
                result = utils.barycorr(midflux, ra, dec, pmra=pmra, pmdec=pmdec, parallax=parallax, rv=rv, zmeas=0.0)
                zb = result/c
                h[0].header['BARYCOR' + str(i)] = zb
            elif h[0].header['BARYCOR' + str(i)] == 'UNKNOWN':
                # do the barycentric correction
                ra = h[0].header['TARGRA' + str(i)]
                dec = h[0].header['TARGDEC' + str(i)]
                try: pmra = h[0].header['PMRA' + str(i)]
                except: pmra = 0.0
                try: pmdec = h[0].header['PMDEC' + str(i)]
                except: pmdec = 0.0
                try: parallax = h[0].header['PARLAX' + str(i)]
                except: parallax = 0.0
                try: rv = h[0].header['RV' + str(i)]
                except: rv = 0.0
            
                result = utils.barycorr(midflux, ra, dec, pmra=pmra, pmdec=pmdec, parallax=parallax, rv=rv, zmeas=0.0)
                zb = result/c
                h[0].header['BARYCOR' + str(i)] = zb
            else:
                zb = h[0].header['BARYCOR' + str(i)]

#            active = np.append(active ,h[0].header['FAUSTAT' + str(i)] == 'GUIDING')
            if h[0].header['FAUSTAT' + str(i)] == 'GUIDING' or 'daytimeSky' in h[0].header['OBJECT' + str(i)]:

                rvs = c*((1.0+chrec['z' + str(i)])*(1.0+zb)-1.0)

                # if all chunks are bad, skip it
                if len(np.where(np.isnan(rvs))[0]) == len(rvs): 
                    nobs-=1
                    continue

                wtsarray = chrec['wt' + str(i)] # MW
                wtsarray[wtbad] = -np.inf  # MW
                chunkweights_return['T'+str(i)].append( wtsarray ) # MW

                jdutcs = np.append(jdutcs,midflux)
                telescope = np.append(telescope,i)

                if len(vji) == 0: vji = rvs
                else: vji = np.vstack((vji,rvs))

                if len(wji) == 0: wji = chrec['wt' + str(i)]
                else: wji = np.vstack((wji,chrec['wt' + str(i)]))
            
                if len(chiji) == 0: chiji = chrec['chi' + str(i)]
                else: chiji = np.vstack((chiji,chrec['chi' + str(i)]))

                if len(ctsji) == 0: ctsji = chrec['cts' + str(i)]
                else: ctsji = np.vstack((ctsji,chrec['cts' + str(i)]))

#                if len(alphaji) == 0: alphaji = chrec['alpha' + str(i)]
#                else: alphaji = np.vstack((alphaji,chrec['alpha' + str(i)]))

#                if len(ipsigmaji) == 0: ipsigmaji = chrec['sigma' + str(i)]
#                else: ipsigmaji = np.vstack((ipsigmaji,chrec['sigma' + str(i)]))

                if len(slopeji) == 0: slopeji = chrec['slope' + str(i)]
                else: slopeji = np.vstack((slopeji,chrec['slope' + str(i)]))
            #else: nobs-=1 # MW: commented
            else: # MW
                nobs -= 1

        h.flush()
        h.close()

    if nspectra == 0:
        print "No good spectra for " + objname
        sys.exit()

    nchunks = len(chrec)
    vij = np.transpose(vji)

    #ipdb.set_trace()

    #wij = np.transpose(wji)
    chiij = np.transpose(chiji)
    ctsij = np.transpose(ctsji)
#    alphaij = np.transpose(alphaji)
#    ipsigmaij = np.transpose(ipsigmaji)
    slopeij = np.transpose(slopeji)
    
    snr = np.sqrt(np.nansum(ctsij,axis=0))

    if chithresh > 0: # MW
	    # reject chunks with the worst fits (step 4)
	    chimed = np.nanmedian(chiij,axis=1)
	    #print chimed ## MW: commented
	    hichi = np.nanpercentile(chimed,100.0-chithresh)
	    bad = np.where(chimed >= hichi)  
	    vij[bad,:] = np.nan  

    # Starting to vank telescopes independently
    vj = np.zeros(nobs)
    sigmavj = np.zeros(nobs)

    for i in np.arange(0, nobs):
        pass#print(len(np.where(np.isnan(vij[:,i]))[0])) # MW: commented and passed

    #ipdb.set_trace()
    vij_return = {} # MW
    sigmaij_return = {} # MW
    # loop through telescopes
    for i in range(1, ntel + 1):
        match = np.where(telescope == i)
        #thisvj, thissigmavj = vanktel(vij, i, telescope, sigmaithresh=sigmaithresh) # MW: commented
        thisvj, thissigmavj, vij_return['T'+str(i)], sigmaij_return['T'+str(i)] = vanktel(vij, i, telescope, sigmaithresh=sigmaithresh, nobslim=nobslim) # MW
        vj[match] = thisvj
        sigmavj[match] = thissigmavj

    # back to original vankA.py code

    print objname + " RMS: " + str(np.nanstd(vj))
    
    # plot scatter vs time
    colors = ['','r','g','b','orange']
    period = 7.126816
    #period = 5189
    for i in range(1, ntel + 1):
        match = np.where(telescope == i)
        #plt.plot((np.mod(jdutcs[match]-2457389, period))/period,vj[match],'o',color=colors[i])
        plt.plot(jdutcs[match]-2457389,vj[match],'o',color=colors[i])

    plt.title(objname)
    plt.xlabel('Days since UT 2016-01-01')
    #plt.xlabel('Phase') # MW: commented
    plt.ylabel('RV (m/s)')
    #plt.ylim(-200,200)
    if plot: plt.savefig(objname + '.' + exten + '.png') # MW
    plt.close()
    
    nightlyrvs = {'all':np.array([]),
                  'T1':np.array([]),
                  'T2':np.array([]),
                  'T3':np.array([]),
                  'T4':np.array([])}

    # bin per telescope per night
    for jd in range(2457480,2457571): # MW: range(2457480, 2457755) this goes from April 1st 2016 to the end of 2016. #range(2457754, 2458120) this is all of 2017. #range(2457480,2457571) this goes to July 1st 2016   # This range determines whether or not you correctly calculate the RMS (or if you can calculate an RMS at all). #range(2457997,2458089) is from Sept 1st 2017 to Dec 1st 2017   #range(2457389, 2458484) is from Jan 1st 2016 to Jan 1st 2019
        for i in range(1,ntel+1):
            #ipdb.set_trace()
            match = np.where((jdutcs >= jd) & (jdutcs < (jd+1)) & np.isfinite(vj) & (telescope==i))
            night = np.mean(jdutcs[match])-2457389
            nightlyrv = np.sum(vj[match]/sigmavj[match]**2)/np.sum(1.0/sigmavj[match]**2)
            plt.plot([night],[nightlyrv],'o',color=colors[i])
            nightlyrvs['T' + str(i)] = np.append(nightlyrvs['T' + str(i)],nightlyrv)
            
        match = np.where((jdutcs >= jd) & (jdutcs < (jd+0)) & np.isfinite(vj))
        night = np.mean(jdutcs[match])-2457389
        nightlyrv = np.sum(vj[match]/sigmavj[match]**2)/np.sum(1.0/sigmavj[match]**2)
        plt.plot([night],[nightlyrv],'ko')
        nightlyrvs['all'] = np.append(nightlyrvs['all'],nightlyrv)

    ### MC, crude sigma clip for T2 nightly RV...    
    vtmp = 1.0*nightlyrvs['T2']
    print vtmp
    vsrt = np.sort(vtmp)
    vmid = vsrt[int(0.15*len(vsrt)):int(0.85*len(vsrt))] ##Just look at the middle 70%
    vmed = np.nanmedian(vmid)
    midstd = np.nanstd(vmid)
    vmsk = (vtmp < vmed-12*midstd) + (vtmp > vmed + 12*midstd)#*(np.isnan(vtmp))
    vtmp[vmsk] = np.nan
    nightlyrvs['T2'] = vtmp
    print vmed, midstd, np.sum(vmsk) 
    plt.title(objname)
    plt.xlabel('Days since UT 2016-01-01')
    plt.ylabel('Nightly binned RV (m/s)')
    if plot: plt.savefig(objname + '.' + exten + '.binned.png') # MW
    plt.close()
    print objname + ' nightly binned RMS:'
    print "All: " + str(np.nanstd(nightlyrvs['all']))
    print "T1: " + str(np.nanstd(nightlyrvs['T1']))
    print "T2: " + str(np.nanstd(nightlyrvs['T2']))
    print "T3: " + str(np.nanstd(nightlyrvs['T3']))
    print "T4: " + str(np.nanstd(nightlyrvs['T4']))
    rms_return = { "All": np.nanstd(nightlyrvs['all']), "T1": np.nanstd(nightlyrvs['T1']), "T2": np.nanstd(nightlyrvs['T2']),  "T3": np.nanstd(nightlyrvs['T3']),  "T4": np.nanstd(nightlyrvs['T4']) }  # MW


    # MW: plot with a specified binning number (instead of binning by night)
    jdsrt = np.argsort( jdutcs ) # Sort by date before you bin anything

    tel_matches = [ np.where( np.isfinite(vj[jdsrt]) & (telescope[jdsrt]==i) )[0] for i in range(1, ntel+1) ]
    tel_b_tots = [ len(tel_matches[i_i]) for i_i in range(0, 4) ]

    rvs_binned = { "All": np.array([]), "T1": np.array([]), "T2": np.array([]), "T3": np.array([]), "T4": np.array([]) }
    #plt.figure()
    for b_i in range(0, len(vj[jdsrt][ np.isfinite(vj[jdsrt]) ]), cbinning ):

        for i_i in range(0, 4):
            
            if b_i >= tel_b_tots[i_i]:           vj_b, sigmavj_b, thesejd = np.array([]), np.array([]), -np.inf
            elif b_i+cbinning > tel_b_tots[i_i]: vj_b, sigmavj_b, thesejd = vj[jdsrt][ tel_matches[i_i] ][b_i: ], sigmavj[jdsrt][ tel_matches[i_i] ][b_i: ], jdutcs[jdsrt][ tel_matches[i_i] ][b_i: ]
            else:                              vj_b, sigmavj_b, thesejd = vj[jdsrt][ tel_matches[i_i] ][b_i: b_i+cbinning], sigmavj[jdsrt][ tel_matches[i_i] ][b_i: b_i+cbinning], jdutcs[jdsrt][ tel_matches[i_i] ][b_i: b_i+cbinning]

            if len(vj_b) > 0:
                bdatum = np.sum(vj_b/sigmavj_b**2)/np.sum(1.0/sigmavj_b**2)
                thisjd = np.mean(thesejd)-2457389
                rvs_binned['T'+str(i_i+1)] = np.append( rvs_binned['T'+str(i_i+1)], bdatum )
                plt.plot( [thisjd], [bdatum], 'o', color=colors[i_i+1])
                
        # Now do bin for the "All" telescopes line 
        if b_i+cbinning > len(vj[jdsrt][ np.isfinite(vj[jdsrt]) ]): vj_b, sigmavj_b, thesejd = vj[jdsrt][b_i: ], sigmavj[jdsrt][b_i: ], jdutcs[jdsrt][b_i: ]   
        else:                                                       vj_b, sigmavj_b, thesejd = vj[jdsrt][b_i: b_i+cbinning], sigmavj[jdsrt][b_i: b_i+cbinning], jdutcs[jdsrt][b_i: b_i+cbinning]
        
	bdatum = np.sum(vj_b/sigmavj_b**2)/np.sum(1.0/sigmavj_b**2)
	thisjd = np.mean(thesejd)-2457389
        rvs_binned['All'] = np.append( rvs_binned['All'], bdatum )
	plt.plot( [thisjd], [bdatum], 'ko')
    
    ### MC, crude sigma clip for T2 nightly RV...    
    vtmp = 1.0*rvs_binned['T2']
    print vtmp
    vsrt = np.sort(vtmp)
    vmid = vsrt[int(0.15*len(vsrt)):int(0.85*len(vsrt))] ##Just look at the middle 70%
    vmed = np.nanmedian(vmid)
    midstd = np.nanstd(vmid)
    vmsk = (vtmp < vmed-12*midstd) + (vtmp > vmed + 12*midstd)#*(np.isnan(vtmp))
    vtmp[vmsk] = np.nan
    rvs_binned['T2'] = vtmp
    print vmed, midstd, np.sum(vmsk)


    plt.title(objname)
    plt.xlabel('Days since UT 2016-01-01')
    plt.ylabel( 'Chronologically binned (by {}) RV (m/s)'.format(cbinning) )
    if plot: plt.savefig( '{}.{}.cbinned.png'.format(objname, exten) )
    plt.close()
    print "--------------"
    print objname + ' chronologically binned (by {}) RMS:'.format(cbinning)
    print "All: " + str(np.nanstd(rvs_binned['All']))
    print "T1: " + str(np.nanstd(rvs_binned['T1']))
    print "T2: " + str(np.nanstd(rvs_binned['T2']))
    print "T3: " + str(np.nanstd(rvs_binned['T3']))
    print "T4: " + str(np.nanstd(rvs_binned['T4']))
    print "--------------"
       
 
    # create a histogram of scatter within a night
    mindate = int(math.floor(min(jdutcs)))
    maxdate = int(math.ceil(max(jdutcs)))

    jdbin = []
    rvbin = []
    errbin = []
    sigmabin = []
    for i in range(mindate,maxdate):
        match = np.where((jdutcs >= i) & (jdutcs < (i+1)) & (np.isfinite(vj)))
        if len(match[0]) > 1:
            jdbin.append(mindate)
            rvbin.append(np.mean(vj[match]))
            errbin.append(np.std(vj[match]))
            sigmabin.append(np.mean(sigmavj[match]))

#    print jdbin, rvbin, errbin, sigmabin
    hist, bins = np.histogram(errbin, bins=20)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    
    plt.bar(center, hist, align='center', width=width)
    plt.xlabel('Intranight RMS (m/s)')
    plt.ylabel('# of Nights')
    if plot: plt.savefig(objname + '.' + exten + '.hist.png') # MW
    plt.close()

    
    """ Amber Changed this    # plot median sigma as a function of chunk number
    sigmawav = []
    sigmaaall = []
    sigmatel = []
    for tel in range(1,ntel+1):
        match = np.where(telescope == tel)
        for chunk in range(np.shape(ipsigmaij)[0]):
            plt.plot(chunk,np.nanmedian(ipsigmaij[chunk,match]),'o',color=colors[tel])
            sigmawav.append()
            sigmaall.append(np.nanmedian(ipsigmaij[chunk,match]))
            sigmatel.append(tel)
            
    plt.title(objname)
    plt.xlabel('Chunk number')
    plt.ylabel('sigma')
    plt.savefig(objname + '.' + exten + '.sigmavchunk.png')
    plt.close()
    """

#    # plot median alpha as a function of chunk number
#    for tel in range(1,ntel+1):
#        match = np.where(telescope == tel)
#        for chunk in range(np.shape(alphaij)[0]):
#                         plt.plot(chunk,np.nanmedian(alphaij[chunk,match]),'o',color=colors[tel])
#    plt.title(objname)
#    plt.xlabel('Chunk number')
#    plt.ylabel('alpha')
#    plt.savefig(objname + '.' + exten + '.alphavchunk.png')
#    plt.close()

    # plot median slope as a function of chunk number
    for tel in range(1,ntel+1):
        match = np.where(telescope == tel)
        for chunk in range(np.shape(slopeij)[0]):
            plt.plot(chunk,np.nanmedian(slopeij[chunk,match]),'o',color=colors[tel])
    plt.title(objname)
    plt.xlabel('Chunk number')
    plt.ylabel('slope')
    if plot: plt.savefig(objname + '.' + exten + '.slopevchunk.png') # MW
    plt.close()


#    # plot sigma for chunk 150 over time
#    chunk = 150
#    for t in jdutcs:
#        for tel in range(1,ntel+1):
#            match = np.where((telescope == tel) & (jdutcs == t))
#            plt.plot(t-2457389,np.nanmedian(ipsigmaij[chunk,match]),'o',color=colors[tel])
#    plt.title(objname)
#    plt.xlabel('Days since UT 2016-01-01')
#    plt.ylabel('sigma')
#    plt.savefig(objname + '.' + exten + '.sigma150vtime.png')
#    plt.close()



    # plot scatter within a night vs SNR
    plt.plot(sigmabin, errbin,'bo')
    plt.title(objname)
    plt.xlabel('Sigma_v_j')
    plt.ylabel('Intranight RMS (m/s)')
    if plot: plt.savefig(objname + '.' + exten + '.SNR.png') # MW
    plt.close()

    # subtract a linear trend from vj
    good = np.where(np.isfinite(vj))
    t0 = np.nanmedian(jdutcs[good])
    coeffs = np.polyfit(jdutcs[good]-t0,vj[good],1)
    vjtrend = coeffs[1] + (jdutcs[good]-t0)*coeffs[0] 

    if plot: # MW
	    # output text files of JD, rv, rverr
	    f = open(objname + '.' + exten + '.dat','w')
	    for i in range(len(vj)):
		if np.isfinite(vj[i]):
		    f.write('{0:f} {1:f} {2:f} T{3:d} {4:s}'.format(jdutcs[i],vj[i],sigmavj[i],int(telescope[i]),'\n'))
	    f.close()

    # plot scatter minus trend vs time
    plt.plot(jdutcs[good]-2457389,vj[good]-vjtrend,'bo')
    plt.title(objname)
    plt.xlabel('Days since UT 2016-01-01')
    plt.ylabel('RV (m/s)')
    if plot: plt.savefig(objname + '.' + exten + '.detrended.png') # MW
    plt.close()
    print objname + " Detrended RMS: " + str(np.nanstd(vj[good]-vjtrend))

    '''for i in range(1,ntel+1):
        match = np.where((telescope == i) & np.isfinite(vj))
#        rate = 1.0/(jdutcs[match]-np.roll(jdutcs[match],1))
#        taus = np.logspace(0, 3, 50)
        taus = np.arange(1,len(vj[match])/3)
        (t2, ad, ade, adn) = allantools.oadev(vj[match], rate=1, data_type="freq", taus=taus)
        plt.loglog(t2,ad,'o',color=colors[i])

        # overplot the line
        y = np.asarray([pow(tt,-0.5) for tt in taus])*ad[0]
        plt.loglog(taus,y,'-',color=colors[i])
        

    match = np.where(np.isfinite(vj))
    taus = np.arange(1,len(vj[match])/3)
    (t2, ad, ade, adn) = allantools.oadev(vj[match], rate=1, data_type="freq", taus=taus)
    plt.loglog(t2,ad,'ko')
    # overplot the line
    y = np.asarray([pow(tt,-0.5) for tt in taus])*ad[0]
    plt.loglog(taus,y,'k-')

    plt.xlabel('N bin')
    plt.ylabel('Precision (m/s)')
        

    plt.savefig(objname + '.' + exten + '.allan.png')
    plt.close()
    '''
    # MW: the following stuff is for allan variance plot. I copied this from vank2_mw and pasted it here.
    ## MW: save the allan-variance data 
    av_return = {}
    for i in range(1,ntel+1):
    #for i in [2]:#range(2,2):
        match = np.where((telescope == i) & np.isfinite(vj))
#        rate = 1.0/(jdutcs[match]-np.roll(jdutcs[match],1))
#        taus = np.logspace(0, 3, 50)
        
        if len(vj[match]) > 0: # MW
		taus = np.arange(1,len(vj[match])/3)
		(t2, ad, ade, adn) = allantools.oadev(vj[match], rate=1, data_type="freq", taus=taus)
		plt.loglog(t2,ad,'o',color=colors[i])

		# overplot the line
		y = np.asarray([pow(tt,-0.5) for tt in taus])*ad[0] ## MW: this crashes if a telescope has no observations, i.e., vj[match] is empty
		plt.loglog(taus,y,'-',color=colors[i])

                av_return['T'+str(i)] = {'ad':ad, 't2':t2, 'y':y, 'taus':taus}
        
        else: # MW
                print('T{} has no observations. I am skipping this allan variance data.'.format(i)) 
    
    '''
    match = np.where(np.isfinite(vj))
    taus = np.arange(1,len(vj[match])/3)
    (t2, ad, ade, adn) = allantools.oadev(vj[match], rate=1, data_type="freq", taus=taus)
    plt.loglog(t2,ad,'ko')
    # overplot the line
    y = np.asarray([pow(tt,-0.5) for tt in taus])*ad[0]
    plt.loglog(taus,y,'k-')
    '''

    plt.xlabel('N bin')
    plt.ylabel('Precision (m/s)')


    if plot: plt.savefig(objname + '.' + exten + '.allan.png')  
    plt.close()

   

#    ipdb.set_trace()

#    rate = 1.0/float(data_interval_in_s)
#    taus = [1,2,3,4,8,16]
#    allantools.adev(frequency=fract_freqdata,rate=rate,taus=taus)


    
    #vji_return = { 'T1': vji[ np.where( telescope==1 )[0] ], 'T2': vji[ np.where( telescope==2 )[0] ], 'T3': vji[ np.where( telescope==3 )[0] ], 'T4': vji[ np.where( telescope==4 )[0] ] }  # MW
    #vij_return = { 'T1': vij[: , np.where( telescope==1 )[0] ], 'T2': vij[: , np.where( telescope==2 )[0] ], 'T3': vij[: , np.where( telescope==3 )[0] ], 'T4': vij[: , np.where( telescope==4 )[0] ] }  # MW: the vij has every chunk with the same RV offset

    #sigmaij_return = { 'T1': sigmaij_r[: , np.where( telescope==1 )[0] ], 'T2': sigmaij_r[: , np.where( telescope==2 )[0] ], 'T3': sigmaij_r[: , np.where( telescope==3 )[0] ], 'T4': sigmaij_r[: , np.where( telescope==4 )[0] ] }  # MW
    for tID in range(1,5): chunkweights_return['T'+str(tID)] = np.array( chunkweights_return['T'+str(tID)] ).T   
 
    stuff_return = { 'rms': rms_return, 'chunkrvs_vij': vij_return , 'chunkerrs_sigmaij': sigmaij_return, 'chunkweights_c_f': chunkweights_return, 'av':av_return } #, 'badwt_f_t': badwt_f_tD_return } # MW
    return stuff_return # MW




def vanktel(vij, tel, telescope, sigmaithresh=None, nobslim=None):
    """
    Sharon X. Wang, vanking vij matrix from a single telescope

    :param vij: RV matrix from all telescopes, i-chunk, j-observation
    :param tel: index of telescope, [1,2,3,4] for the four MINERVA telescopes
    :param telescope: array of telescope index, telling us which telescope each observation comes from
    :param sigmaithresh: sigma/scatter threshold, as in vank()
    :return: vj, sigmavj: RV array and array of internal errors
    """
    nchunks = len(vij[:, 0])  # total number of chunks
    indtel = np.where(telescope == tel)  # indices for slicing out this telescope
    #nobs = len(indtel)  # total number of observations from this telescope # MW: commented. I think this is a typo
    nobs = len(indtel[0])  # total number of observations from this telescope # MW
    vijtel = vij[:, indtel[0]]  # RV matrix for this telescope

    # MW: vet the chunks. chaos ensues when a chunk is left with only 2 observations of RVs
    totNi = np.array([ len( np.where( np.isfinite(vijtel[i_i]) == True )[0] ) for i_i in range(nchunks) ])
    olim = nobslim * np.median( totNi[totNi > 0] )  # I don't care about the chunks that have zero RVs. Those aren't the ones contributing to the chaos.  Because I now exclude the zeroes, taking the median will work.  I could just find the total number of observations (because each chunk must of the same number) and then multiply that by the "limit" fraction but I think that's a bad idea.  After a certain weight threshold is chosen, for every threshold set greater than that one there may be a few observations of each chunk that are always rejected. In other words, certain observations--no matter the chunk--have terribly low S/N and thus will just inflate these matrices with empty useless space.  Therefefore, finding the average number of observations per chunk is a good proxy for the maximum amount of non-rejected observations.
    if olim < 5.0: olim = 5.0  # There may be a case when a certain telescope only has 8 total observations for each chunk.  In that case, we definitely want to have 5 RVs at minimum no matter what.
    for i_i in range(nchunks):
        if totNi[i_i] < olim: vijtel[i_i] = np.nan  # now, instead of only have a couple RVs, the chunk has no RVs

    # adjusting zero point of all chunks
    # bad telescope plagues this procedure, so better have this separately!
    #ipdb.set_trace()
    vijtel -= np.transpose(np.tile(np.nanmean(vijtel, axis=1), (nobs, 1)))

    if len(np.where(np.isnan(vijtel))[0]) == len(vijtel):
        vj = np.zeros(nobs)
        vj[:] = np.nan
        sigmavj = np.zeros(nobs)
        sigmavj[:] = np.nan
        return vj, sigmavj

    # Below: direct copy from vankA.py # MW: not anymore

    # compute chunk weights (step 6):
    # median velocities for each observation
    vjmed = np.nanmedian(vijtel, axis=0)  

    # compute the matrix of velocity differences
    Deltaij = vijtel - np.tile(vjmed, (nchunks, 1))  
    sigmai = np.nanstd(Deltaij, axis=1) 
    #totNi = np.array([ len(np.where( np.isfinite(Deltaij[i_i]) == True )[0]) for i_i in range(nchunks) ]) # MW
    #sigmai = np.nanstd(Deltaij, axis=1) * totNi / (totNi-1) / np.sqrt( totNi ) # MW: testing.  The weights are defined by drastically different amounts of data thus I think these error bars need to be weighted by that.  This should be sqrt( N ) or something like that. Instead of using the standard deviation, I'll try the square root of the sample variance, and then find the standard deviation of the mean by diving by root_N

    #ipdb.set_trace()

    ''' MW: block-commented
    # reject the highest sigmai (step 7)
    bad = np.where((sigmai == 0.0) | np.isnan(sigmai))
    sigmai[bad] = np.inf
    hisigma = np.nanpercentile(sigmai, 100.0 - sigmaithresh) # MW: because the already-rejected chunks/obs have been changed from nan to inf, now this percentile calculations is including (essentially) the nan's.
    if np.isnan(hisigma):
        hisigma = np.inf
    bad = np.where(sigmai >= hisigma)
    sigmai[bad] = np.inf
    '''
    # reject the highest sigmai (step 7)
    bad = np.where( sigmai == 0.0 )
    sigmai[bad] = np.inf
    hisigma = np.nanpercentile(sigmai, 100.0 - sigmaithresh) # MW: because the already-rejected chunks/obs have been changed from nan to inf, now this percentile calculations is including (essentially) the nan's.
    if np.isnan(hisigma):
        hisigma = np.inf
    bad = np.where( np.isnan(sigmai) )
    sigmai[bad] = np.inf
    bad = np.where(sigmai >= hisigma)
    sigmai[bad] = np.inf


    # compute rj (eq 2.7)
    rj = np.nanmedian(np.abs(Deltaij) / np.transpose(np.tile(sigmai, (nobs, 1))), axis=0)  # eq 2.7
    sigmaij = np.transpose(np.tile(sigmai, (nobs, 1))) * np.tile(rj, (nchunks, 1))  # eq 2.5

    # prevent nans
    bad = np.where(sigmaij == 0.0)
    sigmaij[bad] = np.inf

    wij = 1.0 / sigmaij ** 2

    #ipdb.set_trace()

    vj = np.nansum(vijtel * wij, axis=0) / np.nansum(wij, axis=0) 
    sigmavj = 1.0 / np.sqrt(np.nansum(wij, axis=0))  
    
    #ipdb.set_trace()

    #return vj,sigmavj # MW: commented
    return vj,sigmavj, vijtel,sigmaij # MW




if __name__ == "__main__":
                       
#    filename = '/Data/kiwispec-proc/n20160524/n20160524.HD185144.0023.proc.chrec.npy'
#    vank(filename)
#    objnames = ['HD10700','HD9407','HD62613','HD122064','HD191408A','HD185144','HD217107','daytimeSky']
    
    objnames =  ['HD97601']#['HD10700']#['HD217107']#['HD217014']#['HD97601']#['HD142245']#['HD122064']#,'HD185144']#['daytimeSky','HD122064','HD185144'] ['HD97601']#
    for objname in objnames:
        print objnames
        vank(objname)
    sys.exit()

    targets = targetlist.rdlist()
    for objname in targets['name']:
        vank(objname)
