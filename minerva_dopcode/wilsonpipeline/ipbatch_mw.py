#!/usr/bin/env python

'''
Algorithm:
--- Initial data's normalization from gaussian IP
----- Arbitrarily define the center for all plots.

--- Use daytime sky spectra and fit Model to it and get normalization.

--- Iterative process for optimal IP

--- Return a function that already has zero as its centroid

Procedure:  Assume an IP, generate Model, get residuals.  Then normalize Data with Model's optimal parameters to get new IP, then generate Model, get residuals.


Examples
--------
>>>from ipbatch_mw import *
>>>IPfunc = splineip(4,15,1100,128)


>>>nohup python minerva_rv.py > IO_IP/Results/nohup_rv.log &

'''

import numpy as np
import matplotlib.pyplot as plt  #Super
#import seaborn.apionly as sn # can't currently be imported in #Minerva #Super
from astropy.io import fits
from scipy import interpolate
from scipy.optimize import leastsq
from scipy.integrate import quad
import copy, os, shutil
import datetime
import matplotlib  #Super
from matplotlib import animation  #Super
#from IPython.core.debugger import Tracer # use as "Tracer()()" and then use the command "continue"  #MInerva #Super
import pickle
#import ipdb   
#from matplotlib.backends.backend_pdf import PdfPages  #Super #Minerva


# ==============================

def splineip(telescope, order, startcol, chunksize, clobberIP=False, testIP=False): 
	'''
	Instrumental profile is deduced from the cross-dispersion direction. This function can display and/or save plots showing every step of the code: the alignment of all reference spectra and their crosscuts, the spline(gaussian) IP, and the iterative process to find a better IP. If "testIP" is True, then the newly found IP will be tested on the given stellar spectra.  These plots can be saved into a video.  This is also where the header information of the FITS files can be plotted (one plot per header keyword).  Also, text files can be written in order to keep track of all of the testing results: optimal Model parameters and chi-squared on residuals (and perhaps error/informative messages).

	Parameters
	----------
	telescope:  int
			Telescope number can be 1 to 4.

	order:	    int
			Order can be 0 to 28.

	startcol:   int
			This is the first column of the chunk.

	chunksize:  int
			This is the width of the chunk.  # I don't recommend anything above a few angstroms (2 or 4 A) but technically we don't know the wavelength calibration ahead of time since I'm dealing with raw spectra.  You would need to find the iodine lines to do that calibration. # The wavelength to pixel solution can be found in the reduced spectras' extension(s).  It is 0.02 Angstroms per pixel. The minerva doppler code uses 128 pixels as a chunk.

	testIP:      bool
			If True, then the newly found IP is put to the test on stellar spectra specified in the "starsOnly" file.

	clobberIP:  bool
			If True, then re-calculate IP even if the pickle file for the IP is already present. That file will be overwritten.

	Return
	------
	pre_spl_mod_norm_mr: LSQ Univariate spline function

	'''
	# this first big variables section is totally unnecessary (could've just made the dictionary and be done with it) but it does make the script more easily manageable

	variables = {}

	IOdir   = 'IO_IP/' #'IO_IP/'  #'./'  #'/n/regal/jjohnson_lab/mwilson21/data/'   # location of all the "pickle", "perfect", "clear", and "lis" files that already are and will be created #Minerva  #Super
	datdir  = '/Data/kiwispec/n20171230/' #'/Data/kiwispec/n20171230/'  #'../'  #'/n/regal/jjohnson_lab/mwilson21/data/' # each FITS file may come from a different directory. That's fine as long as the number of characters in the directory path is the same for each because this variable is only here to get the length of the string.  The lists themselves contain the true path for each file. # if the directory structure ever changes, the minervaDirs function must change too  #Minerva  #Super
	minerva = True #True #False # if True, use minerva data directory structure #Minerva

	lisID 				          = 'Y17M9-Y18M6' #'testY18M6D20F48'#'Y17M10-Y18M6' #'testY17M10D3' #'Y17M10' #'Y16M4-Y16M7'  #'2016'  is a great set for data in April and May (and maybe June). Unfortunately, between June 1st and Oct. 2016 the IP changed a lot for some reason so October isn't included.
	perfectlisID			      = 'Y17M9-Y18M6' #'Y17M10-Y18M6'  #'Y17M10' #'Y16M4-Y16M7' #'Y16M5' # for the sake of time (if IP file doesn't exist), this can just be "lisID" 	# remember, if you change this and "lisID", but the IPs are already filed, then those IPs are based on the previous "perfectlisID"

	variables['runningdir']               = None #IOdir+'Run_m3_t{}_o{}_sc{}_cs{}_{}_rpm20_ppm25/'.format(telescope,order,startcol,chunksize,lisID) #None #'Run_Latest/' # Beware of Nov 11th, 2016.  The shift in traces is too vast.  I would have to have some serious IF statements if I wanted to have my program evaluate all data in 2016 and 2017 simultaneously, particularly because one should not assume the instrumental profile is stable for that long.  #Minerva #Super

	variables['this_scriptname']	      = './ipbatch_mw.py'

	variables['files_list_filename']      = IOdir+'Files_'+lisID+'_starsOnly.lis' # Make sure these names are only names (i.e., they do not include the directory path) # these must be ".pickle" files because ".pickle" files assures me that they have been vetted for file & file-data corruption & that their cosmic rays have been found
	variables['refs_list_filename']       = IOdir+'Files_'+lisID+'_daytimeSky.lis' # Make sure these names are only names (i.e., they do not include the directory path) # must have ".pickle" extension due to vetting and cosmic rays  # Unfortunately, there are a lot of daytime sky spectra in 2016 that have low SNR, which defeats the main point of getting/using daytime sky spectra. Also, there are sky spectra in Jan. 2016 but that month does not have a single "perfect" file (yet?). This is a major problem for stellar spectra of Jan. 2016.
	variables['refsPerMo']		      = 20 # 10 is good if there was no SNR-like threshold. 20 seems good for 2016. 2016 has plenty of daytime sky spectra with low counts so the SNR-like threshold was necessary so take that into account when setting this variable

	variables['perfect_list_filename']    = IOdir+'Files_'+perfectlisID+'_perfect.lis' # Make sure these names are only names (i.e., they do not include the directory path) # these might as well be daytime sky spectra but must have the ".perfect" extension.  Not all daytime sky spectra were distinguished as "perfect." # these files already have a "pickle" file and are bright enough such that a blue circle is found at every column for every trace. # These files determine what rows each traces starts at and thus determines the specific rows (~10) about that center that will be used in the analysis.  # horrible things can happen to the fitting of crosscuts when these "perfect files" are after June 2017 but the reference spectra are before June 2017, due to the ~2 pixel traces' shift.

	variables['perfectsPerMo']	      = 25 # 5 is good for 2017. 5 and 10 are NOT good enough for 2016!! Just stick with 25 to be safe. You can never have too much of these. The only reason why there's a limit is because of the amount of time it takes to run the code.

	# It seems to me like the changing shape of the residuals for daytime sky spectra over time is due to the changing gaussian-estimated centroid of the absorption features.  This is why under low SNR the residuals have a conspicuous (gauss-hermite) shape.  Under high SNR conditions, it seems that the absorption features reveal themselves through the gradual increase of residuals meeting a sharp drop in the residuals.  The strength of the residuals (and thus, under low SNR conditions, the shape of residuals) is due to the SNR of all crosscuts. 

	# 2016, July ~18th: the spectrograph was vented. 
	# 2016, November 11th: team began to take spectrograph apart and reassembled everything on the 23rd.
	# 2016, April ?:  According to my <centroid>-time tracer plot, this is when the infamous power outage occurred and shifted traces by 1.5 or 2 pixels.  This is the same time that the daytime sky spectra start using the exposure meter "Expmeter."
	# Problematic: 'n20160606.daytimeSkyExpmeter.0072.fits' has incredibly weird data.  Wow! Looking at the file, the traces look catastrophic!  A horrible file that is. 
	# The daytime sky spectra of May 23rd (#63) and May 31st (#85 and #86) of 2016 seem to be incredibly slanted. Even the DS9 images show the severely slanted absorption features.
	# There is a vehement bleeding of a cosmic ray in trace ID_103 startcol_576 for n20160401.daytimeSky.0053.fits that can massively ruin the IP for that chunk. 

	variables['copy_filenames']				 = [  variables['this_scriptname'], \
											variables['files_list_filename'], variables['refs_list_filename'], 
											variables['perfect_list_filename']  ]
							
	

	variables['Nrows'] = 11 # is there a way to softcode this (per column maybe)? # for sake of symmetry, I like making this an odd number.
	variables['addtoNhalfrows'] = 4 # Be careful! this could leak over into another trace. (this will inevitably happen at the very top of the fits files so prepare for that.) The leakage could be reduced if the midrows, i.e. the "perfect" files, are chosen perfectly by matching the dates perfectly.

	variables['cent_pm'] =  3 # this is merely for the alignment process. ~4 pixels is the largest shift in traces I've seen (I think) between 2 adjacent months. This has nothing to do with the length of the more-rows data.

	variables['ampguess'] = 28000 #2000 # 28000 is a great guess for when daytime sky spectra is the reference
	variables['biasguess'] = 1000 #400  # 1000 is a great guess for when daytime sky spectra is the reference
	variables['sigmaguess'] = 2.5 # this was an eyeball estimate but seems to work wonderfully for when daytime sky spectra is the reference

	variables['thresh_chi'] = 1e-4 #1e-11 # difference between reduced chi-squares  # The Iterat/Y18_spline06/ directory shows that when the spline's breakpoints' spacing is 0.6, a sufficient "\chi^2 difference" is when is it is less than 1.0x10^-5 (5 decimal places). 
	#thresh = 1e-4 # difference between standard deviations. 
	

	satlim, cntslim = (55000, 15000) # 15000 is a great choice for "np.max(all data) must be greater than this counts limit"


	variables['T_or_F'] = True # this is for the option "chisqr" in the "metricfunct" definition
	if variables['T_or_F']:	variables['metricsign'] = r'\chi^2' 
	else:	variables['metricsign'] = r'\sigma'
	variables['plotmarkersize'] = 3


	variables['splinedegree'], variables['knotinterval'] = (3, 0.6)   # the iterative step varies by this to an insignificant degree. Any knot interval less than 0.6 will be eventually (or quickly) cause the chi-squared to get worse monotonically but the overall change in chi-squared space is insignificant.


	variables['gain'] = 1.3 # The files' headers never adjust this value.


	variables['str_modelclass'] = 'dx_N_b'  #dx_N_b_c4_c3_gsig  #dx_N_b #dx_N_b_c4_gsig
	variables['mymodguesses'] =  np.array([ 0, 0.0, 0.0])  # { shift, normalization constant, offset, (Hermite constant_4th, and Gaussian sigma) or (Hermite constant_4th, Hermite constant_3rd, Gaussian sigma) }  #np.array([ 0, 0.0, 0.0]) #np.array([ 0, 0.0, 0.0, 1, 1]) #np.array([ 0, 0.0, 0.0, 0.01, 0.01, 1 ]) 
	variables['N_fittedparameters'] = len(variables['mymodguesses'])

	variables['chi_uplim'] = 20	

	# all terminal or qt-console related visuals    #Super #Minerva
	variables['printing'], variables['rapid_printing'] = True, False
	variables['plotting'], variables['rapid_plotting'] = False, False


	variables['Expmeter_only'], variables['noExpmeter_date'] = False, datetime.date(2016,4,20) #datetime.date(2016,6,7) 




	variables['trace_ID'] = ( (28-order)*4 + 2) - telescope # this is hard-coded based on the trace ID setup with 113 total traces specified by traces_starts.reg when its generated from 2016 daytime sky spectra.  The "choose_perfect" function knows to correct this equation if setup is different.
	#if variables['printing']: print( 'Trace ID: {}'.format(variables['trace_ID']) ) This actually prints excessively/rapidly. Must be some part of the software repeatedly calling for this instrumental profile. 
	
	runningdir = variables['runningdir']
	if clobberIP is False:
		try: # see if the IP file is in the IO directory or the running directory.
			if runningdir is None: 
				dumpfile = open("{}IP_m{}_t{}_o{}_sc{}_cs{}_{}.pickle".format(IOdir, variables[
				'N_fittedparameters'], telescope, order, startcol, chunksize, lisID) , 'rb' )

			else:
				dumpfile = open("{}IP_m{}_t{}_o{}_sc{}_cs{}_{}.pickle".format(runningdir, variables[
				'N_fittedparameters'], telescope, order, startcol, chunksize, lisID) , 'rb' )

			IP_norm_mr = pickle.load( dumpfile )["IP"]
			dumpfile.close()
			
			return IP_norm_mr

		except IOError: # calculate the IP
			pass


	str_modelclass = variables['str_modelclass']
	if str_modelclass is 'dx_N_b':              modelclass = dx_N_b
	elif str_modelclass is 'dx_N_b_c4_gsig':    modelclass = dx_N_b_c4_gsig
	elif str_modelclass is 'dx_N_b_c4_c3_gsig': modelclass = dx_N_b_c4_c3_gsig
	# correction for daytime sky spectra
	modelclass_DS = dx_N_b

	nonHermite_class = dx_N_b
	if modelclass == nonHermite_class: Hermite = False
	else: Hermite = True

	
	modelclass_called = modelclass() # call so that the 'self' can be understood.
	gen_normeddata = modelclass_called.gen_normeddata
	modelspl = modelclass_called.modelspl
	#splresidual = modelclass_called.splresidual
	modelsplfit = modelclass_called.modelsplfit
	
	# correction for daytime sky spectra
	modelclass_called_DS = dx_N_b()
	gen_normeddata_DS = modelclass_called_DS.gen_normeddata
	modelspl_DS = modelclass_called_DS.modelspl
	modelsplfit_DS = modelclass_called_DS.modelsplfit 


	printing = variables['printing']
	rapid_printing = variables['rapid_printing']
	plotting = variables['plotting']
	rapid_plotting = variables['rapid_plotting']


	if plotting or runningdir is not None:
		#matplotlib.use("Agg")
		plt.ion()
		sn.set_style('white')


	#runningdir = variables['runningdir'] 
	if runningdir is not None:
		try:
			shutil.rmtree(runningdir)
		except WindowsError:
			pass

		#os.system('mkdir '+runningdir) # can't handle slashes
		os.mkdir(runningdir)

		copy_filenames = variables['copy_filenames']
		for i in range(len(copy_filenames)): shutil.copy(copy_filenames[i], runningdir)

	'''
	if minerva: 
		dataDirs = minervaDirs
	else:
		dataDirs = myDirs
	'''
	
	refsPerMo = variables['refsPerMo']
	refs_list_filename = variables['refs_list_filename']
	ref_list = np.loadtxt(refs_list_filename, dtype=str, ndmin=1) # ndmin=1 guarantees that the inside will be a list instead of just a string
	if minerva:
		ref_list = minervaDirs( ref_list ) # now the originally specified datdir can be used. 	
	else:
		ref_list = myDirs( ref_list, datdir )

	ref_list = np.sort( ref_list )
	'''
	ref_list_dates = name2date( ref_list, datdir )
	ref_list = ref_list[ choose_indsMonth( ref_list_dates, refsPerMo ) ] # the default is a uniform distribution. # random indices may unsort array
	ref_list = extension(ref_list, '.pickle' )
	ref_list = np.sort( ref_list )
	numRefs  = len(ref_list)
	'''
	ref_list = extension(ref_list, '.pickle' )
	numRefs  = len(ref_list)

	if testIP:
		files_list_filename = variables['files_list_filename']
		f_list = np.sort( np.loadtxt(files_list_filename, dtype=str, ndmin=1) )  # Thanks to 'sort' this is alphanumerically sorted--if it wasn't already.
		f_list = extension(f_list, '.pickle') 
		if minerva: 
			f_list = minervaDirs( f_list )
		else:
			f_list = myDirs( f_list, datdir )

	perfectsPerMo = variables['perfectsPerMo']
	perfect_list_filename = variables['perfect_list_filename']
	perfect_list = np.sort( np.loadtxt(perfect_list_filename, dtype=str, ndmin=1) )
	if minerva:
		perfect_list = minervaDirs( perfect_list )
	else:
		perfect_list = myDirs( perfect_list, datdir )

	perfect_list_dates = name2date( perfect_list, datdir )
	perfect_list = perfect_list[ choose_indsMonth( perfect_list_dates, perfectsPerMo ) ] # random indices may unsort array
	perfect_list = np.sort( perfect_list )
	perfect_list = extension( perfect_list, '.perfect' ) 


	# ds9 starts at (1,1), which is where the first counts measurement of a full pixel is.
	#order = variables['order'] 
	#telescope = variables['telescope']
	trace_ID = variables['trace_ID'] 
	#startcol = variables['startcol']
	cols = range(startcol, startcol+chunksize) 
	if variables['printing']: print( 'Trace ID: {}'.format(variables['trace_ID']) ) # This will only print if the IP file doesn't exist


	Nrows = variables['Nrows'] 
	rowminus = np.floor(float(Nrows)/2).astype(int)
	rowplus = np.ceil(float(Nrows)/2).astype(int) # this works if Nrows is even too.
	addtoNhalfrows = variables['addtoNhalfrows']
	Nmorerows = Nrows + 2*addtoNhalfrows

	kwargs_perfect = { 'ID': trace_ID, 'columns': cols, 'rplus': rowplus, 'rminus': rowminus, 'plusNhalfrows': addtoNhalfrows, 'datdir':datdir, 'IOdir': IOdir }


	cent_pm = variables['cent_pm']
	centguess = -np.inf # the optimizer is very biased by the guess (presumably because the number of data points are so few) therefore it is best to pick a guess corresponding to the first or middle-ish crosscut initially and then just change the guess after every crosscut # OR, by using the Chebyshev polynomial (or the blue circles) as a model of the trace prior to gaussian-fitting, you have a great centroid-guess per column
	ampguess = variables['ampguess'] #2000 # 28000 is a great guess for when daytime sky spectra is the reference
	biasguess = variables['biasguess'] #400  # 1000 is a great guess for when daytime sky spectra is the reference
	sigmaguess = variables['sigmaguess']

	#thresh_std = 1e-14 # difference between standard deviations
	thresh_chi = variables['thresh_chi'] # difference between reduced chi-squares

	chi_uplim = variables['chi_uplim']

	T_or_F = variables['T_or_F'] # this is for the option "chisqr" in the "metricfunct" definition
	metricsign = variables['metricsign']
	plotmarkersize = variables['plotmarkersize']


	strs_freepars_keys = modelclass.strs_freepars_keys # instance variable
	#strs_freepars = modelclass.strs_freepars
	str_freepars =  modelclass.str_freepars
	str_model =  modelclass.str_model
	str_reciprocal =  modelclass.str_reciprocal

	# other correction for daytime sky spectra
	strs_freepars_keys_DS = modelclass_DS.strs_freepars_keys 
	str_freepars_DS =  modelclass_DS.str_freepars
	str_model_DS =  modelclass_DS.str_model
	str_reciprocal_DS =  modelclass_DS.str_reciprocal

	splinedegree, knotinterval = ( variables['splinedegree'], variables['knotinterval'] ) 


	gain = variables['gain'] # The files' headers never adjust this value. gain = [ electrons/ADU ] 


	mygaussguessesB = np.array([ ampguess, centguess, sigmaguess, biasguess ], dtype=float) 
	mygaussguesses = np.array([ ampguess, centguess, sigmaguess, 0 ], dtype=float)  



	#=============================  Ascertain IP  ========================================




	# vet files based on arbitrary thresholds: saturation and a SNR-like threshold (because 2016 has lots of low SNR daytime sky spectra) # this process goes incredibly slow. It's because of the medianCRs function being ran on every file.

	d_f_inds = []

	for f_i in range(numRefs):

		try:
			clearfile = open( '{}{}.slim{}_clim{}.clear'.format( IOdir, ref_list[f_i][len(datdir):], satlim, cntslim ) , 'r') # the IOError will have occurred here.
			clearfile.close()


			# check if this specific chunk has high SNR.  This is important for May 31st in particular.
			file_info = fits.open( ref_list[f_i] )
			dat = medianCRs( ref_list[f_i], datdir, IOdir )

			midrows, rows_c, morerows_c = choose_perfectfile( perfect_list, name2date(ref_list[f_i], datdir), True, **kwargs_perfect )
			#check_arr = dat[np.min(np.array(morerows_c)) : np.max(np.array(morerows_c))+1, cols] # this isn't good because morerows_c runs into another telescope's trace when dealing with low orders (i.e. top of FITS file)
			check_arr = dat[np.min(np.array(rows_c)) : np.max(np.array(rows_c))+1, cols]
			
			if  np.max(check_arr) < cntslim - abs(14-order)/15.0*cntslim:#abs(14-order)/14.0*cntslim/2.0:  # This seems to be fine with the cntslim equalling 15 thousand but I believe the 14th (middle-ish) order is not the brightest.  I think the counts are greatest at the bottom of the 2D spectra, i.e., the highest-number orders.
				d_f_inds.append( f_i )
				if rapid_printing: print(ref_list[f_i]+' at this specific chunk is disqualified.')

		except IOError:

			try:
				clearfile = open( '{}{}.slim{}_clim{}.notclear'.format( IOdir, ref_list[f_i][len(datdir):], satlim, cntslim ) , 'r')
				clearfile.close()

				d_f_inds.append( f_i )
				if rapid_printing: print(ref_list[f_i]+' was already disqualified.')

			except IOError:

				file_info = fits.open( ref_list[f_i] )
				dat = medianCRs( ref_list[f_i], datdir, IOdir )

				if (dat > satlim).any() or np.max(dat) < cntslim: # check for saturation and low SNR # there is, of course, a much better way to set a reasonable SNR threshold! Due to vetting, any cosmic rays should be gone or insignificant.

					clearfile = open( '{}{}.slim{}_clim{}.notclear'.format( IOdir, ref_list[f_i][len(datdir):], satlim, cntslim ) , 'w')
					clearfile.close()

					d_f_inds.append( f_i )
					if rapid_printing: print(ref_list[f_i]+' is disqualified.')

				else:
					clearfile = open( '{}{}.slim{}_clim{}.clear'.format( IOdir, ref_list[f_i][len(datdir):], satlim, cntslim ) , 'w')
					clearfile.close()



	d_f_inds = np.array(d_f_inds)
	if len(d_f_inds) > 0:
		numRefs = numRefs-len(d_f_inds)
		ref_list = np.delete( ref_list, d_f_inds)


	# Now make final list of reference spectra

	ref_list_dates = name2date( ref_list, datdir )
	ref_list = ref_list[ choose_indsMonth( ref_list_dates, refsPerMo ) ] # the default is a uniform distribution. # random indices may unsort array
	ref_list = np.sort( ref_list )
	numRefs  = len(ref_list)



	# The centroid of the trace per column changes from file to file. This is most likely the data itself changing its centroid and if that is true then consequently the IP is changing from file to file (hopefully not significantly).

	# The trace-columns need to be aligned via the centroid per column.  Use a gaussian model to find the centroid parameter.  The gaussian fit may be contaminated by the beginnings a cosmic ray though. So after fitting all of the centroids, see if any centroid is an outlier.  That can be found based on the residuals from the fitted centroids and a Chebyshev polynomial as the model that fits the trace across the entire file.  As a start, you can use the blue circles as the model instead of the Chebyshev polynomial.

	alignedrows_c_f    = np.ones( (numRefs, len(cols), Nrows) )*-np.inf
	alignedrows_c_mr_f = np.ones( (numRefs, len(cols), Nmorerows) )*-np.inf 

	for f_i in range(numRefs):
		file_info = fits.open( ref_list[f_i] )
		dat = medianCRs( ref_list[f_i], datdir, IOdir )

		midrows, rows_c, morerows_c = choose_perfectfile( perfect_list, name2date(ref_list[f_i], datdir), True, **kwargs_perfect )

		overscan = dat[0:,2048:] # the overscan region begins at ds9_x=2049.
		sigma_read = np.sqrt(  np.mean( ( gain*overscan.astype(float) - gain*np.mean(overscan.astype(float)) )**2 ) ) # standard deviation (i.e. RMS about zero) of photons in overscan region
		sigma_read_ADU = sigma_read/gain

		
		for c_i in range(len(cols)):
			mygaussguessesB[1] = midrows[c_i] # This can be the Chebyshev model estimate instead.
			mygaussguesses[1] = midrows[c_i]

			rows = rows_c[c_i]
			rowsfl = rows.astype(float)
			morerows = morerows_c[c_i]
			morerowsfl = morerows.astype(float)

			arrB = dat[rows[0]:rows[-1]+1, cols[c_i]].astype(float) # include the last element because that's simply how many rows I actually want to look at.
			arrB_mr = dat[morerows[0]:morerows[-1]+1, cols[c_i]].astype(float) # "morerows" gives me "moredata"

			# find optimal centroid
			# Just in case, this is when the fiber traces all shift by ~2 pixels, you should use as many rows as possible to find the centroid.
			pars_mr = gaussfit(morerows, arrB_mr, 1, mygaussguessesB, midrows[c_i]-cent_pm, midrows[c_i]+cent_pm)  # finds centroid better when there are no errorbars
			mygaussguessesB[1] = pars_mr['b_centroid']
			mygaussguesses[1] = pars_mr['b_centroid']
			arr_noB_mr = arrB_mr - pars_mr['b_offset']  # bias subtraction before normalization can be found.
			arr_noB = arrB - pars_mr['b_offset']  # some de-biased data points might be negative now.

			sigma_Snoise_col_mr = np.sqrt( gain*( np.abs(arr_noB_mr) ) ) # Poisson noise is now in units of electons (or photons) but data is still in ADU

			sigma_Snoise_col_mr_ADU = sigma_Snoise_col_mr / np.sqrt(gain) 
			pars_mr = gaussfit(morerows, arr_noB_mr, np.sqrt(sigma_Snoise_col_mr_ADU**2 + sigma_read_ADU**2), mygaussguesses, midrows[c_i]-cent_pm, midrows[c_i]+cent_pm) 
			b_cent = pars_mr['b_centroid']

			if (c_i == 0) and (f_i == 0): initcentroid = b_cent # arbitarily chosen zero-point for plots

			alignshift = b_cent - initcentroid
			alignedrows_c_f[f_i, c_i] = rowsfl - alignshift
			alignedrows_c_mr_f[f_i, c_i] = morerowsfl - alignshift


		# HERE, after the c_i-FOR-loop is done, you should check to see if their are any outliers!!! For any crosscut that has a centroid-outlier, get rid of the cosmic ray and then re-find the centroid



	alignedrows_refs = alignedrows_c_f.reshape( numRefs*len(cols)*Nrows )
	alignedrows_mr_refs = alignedrows_c_mr_f.reshape( numRefs*len(cols)*Nmorerows )

	# ==================================

	dates_refnames = [] 
	refnames_header = []
	refnames = ref_list

	dat_list = []
	sigma_read_files       = np.zeros( numRefs )
	sigma_read_ADU_files   = np.zeros( numRefs )
	sigma_reads_f          = np.zeros( (numRefs, len(cols)*Nrows) )
	sigma_reads_ADU_f      = np.zeros( (numRefs, len(cols)*Nrows) )
	sigma_reads_mr_f       = np.zeros( (numRefs, len(cols)*Nmorerows) )
	sigma_reads_ADU_mr_f   = np.zeros( (numRefs, len(cols)*Nmorerows) )

	for f_i in range(len(ref_list)):
		refname = ref_list[f_i]
		dates_refnames.append( name2date(refname, datdir) ) 
		file_info = fits.open( refname )
		refnames_header.append( file_info[0].header )
		dat = medianCRs( refname, datdir, IOdir )
		overscan = dat[0:,2048:] # the overscan region begins at ds9 x=2049.
		sigma_read = np.sqrt(  np.mean( ( gain*overscan.astype(float)- gain*np.mean(overscan.astype(float)) )**2 ) ) # standard deviation (i.e. RMS about zero) of photons in overscan region

		dat_list.append( dat )
		sigma_read_files[f_i] = sigma_read
		sigma_reads_f[f_i] = sigma_read*np.ones( len(cols)*Nrows )
		sigma_reads_mr_f[f_i] = sigma_read*np.ones( len(cols)*Nmorerows )

		sigma_read_ADU_files[f_i] = sigma_read/gain
		sigma_reads_ADU_f[f_i] = sigma_reads_f[f_i]/gain
		sigma_reads_ADU_mr_f[f_i] = sigma_reads_mr_f[f_i]/gain


	sigma_reads_refs        = sigma_reads_f.reshape( numRefs*len(cols)*Nrows )
	sigma_reads_ADU_refs    = sigma_reads_ADU_f.reshape( numRefs*len(cols)*Nrows )
	sigma_reads_ADU_mr_refs = sigma_reads_ADU_mr_f.reshape( numRefs*len(cols)*Nmorerows )  


	# =====================


	# Make sure each array's crosscut contains information from all reference spectra

	arr_c_f                     = np.ones( (numRefs, len(cols), Nrows) )*-np.inf
	arr_c_mr_f                  = np.ones( (numRefs, len(cols), Nmorerows) )*-np.inf
	arr_c_normed_f              = np.ones( (numRefs, len(cols), Nrows) )*-np.inf 
	arr_c_normed_mr_f           = np.ones( (numRefs, len(cols), Nmorerows) )*-np.inf 

	sigma_Snoise_c_f            = np.zeros( (numRefs, len(cols), Nrows) )
	sigma_Snoise_c_mr_f         = np.zeros( (numRefs, len(cols), Nmorerows) )
	sigma_Snoise_c_ADU_f        = np.zeros( (numRefs, len(cols), Nrows) )
	sigma_Snoise_c_mr_ADU_f     = np.zeros( (numRefs, len(cols), Nmorerows) )

	for f_i in range(numRefs):

		midrows, rows_c, morerows_c = choose_perfectfile( perfect_list, name2date(ref_list[f_i], datdir), True, **kwargs_perfect )

		alignedrows_c = alignedrows_c_f[f_i]
		alignedrows_c_mr = alignedrows_c_mr_f[f_i]

		for c_i in range(len(cols)):
			rows = rows_c[c_i]
			rowsfl = rows.astype(float)
			morerows = morerows_c[c_i]
			morerowsfl = morerows.astype(float)

			arrB = dat_list[f_i][rows[0]:rows[-1]+1, cols[c_i]].astype(float) 
			arrB_mr = dat_list[f_i][morerows[0]:morerows[-1]+1, cols[c_i]].astype(float)  # "morerows" gives me "more data"

			# find optimal centroid and a good guess at a normalized curve
			# Just in case, this is when the fiber traces all shift by ~2 pixels, you should use as many rows as possible to find the centroid.
			#mygaussguessesB[1] = midrows[c_i] 
			#mygaussguesses[1] = midrows[c_i]
			mygaussguessesB[1] = initcentroid
			mygaussguesses[1] = initcentroid

			pars_mr = gaussfit(alignedrows_c_mr[c_i], arrB_mr, 1, mygaussguessesB, initcentroid-cent_pm, initcentroid+cent_pm)  # finds centroid better when there are no errorbars
			mygaussguessesB[1] = pars_mr['b_centroid']
			mygaussguesses[1] = pars_mr['b_centroid']
			arr_noB_mr = arrB_mr - pars_mr['b_offset']  # bias subtraction before normalization can be found.
			arr_noB = arrB - pars_mr['b_offset']  # some de-biased data points might be negative now.

			sigma_Snoise_c_f[f_i, c_i] = np.sqrt( gain*(np.abs(arr_noB)) ) # Poisson noise = SQRT[N_photons=N_electrons] # bias and other parameters of gaussian model were found via more-rows data.
			sigma_Snoise_c_mr_f[f_i, c_i] = np.sqrt( gain*(np.abs(arr_noB_mr)) ) # Not sure what to do with negative values so for now I'll just take the absolute value.?

			sigma_Snoise_c_ADU_f[f_i, c_i] = sigma_Snoise_c_f[f_i, c_i] / np.sqrt(gain)
			sigma_Snoise_c_mr_ADU_f[f_i, c_i] = sigma_Snoise_c_mr_f[f_i, c_i] / np.sqrt(gain)

			pars_mr = gaussfit(alignedrows_c_mr[c_i], arr_noB_mr, np.sqrt(sigma_Snoise_c_mr_ADU_f[f_i, c_i]**2 + sigma_read_ADU_files[f_i]**2), mygaussguesses, initcentroid-cent_pm, initcentroid+cent_pm) 
			area_tot_mr = quad( gauss, np.min(alignedrows_c_mr[c_i]), np.max(alignedrows_c_mr[c_i]), args=(pars_mr['b_amp'], pars_mr['b_centroid'], pars_mr['b_gsigma'], pars_mr['b_offset']) )[0]  # the b_offset here should be ~0 because it was found from data that was bias-subtracted
			arr_normed_mr = arr_noB_mr / area_tot_mr
			
			ali_inds_mr_to_r = ( alignedrows_c_mr[c_i] >= np.min(alignedrows_c[c_i]) ) & ( alignedrows_c_mr[c_i] <= np.max(alignedrows_c[c_i]) ) # the numbers within the aligned "morerows" encompass the exact same numbers of the aligned "rows" array, therefore, the number elements should now be the same. 
			'''
			pars = gaussfit(rows_refs, arr_noB, np.sqrt(sigma_Snoise_c[c_i]**2 + sigma_reads**2), mygaussguesses) 
			area_tot = quad( gauss, float(rows[0]), float(rows[-1]), args=(pars['b_amp'], pars['b_centroid'], pars['b_gsigma'], pars['b_offset']) )[0]  # the b_offset here should be ~0 because it was found from data that was bias-subtracted
			arr_normed = arr_noB / area_tot
			'''
			arr_normed = arr_normed_mr[ali_inds_mr_to_r] # in theory, not normalizing via the ~11 pixels alone sounds sketchy but isn't once you remember that the "morerows" data should be at 0 (after de-biasing) anyway and thus the area under those edges of the curve are negligible.  This strategy is only used for the gaussian IP anyway.  Furthermore, used "moredata" can provide more accurate optimal-parameters simply because the data reveals more of the curve.  Therefore, truncating that normalized curve is actually a better strategy considering the very few (~11) data points you're dealing with. 

			#crosscenters_c[c_i] = b_cent # this "crosscenter" is not useful whenever there is more than one reference
			arr_c_f[f_i, c_i] = arrB
			arr_c_mr_f[f_i, c_i] = arrB_mr
			arr_c_normed_f[f_i, c_i] = arr_normed
			arr_c_normed_mr_f[f_i, c_i] = arr_normed_mr




	# View the references' spline(Gaussian) IP
	# For each crosscut, use the Model to find the optimal parameters 
	# With the optimal parameters, normalize the data to define a new IP-splined

	g_alignedrows_refs      = alignedrows_c_f.reshape( numRefs*len(cols)*Nrows ) # each array is concatenated onto the preceding array. Array( [A,B], [C,D] ]) --> Array([A,B,C,D])
	arr_refs                = arr_c_f.reshape( numRefs*len(cols)*Nrows )
	sigma_Snoise_refs       = sigma_Snoise_c_f.reshape( numRefs*len(cols)*Nrows )
	sigma_Snoise_ADU_refs   = sigma_Snoise_c_ADU_f.reshape( numRefs*len(cols)*Nrows )
	g_arr_normed_refs       = arr_c_normed_f.reshape( numRefs*len(cols)*Nrows )
	g_srtinds_refs          = np.argsort(g_alignedrows_refs)
   

	N_fittedparameters_DS = 3 #variables['N_fittedparameters'] # with my Model(dx, N, b), the IP acts as a fixed parameter.


	sigmas1st_refs = np.sqrt( sigma_Snoise_refs**2 + sigma_reads_refs**2 )
	sigmas1st_ADU_refs = np.sqrt( sigma_Snoise_ADU_refs**2 + sigma_reads_ADU_refs**2 ) 


	plt.figure(dpi=100)
	plt.plot(g_alignedrows_refs[g_srtinds_refs], arr_refs[g_srtinds_refs], '+', markersize=plotmarkersize) # For multiple references, each numerical row is duplicated numRefs times.
	plt.title('alignment')
	if plotting: plt.show()
	else: plt.close()

	# spline fitting does not account for errorbars when finding a best-model.  But that's where chi-squared comes in. # On second thought, I only use spline fitting whenever the data is already normalized so there shouldn't be any errorbars unless those are scaled down too.  
	spline_errs = True
	while spline_errs: # necessary for order #2 startcol #1856.
		try:
			brkpts = np.arange(g_alignedrows_refs[g_srtinds_refs][1], g_alignedrows_refs[g_srtinds_refs][-1], knotinterval) # np.arange will always exclude that end-number specified here, so these break points will never include the first or last x-value points.

			splnorm = interpolate.LSQUnivariateSpline(g_alignedrows_refs[g_srtinds_refs] - initcentroid, g_arr_normed_refs[g_srtinds_refs], brkpts - initcentroid, k=splinedegree) # model the normalized date's data  # the "interpolate" module forces you to have the x-array in numerical order.

			g_yspline = splnorm(g_alignedrows_refs[g_srtinds_refs] - initcentroid, nu=0, ext='raise') # nu is the order of the derivative of the spline
			spline_errs = False

		except ValueError:
			knotinterval += 0.1
			if printing: print('Problem. A spline error. Changing break points. knotinterval={}'.format(knotinterval))


	firstRefdate = '/'.join([ str(dates_refnames[0].month), str(dates_refnames[0].day), str(dates_refnames[0].year)[2:] ])+"#{}".format(int(refnames[0][-9:-5]))  
	lastRefdate = '/'.join( [ str(dates_refnames[-1].month), str(dates_refnames[-1].day), str(dates_refnames[-1].year)[2:] ])+"#{}".format(int(refnames[-1][-9:-5])) 

	if plotting:
		plt.figure(dpi=100)  
		plt.plot(g_alignedrows_refs[g_srtinds_refs] - initcentroid, g_arr_normed_refs[g_srtinds_refs], '+', markersize=plotmarkersize, label='gaussian-normalized')
		plt.plot(g_alignedrows_refs[g_srtinds_refs] - initcentroid, g_yspline, label='spline(Gaussian)')
		plt.xlabel('row')
		plt.ylabel('normalized counts')
		plt.title('Reference spectra. Gaussian aligned and normalized.\n'+'{}--{}'.format(firstRefdate, lastRefdate))
		plt.legend()
		plt.show()

	# find initial guesses for the Model parameters

	g_alignedrows_mr_refs     = alignedrows_c_mr_f.reshape( numRefs*len(cols)*Nmorerows )
	arr_mr_refs               = arr_c_mr_f.reshape( numRefs*len(cols)*Nmorerows )
	sigma_Snoise_mr_refs      = sigma_Snoise_c_mr_f.reshape( numRefs*len(cols)*Nmorerows )
	sigma_Snoise_mr_ADU_refs  = sigma_Snoise_c_mr_ADU_f.reshape( numRefs*len(cols)*Nmorerows )
	g_arr_normed_mr_refs      = arr_c_normed_mr_f.reshape( numRefs*len(cols)*Nmorerows )
	g_srtinds_mr_refs         = np.argsort(g_alignedrows_mr_refs)


	spline_errs_mr = True
	while spline_errs_mr:
		try:

			brkpts_mr = np.arange(g_alignedrows_mr_refs[g_srtinds_mr_refs][1], g_alignedrows_mr_refs[g_srtinds_mr_refs][-1], knotinterval) 

			splnorm_mr = interpolate.LSQUnivariateSpline(g_alignedrows_mr_refs[g_srtinds_mr_refs] - initcentroid, g_arr_normed_mr_refs[g_srtinds_mr_refs], brkpts_mr - initcentroid, k=splinedegree) # This is the Instrumental Profile.
			spline_errs_mr = False

		except ValueError:
			knotinterval += 0.1
			if printing: print('Problem. A spline error. Changing "morerows" break points. knotinterval={}'.format(knotinterval))


	# Right here I could re-plot the Gaussian-normalized but with splnorm_mr instead. 


	# For each file (and each crosscut) the same IP is used but different Model-parameters

	arr_c_model_f         = np.ones( (numRefs, len(cols), Nrows) )*-np.inf
	residuals_c_f         = np.ones( (numRefs, len(cols), Nrows) )*-np.inf
	shift_c_files         = np.ones( (numRefs, len(cols)) )*-np.inf
	normalize_c_files     = np.ones( (numRefs, len(cols)) )*-np.inf
	offset_c_files        = np.ones( (numRefs, len(cols)) )*-np.inf
	mymodguesses_files    = np.ones( (numRefs, N_fittedparameters_DS) )*-np.inf
	for f_i in range(numRefs):

		refname = ref_list[f_i]

		mymodguesses_DS = np.array([ 0, 0.0, 0.0])#variables['mymodguesses'] # shift, normalization constant, offset, Hermite constant_4th, Hermite constant_3rd, Gaussian sigma

		# combine/overplot/concatenate all crosscuts
		g_alignedrows_mr = alignedrows_c_mr_f[f_i].reshape(          len(cols)*Nmorerows ) 
		arr_mr = arr_c_mr_f[f_i].reshape(                            len(cols)*Nmorerows )
		sigma_Snoise_mr_ADU = sigma_Snoise_c_mr_ADU_f[f_i].reshape(  len(cols)*Nmorerows )
		g_srtinds_mr = np.argsort(g_alignedrows_mr)

		sigma_read_ADU = sigma_read_ADU_files[f_i]

		params = gaussfit(g_alignedrows_mr[g_srtinds_mr], arr_mr[g_srtinds_mr], \
			np.sqrt(sigma_Snoise_mr_ADU[g_srtinds_mr]**2 + sigma_read_ADU**2), mygaussguessesB, \
			initcentroid-cent_pm, initcentroid+cent_pm)
		mymodguesses_DS[2] = params['b_offset'] # bias , a.k.a. offset

		params = gaussfit(g_alignedrows_mr[g_srtinds_mr], arr_mr[g_srtinds_mr] - params['b_offset'], \
			np.sqrt(sigma_Snoise_mr_ADU[g_srtinds_mr]**2 + sigma_read_ADU**2), mygaussguesses, \
			initcentroid-cent_pm, initcentroid+cent_pm)
		area_total = quad( gauss, g_alignedrows_mr[g_srtinds_mr][0], g_alignedrows_mr[g_srtinds_mr][-1], \
		 args=(params['b_amp'], params['b_centroid'], params['b_gsigma'], params['b_offset']) )[0]
		mymodguesses_DS[1] = area_total 

		mymodguesses_files[f_i] = mymodguesses_DS # this is just for the zeroeth iteration of the iterative process

		# thanks to the gaussian approximation, we have a starter instrumental profile to work with in fitting the Data with my Model(dx, N, b)

		arr_c_modelnormed      = np.ones( (len(cols), Nrows) )*-np.inf
		arr_c_modelnormed_mr   = np.ones( (len(cols), Nmorerows) )*-np.inf
		for c_i in range(len(cols)):
			# now find the best parameters for each crosscut's normalization
			dxmaxs = find_dxlims(alignedrows_c_f[f_i, c_i], g_alignedrows_mr_refs[g_srtinds_mr_refs])
			kwargs_modelsplfit = {'interpnorm': splnorm_mr, 'guesses': mymodguesses_DS, 'dxlims': dxmaxs}
			'''if Hermite: kwargs_modelsplfit['centroid'] = 0#initcentroid '''

			pars = modelsplfit_DS( alignedrows_c_f[f_i, c_i] - initcentroid,  arr_c_f[f_i, c_i], \
			np.sqrt(sigma_Snoise_c_ADU_f[f_i, c_i]**2 + sigma_read_ADU**2), **kwargs_modelsplfit ) # this alignedrows[c_i] is already in numerical order 

			#shifted_x_column = alignedrows_c[c_i] - pars['dx']
			#yspline_splnorm = splnorm_mr(shifted_x_column, nu=0, ext='raise') # Now that I've found the discrete points from my "continuous model function"  and its shape, I can shift it onto the Data's x-axis now, because truthfully we wanted to shift the model onto the Data's x-axis anyway. This could have been done straightforward and easily if a gaussian was my model because coding-wise I could have just manipulated the centroid parameter (but I can't do that with this spline function stuff).   # consider using LSQ univariate spline instead. with that you can specify break pts.

			par_args = tuple([ pars[strs_freepars_keys_DS[p_i]] for p_i in range(N_fittedparameters_DS)  ])  
			# par_args = ( pars['dx'], pars['N'], pars['b']  )

			kwargs_modelspl = {'x': alignedrows_c_f[f_i, c_i] - initcentroid, 'interpnorm': splnorm_mr}
			kwargs_normeddata = {} # empty kwargs unpacks into nothing and thus doesn't crash the function.
			kwargs_normeddata_mr = {} 
			'''
			if Hermite: 
				kwargs_modelspl['centroid'] = 0#initcentroid
				kwargs_normeddata = {'dx': pars['dx'], 'x': alignedrows_c_f[f_i, c_i]-initcentroid, 'centroid': 0}#initcentroid}
				kwargs_normeddata_mr = {'dx': pars['dx'], 'x': alignedrows_c_mr_f[f_i, c_i]-initcentroid, 'centroid': 0}#initcentroid}
			'''

			arr_c_model_f[f_i, c_i] = modelspl_DS(*par_args, **kwargs_modelspl) # this is what's done in the optimizer # This is now on the alignedrows_c[c_i] axis, i.e. the Data's x-axis # the approximate centroid accounts for the "dx" shift because the shift is supposed to kick in for lopsided curves where the gaussian's centroid approximation may fail due to its attempt to fit the asymmetrical wings.
			residuals_c_f[f_i, c_i] = arr_c_f[f_i, c_i] - arr_c_model_f[f_i, c_i]

			# now record the updated-normalized estimation 
			shift_c_files[f_i, c_i]     = pars['dx']
			normalize_c_files[f_i, c_i] = pars['N']
			offset_c_files[f_i, c_i]    = pars['b']

			arr_c_modelnormed[c_i] = gen_normeddata_DS(arr_c_f[f_i, c_i], *par_args[1:], **kwargs_normeddata) # ydata = N*yspline_normalized + b, therefore do reciprocal to get new normalization  

			arr_c_modelnormed_mr[c_i] = gen_normeddata_DS(arr_c_mr_f[f_i, c_i], *par_args[1:], **kwargs_normeddata_mr) # The improved instrumental profile is this.  It should be as much of a never-ending, continuous function as possible. These optimal parameters were found with a curve of less data but the "more rows" data should only add a bunch of near-offset/zero values to both sides of the curve anyway, thus these parameters would in theory be the same if they were derived with the "more rows" Data from the start, but that can't be done because of this stupid spline function interpolation crap when attempting to shift x-axis.

		if f_i == 0 and plotting:

			plt.figure(dpi=100)
			for c_i in range(len(cols)):
				plt.plot(alignedrows_c_mr_f[f_i, c_i] - initcentroid, arr_c_mr_f[f_i, c_i], '+', markersize=plotmarkersize )
				
			plt.xlabel('row')
			plt.ylabel('counts')
			plt.title( '{}. Crosscut aligned Data (+).\nMore rows.'.format( ref_list[f_i][len(datdir):]) )
			plt.show()


			plt.figure(dpi=100)
			for c_i in range(len(cols)): 
				plt.plot(alignedrows_c_f[f_i, c_i] - initcentroid, arr_c_model_f[f_i, c_i], 'x', markersize=plotmarkersize)
				plt.plot(alignedrows_c_f[f_i, c_i] - initcentroid, arr_c_f[f_i, c_i], '+', markersize=plotmarkersize )
				
			plt.xlabel('row')
			plt.ylabel('counts')
			plt.title(  '{}. \nCrosscut aligned Data (+) and crosscut models (x). \nmodels=Model({},IP=spline(Gaussian))'.format(ref_list[f_i][len(datdir):], str_freepars_DS)  )
			plt.show()


		# see how much the first date's data spline fit differs from the gaussian fit. 

		# this concatenation does the same thing the same way as the np.reshape. I could use that here instead but concatenating is obviously more visually intuitive.
		arr = arr_c_f[f_i].reshape( len(cols)*Nrows )
		arr_model = arr_c_model_f[f_i].reshape( len(cols)*Nrows )
		arr_modelnormed = []
		arr_modelnormed_mr = []
		for c_i in range(len(cols)):
			arr_modelnormed = np.concatenate((arr_modelnormed, arr_c_modelnormed[c_i]))
			arr_modelnormed_mr = np.concatenate((arr_modelnormed_mr, arr_c_modelnormed_mr[c_i]))

		
		sigma_Snoise_ADU = sigma_Snoise_c_ADU_f[f_i].reshape( len(cols)*Nrows )
		sigmas1st_ADU = np.sqrt( sigma_Snoise_ADU**2 + sigma_read_ADU**2 )

		g_alignedrows = alignedrows_c_f[f_i].reshape( len(cols)*Nrows )
		g_srtinds = np.argsort( g_alignedrows )


		firstresiduals = arr - arr_model
		firstmetric = metricfunc(firstresiduals[g_srtinds], sigmas1st_ADU[g_srtinds], N_fittedparameters_DS, float(len(firstresiduals)), chisqr=T_or_F) #metricfunc(firstresiduals, sigmas1st_ADU[g_srtinds], N_fittedparameters, float(len(firstresiduals))/numRefs, chisqr=T_or_F) # Make sure to change the threshold if you change between standard deviation and chi^2.  
		if printing and rapid_printing: print('first metric: {:.10e}'.format(firstmetric))


		if runningdir is not None and f_i == 0: pdf_pages = PdfPages( "{}IP_m{}_t{}_o{}_sc{}_cs{}_{}.pdf".format(runningdir, N_fittedparameters_DS, telescope, order, startcol, chunksize, lisID)  )

		fig = plt.figure(dpi=100)
		ax1 = fig.add_axes( (.1, .3, .8, .6) ) # xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
		plt.plot(g_alignedrows[g_srtinds] - initcentroid, arr[g_srtinds], '+', markersize=plotmarkersize, label='Data (aligned peaks)')
		plt.plot(g_alignedrows[g_srtinds] - initcentroid, arr_model[g_srtinds], 'x', markersize=plotmarkersize, label="Model("+str_freepars_DS+",IP=spline(Gaussian))")
		ax1.set_xticklabels([])
		plt.ylabel('counts')
		#plt.title('Reference spectra. Model = '+str_model+'\n'+'{}--{}'.format(firstRefdate, lastRefdate))
		plt.title('{}. Model = {}\n'.format(ref_list[f_i][len(datdir):], str_model_DS) )
		plt.legend(loc=2)

		ax2 = fig.add_axes( (.1, .1, .8, .2) )
		plt.plot( g_alignedrows[g_srtinds] - initcentroid, firstresiduals[g_srtinds], '+', markersize=plotmarkersize)
		plt.axhline(y=0.0, linestyle='--', color='k')
		plt.text(0.9, 0.85, r"${}$={:.2f}".format(metricsign,firstmetric), horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes) # the ":.2f" rounds off at 2 decimal places
		plt.xlabel('row')

		if runningdir is not None: pdf_pages.savefig( fig )
		if runningdir is not None and f_i == numRefs-1: pdf_pages.close()
		if plotting: plt.show()
		else: plt.close()


	# Now find the chi-squared value when all of the residuals are used.  
	# The sorting_indices aren't actually necessary when you're not deduced a spline interpolation model

	arr_model_refs = arr_c_model_f.reshape( numRefs*len(cols)*Nrows )
	residuals_refs = residuals_c_f.reshape( numRefs*len(cols)*Nrows )

	firstmetric_ultimate = metricfunc(residuals_refs, sigmas1st_ADU_refs, N_fittedparameters_DS, float(len(residuals_refs)), chisqr=T_or_F)

	if printing: print('first metric (all references): {:.10e}'.format(firstmetric_ultimate))

	if plotting:
		fig = plt.figure(dpi=100)
		ax1 = fig.add_axes( (.1, .3, .8, .6) ) # xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
		plt.plot(g_alignedrows_refs - initcentroid, arr_refs, '+', markersize=plotmarkersize, label='Data (aligned peaks)')
		plt.plot(g_alignedrows_refs - initcentroid, arr_model_refs, 'x', markersize=plotmarkersize, label="Model("+str_freepars_DS+",IP=spline(Gaussian))")
		ax1.set_xticklabels([])
		plt.ylabel('counts')
		plt.title('Reference spectra. Model = '+str_model_DS+'\n'+'{}--{}'.format(firstRefdate, lastRefdate))
		plt.legend(loc=2)

		ax2 = fig.add_axes( (.1, .1, .8, .2) )
		plt.plot( g_alignedrows_refs- initcentroid, residuals_refs, '+', markersize=plotmarkersize)
		plt.axhline(y=0.0, linestyle='--', color='k')
		plt.text(0.9, 0.85, r"${}$={:.2f}".format(metricsign,firstmetric_ultimate), horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes) # the ":.2f" rounds off at 2 decimal places
		plt.xlabel('row')

		plt.show()



	#----- Now start iterative process. -----
	# The whole point to find a different IP each iteration (although you can change your guesses each iteration too).

	# Iteratively improve instrumental profile of the 1st date by checking a Data-model metric.  However, you never really want to touch the data in any way whatsoever, therefore do not use the normalized data in the final metric.
	# But based on Jason's email, Data_normalized - IP may be the correct choice for this iterative step.
	# I think the IP should be finalized when it best matches the data, i.e. my Model(dx,N,b) ultimately determines if my proposed IP is a good one. 
	# Wait, is this about improving the IP AS WELL AS stopping the iteration only when the new model is no longer different than the old?  I think you improve the IP until your resultant metric is not much different than the last IP's resultant metric.  If so, one way of doing this is to get residuals of Model_new - Model_previous, which is what I've been doing the past few months.

	# use some kind of metric to gauge the summarized difference in both models.  To get the same x-axis for both, interpolate one model data onto another.  Perhaps, use standard deviation (if you're working with residuals) or maximum difference (absolute value in reference to a zero point) 
	# difference in chi^2 ideally (from one person's perspective) 1e-8 or 1e-4 . 

	# Use reduced chi^2
	#	chi_r^2 = chi^2 / nu , nu = N_datapts - N_fittedparameters
	#   chi^2 = SUM[ ( (D_i - M_i)/sig_i )^2 ]
	#   sig_i = SQRT[ sig_RN^2 + sig_noise_i^2]
	#   sig_RN: read noise = RMS of overscan region
	#   sig_noise = SQRT[ N_photons ] , N_stellarphotons = gain*(counts-bias) , gain==[electron/ADU] , Analog to Digital Unit = Counts


	test_spl_mod_norm_mr = copy.copy( splnorm_mr ) # Although redundant, let's let the the zeroth iteration calculate the Model and residuals for this.  
	pre_arr_modelnormed_mr_refs = copy.copy( g_arr_normed_mr_refs ) # the [g_srtinds_mr_refs] indices will have to be used on this
	#pre_yspline_spl_mod_norm = copy.copy( yspline_splnorm ) # for plotting purposes only
	pre_arr_modelnormed_refs = copy.copy( g_arr_normed_refs ) 

	#pre_shift_c_files = shift_c_files
	test_shift_c_files = np.ones( np.shape(shift_c_files) )*-np.inf
	test_normalize_c_files = np.ones( np.shape(normalize_c_files) )*np.nan
	test_offset_c_files = np.ones( np.shape(offset_c_files) )*np.nan

	thresh = thresh_chi # "thresh" is already defined, based on choice of chunk and metric.

	#g_srtinds_mr_to_r_refs = (g_alignedrows_mr_refs[g_srtinds_mr_refs] >= np.min(g_alignedrows_refs)) & (g_alignedrows_mr_refs[g_srtinds_mr_refs] <= np.max(g_alignedrows_refs)) # this is only for plotting purposes

	diff = -100.0 # difference between chi^2 or standard deviations 
	cnt = 0
	if minerva is False:
		test_metrics = np.array([]) 
		diffs        = np.array([]) 
		yay = False
	while np.abs(diff) >= thresh: 
		
		if cnt > 0: # record stuff from previous iteration because that stuff gave us a better Model.
			pre_spl_mod_norm_mr = copy.copy( test_spl_mod_norm_mr )
			pre_metric = test_metric 
			pre_shift_c_files = copy.copy( test_shift_c_files )
			pre_normalize_c_files = copy.copy( test_normalize_c_files )
			pre_offset_c_files = copy.copy( test_offset_c_files )
			#pre_residuals = copy.copy( residuals_refs )
			test_spl_mod_norm_mr = copy.copy( next_spl_mod_norm_mr )
			pre_arr_modelnormed_mr_refs = copy.copy( test_arr_modelnormed_mr_refs ) # the [g_srtinds_mr_refs] indices will have to be used on this
			pre_arr_modelnormed_refs = copy.copy( test_arr_modelnormed_refs ) # for plotting purposes

		if cnt >= 2 and runningdir is not None:
			os.remove(runningdir+'latest_IP_1st_'+str(cnt%2)+'.png')
			os.remove(runningdir+'latest_Model_'+str(cnt%2)+'.png')	

		residuals_c_f                 = np.ones( (numRefs, len(cols), Nrows) )*-np.inf
		test_arr_c_modelnormed_mr_f   = np.ones( (numRefs, len(cols), Nmorerows) )*-np.inf 
		test_arr_c_modelnormed_f      = np.ones( (numRefs, len(cols), Nrows) )*-np.inf 
		test_arr_c_model_f            = np.ones( (numRefs, len(cols), Nrows) )*-np.inf
		normalizations_c_f            = np.ones( (numRefs, len(cols), Nrows) )*np.nan
		for f_i in range(numRefs):

			mymodguesses_DS = mymodguesses_files[f_i]
			
			sigma_read_ADU = sigma_read_ADU_files[f_i]
			
			for c_i in range(len(cols)):
				# now find the best parameters for each crosscut's normalization
				if cnt > 0: # The 'cnt>0' assures that the zeroeth iteration is truly identical to the previous plot.
					mymodguesses_DS[0] = pre_shift_c_files[f_i, c_i] 
					mymodguesses_DS[1] = pre_normalize_c_files[f_i, c_i] 
					mymodguesses_DS[2] = pre_offset_c_files[f_i, c_i]
				

				dxmaxs = find_dxlims(alignedrows_c_f[f_i, c_i], g_alignedrows_mr_refs[g_srtinds_mr_refs])
				kwargs_modelsplfit = {'interpnorm': test_spl_mod_norm_mr, 'guesses': mymodguesses_DS, 'dxlims': dxmaxs}
				'''if Hermite: kwargs_modelsplfit['centroid'] = 0#initcentroid '''

				pars = modelsplfit_DS( alignedrows_c_f[f_i, c_i] -initcentroid, arr_c_f[f_i, c_i], \
				np.sqrt(sigma_Snoise_c_ADU_f[f_i, c_i]**2 + sigma_read_ADU**2), **kwargs_modelsplfit ) # you could adjust all of the guesses here per c_i

				test_shift_c_files[f_i, c_i] = pars['dx'] # record the dx parameter just in case we have another iteration
				test_normalize_c_files[f_i, c_i] = pars['N']
				test_offset_c_files[f_i, c_i] = pars['b']

				normalizations_c_f[f_i, c_i] = pars['N']*np.ones( Nrows )

				#shifted_x_column = alignedrows_c[c_i] - pars['dx']
				#yspline_spl_mod_norm_column = test_spl_mod_norm_mr(shifted_x_column, nu=0, ext='raise')
				par_args = tuple([ pars[strs_freepars_keys_DS[p_i]] for p_i in range(N_fittedparameters_DS)  ])

				kwargs_modelspl = {'x': alignedrows_c_f[f_i, c_i] -initcentroid, 'interpnorm': test_spl_mod_norm_mr}
				kwargs_normeddata = {}  # for plotting purposes
				kwargs_normeddata_mr = {} # empty kwargs unpacks into nothing and thus doesn't crash the function. 
				'''
				if Hermite: 
					kwargs_modelspl['centroid'] = 0#initcentroid
					kwargs_normeddata = {'dx': pars['dx'], 'x': alignedrows_c_f[f_i, c_i]-initcentroid, 'centroid': 0} # for plotting purposes 
					kwargs_normeddata_mr = {'dx': pars['dx'], 'x': alignedrows_c_mr_f[f_i, c_i]-initcentroid, 'centroid': 0}#initcentroid}
				'''

				test_arr_c_model_f[f_i, c_i] = modelspl_DS(*par_args, **kwargs_modelspl) 
				residuals_c_f[f_i, c_i] = arr_c_f[f_i, c_i] - test_arr_c_model_f[f_i, c_i]

				test_arr_c_modelnormed_mr_f[f_i, c_i] = gen_normeddata_DS(arr_c_mr_f[f_i, c_i], *par_args[1:], **kwargs_normeddata_mr)  # Preparation for possible next iteration  # par_args[0] will always be the 'dx' regardless of the Class-Model chosen.

				test_arr_c_modelnormed_f[f_i, c_i] = gen_normeddata_DS(arr_c_f[f_i, c_i], *par_args[1:], **kwargs_normeddata) # for plotting purposes


		residuals_refs = residuals_c_f.reshape( numRefs*len(cols)*Nrows )
		normalizations_refs = normalizations_c_f.reshape( numRefs*len(cols)*Nrows )
		test_arr_model_refs = test_arr_c_model_f.reshape( numRefs*len(cols)*Nrows )
		test_arr_modelnormed_mr_refs = test_arr_c_modelnormed_mr_f.reshape( numRefs*len(cols)*Nmorerows )
		test_arr_modelnormed_refs = test_arr_c_modelnormed_f.reshape( numRefs*len(cols)*Nrows )

		#residuals_refs = arr_refs - test_arr_model_refs

		next_spl_mod_norm_mr = interpolate.LSQUnivariateSpline(g_alignedrows_mr_refs[g_srtinds_mr_refs] -initcentroid, test_arr_modelnormed_mr_refs[g_srtinds_mr_refs], brkpts_mr -initcentroid, k=splinedegree ) # test instrumental profile for next iteration.

		test_metric = metricfunc(residuals_refs, sigmas1st_ADU_refs, N_fittedparameters_DS, float(len(residuals_refs)), chisqr=T_or_F) #metricfunc(residuals, sigmas1st_ADU[g_srtinds], N_fittedparameters, float(len(residuals))/numRefs, chisqr=T_or_F) # Make sure to change the threshold if you change between standard deviation and chi^2.  
		if T_or_F:
			if printing and rapid_printing: print('testing Chi^2: '+ "{:.10e}".format(test_metric))
		else:
			if printing and rapid_printing: print('testing St.d.: '+ "{:.10e}".format(test_metric))


		if cnt == 0:
			if printing: print('Zeroth iteration')

		else:		
			diff = test_metric - pre_metric
			if T_or_F:
				if printing and rapid_printing: print('Chi^2_testing - Chi^2_previous: '+"{:.10e}".format(diff))
			else:
				if printing and rapid_printing: print('St.d._testing - St.d._previous: '+"{:.10e}".format(diff))
		

		#yspline_spl_mod_norm = test_spl_mod_norm_mr(g_alignedrows_mr_refs[g_srtinds_mr_refs][g_srtinds_mr_to_r_refs] - initcentroid, nu=0, ext='raise') # only for plotting purposes


		yspline_spl_mod_norm = test_spl_mod_norm_mr(g_alignedrows_refs[g_srtinds_refs] - initcentroid, nu=0, ext='raise') # only for plotting purposes
		residualsIP_refs = pre_arr_modelnormed_refs[g_srtinds_refs] - yspline_spl_mod_norm # only for plotting purposes
		test_IPmetric = metricfunc(residualsIP_refs, sigmas1st_ADU_refs[g_srtinds_refs]/normalizations_refs[g_srtinds_refs], N_fittedparameters_DS, float(len(residualsIP_refs)), chisqr=T_or_F)  # Used propagated error to find scaled down error bars. # Because the array of errorbars did not originate from "mr" data, I felt like making the residuals not originate from such data too.

		'''
		figip = plt.figure(dpi=100, figsize=(5,4)) 
		plt.plot(g_alignedrows_mr_refs[g_srtinds_mr_refs][g_srtinds_mr_to_r_refs] - initcentroid, pre_arr_modelnormed_mr_refs[g_srtinds_mr_refs][g_srtinds_mr_to_r_refs] , '+', markersize=plotmarkersize, label='Normalized data' )
		plt.plot(g_alignedrows_mr_refs[g_srtinds_mr_refs][g_srtinds_mr_to_r_refs] - initcentroid, yspline_spl_mod_norm, label=r'Latest test IP$_{1st}$' )
		plt.ylabel('normalized counts')
		plt.xlabel('row')
		plt.title('Model = '+str_model+'\n'+'{}--{}'.format(firstRefdate, lastRefdate))
		plt.legend()
		if runningdir is not None: plt.savefig(runningdir+'latest_IP_1st_'+str(cnt%2)+'.png') 
		if plotting and rapid_plotting: plt.show()
		else: plt.close()
		'''
		figmodel = plt.figure(dpi=100, figsize=(5,4)) # The 'figsize' is needed here because of the long-numbered y-axis. 
		ax1 = figmodel.add_axes( (.2, .3, .75, .6) ) # xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
		plt.plot(g_alignedrows_refs[g_srtinds_refs] - initcentroid, pre_arr_modelnormed_refs[g_srtinds_refs], '+', markersize=plotmarkersize, label='Normalized Data' )
		#plt.plot(g_alignedrows_refs[g_srtinds_refs] - initcentroid, yspline_spl_mod_norm, label=r'Latest test IP$_{1st}$' )
		plt.plot(g_alignedrows_refs[g_srtinds_refs] - initcentroid, yspline_spl_mod_norm, label=r'IP' )
		ax1.set_xticklabels([])
		plt.ylabel('Normalized Counts')
		plt.legend(loc=1)
		plt.title('Model = '+str_model_DS+'\n'+'{}--{}'.format(firstRefdate, lastRefdate))
		
		ax2 = figmodel.add_axes( (.2, .1, .75, .2) )
		plt.plot(g_alignedrows_refs[g_srtinds_refs] - initcentroid, residualsIP_refs, '+', markersize=plotmarkersize)
		plt.axhline(y=0.0, linestyle='--', color='k')
		plt.text(0.9, 0.85, r"${}$={:.2f}".format(metricsign, test_IPmetric), horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes) # the ":.2f" rounds off at 2 decimal places
		plt.xlabel('Row')
		if runningdir is not None: plt.savefig(runningdir+'latest_IP_1st_'+str(cnt%2)+'.png') 
		if plotting and rapid_plotting: plt.show()
		else: plt.close()


		figmodel = plt.figure(dpi=100, figsize=(5,4)) # The 'figsize' is needed here because of the long-numbered y-axis. 
		ax1 = figmodel.add_axes( (.2, .3, .75, .6) ) # xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
		plt.plot(g_alignedrows_refs[g_srtinds_refs] - initcentroid, arr_refs[g_srtinds_refs], '+', markersize=plotmarkersize, label='Data (aligned peaks)')
		plt.plot(g_alignedrows_refs[g_srtinds_refs] - initcentroid, test_arr_model_refs[g_srtinds_refs], 'x', markersize=plotmarkersize, label=r"Model({})".format(str_freepars_DS)+r" from latest test IP$_{1st}$")
		ax1.set_xticklabels([])
		plt.ylabel('counts')
		plt.legend(loc=2)
		plt.title('Model = '+str_model_DS+'\n'+'{}--{}'.format(firstRefdate, lastRefdate))

		ax2 = figmodel.add_axes( (.2, .1, .75, .2) )
		plt.plot( g_alignedrows_refs[g_srtinds_refs] - initcentroid, residuals_refs[g_srtinds_refs], '+', markersize=plotmarkersize)
		plt.axhline(y=0.0, linestyle='--', color='k')
		plt.text(0.9, 0.85, r"${}$={:.2f}".format(metricsign, test_metric), horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes) # the ":.2f" rounds off at 2 decimal places
		plt.xlabel('row')
		if runningdir is not None: plt.savefig(runningdir+'latest_Model_'+str(cnt%2)+'.png')  
		if plotting and rapid_plotting: plt.show()
		else: plt.close()


		if minerva is False:
			test_metrics = np.append(test_metrics, test_metric) 
			diffs        = np.append(diffs, diff)
		

		'''   Check if the spline breakpoints' interval causing a plateau or a runaway in the chi-squared # debugging
		cnt += 1 
		if diff > 0:  
			yay = True
		if cnt > 100 and yay:
			Tracer()() # plt.figure() # this may never happen if the spline breakpoints' interval is 0.6 or more.
			break
		'''
		#if diff > 0 or np.abs(diff) < thresh: #if np.abs(diff) <= thresh: # WHILE Loop is about to break
		#	if printing: print('While Loop breaking by manual break.')
		if np.abs(diff) <= thresh: # WHILE Loop is about to break

			if runningdir is not None:
				if diff > 0: # delete current one because this has worse chi-squared
					os.remove(runningdir+'latest_IP_1st_'+str(cnt%2)+'.png')
					os.remove(runningdir+'latest_Model_'+str(cnt%2)+'.png')	
				else: # delete previous one because this has better chi-squared
					os.remove(runningdir+'latest_IP_1st_'+str(1 - cnt%2)+'.png')
					os.remove(runningdir+'latest_Model_'+str(1 - cnt%2)+'.png')


			#break # break the WHILE loop if the new IP is worst than the previous one because that is the only thing about this iterative process that is changing per iteration.  Therefore, if the residuals are worse now, then they won't get any better than they currently are unless I keep the current IP and change something else (maybe a parameter) within the WHILE Loop  # If this happens on the first iteration then that means the Gaussian IP was better than the first Model(dx,N,b) fit.
		else:	
			cnt += 1
		#'''
	
	'''   check for plateau or runaway # debugging   
	plt.figure(dpi=100)
	plt.plot(test_metrics, '+')
	plt.ylabel(r'$\chi^2$')
	plt.show()
	plt.figure(dpi=100)
	plt.plot(diffs[1:], '+')
	plt.ylabel('diffs[1:]')
	plt.show()
	plt.figure(dpi=100)
	plt.plot(test_metrics[1:], '+')
	plt.ylabel(r'$\chi^2$[1:]')
	plt.show()
	plt.figure(dpi=100)
	plt.plot(diffs[2:], '+')
	plt.ylabel('diffs[2:]')
	plt.show()

	plt.figure(dpi=100)
	plt.semilogy(test_metrics, '+')
	plt.ylabel(r'$\chi^2$')
	plt.show()
	plt.figure(dpi=100)
	plt.semilogy(diffs[1:], '+')
	plt.ylabel('diffs[1:]')
	plt.title('no negatives')
	plt.show()
	plt.figure(dpi=100)
	plt.semilogy(test_metrics[1:], '+')
	plt.ylabel(r'$\chi^2$[1:]')
	plt.show()
	plt.figure(dpi=100)
	plt.semilogy(diffs[2:], '+')
	plt.ylabel('diffs[2:]')
	plt.title('no negatives')
	plt.show()
	#'''


	'''
	initcentroid = initcentroid
	g_alignedrows_mr_refs = g_alignedrows_mr_refs

	# variables used for plotting in a tracers/summary plot
	dates_refnames = dates_refnames
	splinedegree = splinedegree 
	knotinterval = knotinterval
	pre_metric = pre_metric
	'''

	if minerva is False:
		saveiterations = {"test_metrics": test_metrics, "diffs": diffs}
		if runningdir is None:
			savefile = open("{}Iterations.pickle".format(IOdir), 'wb')
		else:
			savefile = open("{}Iterations.pickle".format(runningdir), 'wb')

		pickle.dump(saveiterations, savefile)
		savefile.close()

	savevar = {"IP": pre_spl_mod_norm_mr}
	if runningdir is None: 
		dumpfile = open("{}IP_m{}_t{}_o{}_sc{}_cs{}_{}.pickle".format(IOdir, N_fittedparameters_DS, telescope, order, startcol, chunksize, lisID) , 'wb' )
	else:
		dumpfile = open("{}IP_m{}_t{}_o{}_sc{}_cs{}_{}.pickle".format(runningdir, N_fittedparameters_DS, telescope, order, startcol, chunksize,lisID) , 'wb' )

	pickle.dump(savevar, dumpfile)
	dumpfile.close()



	if runningdir is not None:
		file_pars_chi = open(runningdir+'pars_chi.txt','w')
		file_pars_goodchi = open(runningdir+'pars_goodchi.txt','w')
		file_messages = open(runningdir+'messages.txt','w')

		chunk_string = 'chunk: { trace_ID='+str(trace_ID)+', cols=range('+str(cols[0])+','+str(cols[-1])+'+1), Nrows='+str(Nrows)+', Nmorerows='+str(Nmorerows)+' }' 
		file_pars_chi.write( chunk_string )
		file_pars_goodchi.write( chunk_string )
		file_messages.write( chunk_string )

		file_pars_chi.write('\nNumber of reference spectra used: '+str(len(ref_list)) )
		file_pars_chi.write('\nNumber of perfect spectra used: '+str(len(perfect_list)) )
		file_pars_chi.write('\n'+'\n'.join(ref_list)+'\n------' ) # show all reference spectra individually line by line
		file_pars_chi.write('\n'+'\n'.join(perfect_list)+'\n------' )
		#remaining_string = ", ".join([ p+' = N/A' for p in strs_freepars_keys ])

		file_pars_goodchi.write('\nchi_uplim = '+str(chi_uplim) )



	if testIP is False: 

		if runningdir is not None:
			file_pars_chi.close()
			file_pars_goodchi.close()
			file_messages.close()
		
		return pre_spl_mod_norm_mr



	#=========== Use the IP on stellar spectra as a test =========================

			

	fnames_header = []
	dates_fnames = []

	goodchi_mask          = np.ones( len(f_list) ).astype(bool) # 1 is True, 0 is False
	metrics_files         = np.ones( len(f_list) )*-np.inf
	crosscenters_files    = np.ones( len(f_list) )*-np.inf  # this is a useless monitor for data of multiple references. So only use this for the spectra in this analysis script.
	rows_f               = np.ones( (len(f_list), len(cols)*Nrows) )*-np.inf
	residuals_f          = np.ones( (len(f_list), len(cols)*Nrows) )*-np.inf
	counts_f             = np.ones( (len(f_list), len(cols)*Nrows) )*-np.inf
	counts_model_f       = np.ones( (len(f_list), len(cols)*Nrows) )*-np.inf
	counts_modelnormed_f = np.ones( (len(f_list), len(cols)*Nrows) )*-np.inf

	spl_mod_norm_mr_1st = copy.copy( pre_spl_mod_norm_mr )
	metric_1st = copy.copy( pre_metric ) # only used for the tracers/summary plots

	N_fittedparameters = variables['N_fittedparameters']

	perfectdate = name2date(f_list[0], datdir) # this is merely to help with efficiency
	midrows, rows_c, morerows_c = choose_perfectfile( perfect_list, perfectdate, getrows=True, **kwargs_perfect )
	perfect_calls = 1 # this is merely to help with efficiency

	maxabsresidual = 100 # for ylim plotting purposes # arbitrary number is set here but will be changed later.
	#xlim_left = int(np.min(g_alignedrows_refs - initcentroid))-2 
	#xlim_right = int(np.max(g_alignedrows_refs - initcentroid))+2
	xlim_left = int(np.min(g_alignedrows_mr_refs - initcentroid))-0.5 
	xlim_right = int(np.max(g_alignedrows_mr_refs - initcentroid))+0.5

	






	for f_i in range(len(f_list)):
		fname = f_list[f_i]

		ali_rows_c            = np.ones( (len(cols), Nrows) )*-np.inf
		ali_rows_c_mr         = np.ones( (len(cols), Nmorerows) )*-np.inf
		counts_c              = np.ones( (len(cols), Nrows) )*-np.inf
		counts_c_mr           = np.ones( (len(cols), Nmorerows) )*-np.inf
		counts_c_model        = np.ones( (len(cols), Nrows) )*-np.inf
		counts_c_modelnormed  = np.ones( (len(cols), Nrows) )*-np.inf
		sigmas_c              = np.ones( (len(cols), Nrows) )*-np.inf
		sigmas_c_ADU          = np.ones( (len(cols), Nrows) )*-np.inf
		crosscenters_c        = np.ones(len(cols))*-1*np.inf
		crossdx_c             = np.ones(len(cols))*-1*np.inf
		normalizations_c      = np.ones( (len(cols), Nrows) )*np.nan

		mymodguesses = variables['mymodguesses'] # each crosscut will have reasonable normalization and bias-offset guesses before being analyzed.

		file_info = fits.open( fname )
		fnames_header.append( file_info[0].header )
		dates_fnames.append( name2date(fname, datdir) )

		dat = medianCRs( fname, datdir, IOdir )
		overscan = dat[0:,2048:].astype(float) # the overscan region begins at ds9 x=2049.
		sigma_read = np.sqrt(  np.mean( (gain*overscan- gain*np.mean(overscan))**2 )  ) # Is read noise the RMS of the counts or the photons?  I'm going to go with photons, but use counts when performing calculations.
		sigma_read_ADU = sigma_read/gain

		if name2date(fname, datdir) > perfectdate: 

			midrows, rows_c, morerows_c, pname = choose_perfectfile( perfect_list[perfect_calls:], name2date(fname, datdir), getrows=True, getall=True, **kwargs_perfect ) # in the function, I arbitrarily set it such that the perfectdate will be equal to or most likely end up being right after the current file's date, hence the condition of this IF statement.

			if perfect_calls is not len(perfect_list)-1: perfect_calls += 1 # This isn't nearly as efficient as it looks
			#perfectdate = name2date( choose_perfectfile( perfect_list[perfect_calls:], name2date(fname, datdir), getrows=False, datdir=datdir) , datdir )
			perfectdate = name2date( pname, datdir )


		
		for c_i in range(len(cols)):
			mygaussguessesB[1], mygaussguesses[1] = midrows[c_i], midrows[c_i]

			rows = rows_c[c_i]
			rowsfl = rows.astype(float)
			morerows = morerows_c[c_i]
			morerowsfl = morerows.astype(float)

			counts_col = dat[rows[0]:rows[-1]+1, cols[c_i]].astype(float) # the "dat" variable is already a float, but I'd rather be safe than sorry
			counts_mr_col = dat[morerows[0]:morerows[-1]+1, cols[c_i]].astype(float)
			counts_c[c_i] = counts_col 
			counts_c_mr[c_i] = counts_mr_col 	

			mygaussguessesB[0], mygaussguesses[0] = tuple(np.ones(2)*( np.max(counts_mr_col)-np.min(counts_mr_col) )) # get a decent guess for amplitude
			pars_mr = gaussfit(morerowsfl, counts_mr_col, 1, mygaussguessesB, midrows[c_i]-cent_pm, midrows[c_i]+cent_pm)
			
			#--------- Get good guesses for dx, N, b, c4, c3, g_sigma for each and every crosscut # well... except for "dx"  ---------

			mymodguesses[2] = pars_mr['b_offset']  
			mygaussguessesB[1] = pars_mr['b_centroid'] # This new guess (found without error bars) helps the fitting a lot once the residuals are weighted by noise.
			mygaussguesses[1] = pars_mr['b_centroid']

			counts_noB_col = counts_c[c_i] - pars_mr['b_offset']  # some de-biased data points may be negative now.
			
			signoise_col = np.sqrt( gain*(np.abs(counts_noB_col)) ) # Poisson noise = SQRT[N_photons=N_electrons]  
			sigmas_c[c_i] = np.sqrt(signoise_col**2 + sigma_read**2)  

			signoise_col_ADU = signoise_col/np.sqrt(gain)  
			sigmas_c_ADU[c_i] = np.sqrt(signoise_col_ADU**2 + sigma_read_ADU**2) 

			counts_noB_mr_col = counts_mr_col - pars_mr['b_offset']
			signoise_mr_col = np.sqrt( gain*(np.abs(counts_noB_mr_col)) )  # ask Jason what to do about this.?
			signoise_mr_col_ADU = signoise_mr_col/np.sqrt(gain)

			#if ( counts_noB_mr_col < -1*np.max(counts_noB_mr_col)/10).any(): print( counts_noB_mr_col[counts_noB_mr_col <0], fname, c_i) # debugging

			pars_mr = gaussfit(morerowsfl, counts_noB_mr_col, np.sqrt(signoise_mr_col_ADU**2 + sigma_read_ADU**2), mygaussguesses, midrows[c_i]-cent_pm, midrows[c_i]+cent_pm) # This often does HORRRIBLE when the residuals is weighted by noise, unless the guesses are REALLY good.
			area = quad( gauss, morerowsfl[0], morerowsfl[-1], args=(pars_mr['b_amp'], pars_mr['b_centroid'], pars_mr['b_gsigma'], pars_mr['b_offset']) )[0]
			mymodguesses[1] = area 

			#------------

			b_cent = pars_mr['b_centroid'] 
			crosscenters_c[c_i] = b_cent

			shiftalign = b_cent - initcentroid   # the peak of each crosscut's curve is aligned to the very first date's first crosscut's curve's peak (the absolute number pixel, that is; not the fractional pixel position with respect to range of pixels).
			ali_rows_c[c_i] = rowsfl - shiftalign 
			ali_rows_c_mr[c_i] = morerowsfl - shiftalign

			dxmaxs = find_dxlims(ali_rows_c[c_i], g_alignedrows_mr_refs[g_srtinds_mr_refs])
			kwargs_modelsplfit = {'interpnorm': spl_mod_norm_mr_1st, 'guesses': mymodguesses, 'dxlims': dxmaxs}
			if Hermite: kwargs_modelsplfit['centroid'] = 0#initcentroid
			
			ps = modelsplfit( ali_rows_c[c_i]-initcentroid, counts_c[c_i], np.sqrt(signoise_col_ADU**2 + sigma_read_ADU**2), **kwargs_modelsplfit )  # Undo all important stuff if this raises the extrapolation value error
			
			crossdx_c[c_i] = ps['dx']

			normalizations_c[c_i] = ps['N']
		
			#shifted_x_col = ali_rows_c[c_i] - ps['dx']
			#yspline_splmodelnorm = spl_mod_norm_mr_1st(shifted_x_col, nu=0, ext='raise') 
			p_args = tuple([ ps[strs_freepars_keys[p_i]] for p_i in range(N_fittedparameters)  ]) 

			kwargs_modelspl = {'x': ali_rows_c[c_i]-initcentroid, 'interpnorm': spl_mod_norm_mr_1st}
			kwargs_normeddata = {} # empty kwargs unpacks into nothing and thus doesn't crash the function.
			if Hermite: 
				kwargs_modelspl['centroid'] = 0#initcentroid
				kwargs_normeddata = {'dx': ps['dx'], 'x': ali_rows_c[c_i]-initcentroid, 'centroid': 0}#initcentroid}

			counts_c_model[c_i] =  modelspl(*p_args, **kwargs_modelspl) 

			counts_c_modelnormed[c_i] =  gen_normeddata(counts_c[c_i], *p_args[1:], **kwargs_normeddata)  
	



		counts             = counts_c.reshape( len(cols)*Nrows )
		counts_model       = counts_c_model.reshape( len(cols)*Nrows )
		counts_modelnormed = counts_c_modelnormed.reshape( len(cols)*Nrows )
		sigmas             = sigmas_c.reshape( len(cols)*Nrows )
		sigmas_ADU         = sigmas_c_ADU.reshape( len(cols)*Nrows )
		ali_rows           = ali_rows_c.reshape( len(cols)*Nrows )
		normalizations     = normalizations_c.reshape( len(cols)*Nrows )

		sortedinds = np.argsort( ali_rows )
		
		rows_f[f_i] = ali_rows[sortedinds] 

		residu = counts - counts_model
		residuals_f[f_i] = residu[sortedinds] 

		date_metric = metricfunc(residu[sortedinds], sigmas_ADU[sortedinds], N_fittedparameters, float(len(residu)), chisqr=T_or_F)
		metrics_files[f_i] = date_metric 
		
		if runningdir is not None:
			remaining_string = ", ".join([ strs_freepars_keys[p_i]+" = {:.2e}".format( ps[strs_freepars_keys[p_i]] ) for p_i in range(N_fittedparameters) ]) # write out each parameter with their value
			file_pars_chi.write('\n'+fname+': {} = {:.3f}'.format(metricsign, date_metric)+', '+remaining_string) 

		if date_metric < chi_uplim: # if date_metric is nan then this condition will not be satisfied nor will the program crash

			if runningdir is not None: file_pars_goodchi.write('\n'+fname+': {} = {:.3f}'.format(metricsign, date_metric)+', '+remaining_string)

			checkabsmax = np.max( np.abs(residu) )
			if checkabsmax > maxabsresidual: maxabsresidual = checkabsmax

		else:
			goodchi_mask[f_i] = False

		counts_modelnormed_f[f_i] = counts_modelnormed[sortedinds] 
		counts_f[f_i] = counts[sortedinds] 
		counts_model_f[f_i] = counts_model[sortedinds] 

		crosscenters_files[f_i] = np.mean(crosscenters_c - crossdx_c) 
		
		if plotting and rapid_plotting: # the plotting condition is a fail-safe of common sense
			fig, (ax1, ax2, ax3) = plt.subplots(3,1, gridspec_kw = {'height_ratios':[2,3,1]}, figsize=(6,7), dpi=120, sharex=True, sharey=False)
			#fig.set_dpi(120)
			#fig.set_size_inches(6,8)
			
			ax1.plot( ali_rows[sortedinds] - initcentroid, counts_modelnormed[sortedinds], '+', markersize=plotmarkersize, label="{}".format(str_reciprocal) )
			ax1.set_ylabel('normalized counts')
			ax1.set_title(r'Used IP$_{1st}$ to find optimal parameters of Model'+'({})\n{}'.format(str_freepars,fname[len(datdir):]) )
			ax1.legend(loc=1)

			ax2.plot( ali_rows[sortedinds] - initcentroid, counts[sortedinds], '+', markersize=plotmarkersize, label="Data (aligned peaks)")
			ax2.plot( ali_rows[sortedinds] - initcentroid, counts_model[sortedinds], 'x', markersize=plotmarkersize, label="Model({})".format(str_freepars) )
			ax2.set_ylabel('counts')
			ax2.legend(loc=2)

			ax3.plot( ali_rows[sortedinds] - initcentroid, residu[sortedinds], '+')
			ax3.axhline(y=0.0, linestyle='--', color='k')
			ax3.text(0.9, 0.85, r'${}$={:.2f}'.format(metricsign, date_metric), horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)
			ax3.set_xlabel('row')
			ax3.set_xlim(xlim_left, xlim_right)

			fig.subplots_adjust(hspace=0)
			plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
			plt.show()


			# Extra plot

			figmodel = plt.figure(dpi=100, figsize=(5,4)) # The 'figsize' is needed here because of the long-numbered y-axis. 
			ax1 = figmodel.add_axes( (.2, .3, .75, .6) ) # xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
			plt.plot(ali_rows[sortedinds] - initcentroid, counts_modelnormed[sortedinds], '+', markersize=plotmarkersize, label='Normalized Data' )
			plt.plot(ali_rows[sortedinds] - initcentroid, spl_mod_norm_mr_1st(ali_rows[sortedinds] - initcentroid), label=r'IP' )
			ax1.set_xticklabels([])
			plt.ylabel('Normalized Counts')
			#plt.legend()

			residsIP = counts_modelnormed - spl_mod_norm_mr_1st(ali_rows - initcentroid)
			IPmetric = metricfunc(residsIP, sigmas_ADU/normalizations, N_fittedparameters, float(len(residsIP)), chisqr=T_or_F)  # Used propagated error to find scaled down error bars.
			
			ax2 = figmodel.add_axes( (.2, .1, .75, .2) )
			plt.plot(ali_rows[sortedinds] - initcentroid, residsIP[sortedinds], '+', markersize=plotmarkersize)
			plt.axhline(y=0.0, linestyle='--', color='k')
			plt.text(0.9, 0.85, r"${}$={:.2f}".format(metricsign, IPmetric), horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes) # the ":.2f" rounds off at 2 decimal places
			#ax2.set_ylim(-0.007, 0.007)#(-0.015, 0.015)
			plt.xlabel('Row')
			if runningdir is not None: plt.savefig(runningdir+'testedIPon_{}.png'.format(fname[len(datdir)+1:len(datdir)+9]))
			plt.show()



	goodchi_inds = np.where( goodchi_mask == True )[0]

	if printing: print('len(f_list) = '+str(len(f_list)) ) 
	if printing: print('len(goodchi_inds) = '+str(len(goodchi_inds)) ) 
	if runningdir is not None:
		file_pars_chi.write('\n\nlen(f_list) = '+str(len(f_list)) )
		file_pars_goodchi.write('\n\nlen(goodchi_inds) = '+str(len(goodchi_inds)) )

		file_pars_chi.close()
		file_pars_goodchi.close()
		file_messages.close()

	fnames_header_arr = np.array( fnames_header ) # necessary in order to do list-indexing # For some reason, changing this to an array SOMETIMES changes the structure entirely ... as in the indexing leads to a new numpy record instead of a FITS file header. Perhaps, it may be when certain dates have keywords that others don't, this cause numpy to freak out because it doesn't want a non-rectangle matrix.
	dates_fnames = np.array( dates_fnames ) # necessary in order to do list-indexing

	if runningdir is not None:

		fig, (ax1, ax2, ax3) = plt.subplots(3,1, gridspec_kw = {'height_ratios':[2,3,1]}, figsize=(6,7), dpi=120, sharex=True, sharey=False)
		#fig.set_dpi(120)
		#fig.set_size_inches(6,8)

		pts1, = ax1.plot([], [], '+', markersize=plotmarkersize, label="{}".format(str_reciprocal) ) # the 'r' and the latex dollar signs are already within the variable
		ax1.set_ylabel('normalized counts')
		ax1.legend(loc=2)


		pts2a, = ax2.plot( [], [], '+', markersize=plotmarkersize, label="Data (aligned peaks)")
		pts2b, = ax2.plot( [], [], 'x', markersize=plotmarkersize, label="Model({})".format(str_freepars) )
		ax2.set_ylabel('counts')
		ax2.legend(loc=2)

		pts3, = ax3.plot( [], [], '+', markersize=plotmarkersize)
		ax3.axhline(y=0.0, linestyle='--', color='k')
		ax3.set_xlabel('row')
		ax3.set_xlim(xlim_left, xlim_right)
		metric_text = ax3.text(0.9, 0.85, r'', horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)

		fig.subplots_adjust(hspace=0)
		plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)

		def initialf():		
			metric_text.set_text('')
			pts1.set_data([],[])
			pts2a.set_data([],[])
			pts2b.set_data([],[])
			pts3.set_data([],[])

			return pts1, pts2a, pts2b, pts3, metric_text 



		def animate(i):
			x = rows_f[i] 
			y1 = counts_modelnormed_f[i]
			y2a = counts_f[i]
			y2b = counts_model_f[i]
			y3 = residuals_f[i]
			date_metric = metrics_files[i]
			fname = f_list[i]

			# set y-limits. If you don't, they will fix the limits based on the very 1st plot (which is the empty plot).

			#ax1.set_ylim(-0.03, np.max(y1)+0.03) # This works beautifully # if blit=True, this resetting of the limits would screw up the color for some reason. Or! the problem was with the dpi of the video itself!  
			ax1.set_ylim(-0.03, 0.4)

			#ax2.set_ylim(-1000, np.max( np.concatenate((y2a, y2b)) )+5000 ) # works beautifully
			ax2.set_ylim(0, np.max( np.concatenate((y2a, y2b)) )+200 )  # accounting for so many stellar spectra

			ax3.set_ylim(np.min(y3)- np.abs(np.min(y3))/5.0, np.max(y3) + np.max(y3)/5.0 ) # works beautifully 
			#ax3.set_ylim(-700 , 700)  # works the best based on eyeball estimate of all daytime sky spectra
			#ax3.set_ylim(-200 , 200)


			metric_text.set_text(r'${}$={:.2f}'.format(metricsign, date_metric)) #metric_text.set_text(r'$\chi^2$={0:.2f}'.format(date_metric))
			
			pts1.set_data(x - initcentroid, y1)
			ax1.set_title(r'Used IP$_{1st}$ to find optimal parameters of Model'+'({})\n{}'.format(str_freepars, fname[len(datdir):]) ) 
			pts2a.set_data(x - initcentroid, y2a)
			pts2b.set_data(x - initcentroid, y2b)
			pts3.set_data(x - initcentroid, y3)

			return pts1, pts2a, pts2b, pts3, metric_text

		#anim = animation.FuncAnimation(fig, animate, init_func=initialf, frames=len(fnames)-1, interval=10)# init_func=initialf, frames=len(f_list)-1, interval=300, blit=False)
		anim = animation.FuncAnimation(fig, animate, init_func=initialf, frames=goodchi_inds, interval=10, repeat=False) # blit doesn't seem to change anything. # hopefully, my code calculates the same chi_square for the Reference spectra twice and thus I have no need to show the plot of the zeroeth "goodchi_inds".  However, that likely won't happen because the reference spectra went through an iterative process to get its chi_squared.  
		#anim = animation.FuncAnimation(fig, animate,                               frames=len(f_list)-1, interval=300, blit=True, repeat=False)

		# save the animation as an mp4.  This requires ffmpeg or mencoder to be
		# installed.  The extra_args ensure that the x264 codec is used, so that
		# the video can be embedded in html5.  You may need to adjust this for
		# your system: for more information, see
		# http://matplotlib.sourceforge.net/api/animation_api.html
		anim.save(runningdir+'basic_animation_IP.mp4', fps=3, dpi=300,  extra_args=['-vcodec', 'libx264']) # fps=7
		plt.close()  # The initial and empty plot would have been shown. 


	# Keep going to Minerva Main to get more stellar and daytime sky spectra.

	'''
	if len(goodchi_inds) > 450:
		height = 200
	elif len(goodchi_inds) < 450:
		height = 140  # I don't think it actually goes this high anyway.
	elif len(goodchi_inds) < 350:
		height = 100
	elif len(goodchi_inds) < 150:
		height = 50
	elif len(goodchi_inds) < 75:
		height = 30
	elif len(goodchi_inds) < 40:
		height = 5

	f, axs = plt.subplots(len(goodchi_inds), sharex=True)  
	f.dpi = 200
	f.set_size_inches(7,height) # (width, height)
	plot_cnt = 0
	for f_ind in goodchi_inds:
		axs[plot_cnt].plot( rows_f[f_ind] - initcentroid, residuals_f[f_ind], '+', label=f_list[f_ind])
		axs[plot_cnt].axhline(y=0.0, linestyle='--', color='k')
		axs[plot_cnt].text(0.99, 0.8, r'${}$={:.2f}'.format(metricsign, metrics_files[f_ind]), fontsize=6, horizontalalignment='right', verticalalignment='center', transform=axs[plot_cnt].transAxes)
		axs[plot_cnt].set_ylim(-1*maxabsresidual, maxabsresidual) # just for prettiness and clarity of shape, but shouldn't be used.
		axs[plot_cnt].set_xlim(xlim_left, xlim_right) # Some files forced the "dx" paramater to go wild so this is necessary ideally but shouldn't be used.
		axs[plot_cnt].text(1.01, 0.5, f_list[f_ind], fontsize=9, horizontalalignment='left', verticalalignment='center', transform=axs[plot_cnt].transAxes)
		plot_cnt += 1

	axs[0].set_title('Residuals\n'+r'(IP$_{1st}$'+' came from reference spectra:{}--{})'.format(firstRefdate,lastRefdate))
	plt.xlabel('row')
	plt.savefig(runningdir+'/residuals_row.png')
	if plotting: plt.show() 
	else: plt.close() # The plt.close() is necessary in QT console when you do not want to show a plot.  In other words, merely leaving out "plt.show()" is not good enough.
	'''

	Expmeter_only, noExpmeter_date = variables['Expmeter_only'], variables['noExpmeter_date']

	dates_fnames_goodchis = dates_fnames[goodchi_mask]

	# plot chi-squared values
	sn.set_style('ticks',{'ytick.major.size':0,'ytick.minor.size':0, 'xtick.direction':'in'})
	plt.figure(dpi=150)
	if Expmeter_only:
		plt.plot_date(dates_fnames_goodchis[dates_fnames_goodchis > noExpmeter_date], metrics_files[goodchi_mask][dates_fnames_goodchis > noExpmeter_date], '+')
	else:
		plt.plot_date(dates_fnames_goodchis, metrics_files[goodchi_mask], '+')

	plt.plot_date(dates_refnames, metric_1st*np.ones(numRefs), 'r-')
	plt.xlabel('Date')
	plt.ylabel(r'${}$'.format(metricsign)+r'$_{reduced}$')
	plt.title('spline degree: '+str(splinedegree)+',   break pts interval: '+str(knotinterval))
	plt.gcf().autofmt_xdate()
	if runningdir is not None: plt.savefig(runningdir+'chisqr_date.png')
	if plotting: plt.show()
	else: plt.close() 


	# plot <centroid + dx>(t). The average is over crosscuts for one date.
	sn.set_style('ticks',{'ytick.major.size':0,'ytick.minor.size':0, 'xtick.direction':'in'})
	plt.figure(dpi=150)  
	if Expmeter_only:
		plt.plot_date(dates_fnames_goodchis[dates_fnames_goodchis > noExpmeter_date], crosscenters_files[goodchi_mask][dates_fnames_goodchis > noExpmeter_date], '+')
	else:
		plt.plot_date(dates_fnames_goodchis, crosscenters_files[goodchi_mask], '+')

	plt.xlabel('Date')
	plt.ylabel('row')
	plt.title('row = <centroid + dx> of all crosscuts')
	plt.gcf().autofmt_xdate()
	sn.plt.ticklabel_format(axis='y', style='plain', useOffset=False)
	#plt.ticklabel_format(axis='y', style='plain') # this style forces the numbers to not be in scientific notation, no matter how many digits they contain.
	#plt.gca().get_xaxis().get_major_formatter().set_useOffset(False) # now no offset will be used 
	#plt.gca().get_xaxis().get_major_formatter().set_scientific(False)
	if runningdir is not None: plt.savefig(runningdir+'row_date.png') 
	if plotting: plt.show()
	else: plt.close() 

	
	# plot all of the possible tracers of the problem via file headers
	'''
	-- TEMPA2 vs time  (and with all temp gauges)
	-- SPECPRES vs time
	-- I2TEMPA vs time
	-- *CLOUD vs time
	-- ALT1 (telescope #1) vs time  AND  AZ1 vs time . Maybe trees or the building or that lightning rod is in the image and if so, that would affect the profile ... not exactly sure how though.?

	tempa2 = np.array([ fnames_header[f_i]['TEMPA2'] for f_i in goodchi_inds ])
	tempa1 = np.array([ fnames_header[f_i]['TEMPA1'] for f_i in goodchi_inds ])
	specpres = np.array([ fnames_header[f_i]['SPECPRES'] for f_i in goodchi_inds ])
	i2tempa = np.array([ fnames_header[f_i]['I2TEMPA'] for f_i in goodchi_inds ])
	alt1 = np.array([ fnames_header[f_i]['ALT1'] for f_i in goodchi_inds ])
	az1 = np.array([ fnames_header[f_i]['AZ1'] for f_i in goodchi_inds ])
	hcloud = np.array([ fnames_header[f_i]['HCLOUD'] for f_i in goodchi_inds ])
	acloud = np.array([ fnames_header[f_i]['ACLOUD'] for f_i in goodchi_inds ])
	mcloud = np.array([ fnames_header[f_i]['MCLOUD'] for f_i in goodchi_inds ])
	mincloud = np.array([ fnames_header[f_i]['MINCLOUD'] for f_i in goodchi_inds ])

	tracers = [tempa2, tempa1, specpres, i2tempa, alt1, az1, hcloud, acloud, mcloud, mincloud]
	'''
	tracers_keyword = ['TEMPA2', 'TEMPA1', 'TEMPA3', 'TEMPA4', 'SPECPRES', 'I2TEMPA', 'ALT1', 'AZ1', 'HCLOUD', 'ACLOUD', 'MCLOUD', 'MINCLOUD']


	try:
		tracers_dict = {}

		# find a file that has comments for all keywords
		commentsdone = False
		c_cnt = 0
		while commentsdone == False:
			try:
				for kw in tracers_keyword: test = fnames_header[c_cnt].comments[kw]
				commentsdone = True
			except KeyError:
				c_cnt += 1

		# determine which files have ALL of the keywords
		header_mask = np.ones( len(f_list) ).astype(bool)
		for f_ind in range(len(f_list)):

			try:
				for kw in tracers_keyword:
					test = float( fnames_header[f_ind][kw] ) # Instead of a number, it can say "UNKNOWN".

			except KeyError:
				header_mask[f_ind] = False

			except ValueError: # if it says "UNKNOWN"
				header_mask[f_ind] = False


		fnames_header_masked = [ fnames_header[i] for i in range(len(f_list)) if header_mask[i] == True ] # "if header_mask is True" doesn't work for some reason.
		for kw in tracers_keyword:
			tracers_dict[kw] = { 'identity': fnames_header[c_cnt].comments[kw] , 'values': np.array([ fnames_header_masked[f_ind][kw] for f_ind in range(len(f_list[header_mask])) ]) }  # use any file to get the header keywords but the values should be specific
		'''
		for kw in tracers_keyword:
			tracers_dict[kw] = { 'identity': fnames_header[-1].comments[kw] , 'values': np.array([ fnames_header[f_ind][kw] for f_ind in range(len(f_list)) ]) }  # use any file to get the header keywords but the values should be specific
		

		if Expmeter_only:
			mask = dates_fnames > noExpmeter_date
		else:
			mask = dates_fnames > datetime.date(2015,1,1) # There are no files in/before 2015 so this is just a dummy-date.
		'''
		if Expmeter_only:
			mask = dates_fnames[header_mask] > noExpmeter_date
		else:
			mask = dates_fnames[header_mask] > datetime.date(2015,1,1) # There are no files in/before 2015 so this is just a dummy-date.

		for kw in tracers_keyword:
			sn.set_style('ticks',{'ytick.major.size':0,'ytick.minor.size':0, 'xtick.direction':'in'})
			plt.figure(dpi=100)
			plt.plot_date(dates_fnames[header_mask][mask], tracers_dict[kw]['values'][mask], '+') # here is where it might crash
			plt.xlabel('Date')
			plt.ylabel(tracers_dict[kw]['identity'])
			plt.title(kw)
			plt.gcf().autofmt_xdate()
			if runningdir is not None: plt.savefig(runningdir+kw+'_date.png')
			if plotting: plt.show()
			else: plt.close() 

	except KeyError: # because a file may not have the keyword, for example, n20170225.HD62613.0001.fits or '*daytimeSky.*' files
		# FOR each file name corresponding to goodchi_inds,
		# find 'daytimeSky.' in the string,
		# IF True,
		# re-run  tracers_dict[kw] = { 'identity': fnames_header[0].comments[kw] , 'values': np.array([ fnames_header[f_i][kw] for f_i in goodchi_inds ] IF f_i is not BADFILE) 
		
		pass

	except ValueError: # because a '*daytimeSky.*' file may have the keyword but not a value # because a '*daytimeSky.*' file may place the string "UNKOWN" in the place of the dictionary's 'value' for a specific keyword (e.g. 'TEMPA2').  Now only do Expmeter files.
		print('Because of some kind of error, I\'m only showing Expmeter files for the Tracers\' Plots')
		

		mask = dates_fnames[header_mask] > noExpmeter_date
		for kw in tracers_keyword:
			sn.set_style('ticks',{'ytick.major.size':0,'ytick.minor.size':0, 'xtick.direction':'in'})
			plt.figure(dpi=100)
			plt.plot_date(dates_fnames[header_mask][mask], tracers_dict[kw]['values'][mask], '+')
			plt.xlabel('Date')
			plt.ylabel(tracers_dict[kw]['identity'])
			plt.title(kw)
			plt.gcf().autofmt_xdate()
			if runningdir is not None: plt.savefig(runningdir+kw+'_date.png')
			if plotting: plt.show()
			else: plt.close() 



	return spl_mod_norm_mr_1st






# ========= Functions necessary for various IP models and even other functions ========


def gauss(x_vals, amp, centroid, gsigma, offset):
    return amp*np.exp(-(1.0/2.0)*((centroid-x_vals)/gsigma)**2)+offset


def gaussresidual(vars, x_vals, Data, eps_data, cent_LLimit, cent_ULimit):
    amp = vars[0]
    centroid = vars[1]
    gsigma = vars[2]
    offset = vars[3]

    if cent_LLimit is not None:
    	if amp < 0 or centroid < cent_LLimit or centroid > cent_ULimit: 
			return np.ones( len(Data) )*np.inf # this should stop horrible alignments

    model = gauss(x_vals, amp, centroid, gsigma, offset)

    #print( np.sum( ((Data - model)/eps_data)**2 ) )

    return (Data-model)/eps_data


def gaussfit(x, y, eps_y, guesses, c_LLimit=None, c_ULimit=None):
	vars = guesses
	out = leastsq(gaussresidual, vars, args=(x, y, eps_y, c_LLimit, c_ULimit)) 
	b_amp, b_centroid, b_gsigma, b_offset = tuple( out[0] )

	return {"b_amp":b_amp, "b_centroid":b_centroid, "b_gsigma":b_gsigma, "b_offset":b_offset}


def herm4th(x_vals, centroid):
	return 16*(x_vals-centroid)**4 - 48*(x_vals-centroid)**2 + 12

def herm3rd(x_vals, centroid):
	return 8*(x_vals-centroid)**3 - 12*(x_vals-centroid)



#=============== IP models =============================


class dx_N_b(object):

	strs_freepars_keys = ('dx', 'N', 'b')
	strs_freepars = ('dx', 'N', 'b')
	str_freepars = ",".join( strs_freepars )
	str_model = r'N$\times$IP$_{1st}$(x$_{1st}$ - dx) + b'
	str_reciprocal = '(Data - b)/N'
	

	def gen_normeddata(self, data, N, b):
		return (data - b)/N


	def modelspl(self, dx, N, b, x, interpnorm):
		# The interpnorm is some continous function.  shift the data's x-axis and then find the corresponding values within the normalized curve (i.e. interpolate). 
		# So that no extrapolation occurs, interpnorm should come from the "morerows" data while the Data itself is only ~11 rows long.

		x_shifted = x - dx   

		ysplin = interpnorm(x_shifted, nu=0, ext='raise') # thanks to the dx_max limit, I can use this and know that the error will never occur. The dx parameter merely used for potential skewing of the curves, i.e., correcting--in a minor way--the original centroid estimate.

		return  N*ysplin + b


	def splresidual(self, vars, x_vals, Data, eps_data, interpnorm, dxlim_l, dxlim_r):
		
		dx = vars[0]
		N = vars[1]
		b = vars[2]

		if (dx > 0 and dx > dxlim_l).all() or (dx < 0 and abs(dx) > dxlim_r).all() : return np.ones( len(x_vals) )*np.inf # now chunks that have only sky background noise in it won't be able to be modelled with a straight line and thus they should get a bad chi-squared value.

		model = self.modelspl(dx, N, b, x_vals, interpnorm)  # If dx > 0, you shift the data to the left, then interpolate across that x-axis, thus your (normalized) model (let's say the centroid of it for example) is now shifted to the right of it's original centroid, then the residuals are taken on the original x-axis which now has the model's centroid shifted to the right.

		return (Data - model)/eps_data # model came from the shifted x-axis of the Data. See explanation at the first model(dx,N,b)-fitting.


	def modelsplfit(self, x, y, eps_y, interpnorm, guesses, dxlims):
		vars = guesses
		out = leastsq(self.splresidual, vars, args=(x, y, eps_y, interpnorm, dxlims[0], dxlims[1]))
		dx, N, b = tuple( out[0] )
		return {'dx':dx, 'N':N, 'b':b}



class dx_N_b_c4_gsig(object):
	
	strs_freepars_keys = ('dx', 'N', 'b', 'c4', 'g_sigma')
	strs_freepars = ('dx', 'N', 'b', r'c$_4$', r'$\sigma$')
	str_freepars = ",".join( strs_freepars )
	str_model = r'c$_4$H$_4$G($\sigma$) + N$\times$IP$_{1st}$(x$_{1st}$ - dx) + b'
	str_reciprocal = r'(Data - b - c$_4$H$_4$G($\sigma$))/N'


	def gen_normeddata(self, data, N, b, c4, g_sigma, x, centroid, dx):
		#return ( data - b - c4*herm4th(x-dx, centroid-dx)*gauss(x-dx, 1.0, centroid-dx, g_sigma, 0.0) )/N
		x_n = np.sqrt(2)*x/g_sigma
		N4 = 1.0/np.sqrt(2**4 *np.math.factorial(4))
		return ( data - b - ( 2/(np.pi*g_sigma**2) )**(1.0/4)*c4*N4*herm4th(x_n-dx, centroid-dx)*gauss(x-dx, 1.0, centroid-dx, g_sigma/np.sqrt(2), 0.0) )/N


	def modelspl(self, dx, N, b, c4, g_sigma, x, centroid, interpnorm):
		# The interpnorm is some continous function.  shift the data's x-axis and then finding the corresponding values within the normalized curve (i.e. interpolate). 
		# So that no extrapolation occurs, interpnorm should come from the "morerows" data while the Data itself is only ~11 rows long.

		x_shifted = x - dx   

		ysplin = interpnorm(x_shifted, nu=0, ext='raise') # the 'raise' means to raise a ValueError if any extrapolation occurs # nu is the order of the derivative of the spline

		#return ( c4*herm4th(x_shifted, centroid-dx)*gauss(x_shifted, 1.0, centroid-dx, g_sigma, 0.0) + N*ysplin ) + b # the c4 significantly affects the amplitudes of the 3 middle bumps (in the gauss-hermite curve).  the g_sigma signficantly affects the amplitude of the outmost bump on each side (of the gauss-hermite curve).
		x_sh = np.sqrt(2)*x_shifted/g_sigma
		N4 = 1.0/np.sqrt(2**4 *np.math.factorial(4))
		return ( 2/(np.pi*g_sigma**2) )**(1.0/4)*(  c4*N4*herm4th(x_sh, centroid-dx)*gauss(x_shifted, 1.0, centroid-dx, g_sigma/np.sqrt(2), 0.0) + N*ysplin  ) + b


	def splresidual(self, vars, x_vals, Data, eps_data, centroid, interpnorm, dxlim_l, dxlim_r):
		
		dx = vars[0]
		N = vars[1]
		b = vars[2]
		c4 = vars[3]
		g_sigma = vars[4]


		if (dx > 0 and dx > dxlim_l).all() or (dx < 0 and abs(dx) > dxlim_r).all() : return np.ones( len(x_vals) )*np.inf # now chunks that have only sky background noise in it won't be able to be modelled with a straight line and thus they should get a bad chi-squared value.


		model = self.modelspl(dx, N, b, c4, g_sigma, x_vals, centroid, interpnorm)

		return (Data - model)/eps_data # model came from the shifted x-axis of the Data. See explanation at the first model(dx,N,b)-fitting.


	def modelsplfit(self, x, y, eps_y, centroid, interpnorm, guesses, dxlims):
		vars = guesses
		out = leastsq(self.splresidual, vars, args=(x, y, eps_y, centroid, interpnorm, dxlims[0], dxlims[1]))
		dx, N, b, c4, g_sigma = tuple( out[0] )
		return {'dx':dx, 'N':N, 'b':b, 'c4':c4 , 'g_sigma':g_sigma}



class dx_N_b_c4_c3_gsig(object):
	
	strs_freepars_keys = ('dx', 'N', 'b', 'c4', 'c3', 'g_sigma')
	strs_freepars = ('dx', 'N', 'b', r'c$_4$', r'c$_3$', r'$\sigma$')
	str_freepars = ",".join( strs_freepars )
	str_model = r'(c$_4$H$_4$+c$_3$H$_3$)$\times$G($\sigma$) + N$\times$IP$_{1st}$(x$_{1st}$ - dx) + b'
	str_reciprocal = r'(Data - b - (c$_4$H$_4$+c$_3$H$_3$)$\times$G($\sigma$))/N'


	def gen_normeddata(self, data, N, b, c4, c3, g_sigma, x, centroid, dx):
		#return ( data - b - ( c4*herm4th(x-dx, centroid-dx) + c3*herm3rd(x-dx, centroid-dx) )*gauss(x-dx, 1.0, centroid-dx, g_sigma, 0.0) )/N  # ydata = (c4*H4th+c3*H3rd*)Gauss + N*yspline_normalized + b, therefore do reciprocal to get new normalization
		x_n = np.sqrt(2)*x/g_sigma
		N4 = 1.0/np.sqrt(2**4 *np.math.factorial(4))
		N3 = 1.0/np.sqrt(2**3 *np.math.factorial(3))
		return ( data - b - ( 2/(np.pi*g_sigma**2) )**(1.0/4)*( c4*N4*herm4th(x_n-dx, centroid-dx) + c3*N3*herm3rd(x_n-dx, centroid-dx) )*gauss(x-dx, 1.0, centroid-dx, g_sigma/np.sqrt(2), 0.0) )/N

	def modelspl(self, dx, N, b, c4, c3, g_sigma, x, centroid, interpnorm):
		# The interpnorm is some continous function.  shift the data's x-axis and then finding the corresponding values within the normalized curve (i.e. interpolate). 
		# So that no extrapolation occurs, interpnorm should come from the "morerows" data while the Data itself is only ~11 rows long.

		x_shifted = x - dx   

		ysplin = interpnorm(x_shifted, nu=0, ext='raise') # the 'raise' means to raise a ValueError if any extrapolation occurs # nu is the order of the derivative of the spline

		#return (  ( c4*herm4th(x_shifted,centroid-dx)+c3*herm3rd(x_shifted,centroid-dx) )*gauss(x_shifted, 1.0, centroid-dx, g_sigma, 0.0) + N*ysplin ) + b # the c4 significantly affects the amplitudes of the 3 middle bumps (in the gauss-hermite curve).  the g_sigma signficantly affects the amplitude of the outmost bump on each side (of the gauss-hermite curve).
		x_sh = np.sqrt(2)*x_shifted/g_sigma
		N4 = 1.0/np.sqrt(2**4 *np.math.factorial(4))
		N3 = 1.0/np.sqrt(2**3 *np.math.factorial(3))
		return ( 2/(np.pi*g_sigma**2) )**(1.0/4)*(   ( c4*N4*herm4th(x_sh,centroid-dx)+c3*N3*herm3rd(x_sh,centroid-dx) )*gauss(x_shifted, 1.0, centroid-dx, g_sigma/np.sqrt(2), 0.0) + N*ysplin   ) + b


	def splresidual(self, vars, x_vals, Data, eps_data, centroid, interpnorm, dxlim_l, dxlim_r):
	
		dx = vars[0]
		N = vars[1]
		b = vars[2]
		c4 = vars[3]
		c3 = vars[4]
		g_sigma = vars[5]

		if (dx > 0 and dx > dxlim_l).all() or (dx < 0 and abs(dx) > dxlim_r).all() : return np.ones( len(x_vals) )*np.inf # now chunks that have only sky background noise in it won't be able to be modelled with a straight line and thus they should get a bad chi-squared value.

		model = self.modelspl(dx, N, b, c4, c3, g_sigma, x_vals, centroid, interpnorm)

		return (Data - model)/eps_data # model came from the shifted x-axis of the Data. See explanation at the first model(dx,N,b)-fitting.


	def modelsplfit(self, x, y, eps_y, centroid, interpnorm, guesses, dxlims):
		vars = guesses
		out = leastsq(self.splresidual, vars, args=(x, y, eps_y, centroid, interpnorm, dxlims[0], dxlims[1]))
		dx, N, b, c4, c3, g_sigma = tuple( out[0] )
		return {'dx':dx, 'N':N, 'b':b, 'c4':c4, 'c3':c3, 'g_sigma':g_sigma}



#=============== Miscellaneous ===========================


def metricfunc(D_minus_M, sigs, Npars, Ndatapts, chisqr=True ):
	if chisqr == True:
		return np.sum( (D_minus_M/sigs)**2 ) /( Ndatapts - Npars)
	else:
		return np.sqrt( np.mean(D_minus_M**2) - np.mean(D_minus_M)**2 ) # standard deviation


def mediansmoother(D, First_right_ind=2): 
	'''
	This median filter gets rid of outliers well and thus smooths the dataset if the number of elements is 3 (or any small odd number). Note that this function will intentionally return an array with two less elements than the original data.

	Parameters
	----------
	First_right_ind: int
					Placing it as "2" implies that you want the median between 3 data points repeatedly across the data set.

	Example (Credit: wikipedia)  
	---------------------------
	x = (2, 80, 6, 3)
	y_1 = med(2,2,80) = 2
	y_2 = med(2,80,6) = med(3,6,80) = 6
	y_3 = med(6,3,3) = med(3,3,6) = 3
	therefore, y = (2,6,6,3)
	'''

	left_ind, right_ind = 0, First_right_ind 
	D_smoothed = np.array([])
	while right_ind < len(D):  
		D_smoothed = np.concatenate(( D_smoothed, np.array([ np.median(D[left_ind:right_ind+1]) ]) ))
		left_ind += 1
		right_ind += 1

	return D_smoothed


def medianCRs(fitsfilename, datadir='', IOdir=''):
	# access data
	# access pickle file
	# everwhere there is a cosmic ray, median-smooth it (column-wise, of course)
	# return data


	fitsfile = fits.open(fitsfilename)
	data = fitsfile[0].data
	orig_data = copy.copy( data ) # most likely not necessary

	fitsfilename = fitsfilename[len(datadir):]
	
	picklefile = open(IOdir+fitsfilename+'.pickle', 'rb')
	dict_pickle = pickle.load(picklefile) 
	picklefile.close()
	allTs, allCRs = dict_pickle['allTs'], dict_pickle['allCRs']

	y_rows_N, x_cols_N = np.shape(data)

	for col in range(x_cols_N):
		if len(allCRs[col]) > 0: # I don't think this is necessary.
			for row_CR in allCRs[col]:
				if row_CR == 0 or row_CR == y_rows_N-1: # then keep the value as is.  It's most likely part of a trace.  Even if it isn't, don't take the very top or bottom of the file seriously.  It's a waste of time and energy to analyze such sketchy areas.
					pass
				else:
					data[row_CR, col] = mediansmoother(orig_data[row_CR-1:row_CR+2, col]) # 3 data points	

	return data



def tracktraces(perfect_filename='../n20170307.daytimeSkyExpmeter.0021.fits', ds9=True, datdir='../', IOdir=''):
	# make dictionary of all traces.  By eye, find the row where each trace starts and hard-code it.
	# Per trace, loop through every column of the file in order to make one list (because there will be None-values) for one trace indicating the row per column ... or just make an array in place a np.inf in place of the None-values.  
	# Per column, you can find that specific trace's row by saying WHERE is a row that is 1-below, the same, or 1-above the previous row-trace. # this doesn't work because the previous row-trace may be non-existent. Unless... I find a perfect file that has a blue circle at EVERY column for each trace!

	# consider the strange case of n20171026.daytimeSkyExpmeter.0033.fits.  It's freaky how at the very edge of only 1 trace, a column wasn't detected. Because of its near-perfection, consider making another function exactly like this one except making it for a perfect CHUNK rather than the entire file being perfect.


	# n20170307.daytimeSkyExpmeter.0021.fits # There are 4 traces at the very top of the file not being accounted for because they don't start at the first column. Well... just do np.ones() array to indicate the columns and then concatenate that for the additional traces that didn't start at the first column
	'''
	traces_rowstarts = np.array([ 2042, 2026, 2010, 1994, 1983, 1968, 1952, 1936, 1924, 1908, 1892, 1876,...
	 1864, 1848, 1832, 1816, 1803, 1787, 1771, 1755, 1741, 1725, 1709, 1693, 1678, 1662, 1646, ...
	 1631, 1614, 1599, 1583, 1567, 1550, 1534, 1518, 1503, 1484, 1469, 1453, 1437, 1418, ...
	 1402, 1386, 1371, 1350, 1335, 1319, 1303, 1282   ])
	'''
	
	perfect_filename_pre = perfect_filename	 
	perfect_filename = perfect_filename[len(datdir):]



	picklefile = open(IOdir+perfect_filename+'.pickle', 'rb') # this file is bright and the vetting gave each trace blue circles consistently throughout.
	allTs = pickle.load(picklefile)['allTs']
	picklefile.close()
	
	# find when the traces first have a blue circle (check the ds9 reg file) and this defines the beginning of the trace.  Thus, only traces with a blue circe will be available for analysis.
	
	traces_colstarts = np.concatenate(( np.zeros( len(allTs[0]) ), np.array([91, 418, 819, 1381 ])-1 )).astype(int) # the "-1" is converting DS9 to python values
	traces_rowstarts = np.concatenate(( np.array(allTs[0]), np.array([2047, 2047, 2047, 2047 ])-1 )).astype(int)  # this method is HEAVILY dependent on the success of column-zero. But it doesn't have to be...
	# Jason mentioned that we don't care about the traces that don't expand the width of the fits file.  This means the 4 traces at the very top and 2 traces at the very bottom
	# Looking at the processed spectra and the "minerva_all" file it's clear that Matt chose their to be 29 orders in the FITSs files. The processed files' orders are reversed from the raw spectra.  The top/largest order (index 28) in the 1-D spectra corresponds to the lowest rows of the 2-D raw spectra.  The telescopes, similarly, go from top to bottom with respect to the 2-D raw spectra, thus telescope #1 start at the top and #4 is at the bottom. In other words, the orders are highest at the bottom of the 2-D raw spectra, and the telescopes are highest at the bottom of each order (within the 2-D raw spectra).  The orders however are highest at the top of the 1-D processed spectra. 
	traces_colstarts = traces_colstarts[0:-4] # this corresponds with the processed 1-D spectra orders. # if this ever changes, you might need to re-run this function for all of the pickle files with the "vet_perfection" function
	traces_rowstarts = traces_rowstarts[0:-4] # ideally there are 114 in 2016 and 113 traces in 2017.

	limit = 113 #110 # give some wiggle room because the traces are shifted by ~8 pixels in 2016 and ~2 in early 2017 when compared to late 2017
	if len(traces_colstarts) < limit: # != 113: 
		raise KeyError("There are less than {} traces detected at column-zero. Ideally, there are 113 traces (in 2017) when all 4 telescopes are used.".format(limit)) # 




	#-------- define the traces and their circles in a dictionary --------

	fitsfile = fits.open(perfect_filename_pre)
	data = fitsfile[0].data
	y_rows_N, x_cols_N = np.shape(data)  # therefore the indexing show go as data[row, column]

	traces_dict = {}
	for T_i in np.arange(len(traces_rowstarts)): 
		if T_i > 1: # The lowest 2 trace IDs (when they are T1 and T2) are included in the processed 1-D spectra but since they are cut off, no files will ever be considered "perfect" if I include that when raising the Error # Sometimes (for instance, 2017 daytime sky spectra) the lowest ID isn't detected because it is so low now, and therefore the lowest 2 traces become T4 and T1 traces instead of T1 and T2 traces. But even this new T4 trace is too low and would be cut off and thus ruin the "perfection."
			trace = [ traces_rowstarts[T_i] ] # the first row in the trace is now accounted for
			colstart = traces_colstarts[T_i]
			for col in range(colstart+1, 2048): #x_cols_N): # the "+1" is due to the first one already being accounted for.
				result_inds = np.where( ( allTs[col] >= trace[col-1 -colstart] - 2 )&( allTs[col] <= trace[col-1 -colstart] + 2 ) )[0] # there should only be 1 result anyway. This is the row of the trace at this column

				'''
				if len(result) == 0: # then an absorption feature caused the trace to not be marked, or it's in overscan region.
					trace.append( -np.inf )
				else:
					trace.append( result[0] ) 
				'''

				if len(result_inds) == 0: raise KeyError("One trace/order of this file is missing 1 or more detected columns: trace_#T{}, column={}".format(T_i+1,col)) #break # breaks only the most recent for loop # there is an absorption feature causing the trace to not be marked, or it's in the overscan region.  Because this is a "perfect" file, even the absorption features are bright enough for the original function to pick up the trace-position so this loop will definitely not break until the overscan region is reached. # This break in the loop was only meant for the overscan region, but it turns out that this break helps continue files that are not in face "perfect".  What is happening is that files that have a trace with 1 missing column will continue but that trace will not be included in the final dictionary. Furthermore, because I always create the "trace" list and input the first value, this program never crashes even if there is definitely no trace (for the hard-coded values). As a result, it's label the traces that at least have a start at column-zero (in addition to the traces I hard-coded). NOW, I'm going to have the FOR loop stop at the overscan region.  This way, I can intentionally crash the program with a "raise" instead of merely "break"ing the FOR loop. 

				trace.append( allTs[col][ result_inds[0] ] )  # if the column-loop "breaks" before this then the 

			Tcolumns = np.arange( colstart, colstart+len(trace) )
			traces_dict['trace_#'+str(T_i+1)] = {'cols': Tcolumns, 'rows': np.array(trace)}
	

	#------- Make a DS9 region file of the starts of each trace ---------
	if ds9:

		regfile = open(IOdir+'traces_starts.reg', 'w')
		prologue = '# Region file format: DS9 version 4.1\nglobal dashlist=8 3 width=1 font="helvetica 12 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\nimage\n'
		regfile.write(prologue)

		for starter in range(len(traces_colstarts)):
			#regfile.write(  "\ntext {x} {y} # text={{txt}}".format(  x=str(traces_colstarts[starter]+6), y=str(traces_rowstarts[starter]+1), txt='Trace_#'+str(starter+1)  )  ) # something I imported makes this not work.
			regfile.write(  "\ntext {x} {y} ".format(  x=str(traces_colstarts[starter]+
				+1+45), y=str(traces_rowstarts[starter]+1-2)  )+"# text={Trace_#"+str(starter+1)+"}"  )

		regfile.close()


	return traces_dict



def choose_perfectfile(perfectnames, spectra_date, getrows=True, getall=False, ID=None, columns=None, rplus=None, rminus=None, plusNhalfrows=None, datdir='', IOdir=''):
	# based on a date, choose a file to act as the current "perfect" file. This is necessary because the traces are shifted by ~8 pixels in 2016 and ~2 in early 2017 when compared to late 2017.  Furthermore, my average-centroid plot showed that the traces even gradually change over months-long timescales too.
	# you could be really picky and account for the fits-file number too but that's unnecessary.

	# I could find the perfect-date that is right after the date of interest, or I could fine the perfect-date that is in the same month as the date of interest, or I could find the perfect-date that is right before the date of interest. Due to the abrupt and gradual nature of the shifting traces, I see no benefit in picking one of these ideas over the other. # Well... if you know the spectra_date is definitely somewhere in the perfect-names array then calling it a quits when the dates are equal is a good idea.
	for i in range(len(perfectnames)):
		best_filename = perfectnames[i] # filenames given should already by sorted
		perfectdate = name2date( perfectnames[i], datdir )
		if perfectdate >= spectra_date: break # break loop when the perfect-date is equal to or right after the date of interest.  If the date of interest is too recent for the chosen perfect-files then the perfect-date will be right before the date of interest. # this means that the file-number doesn't matter and thus the first file of a date is always used.



	if getrows is not True:
		return best_filename

	perfect_traces = tracktraces(best_filename, ds9=False, datdir=datdir, IOdir=IOdir )  # The trace for T2 at the very bottom of the file may not be deemed ID #1 because it's 1st column does not have a blue circle (in 2017 "perfect" files).  Therefore, assume the ID given is always for the 113 traces case (i.e. the 2016 case) and thus if this specific "perfect" file has 114 traces then account for the ID discrepancy.  Remember though, the first order at the bottom every FITS file start the ID numbering but those two traces (T1 and, if present, T2) are not present in the actual array. Therefore the array is often 112 elements long while the ID numbers start from ID_#3 and go up to ID_#114.
	if len(perfect_traces.keys()) == 111: match_ID = ID  
	if len(perfect_traces.keys()) == 112: 
		firstmidrows = np.array([ perfect_traces[ 'trace_#3' ]['rows'][0-0] , perfect_traces[ 'trace_#4' ]['rows'][0-0] ])
		if (firstmidrows > 60).all(): match_ID = ID+1 # Example: the ID given (which assumes 113 total traces) is 5. If this perfect file has 114 total traces due to the bottom one being detected, then the corresponding trace is ID_#6.  This crazy scenario happened when finding a "perfect" file for n20160512.HD122064.001[456].fits.  Of course it happened more but this problem was detected thanks to T2 being inactive in that stellar spectra.
		# else: then 1 trace has appeared at the very top of the file while the T2 trace at the very bottom is gone. I have yet to notice this happen though.  

	firstcol = perfect_traces[ 'trace_#'+str(match_ID) ]['cols'][0] # this is for those ~5 traces that do not start at the first column of a FITS file, but this of course works for all other traces too.
	midrows = perfect_traces[ 'trace_#'+str(match_ID) ]['rows'][np.array(columns)-firstcol] # these are the blue circles 

	'''
	Nrows = variables['Nrows'] 
	rowminus = np.floor(float(Nrows)/2).astype(int)
	rowplus = np.ceil(float(Nrows)/2).astype(int) # this works if Nrows is even too.
	#LRrows = np.ceil(Nrows/2).astype(int)
	addtoNhalfrows = variables['addtoNhalfrows']
	Nmorerows = Nrows + 2*addtoNhalfrows
	'''

	rows_c = []
	morerows_c = []
	for c_i in range(len(columns)):
		rows_c.append( np.arange(midrows[c_i]-rminus, midrows[c_i]+rplus).astype(int) )
		morerows_c.append( np.arange(midrows[c_i]-rminus-plusNhalfrows, midrows[c_i]+rplus+plusNhalfrows).astype(int) ) 

	if getall: return midrows, rows_c, morerows_c, best_filename
	return midrows, rows_c, morerows_c




def pickle_ext(filenames):
	# Get rid of the ".pickle" extension

	if type(filenames) == str or type(filenames) == np.string_:
		filename = filenames
		return filename[0:-7]

	fnames=[]
	for i in range(len(filenames)):
		fnames.append( filenames[i][0:-7] )

	return np.array(fnames)


def extension(filenames, ext):
	'''
	e.g.,    ext='.pickle'
	'''
	Nchars = len(ext)

	if type(filenames) == str or type(filenames) == np.string_:
		filename = filenames
		return filename[0:-Nchars]

	fnames = [] # freaky things happen when you just simply replace the original value in the string array, so do this instead.
	for i in range(len(filenames)):
		fnames.append( filenames[i][0:-Nchars] )

	return np.array(fnames)


def name2date(filenames, fdir=''):
	# make date time object from just the fits file's name
	if type(filenames) == str or type(filenames) == np.string_:
		filename = filenames
		filename = filename[len(fdir):]
		return datetime.date( int(filename[1:5]), int(filename[5:7]), int(filename[7:9]) )
	
	dates = []
	for filename in filenames:
		filename = filename[len(fdir):]
		dates.append(  datetime.date( int(filename[1:5]), int(filename[5:7]), int(filename[7:9]) )  )

	return np.array( dates )
	


def choose_indsMonth(date_array, NperMonth):
	# make sure the date np.array is already sorted

	def dt_days(month, year):

		if month == 2:
			days_mo = 28
			if year%4 ==0:
				days_mo = 29
		else:
			'''
			if x.month <=7:
				elif x.month%2 == 1: # odd month		
					days_mo = 31 
				elif x.month%2 == 0:  
					days_mo = 30
			else:
				elif x.month%2 == 1: # odd month		
					days_mo = 30 
				elif x.month%2 == 0:  
					days_mo = 31
			'''
			if month%2 == 1: # odd month
					days_mo = 31 - int(month/8) # from August onward, the days per month flip.
			elif month%2 == 0:  
					days_mo = 30 + int(month/8) # from August onward, the days per month flip.


		dt = datetime.timedelta(days = days_mo)
		return {'dt':dt, 'days_mo':days_mo}


	inityear, initmonth = date_array[0].year, date_array[0].month

	if initmonth > 1:
		llimit = datetime.date(inityear, initmonth-1, dt_days( initmonth-1, inityear )['days_mo'] )
	elif initmonth == 1:
		llimit = datetime.date(inityear-1, 12, 31)

	if initmonth < 12:
		ulimit = datetime.date(inityear, initmonth+1, 1)
	elif initmonth == 12:
		ulimit = datetime.date(inityear+1, 1, 1)


	file_inds  = np.array([])
	todaytime  = datetime.datetime.utcnow() # the file names were made using UTC time zone.
	todaydate  = datetime.date(todaytime.year, todaytime.month, todaytime.day)
	while ulimit < todaydate: # essentially, only when the upper limit's month is greater than today's month will this WHILE loop be broken.

		month_inds = np.where( (date_array > llimit) & (date_array < ulimit) )[0] # even if this returns no results, the concatenation will handle an empty np.array well.    
		if len(month_inds) > NperMonth: # now reduce the number of indices
			month_inds = np.random.choice( month_inds, NperMonth, replace=False)

		# now increment forward by a full month

		llim_1day = llimit + datetime.timedelta(days=1)
		dt_llimit = dt_days( llim_1day.month, llim_1day.year )['dt']

		dt_ulimit = dt_days( ulimit.month, ulimit.year )['dt']

		llimit = llimit + dt_llimit
		ulimit = ulimit + dt_ulimit

		file_inds = np.concatenate(( file_inds, month_inds ))

	return file_inds.astype(int)


def find_dxlims(x_crosscut, x_IP):
	left_diff = np.min(x_crosscut) - np.min(x_IP)
	right_diff = np.max(x_IP) - np.max(x_crosscut)

	return left_diff, right_diff


def minervaDirs(filenames, leading='/Data/kiwispec/'):
	# in this (special) case, filenames do not include path-names because this function will find the path and insert it.
	
	nchars = len('n20171230') # just a mere example is necessary.
	''' This doesn't work because the array is already set and each string has a maximum limit of characters allowed.
	for i in range(len(filenames)):
		filenames[i] = leading+filenames[i][0:nchars]+'/'+filenames[i]

	return filenames
	'''
	newnames = []
	for i in range(len(filenames)):
		newnames.append( leading+filenames[i][0:nchars]+'/'+filenames[i] )

	return np.array( newnames )


def myDirs(filenames, path='../'):
	# in this (special) case, filenames do not include path-names because this function will find the path and insert it.
	
	newnames = []
	for i in range(len(filenames)):
		newnames.append( path+filenames[i] )

	return np.array( newnames )
