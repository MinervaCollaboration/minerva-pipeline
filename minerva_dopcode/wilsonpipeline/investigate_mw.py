
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import vankindp_mw as vank
import ipdb
import pickle
#import matplotlib.cm as cm
import ipdb


class study(object):

	outpath = './Investigating/'

	def weights(self, ps = np.arange(0.0, 100.0, 1.0), obsperc=0.5, objectname = 'HD122064' ):

	
		def rejects(objname, weightperc=0.0, chiperc=0.0, sigmaiperc=0.0, obsperc=0.5):
			rms = vank.vank(objname=objname, weightthresh=weightperc, nobslim=obsperc, plot=False)['rms']
			return rms 


		rms_p = {  'All': np.ones( len(ps) )*-np.inf, 'T1': np.ones( len(ps) )*-np.inf, 'T2': np.ones( len(ps) )*-np.inf, 'T3': np.ones( len(ps) )*-np.inf, 'T4': np.ones( len(ps) )*-np.inf  }

		ts = ['All', 'T1', 'T2', 'T3', 'T4']
		
		#colors = ['','r','g','b','orange'] # same as in vank code
		colors = ['k','r','g','b','orange'] # same as in vank code

		#plt.figure()
		for p_i in range(len(ps)):
			rms = rejects( objname = objectname, weightperc = ps[p_i] , obsperc = obsperc) # this is creating figures so this interrupts this script's plots
			print( 'p was {}'.format(ps[p_i]) ) 
			for c_i in range(len(colors)):
				#plt.plot( p, rms[t] , 'o', color=colors[c_i] )
				rms_p[ ts[c_i] ][p_i] = rms[ ts[c_i] ]


		plt.figure()
		inds = np.zeros(5).astype(int)
		for c_i in range(len(colors)):
			plt.plot( ps , rms_p[ ts[c_i] ], color=colors[c_i], label=ts[c_i] ) 
			ind = np.where( rms_p[ ts[c_i] ] == np.min( rms_p[ ts[c_i] ] ) )[0]
			plt.plot( ps[ind], rms_p[ ts[c_i] ][ind] , 'o', color=colors[c_i] )
			inds[c_i] = ind

		plt.xlim(ps[0]-2 , ps[-1]+2)
		plt.legend( bbox_to_anchor=(0.97, 1.0), loc=2, borderaxespad=0., framealpha=0.7 )
		plt.title( objectname )
		plt.xlabel( 'Weight threshold\nT1: {},    T2: {},   T3: {},    T4: {},    All: {}'.format( round(rms_p[ ts[1] ][inds[1]],2), round(rms_p[ ts[2] ][inds[2]],2), round(rms_p[ ts[3] ][inds[3]],2), round(rms_p[ ts[4] ][inds[4]],2), round(rms_p[ ts[0] ][inds[0]],2) ) )
		plt.ylabel('Nightly binned RMS (m/s)')
		plt.savefig('{}investigate_weights_{}.png'.format(self.outpath, objectname))
		plt.close()


		pfile = open( '{}investigate_weights_{}.pickle'.format(self.outpath, objectname) , 'wb')
		pickle.dump( rms_p , pfile )
		pfile.close()


	def chunkcheck(self, weightperc, obsperc, yl = 'RVs', histo=True, hbins=150, obss = None, objname = 'HD122064'):
			
		stuff = vank.vank(objname=objname, weightthresh=weightperc, nobslim=obsperc, plot=False)
		
		if yl is 'RVs':
			#chunkij, badwt_f_t = stuff['chunkrvs_vij'], stuff['badwt_f_t']
			chunkij = stuff['chunkrvs_vij']
			fID = 'rv'

		elif yl is 'RVerrs': 
			chunkij = stuff['chunkerrs_sigmaij']
			fID = 'rverr'

		elif yl is 'Weights':
			chunkij = stuff['chunkweights_c_f']
			fID = 'wts'


		if histo: chunk_t = {}

		for t_i in [0,1,2,3]:

			if obss is None: # then use all observations 
				obss = range(len( chunkij['T'+str(t_i+1)].T ))
			elif type(obss) is int:
				obss = range(obss)
			#  else: use the obss given as an array (full of indices) 

			#incr = 1.0/len(rvs_time)
			plt.figure()
			for obs_i in obss:

				mask = np.isfinite( chunkij['T'+str(t_i+1)][: , obs_i] )
				mask_inds = np.where( mask == True )[0]  
				#plt.plot( range(len( chunkij['T'+str(t_i+1)][: , obs_i][mask] )), chunkij['T'+str(t_i+1)][: , obs_i][mask], '+' ) #, color=cm.rainbow(incr*obs_i) )
				#plt.plot( chunk_inds[obs_i]['T'+str(t_i+1)][mask] , chunkij['T'+str(t_i+1)][: , obs_i][mask], '+' ) #, color=cm.rainbow(incr*obs_i) )
				plt.plot( mask_inds , chunkij['T'+str(t_i+1)][: , obs_i][mask], '+' ) #, color=cm.rainbow(incr*obs_i) )

			plt.xlabel('chunks')
			plt.ylabel(yl)
			plt.title(objname+'   T'+str(t_i+1)+'   Wt: '+str(weightperc)+'%')
			plt.savefig( '{}investigate_wt{}_{}_T{}_{}.png'.format(self.outpath, weightperc, fID, t_i+1, objname) )
			plt.close()

			
			if histo:
				chunk_t['T'+str(t_i+1)] = np.array([])
				for obs_i in obss:
					chunk_t['T'+str(t_i+1)] = np.concatenate(( chunk_t['T'+str(t_i+1)], chunkij['T'+str(t_i+1)][: , obs_i] ))

				mask = np.isfinite( chunk_t['T'+str(t_i+1)] )
				plt.figure()
				plt.hist( chunk_t['T'+str(t_i+1)][mask], bins=hbins )
				plt.title(objname+'   T'+str(t_i+1)+'   Wt: '+str(weightperc)+'%')
				plt.xlabel(yl)
				plt.ylabel('# w/in chunks & observations')
				plt.savefig( '{}investigate_hist_wt{}_{}_T{}_{}.png'.format(self.outpath, weightperc, fID, t_i+1, objname) )
				plt.close()
				


	def thresholds(self, wt_T=np.arange(0.0, 100.0, 1.0), c_T=1.0, s_T=1.0, obsperc=0.5, objectname = 'HD122064', skewpar=False ):

		def rejects(objname, kwargs=None):
			rms = vank.vank(objname=objname, **kwargs)['rms']
			return rms

 
		kwargs = {"weightthresh": wt_T, "chithresh": c_T, "sigmaithresh": s_T, "nobslim": obsperc, "plot": False, "skewpar": skewpar}
		try: 
			test = len(wt_T)
			parname =  "weightthresh"
			ps = wt_T
			Ts_str = "chithresh={},   sigmaithresh={},   obsperc={}".format(c_T, s_T, obsperc)   
		except:
			pass
		try: 
			test = len(c_T)
			parname =  "chithresh"
			ps = c_T
			Ts_str = "weightthresh={},   sigmaithresh={},   obsperc={}".format(wt_T, s_T, obsperc)   
		except:
			pass
		try: 
			test = len(s_T)
			parname =  "sigmaithresh"
			ps = s_T
			Ts_str = "weightthresh={},   chithresh={},   obsperc={}".format(wt_T, c_T, obsperc)   
		except:
			pass
		try: 
			test = len(obsperc)
			parname =  "nobslim"
			ps = obsperc
			Ts_str = "weightthresh={},   sigmaithresh={},   chithresh={}".format(wt_T, s_T, c_T)   
		except:
			pass


		rms_p = {  'All': np.ones( len(ps) )*np.inf, 'T1': np.ones( len(ps) )*np.inf, 'T2': np.ones( len(ps) )*np.inf, 'T3': np.ones( len(ps) )*np.inf, 'T4': np.ones( len(ps) )*np.inf  }

		ts = ['All', 'T1', 'T2', 'T3', 'T4']
		
		#colors = ['','r','g','b','orange'] # same as in vank code
		colors = ['k','r','g','b','orange'] # same as in vank code

		#plt.figure()
		for p_i in range(len(ps)):
			kwargs[parname] = ps[p_i]
			try:
				rms = rejects( objname = objectname, kwargs = kwargs) # this is creating figures so this interrupts this script's plots
				print( 'p was {}'.format(ps[p_i]) ) 
				for c_i in range(len(colors)):
					#plt.plot( p, rms[t] , 'o', color=colors[c_i] )
					rms_p[ ts[c_i] ][p_i] = rms[ ts[c_i] ]
			except: # sigma threshold causes problems
				pass

		plt.figure()
		inds = np.zeros(5).astype(int)
		besties = np.zeros(5)*np.nan
		for c_i in range(len(colors)):
			#T_i = (c_i+1) % 5
			realmask = np.isfinite( rms_p[ ts[c_i] ] )
			if (realmask == False).all() or len(realmask)==0:
				besties[c_i] = np.nan
			else:
				plt.plot( ps[realmask] , rms_p[ ts[c_i] ][realmask], color=colors[c_i], label=ts[c_i] ) 
				ind = np.where( rms_p[ ts[c_i] ] == np.nanmin( rms_p[ ts[c_i] ] )  )[0]
				if np.shape( ind ) != () : ind = ind[0] # just in case 2 equivalent minima appear
				plt.plot( ps[ind], rms_p[ ts[c_i] ][ind] , 'o', color=colors[c_i] )
				inds[c_i] = ind
				
				besties[c_i] =  round(rms_p[ ts[c_i] ][inds[c_i]],2) #round(rms_p[ ts[T_i] ][inds[T_i]],2)


		if parname is "nobslim": plt.xlim(ps[0]-0.1 , ps[-1]+0.1)
		else: plt.xlim(ps[0]-2 , ps[-1]+2)
		plt.legend( bbox_to_anchor=(0.97, 1.0), loc=2, borderaxespad=0., framealpha=0.7 )
		plt.title( "{}\n {}".format(objectname, Ts_str) )
		#plt.xlabel( '{}\nT1: {},    T2: {},   T3: {},    T4: {},    All: {}'.format( parname, round(rms_p[ ts[1] ][inds[1]],2), round(rms_p[ ts[2] ][inds[2]],2), round(rms_p[ ts[3] ][inds[3]],2), round(rms_p[ ts[4] ][inds[4]],2), round(rms_p[ ts[0] ][inds[0]],2) ) )
		plt.xlabel( '{}\nT1: {},    T2: {},   T3: {},    T4: {},    All: {}'.format( parname, besties[1], besties[2], besties[3], besties[4], besties[0] ) )
		plt.ylabel('Nightly binned RMS (m/s)')
		plt.savefig('{}investigate_{}_{}.png'.format(self.outpath, parname, objectname))
		plt.close()


		pfile = open( '{}investigate_{}_{}.pickle'.format(self.outpath, parname, objectname) , 'wb')
		pickle.dump( rms_p , pfile )
		pfile.close()



	def a_v(self, objectname='HD122064', skewpar=False, cbinning=4):
		
		av_stuff =  vank.vank(objname=objectname, skewpar=skewpar, cbinning=cbinning)['av']

		av_file = open( '{}investigate_allanvariance_{}.pickle'.format(self.outpath, objectname) , 'wb')
		pickle.dump(av_stuff, av_file)
		av_file.close()
		
		return av_stuff


