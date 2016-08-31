import four_chunk_jj8 as fch
from timeit import default_timer as timer
import ipdb
import numpy as np
import corner

obfile = '/Users/xuesongwang/Downloads/minerva_temp/hackweek/n20160619.HD122064.0014.proc.fits'

spec = fch.Spectrum(obfile)

oversamp = [4.0, 6.0, 8.0, 10.0, 20.0]
oversamp = [4.0]

"""
for this_os in oversamp:
    chains = np.load('testmc_data/chain'+str(int(this_os))+'.npy')
    chunkpar = np.load('testmc_data/chunkpar'+str(int(this_os))+'.npy')

    for tel in range(0, 4):
        fig = corner.corner(chains[tel].reshape((-1, 7)), labels=["z","w0","dw",'a','b','sigma','alpha'],
                            truths=chunkpar[7*tel:7*(1+tel)])
        fig.savefig('testmc_data/triangle'+str(int(this_os))+'_'+str(int(tel))+'.png')

"""
for this_os in oversamp:
    chunks = fch.FourChunk(spec, 8, 1344, 128, oversamp=this_os)  # chunk 100, looks like a good one
    startT = timer()
    samplers = chunks.emcee_fitter(20, 5000, 1000)
    endT = timer()
    print 'testmc total time for oversamp '+str(this_os)+': ', (endT-startT)/60., 'minutes'

    # save the chains
    np.save('testmc_data/chain'+str(int(this_os))+'.npy',
            [samplers[0].chain, samplers[1].chain, samplers[2].chain, samplers[3].chain])
    np.save('testmc_data/chunkpar'+str(int(this_os))+'.npy', chunks.par)

    # triangle plot
    for tel in range(0, 4):
        fig = corner.corner(samplers[tel].flatchain, labels=["z","w0","dw",'a','b','sigma','alpha'],
                            truths=chunks.par[7*tel:7*(1+tel)])
        fig.savefig('testmc_data/triangle'+str(int(this_os))+'_'+str(int(tel))+'.png')


ipdb.set_trace()


