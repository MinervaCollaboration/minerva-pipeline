# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 15:29:16 2016

@author: johnjohn
"""
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as pl
from scipy.signal import medfilt

def fun(x, p):
    return p[0] + p[1]*x
    
def fo_loglike(p, *args):
    resid = args[0]
    #resid = p[0] - y
    sigma = np.abs(p[0])
    f = np.abs(p[1])
    s = np.abs(p[2])
    norm = 1./(sigma*np.sqrt(2*np.pi))
    part1 = f/s * np.exp(-0.5 * (resid/(s*sigma))**2)
    part2 = (1-f) * np.exp(-0.5 * (resid/sigma)**2)
    loglike = np.log(norm * (part1 + part2))
    ll = -np.sum(loglike)
    #print p, ll
    return ll
    
def find_outliers(resid):
    x = np.arange(len(resid))
    bnds = ((1e-8, None),
            (0.,1.),
            (1., None))
    #sig0 = np.std(medfilt(resid))
    p0 = np.array([3e-7, 0.03, 5])
    result = minimize(fo_loglike, p0, (resid), method='SLSQP', bounds=bnds, options={'maxiter':50})
    bp = np.abs(result.x)
    bad = np.where(np.abs(resid) > (3*bp[0]))
    nbad = len(bad[0])
    print bad
    pl.plot(resid,'bo')
    pl.plot(x[bad], resid[bad], 'ro')
    pl.show()
    return result
        
x = np.arange(128)

def test(ntrial):
    fail = 0.
    for i in range(ntrial):
        x = np.arange(128)
        noise = np.random.randn(len(x))*0.3
        data = noise 
        bad = [10,69,70,71,73]
        data[bad] += np.random.randn(len(bad))*50
        r = find_outliers(data)
        if not r.success:
            fail += 1
    print 'Failure rate: ', fail/ntrial
    return

'''
x = np.arange(128)
if True:
    noise = np.random.randn(len(x))*0.3
    data = noise 
    bad = [10,69,70,71,73]
    data[bad] += np.random.randn(len(bad))*50

bnds = ((1e-8, None),
        (0.,1.),
        (1., None))
        
p0 = [0.1, 0.2, 200.]
pnames = ['sigma', 'frac', 'sigma_bad']
result = minimize(fo_loglike, p0, (data), method='SLSQP', bounds=bnds, options={'maxiter':50})

bp = find_outliers(data)
print bp


bp = result.x
for i, par in enumerate(bp):
    print pnames[i], ':', np.round(par, decimals=2)
pl.plot(x,data,'o')
mod = np.zeros(len(x)) + bp[0]

resid = mod - data
wbad = np.where(np.abs(resid) > (3*bp[1]))

pl.plot(x, mod)
pl.plot(x[wbad], data[wbad], 'ro')
pl.plot(x, np.zeros(len(x))+np.mean(data), 'r')
pl.show()
'''