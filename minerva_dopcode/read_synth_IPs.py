# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 13:13:36 2016

@author: johnjohn
"""
import numpy as np
from scipy.interpolate import interp1d

pre = 'synthetic_IPs/'
ffile = pre+'files.txt'

files = np.loadtxt(ffile, dtype='string')
ipdict_blank = {'xip':np.array([]), 'ip':np.array([]), 'wav':0., 'order':0.}
nfiles = len(files)
osamp = 50.0
ipdict = [dict() for x in range(nfiles)]
for i, f in enumerate(files):
    ipa = np.loadtxt(pre+f)
    wav = f.split('lam')
    wav = float((wav[1].split('.dat'))[0])
    order = int((f.split('.'))[2])
    ipdict_blank = {'xip':np.array([]), 'ip':np.array([]), 'wav':0., 'order':0.}
    thisipdict = ipdict_blank
    thisipdict['wav'] = wav*10
    thisipdict['order'] = order
    xip_fine = (ipa[:,0] - np.mean(ipa[:,0]))/10.
    ip_fine  = ipa[:,2]
    xip = np.arange(-4.75,5.,1/osamp)
    ifunc = interp1d(xip_fine, ip_fine)
    ip = ifunc(xip)
    thisipdict['xip'] = xip
    thisipdict['ip'] = ip/ip.sum()/osamp
    ipdict[i] = thisipdict

np.save(pre+'ipdict.' + str(int(osamp)).zfill(3) + '.npy', ipdict)
