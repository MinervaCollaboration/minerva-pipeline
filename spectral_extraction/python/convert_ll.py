#!/usr/bin/env python 2.7

#To transfer lamp lines from table1.dat to the format for specex

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

#lamp_lines = np.loadtxt('table1.dat',
#                dtype={'formats':(np.float,np.str)})
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
#valid_max = False
#while not valid_max:
#    idx = np.argmax(its)
#    wl_max = wls[idx]
#    if wl_max>4874 and wl_max<6466:
#        valid_max = True
#    else:
#        wls = np.delete(wls,idx)
#        its = np.delete(its,idx)
        #lamp_lines = np.delete(lamp_lines,wl_max)

#for i in range(rows):
#    if wls[i]>4874 and wls[i]<6466 and itssc[i]>0.1:
#        plt.plot([wls[i],wls[i]],[0,itssc[i]],'r')
#    
#plt.show()
#Bin the ThAr lines for comparison to arc frame
bin_wid = 0.02 #Angstroms
wlrng = np.arange(4874,6466,bin_wid)
itrng = np.zeros((np.shape(wlrng)))
i=0
for wl in wlrng:
    itrng[i] = sum(its[(wls>(wl-bin_wid/2))*(wls<(wl+bin_wid/2))])
    i+=1

#plt.plot(wls,its,wlrng,itrng)
#plt.show()
#col1 = 'arclineid'
#wl_min = 4874 #angstroms
#wl_max = 6466 #angstroms
#wl_idx = 0
#it_idx = 1
##name_idx = 5
#status = "GOOD"
#
#ll_minerva = open('lamplines_minerva.par',"a")
##for i in range(10):
##    ll_minerva.readline()
#
#for j in range(np.shape(lamp_lines)[0]):
#    wl = float(lamp_lines[j,wl_idx])
#    if wl<wl_min or wl>wl_max:
#        continue
#    else:
#        wl = str(wl)
#        it = str(int(lamp_lines[j,it_idx]))
#        name = line_names[j][0]
#        stat = status
#        if int(it)>10000:
#            psf = 'YES_PSF'
#        else:
#            psf = 'NO_PSF'
#        ll_minerva.write("arclineid\t{wl}\t{it}\t{stat}\t{psf}\t\"{name}\"\n".format(wl=wl,it=it,stat=stat,psf=psf,name=name))
        