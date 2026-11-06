import numpy as np
from utils import *

def sumgau(x, pars):
    """
    Sum of Gaussian IP, Keck HIRES style but with floating central width.

    Created by Sharon X. Wang Jan 22 2018
    for MINERVA pipeline, also see John's gpfunc function in four_chunk_jjN.py.
    """

    # psfpix and psfsig are determined by exploring the chisq surface when adding just one pair of satellite gaussians
    # and trying to locate local minima in the chisquare space with Paul Butler's old IDL codes.
    psfpix = np.array([-0.9, -0.5, 0.5, 0.9])
    psfsig = np.array([0.6, 0.35, 0.35, 0.6])

    # jjgauss(x, height, position, width)
    # Central Gaussian
    ip = jjgauss(x, 1.0, 0.0, pars[0])

    # Add satellites
    for i, pix in enumerate(psfpix):
        sig = psfsig[i]
        gau = jjgauss(x, pars[i+1], pix, sig)
        ip += gau

    # normalization
    dx = (x[-1] - x[0]) / (len(x) - 1.0)
    ip /= (ip.sum() * dx)
        
    return ip

def supergau(x, pars):
    """
    Super Gaussian (with changing exponent) with satellite gaussians.

    Code based on class SuperGaussian from Rene Tronsgaard Rasmussen from his newrv2 package built for SONG.
    Sharon Xuesong Wang, Jan 22 2018
    """

    # satellite gaussians
    #                   1     2     3    4    5    6
    #psfpix = np.array([-3.0, -1.3, -0.5, 0.5, 1.3, 3.0])
    #psfsig = np.array([ 1.0,  0.7,  0.3, 0.3, 0.7, 1.0])
    psfpix = np.array([-1.4, -0.45, 0.45, 1.4])
    psfsig = np.array([ 0.7,  0.50, 0.50, 0.7])
    
    # Include fixed satellite parameters
    a = np.concatenate(([1.0], pars[2:len(pars)]), axis=0)  # heights of center gaussian and satellites
    b = np.concatenate(([0.0], psfpix), axis=0)  # position of center gaussian and satellites
    c = np.concatenate(([pars[0]], psfsig), axis=0)  # widths of center gaussian and satellites
    n = np.concatenate(([pars[1]], np.zeros(psfpix.size) + 2), axis=0)  # exponents of gaussians

    # Supergauss function
    xarr = np.transpose([x] * int(psfpix.size+1))
    ip = np.sum(a * np.exp(-0.5 * (np.abs(xarr - b) / c) ** n), axis=1)

    # Replace negative values with zero - let's not do this for MINERVA, SXW
    #ip[np.where(ip < 0.0)] = 0.0

    # Calculate centroid and re-center the LSF - not for MINERVA
    #offset = np.sum(x * ip) / np.sum(ip)
    #ip = func(x + offset)

    # normalization
    dx = (x[-1] - x[0]) / (len(x) - 1.0)
    ip /= (np.sum(ip)*dx)

    return ip
