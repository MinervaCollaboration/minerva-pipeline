#!/usr/bin/env python

'''
# Code to implement B-splines, both in 1D and 2D
# Sort of slow and messy and scipy does most of it better
# I'm leaving this in here just in case someone wants to work with it
# (not recommended...)
'''

#Import all of the necessary packages
from __future__ import division
import numpy as np
from numpy import cos, sin
import matplotlib.pyplot as plt
import special as sf
import lmfit

def basic_spline(t,breakpoints,order=4):
    """ Function to calculate basic splines.
        Follows convention of Carl de Boor essay B(asic)-Spline Basics:
         -> ftp.cs.wisc.edu/Approx/bsplbasic.pdf
        INPUTS:
            t - array of x coordinates to evaluate
            breakpoints - array of breakpoints/knots/x-coordinates
            order - spline order (4 = cubic)
        OUTPUTS:
            B_spline - matrix of splines (band diagonal) (n-1) x m where n=len(breakpoints), m = len(t)
    """
    if np.min(np.ediff1d(breakpoints)) < 0:
        print("B-spline code is only valid for non-decreasing intervals")
        exit(0)
    def w(i,k,t,breakpoints):
        ### REVISIT LATER? Don't think this edge handling is correct
        if i < 0 or breakpoints[i] == breakpoints[i+k-1]:#i+k-1 >= len(breakpoints):
            return np.zeros(len(t))
        if breakpoints[i] != breakpoints[i+k-1]:
            return (t-breakpoints[i])/(breakpoints[i+k-1]-breakpoints[i])
#        elif i+k-1 < len(breakpoints):
#            if breakpoints[i] != breakpoints[i+k-1]:
#                return (t-breakpoints[i])/(breakpoints[i+k-1]-breakpoints[i])
#            else:
#                return zeros(len(t))
#        elif breakpoints[i] != breakpoints[-1]:
#            return (t-breakpoints[i])/(breakpoints[-1]-breakpoints[i])
        else:
            return np.zeros(len(t))
    def X(i,t,breakpoints):
        if i < 0 or i+1 >= len(breakpoints):
            return np.zeros(len(t))
        else:
            return (t>=breakpoints[i])*(t<breakpoints[i+1])
    def B(i,k,t,breakpoints):
        if i < 0 or i+1 >= len(breakpoints):
            return np.zeros(len(t))
        elif k == 1:
            return X(i,t,breakpoints)
        elif k > 1:
            return w(i,k,t,breakpoints)*B(i,k-1,t,breakpoints) + (1 - w(i+1,k,t,breakpoints))*B(i+1,k-1,t,breakpoints)
        else:
            return np.zeros(len(t))
    ### Account for edge effects:
    ### Pad linearly, other options available if desired
    pad = order-1#int(np.ceil(order/2))
    d_bpt = np.mean(np.ediff1d(breakpoints))
    bpt_pad = np.pad(breakpoints,pad,mode='linear_ramp',end_values=((breakpoints[0]-pad*d_bpt,breakpoints[-1]+pad*d_bpt)))
    dt = np.mean(np.ediff1d(t))
    t_per_bpts = int(np.floor(len(t)/len(breakpoints)))
    t_pad = np.pad(t,t_per_bpts*pad,mode='linear_ramp',end_values=((t[0]-pad*dt*t_per_bpts,t[-1]+pad*dt*t_per_bpts)))
    ### Now calculate the spline matrix
    B_spline = np.zeros((len(breakpoints)-1+pad,len(t)))
    for i in range(pad-order+1,pad+len(breakpoints)-1):
        #special handling for order 1 spline
        if pad == 0:
            B_spline[i,:] = B(i,order,t_pad,bpt_pad)
        else:
#            print t_per_bpts
            B_spline[i,:] = B(i,order,t_pad,bpt_pad)[pad*t_per_bpts:-pad*t_per_bpts]
#        plt.plot(t,B_spline[i-pad,:])
#    plt.show()
    return B_spline

def get_spline_1D_fit(xarr, yarr, breakpoints, coeffs, order=4, return_matrix=False):
    spline_interp = np.zeros((len(yarr)))
    ### (Assumes matrix size is relatively small)
    B_spline_matrix = basic_spline(xarr,breakpoints,order=order)
    spline_interp = np.dot(B_spline_matrix.T,coeffs)
    if return_matrix:
        return B_spline_matrix.T
    else:
        return spline_interp

### 1D case, nothing complex
def spline_1D(xarr,yarr,breakpoints,invar=None,order=4, window=100, pad=20, return_coeffs=False, incoeffs=None, inalign=None, fit_bg=False):
    """ Returns the spline fit to xarr, yarr with inverse variance invar for
        the specified breakpoint array (and optionally spline order)
        
        Assumes independent data points.
        
        Code is a little messy now - forced in a way to fit to a set of data, 
        then re-evaulate for arbitrary new inputs.  Works, but could stand
        to be cleaned up/simplified
    """
    spline_interp = np.zeros((len(yarr)))
    ### Break up to increase speed for large matrices
    if len(yarr) < window:
        B_spline_matrix = basic_spline(xarr,breakpoints,order=order)
        if fit_bg:
            B_spline_matrix = np.concatenate((B_spline_matrix,np.ones((1,len(xarr)))))
        if incoeffs is None:
            if invar is None:
                coeffs = np.linalg.lstsq(B_spline_matrix.T, yarr)
                coeffs = coeffs[0]
            else:
                coeffs, chi = sf.chi_fit(yarr,B_spline_matrix.T,np.diag(invar))
        else:
            coeffs = incoeffs
        outcoeffs = 1.0*coeffs
        spline_interp = np.dot(B_spline_matrix.T,coeffs)#[:,0]
        if return_coeffs:
            align_points = None
    else:
        if pad > 0.5*window:
            print "Spline pad is too big for window"
            exit(0)
        st = 0
        if return_coeffs:
            outcoeffs = dict()
            align_points = np.zeros((int(np.ceil(len(xarr)/(window-pad))),4))
        lc = 0
        m2 = 0
        while m2 < len(yarr):
            if inalign is not None:
                if lc == inalign.shape[0]:
                    m2 = len(yarr)
                    continue
            if st == 0:
                end = window+pad
            else:
                end = min(st+window+2*pad,len(yarr))
            xsub, ysub = xarr[st:end], yarr[st:end]
            xsl, xsh = np.min(xsub), np.max(xsub)
            if return_coeffs:
                align_points[lc,:] = [st, end, xsl, xsh]
            if inalign is not None:
                st, end, xsl, xsh = inalign[lc,:]
                st, end = int(st), int(end)
            mask = (xarr >= xsl) * (xarr <= xsh)
            m1 = np.argmax(mask) ## m1 = st
            if np.argmin(mask[m1:]) == 0:
                m2 = len(mask)
            else:
                m2 = np.argmin(mask[m1:])+m1+1 ## m2 = end
            if inalign is not None:
                xsub, ysub = xarr[m1:m2], yarr[m1:m2]
            wd = m2-m1
            bkpts = breakpoints[(breakpoints >= xsl) * (breakpoints < xsh)]
            if len(bkpts) == 0:
                lc+=1
#                m2 == len(yarr)
                continue
            B_spline_matrix = basic_spline(xsub,bkpts,order=order)
            if fit_bg:
                B_spline_matrix = np.concatenate((B_spline_matrix,np.ones((1,len(xarr)))))
            if incoeffs is None:
                if invar is None:
                    coeffs = np.linalg.lstsq(B_spline_matrix.T, ysub)
                    coeffs = coeffs[0]
                else:
                    invsub = invar[st:end]
                    coeffs, chi = sf.chi_fit(ysub,B_spline_matrix.T,np.diag(invsub))
            else:
                coeffs = incoeffs[lc]
            if st == 0:
#                spline_interp[st:end-pad] = np.dot(B_spline_matrix.T,coeffs)[0:window]
                spline_interp[m1:m2-pad] = np.dot(B_spline_matrix.T,coeffs)[0:wd-pad]
            elif end == len(yarr):
                spline_interp[m1+pad:m2] = np.dot(B_spline_matrix.T,coeffs)[pad:]
            else:
#                try:
                spline_interp[m1+pad:m2-pad] = np.dot(B_spline_matrix.T,coeffs)[pad:wd-pad]
            if return_coeffs:
                outcoeffs[lc] = coeffs
            st += window-pad
            lc += 1
    if return_coeffs:
        return outcoeffs, breakpoints, align_points
    else:
        return spline_interp



### 2D case using my own method - use outer product of two 1D splines as bases
### Also have a 2D radial spline option
'''
def spline_2D(img_matrix,invar_matrix,h_breakpoints,v_breakpoints,order=4,return_coeffs=False):
    """ Returns a 2D spline interpolation of an image.  Option to also return scaled spline_coeffs
    
        Assumes a pixel-based image (ie, integer horizontal and vertical
        steps between points)
        Assumes pixel errors are independent
        
        Note, computational time scales up very quickly, suggest modest image
        sizes (~100 x 100 px or less)
    """
    hpix = np.shape(img_matrix)[1] #horizontal pixels
    vpix = np.shape(img_matrix)[0] #vertical pixels
    harr = np.arange(hpix)
    varr = np.arange(vpix)
    ### Given h, v arrays and breakpoints, find splines along both directions
    h_splines = basic_spline(harr,h_breakpoints,order=order)
    v_splines = basic_spline(varr,v_breakpoints,order=order)
    h_sp_len = np.shape(h_splines)[0]
    v_sp_len = np.shape(v_splines)[0]
    dim1 = hpix*vpix
    dim2 = h_sp_len * v_sp_len
    ### Use the h and v splines to construct a 2D profile matrix for linear fitting
    profile_matrix = np.zeros((vpix,hpix,dim2))
    for i in range(h_sp_len):
        for j in range(v_sp_len):
            k = i*v_sp_len + j
            h_sp_tmp = np.reshape(h_splines[i],(1,hpix))
            v_sp_tmp = np.reshape(v_splines[j],(vpix,1))
            profile_matrix[:,:,k] = np.dot(v_sp_tmp,h_sp_tmp)
    
    #Reshape the profile matrix and input image and invar matrices for chi^2 fitting
    profile_matrix = np.reshape(profile_matrix,(dim1,dim2))
    data = np.reshape(img_matrix,(dim1,))
    noise = np.diag(np.reshape(invar_matrix,(dim1,)))
    #Chi^2 fit
    a_coeffs, chi = sf.chi_fit(data,profile_matrix,noise)
    #Evaluate spline fit and reshape back to 2D array
    spline_fit = np.dot(profile_matrix,a_coeffs)
    spline_fit = np.reshape(spline_fit,(vpix,hpix))
    spline_integral = np.sum(spline_fit)
    a_coeffs_scale = a_coeffs/spline_integral
    if return_coeffs:
        return spline_fit, a_coeffs_scale, spline_integral
    else:        
        return spline_fit
    
def spline_2D_eval(coeffs,scale,img_matrix,h_breakpoints,v_breakpoints,order=4):
    """ Returns a spline "fit" for arbitrary coefficients.
        INPUTS:
            coeffs - array of fitted coeffs (of same format as spline_2D)
            scale - float, scale value to match img_matrix
            img_matrix - original matrix, mostly used for matching dimensions
            h/v_breakpoints - horiz/vertical breakpoints used in fit
            order - spline order (4 = cubic)
        OUTPUTS:
            spline_fit - fitted spline using coeffs
    """
    hpix = np.shape(img_matrix)[1] #horizontal pixels
    vpix = np.shape(img_matrix)[0] #vertical pixels
    harr = np.arange(hpix)
    varr = np.arange(vpix)
    ### Given h, v arrays and breakpoints, find splines along both directions
    h_splines = basic_spline(harr,h_breakpoints,order=order)
    v_splines = basic_spline(varr,v_breakpoints,order=order)
    h_sp_len = np.shape(h_splines)[0]
    v_sp_len = np.shape(v_splines)[0]
    dim1 = hpix*vpix
    dim2 = h_sp_len * v_sp_len
    ### Use the h and v splines to construct a 2D profile matrix for linear fitting
    profile_matrix = np.zeros((vpix,hpix,dim2))
    for i in range(h_sp_len):
        for j in range(v_sp_len):
            k = i*v_sp_len + j
            h_sp_tmp = np.reshape(h_splines[i],(1,hpix))
            v_sp_tmp = np.reshape(v_splines[j],(vpix,1))
            profile_matrix[:,:,k] = np.dot(v_sp_tmp,h_sp_tmp)
    
    #Reshape the profile matrix
    profile_matrix = np.reshape(profile_matrix,(dim1,dim2))
    #Evaluate spline fit and reshape back to 2D array
    spline_fit = np.dot(profile_matrix,coeffs)
    spline_fit = np.reshape(spline_fit,(vpix,hpix))
    spline_fit *= scale
    return spline_fit

def build_rarr_thetaarr(img_matrix,params,hlims=None,vlims=None,pts_per_px=1):
    """ Part of profile matrix building for splines.  Slightly more general -
        can be used in any elliptical coordinate system.
        INPUTS:
            img_matrix - image to evaluate coordinates (just used for shape)
            params - center coordinates, ellipticity and position angle
            pts_per_px - allows sub-pixel radius finding
        OUTPUTS:
            r_arr - 2D array, size of img_matrix, with elliptical radius
            theta_arr - 2D array, size of img_matrix, with elliptical angle
            dim1 - #of pixels in img_matrix
            r_inds - sorting array for raveling r_arr
    """
    ### Accepts lmfit.Parameters object or array [hc, vc, q, PA]
    try:
        hc = params['hc'].value
        vc = params['vc'].value
        q = params['q'].value
        q = abs(q)
        PA = params['PA'].value
    except:
        hc, vc, q, PA = params
        q = abs(q)
    hpix = np.shape(img_matrix)[1] #horizontal pixels
    vpix = np.shape(img_matrix)[0] #vertical pixels
    dim1 = vpix*hpix*pts_per_px**2
    if hlims is None and vlims is None:
        h_matrix = np.tile(np.arange(0,hpix,1/pts_per_px),(vpix*pts_per_px,1))
        v_matrix = np.tile(np.arange(0,vpix,1/pts_per_px),(hpix*pts_per_px,1)).T
    else:
        h_arr, v_arr = np.linspace(hlims[0], hlims[1], hpix*pts_per_px), np.linspace(vlims[0], vlims[1], vpix*pts_per_px)
        h_matrix, v_matrix = np.meshgrid(h_arr, v_arr)
    x_ell = (v_matrix-vc)*cos(PA) + (h_matrix-hc)*sin(PA)
    y_ell = (h_matrix-hc)*cos(PA) - (v_matrix-vc)*sin(PA)
    r_matrix = np.sqrt((x_ell)**2*q + (y_ell)**2/q)
#    if np.min(r_matrix) > 2:
#        plt.imshow(img_matrix)
##        plt.plot(hc,vc,'b.')
#        print hc, vc
#        plt.show()
#        plt.close()
    theta_matrix = np.arctan(y_ell/x_ell)
    r_inds = np.argsort(np.ravel(r_matrix))
    r_arr = np.ravel(r_matrix)[r_inds]
    theta_arr = np.ravel(theta_matrix)[r_inds]
    return r_arr, theta_arr, dim1, r_inds
    
def build_radial_profile(r_arr,theta_arr,r_breakpoints,theta_orders,dim1,order=4, fit_bg=False):
    """ Builds the profile matrix for spline fitting.
        INPUTS:
            r_arr - 2D array with radius to each pixel
            theta_arr - 2D array with angle to each pixel
            r_breakpoints - breakpoints for radial b-splines
            theta_orders - which angular moments to use (integers)
            dim1 - vertical size of profile matrix
            order - b-spline order for fitting (4, default, is cubic spline)
            fit_bg - boolean to include constant background term
        OUTPUTS:
            profile_matrix - for extraction, etc.
    """
    r_splines = basic_spline(r_arr,r_breakpoints,order=order)
    r_sp_len = np.shape(r_splines)[0]
    dim2 = (r_sp_len+fit_bg)*len(theta_orders)
    profile_matrix = np.zeros((dim1,dim2))
    for i in range(dim2):
        spline_dim = np.mod(i,(r_sp_len+fit_bg))
        ### Add constant background term
        if fit_bg and spline_dim == r_sp_len:
            profile_matrix[:,i] = np.ones(np.shape((r_splines[0])))
            continue
        theta_ind = int(np.floor(i/(r_sp_len+fit_bg)))
        theta_i = theta_orders[theta_ind]
        if theta_i == 0:
            profile_matrix[:,i] = r_splines[spline_dim]
        elif theta_i < 0:
            profile_matrix[:,i] = r_splines[spline_dim,:]*sin(-theta_i*theta_arr)
        else:
            profile_matrix[:,i] = r_splines[spline_dim,:]*cos(theta_i*theta_arr)
    return profile_matrix
    
def spline_2D_radial(img_matrix,invar_matrix,r_breakpoints,params,theta_orders=[0],order=4,return_coeffs=False,spline_coeffs=None,sscale=None,fit_bg=False, hlims=None, vlims=None,  pts_per_px=1):
    """ Follows Bolton et. al. 2005 (and similar) to implement radial splines
        Here I use vertical(v) and horizontal(h).  Usually v=x, h=y in Cartesian.
        INPUTS:
            img_matrix - 2D image (data)
            invar_matrix - 2D inverse variance (assuming independent points)
            r_breakpoints - radial breakpoints
            params - dict of [vc, hc, q, PA], best fits
            theta_orders - list [0,...] allows fitting of multipole orders
            order - order of spline (4=cubic is default)
            return_coeffs - returns the coefficients from the spline fit
            spline_coeffs - if array is entered returns model with coeffs instead of fitting
            sscale - if value (float) is included this scales the spline model
            fit_bg - boolean to fit background (default=False)
            pts_per_px - allows model building on sub pixel resolution
        OUTPUTS:
            spline fit - fitted image using radial bsplines
            if return_coeffs=True also return:
            spline_coeffs - spline coefficients
            coeff_scale - scale factor (coeffs normalized to one)
            if spline_coeffs is not None:
            spline_fit - model using previously determined coefficients
    """
    hpix = np.shape(img_matrix)[1]*pts_per_px #horizontal pixels
    vpix = np.shape(img_matrix)[0]*pts_per_px #vertical pixels
    dim1 = vpix*hpix*pts_per_px
    r_arr, theta_arr, dim1, r_inds = build_rarr_thetaarr(img_matrix,params,hlims=hlims,vlims=vlims,pts_per_px=pts_per_px)
    ### Given h, v arrays and breakpoints, find splines along both directions
    ### Use the h and v splines to construct a 2D profile matrix for linear fitting
    profile_matrix = build_radial_profile(r_arr,theta_arr,r_breakpoints,theta_orders,dim1,order=order,fit_bg=fit_bg)
#    if np.max(profile_matrix)==0:
#        plt.plot(r_arr)
#        plt.show()
    #    if fit_bg:
#        profile_matrix = np.hstack((profile_matrix,np.ones((dim1,1))))
#    print profile_matrix[:,0]
#    print profile_matrix[:,-1]

    #Reshape the profile matrix and input image and invar matrices for chi^2 fitting
    #Evaluate spline fit and reshape back to 2D array
    if spline_coeffs is None:
        data = np.ravel(img_matrix)[r_inds]
        noise = np.diag(np.ravel(invar_matrix)[r_inds])
        #noise = np.ravel(invar_matrix)[r_inds]
        #Chi^2 fit
        a_coeffs, chi = sf.chi_fit(data,profile_matrix,noise)
#        try:
#            a_coeffs = sf.extract_2D(data,profile_matrix,noise,return_no_conv=False)
#        except:
#            plt.plot(r_arr)
#            plt.show()
#            plt.close()
#            plt.imshow(profile_matrix,interpolation='none')
#            plt.show()
#            plt.close()
        spline_fit = np.dot(profile_matrix,a_coeffs)
        resort_inds = np.argsort(r_inds)
        spline_fit = np.reshape(spline_fit[resort_inds],(vpix,hpix))
#        res = img_matrix - spline_fit
#        vis = np.hstack((img_matrix,spline_fit,res))
#        plt.imshow(vis,interpolation='none')
#        plt.show()
#        plt.close()
        spline_integral = np.sum(spline_fit)
        a_coeffs_scale = a_coeffs/spline_integral
        if return_coeffs:
            return spline_fit, a_coeffs_scale, spline_integral
        else:        
            return spline_fit
    else:
        spline_fit = np.dot(profile_matrix,spline_coeffs)
        resort_inds = np.argsort(r_inds)
        spline_fit = np.reshape(spline_fit[resort_inds],(vpix,hpix))
        if sscale is None:
            return spline_fit
        else:
            spline_fit *= sscale
            return spline_fit

def spline_residuals(params, data, invar, breakpoints, theta_orders=[0], spline_coeffs=None, sscale=None):
    """ Function for fitting elliptical spline profile.  Returns residuals.
        INPUTS:
            params - parameter array from lmfit (or not if using x_coords)
            data - 2D data image
            invar - 2D array with per pixel error estimates of data
        OUTPUTS:
            residuals - residual array scaled by errors
    """
    spline_fit = spline_2D_radial(data,invar,breakpoints,params,theta_orders=theta_orders, spline_coeffs=spline_coeffs,sscale=sscale)
    residuals = (spline_fit-data)*np.sqrt(invar)
#    plt.imshow(np.hstack((data,spline_fit,residuals)))
#    plt.show()
    residuals = np.ravel(residuals)
    return residuals
    
def spline_poly_residuals(params,data,invar,breakpoints,x_coords,spline_coeffs,theta_orders=[0],cpad=5,fit_bg=False):
    """ Specific function (maybe can generalize later?) for fitting elliptical
        parameters for a spline function, with quadratic variation over an
        order.
        INPUTS:
            params - parameters (must be lmfit object with 12 values, detailed below)
            data - scaled data over x_coords (3D array)
            invar - scaled inverse variance over x_coords (3D array)
            breakpoints - radial breakpoints for building profile matrix
            x_coords - pixel positions normalized to interval [-1,1]
            ### centers - 2D array of horizontal/vertical centers (in pixels)
                      (col 0 = hcenter, col1 = vcenter)
            spline_coeffs - array from fitting
            theta_orders - optionally can set additional moments for angles
        OUTPUTS:
            residauls - error scaled residual array for lmfit.minimize()
    """
    ### The following only works for a very specific form of params
    ### It will be a len 12 array with indices hc0, hc1, hc2, vc0, vc1, vc2
    ### q0, q1, q2, and PA0, PA1, PA2 (lowest value is constant, highest 
    ### quadratic).  x_coords must be on interval [-1,1]
    len1 = np.size(data[:,:,0])
    residuals = np.zeros((len1*len(x_coords)))
    for i in range(len(x_coords)):
        x_coord = x_coords[i]
#        hc = params['hc0'] + params['hc1']*x_coord + params['hc2']*x_coord**2
#        vc = params['vc0'] + params['vc1']*x_coord + params['vc2']*x_coord**2
#        hc = centers[i,0]
#        vc = centers[i,1]
        q = params['q0'] + params['q1']*x_coord + params['q2']*x_coord**2
        PA = params['PA0'] + params['PA1']*x_coord + params['PA2']*x_coord**2
        params1 = lmfit.Parameters()
#        params1.add('vc', value = vc-int(np.floor(vc))+cpad)
#        params1.add('hc', value = hc-int(np.floor(hc))+cpad)
        params1.add('vc', value = params['vc{}'.format(i)].value)
        params1.add('hc', value = params['hc{}'.format(i)].value)
#        print params1['vc'].value
#        print params1['hc'].value
        params1.add('q',value = q, min=0, max=1)
        params1.add('PA',value = PA, min=0, max=360)
#        print params1['hc'].value, params1['vc'].value
        r_arr, theta_arr, dim1, r_inds = build_rarr_thetaarr(data[:,:,i],params1)
#        print r_arr
        profile_base = build_radial_profile(r_arr,theta_arr,breakpoints,theta_orders,dim1,order=4)
        profile_matrix = np.hstack((profile_base,profile_base*x_coord,profile_base*(x_coord**2)))
        if fit_bg:
            bg_base = np.zeros((len1,len(x_coords)))
            bg_base[:,i] = 1
            profile_matrix = np.hstack((profile_matrix,bg_base)) 
#        print profile_matrix[5]
        spline_fit = np.dot(profile_matrix,spline_coeffs)
#        if i == len(x_coords)-1:
##            print params1['vc'].value
##            print params1['hc'].value
##            print r_arr[0:10]
##            print profile_base[0,:]
##            print profile_matrix[0,:]
##            print spline_fit[0]
##            print np.sum(data[:,:,i])
##            print np.sum(spline_fit)
#            resort_inds = np.argsort(r_inds)
#            nf = np.resize(spline_fit[resort_inds],np.shape(data[:,:,0]))
#            plt.imshow(np.hstack((data[:,:,i],nf)),interpolation='none')
#            plt.show()
#        print spline_coeffs[0]
#        print np.max(spline_fit), np.min(spline_fit)
        data_for_res = np.ravel(data[:,:,i])[r_inds]
        inv_for_res = np.ravel(invar[:,:,i])[r_inds]
        residuals[i*len1:(i+1)*len1] = (data_for_res-spline_fit)*np.sqrt(inv_for_res)
#        resort_inds = np.argsort(r_inds)
#        nd = np.resize(data_for_res[resort_inds],np.shape(size_arr))
#        nf = np.resize(spline_fit[resort_inds],np.shape(size_arr))
#        print nf
#        plt.imshow(np.hstack((nd,nf)))
#        print r_arr, theta_arr
#        plt.imshow(nf)
#        plt.show()
#        plt.close()
#    residuals = np.ravel(residuals)
    return residuals
    
def make_spline_model(params,coeff_matrix,center,hpoint,img_shape,r_breakpoints,theta_orders,fit_bg=False,pts_per_px=1):
    """ Builds spline profile model.
        INPUTS:
            params - lmfit parameter object with q0/1/2 and PA0/1/2
            coeff_matrix - (N_splines x 3) array (quadratic coefficients)
            center - [hc,vc] of point to evaluate, relative coordinates
            hpoint - scaled point to evaluate (degenerate with hc)
            img_shape - [rows, cols] of model
            r_breakpoints - must match those used to build coeffs_matrix
            theta_orders - must match that used to build coeffs_matrix
            fit_bg - boolean, whether to fit constant background separately
            pts_per_px - allows finer grid (for model building).  Default is 1
        OUTPUTS:
            model - params/coeffs evaluated at hpoint
    """
    new_params = lmfit.Parameters()
    if hpoint > 1 or hpoint < -1:
        print("hpoint variable must be on interval [-1,1]")
        exit(0)
    q = params['q0'].value + params['q1'].value*hpoint + params['q2'].value*hpoint**2
    PA = params['PA0'].value + params['PA1'].value*hpoint + params['PA2'].value*hpoint**2
    new_params.add('q', value = q)
    new_params.add('PA', value = PA)
    new_params.add('hc', value = center[0]+img_shape[0]/2)
    new_params.add('vc', value = center[1]+img_shape[0]/2)
    spline_coeffs = np.dot(coeff_matrix,np.array(([1,hpoint,hpoint**2])))
    ### Need dummy matrices for shape
    img_matrix = np.ones((img_shape))
    invar_matrix = np.ones((img_shape))
    model = spline_2D_radial(img_matrix,invar_matrix,r_breakpoints,new_params,theta_orders,order=4,return_coeffs=False,spline_coeffs=spline_coeffs,sscale=None,fit_bg=fit_bg,pts_per_px=pts_per_px)
#    plt.imshow(model,interpolation='none')
#    plt.show()
#    plt.close()
    return model
#'''