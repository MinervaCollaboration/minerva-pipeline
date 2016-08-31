
import numpy as np
import scipy
import scipy.misc
import scipy.integrate
import ipdb
import matplotlib.pyplot as plt

"""
What do we ultimately want:
I think we want a function that calculates the first 15 gauss-hermite poly-
nomials for some range of x values. are x values the pixels? 
we are given a set of parameters, and the x values. these parameters are used
to 
Where are we:
-i'm getting an output from ghfunc, no idea if right
-trying to test from idl code, with poor results of matching
What we have left:
-need to make sure that all the arrays are oriented correctly, in terms of 
 transposes, etc. i think this is really where i'm falling short
"""


def ghfunc_old(x,p,par,oldwidth,herm_arr,pow_arr,zarr,normarr):
 
    nx = x.shape[0]
    #S amplitude is a bit strange to me, I'm not sure what it is for.
    #S I think it is the coeffs for each GH, but that doesn't make sense to me.
    #S it is an array of shape (15)X(nx) 
    #S p is the output GH's with out their coeffs in a (121,15) shaped array
    
    #S just want parameters, 
    amp = fan(np.concatenate(([1.],par[1:11],par[15:19])),nx)
    if par[0] != oldwidth:
        width = par[0]
        p = set_param(x,herm_arr,pow_arr,zarr,normarr,width=width)
        oldwidth = par[0]
    #S which axis is this summed over? I believe the first due to idl docs on 
    #S 'total'
    ipro = np.sum(amp*p,axis=1)
    return ipro/scipy.integrate.simps(ipro, x=x), oldwidth

#do ghfunc except param, just ignore. make all attributes their own variables
#

def ghfunc(x,par,param={}):
    amp = [1.0,par[1:-1]]

    # if the width has changed (not equal to par[0]) or param has not been
    # initialized, we need to (re)calculate param
    try: 
        if par[0] != param['width']:
            set_param(x,par,param)
    except:
        set_param(x,par,param)

    ipro = np.sum(amp*param['p'],axis=1)
    return ipro/scipy.integrate.simps(ipro, x=x)


def set_param(x,par,param={}):#herm_arr,pow_arr,zarr,normarr,width=25.):

    # param not set, calculate it
    if len(param) == 0:
        nx = len(x)

        # number of GH polynomials to compute
        # the first element is the gaussian width, rest are GH polynomial amplitudes
        n = len(par)-1 


        param['coeffs'] = make_herm(n-1) 
        pow = np.arange(n)
        param['powers'] = fan(pow, nx,transpose=True)
        xarr = fan(x,n)
        param['zarr'] = xarr**coeffs['powers']
        norm = 1.0/np.sqrt(2.0**pow * np.sqrt(np.pi) * scipy.misc.factorial(pow))
        param['normarr'] = fan(norm,nx,transpose=True) 

    param['width'] = par[0]
    beta = np.zeros(nx, dtype=np.float64) + width
    betarr = fan(beta,n)**param['powers']
    gauss = np.exp(-(x * beta)**2/2.0)
    gaussarr = fan(gauss,n)
    t = np.dot(param['coeffs'],(param['zarr'] * betarr))
    param['p'] = param['nornarr'] * gaussarr * t

    return param

def make_herm(n,transpose=False):

    """
    Create an array of the first n hermite polynomial coefficients. 
    Returns an array of (n+1)X(n+1) of the coeffs, and can be transposed.
    Emulating code from johnjohn's make_herm.pro
    For reference, the polynomials H_N for order N are
    H_0 = 1
    H_1 = 2x
    H_2 = 4x**2 -2
    H_3 = 8x**3 -12x
    H_4 = 16x**4 - 48x**2 + 12
    def make_herm(n,transpose=False):
    return h
    """
    #S we need special cases to handle the coefficients less than two, as the
    #S recursion formula works only for n>2. These cases aren't hard though!

    #S make the first element equal to 1
    h = np.zeros([n+1,n+1],dtype=np.float64)
    h[0,0] = 1.
    
    #S if the array is large enough, make element_2,2 equal to 2
    if n > 0:
        h[1,1] = 2.
    #S formula seems to work, found a different one on wikipedia. this one from
    #S make_herm.pro, maybe just the same result? need to work them out to 
    #S equivalence. this returns correct array up to H_10
    if n > 1:
        for ind in range(2,n+1):
            h[ind,:] = np.roll(h[ind-1,:],1)*2.-2.*float(ind-1)*h[ind-2,:]
    #S if we want the transpose
    if transpose:
        return h.T

    #S otherwise just send out the h array
    else:
        return h
    
def fan(array,nfan,transpose=False):
    #S function to emulate fan.pro
    ones = np.ones(nfan)
    fan_array = np.outer(ones,array)
    if transpose:
        fan_array = fan_array.T
    return fan_array
    
if __name__ == '__main__':
    ipdb.set_trace()
    x1 = np.arange(-5,5.25,.25/2.)
    h,p,z,n,x=make_arrs(x1,7)
    z = set_param(x1,h,p,z,n,width=1.)
    
#    x1000 = fan(np.linspace(-5,5,1000),7)
#    p1000 = fan(np.arange(7),1000,transpose=True)
#    hp1000 = np.dot(h,x1000**p1000)
    hp = np.dot(h,x1**p)
    c = np.array([1.,.27,.2,.2,.05])
    plt.plot(x1,c[0]*z[0,:]+c[1]*z[1,:]+c[2]*z[2,:]+c[3]*z[3,:])
    for i in np.arange(4):
        plt.plot(x1,c[i]*z[i,:],'--', label=r'n=%i,$C_{%i}$=%0.1f'%(i,i,c[i]))
    plt.title('fourth order series')
    plt.legend()
    plt.show()
    ipdb.set_trace()
    for i in range(5): plt.plot(x1,hp[i,:],label=r'$n=$'+str(i))
    plt.legend()
    plt.title(r'$H_n(x)$')
    plt.axis([-3,3,-25,25])
    plt.show()
    ipdb.set_trace()
    for i in range(5): plt.plot(x1,z[i,:],label=r'$n=$'+str(i))
    plt.title(r'$e^{-(\beta x)^2/2}$$H_n(\beta x)$')
    plt.legend()
    plt.axis([-5,5,-1,1])
    plt.show()
    ipdb.set_trace()
    plt.plot(x1,z[0,:],'--',label=r'$C_0=1,C_1=0,C_2=0$')
    plt.plot(x1,z[0,:]+.2*z[1,:],'--',label=r'$C_0=1,C_1=0.2,C_2=0$')
    plt.plot(x1,z[0,:]-.2*z[1,:],label=r'$C_0=1,C_1=-0.2,C_2=0$')
    plt.plot(x1,z[0,:]+.2*z[1,:]+.2*z[2,:],label=r'$C_0=1,C_1=0.2,C_2=0.2$')
    plt.title(r'IP($x$) = $\sum C_ne^{-(\beta x)^2/2}$'\
                  +r'$H_n(\beta x)$')
    plt.legend()
    plt.show()
    ipdb.set_trace()
    ghfunc(x,z,np.arange(20),25.,h,p,z,n)
    
#def ghfunc(x,p,par,oldwidth,herm_arr,zarr,normarr):
