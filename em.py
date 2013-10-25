"""expectation minimization of gaussian mixture"""

import numpy as np

with open('Majid.txt') as nf:
    nums=np.array(nf.read().split(' ')[:-1],dtype=float)


sig=1

def p(xi,mu): return np.exp(-(xi-mu)**2/(2*sig**2))

def Ej(xi,muj,mu2):
    pj=p(xi,muj)
    return pj/(pj+p(xi,mu2))

def muj(x,muj,mu2): return np.sum(Ej(x,muj,mu2)*x)/np.sum(Ej(x,muj,mu2)) 


def EM():
    na=np.average(nums)
    nsd=np.std(nums)
    mu1=na-2*nsd; mu2=na+2*nsd #guess in each direction from mean
    
    for i in xrange(100):
        #E1,E2=Ej(nums,mu1,mu2),Ej(nums,mu2,mu1)
        mu1n,mu2n=muj(nums,mu1,mu2),muj(nums,mu2,mu1)
        if abs(mu1n-mu1)/mu1n<.01 and abs(mu2n-mu2)/mu2n<.01:
            #print i
            return mu1n, mu2n
        else: mu1,mu2=mu1n,mu2n

