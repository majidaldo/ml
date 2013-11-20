import rd
import numpy as np
import math

from numba import autojit,jit,typeof,float_

mf=np.float32
mui=np.uint16
mi=np.int8

gamma=.0521
@autojit
def Kradial(xi,xj):
    #gamma=kwargs.setdefault('gamma',.0521)
    xd=xi-xj
    xd*=xd #inplace #for i in xrange(len(xd)): xd[i]*=xd[i]
    xd= np.sum(xd)
    return math.e**(-gamma*xd)


alpha=.0156
beta=0.0
d=3.0
@autojit
def Kpoly(xi,xj): return (alpha*np.dot(xi,xj)+beta)**d
    

def genyx(dataset,yp1,ym1):
    """yp, ym refer to a digit"""
    ypm={yp1:+1,ym1:-1}
    for ay in [yp1,ym1]:
        for ax in dataset[ay]:
            yield ypm[ay], ax
def listyxs(*args):
    return np.array(list(genyx(*args))
                   ,dtype=[('y',mi),('x',mf,len(rd.train['0'][0]))])


from itertools import izip
from itertools import combinations as combos
#from itertools import combinations_with_replacement as comboswr
#from math import factorial as wow


def listijs(n): return np.array(#indexing
         list(combos(xrange(n),2)) #first ninej are the combos
        +list(izip(  xrange(n),xrange(n)  ))
        ,dtype=mui
        )

#lijs=
classes=['5','2']
yxs= listyxs(rd.train,*classes)
lijs=listijs(  len(yxs)  ) #indexing
yxstest= listyxs(rd.test,*classes)


@autojit
def yKcalc(xs,ys,kernel):
    #li=listiter(len(xs))
    yKs=np.empty(len(lijs),dtype=mf)
    #ij=0
    #for i in lis:
    #    for j in ljs:
    for i in xrange(len(yKs)):#j=i[1],i=i[0]
           yKs[i]= ys[lijs[i][0]]*ys[lijs[i][1]]\
           *kernel(xs[lijs[i][0]],xs[lijs[i][1]])
    return yKs

#ugly code due to numba!

#choose kernel
kernel=Kradial
#kernel=Kpoly
yKs=yKcalc(yxs['x'],yxs['y'],kernel)


ic={}#index cache
ykc={}#yi*yj*K cache
#@jit(float_(float_[:],float_[:,:],float_[:],function(float_,[float_,float_])))
@autojit
def of(alphas):
    n=len(alphas)
    sa= np.sum(alphas)
#    ic.setdefault(n
#        ,np.array(
#        list(combos(xrange(n),2)) #first ninej are the combos
#        +list(izip(  xrange(n),xrange(n)  ))
#        ,dtype=mui
#        )   
#        )
    ninej=(n*n-n)/2#wow(n)/(2*wow(n-2)) #number of unique i,j combos, i!=j
    sayKs=0.0
    for i in xrange(ninej):
        sayKs+=yKs[i]*alphas[lijs[i][0]]*alphas[lijs[i][1]]
    sayKs*=2 #symmetric loop
    for i in xrange(ninej,n-1):
        sayKs+=yKs[i]*alphas[lijs[i][0]]*alphas[lijs[i][1]]#argh! wrote it 2x
    return sa-.5*sayKs


C=100
def istar(alphas):
    alphas=np.array(alphas)
    istr=alphas>0
    istr*=alphas<C
    return np.where(istr==True)[0]
#@autojit
def sum_yaKx(yas,kernel,xs,x):#(ys,alphas,kernel,xs,x):
    #ys*=alphas;yas=ys
    #yas=np.empty_like(yas,dtype=mf)
    sm=0.0
    for i,ya in enumerate(yas): sm+=ya*kernel(xs[i],x)
    #yaks=yas #yi*alphai*K(xi,x)
    return sm #np.sum(yaks)
#@autojit
def separate(xc,alphas):#(ys,alphas,xs):#alphas from a kernel
    """returns: was it of the type classified with a 1?"""
    alphas=np.array(alphas)
    ys=yxs['y']
    xs=yxs['x']
    istr=istar(alphas)
    yss=ys[istr]; alphass=alphas[istr]; xss=xs[istr]
    yass=alphass*yss;
    # yass*=alphas WRONG inplace operation got me here bc ys is int!
    bs=[]
    for abi in xrange(len(yss)):
        b=sum_yaKx(yass,kernel,xss,xss[abi])-yss[abi] #take first istar
        bs.append(b)
    b=np.average(bs)
    print bs
    #b=sum_yaKx(yass,kernel,xss,xss[0])-yss[0] #take first istar they 
    #should all be same
    dec=np.empty(len(xc),dtype=bool)
    print 'b=',b
    for i,x in enumerate(xc):
        if (sum_yaKx(yass,kernel,xss,x)-b)>=0: dec[i]=True
        else: dec[i]=False 
    return dec

def evalsep(yxtest,alphas):
    ysb=yxtest['y']
    correct=np.zeros_like(ysb,dtype=bool)
    d=separate(yxtest['x'],alphas)
    for i in xrange(len(d)):
        if    ysb[i]== 1 and d[i]==True:  correct[i]=True 
        elif  ysb[i]==-1 and d[i]==False: correct[i]=True
    return correct
    
    
