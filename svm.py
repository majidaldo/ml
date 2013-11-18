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
yxs= listyxs(rd.train,'5','2')
lijs=listijs(  len(yxs)  ) #indexing


@autojit
def yKcalc(xs,ys,kernel):
    #li=listiter(len(xs))
    yKs=np.empty(len(lijs),dtype=mf)
    #ij=0
    #for i in lis:
    #    for j in ljs:
    for i in xrange(len(yKs)):
           yKs[i]= ys[lijs[i][0]]*ys[lijs[i][1]]\
           *kernel(xs[lijs[i][0]],xs[lijs[i][1]])
    return yKs

#ugly code due to numba!
kernel=Kradial
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
    sayKs=0
    for i in xrange(ninej):
        sayKs+=yKs[i]*alphas[lijs[i][0]]*alphas[lijs[i][1]]
    sayKs*=2 #symmetric loop
    for i in xrange(ninej,n-1):
        sayKs+=yKs[i]*alphas[lijs[i][0]]*alphas[lijs[i][1]]#argh! wrote it 2x
    return sa-.5*sayKs


C=13
def istar(alphas):
    istr=alphas>0
    istr*=alphas<C
    return np.where(istr==True)
#@autojit
def yaKx(yas,kernel,xs,x):#(ys,alphas,kernel,xs,x):
    #ys*=alphas;yas=ys
    for i,ya in enumerate(yas): yas[i]=ya*kernel(xs[i],x)
    yaks=yas #yi*alphai*K(xi,x)
    return np.sum(yaks)
#@autojit
def classify(xc,alphas):#(ys,alphas,xs):#alphas from a kernel
    ys=yxs['y']
    xs=yxs['x']
    istr=istar(alphas)
    print 'istars=',len(istr)
    
    yss=ys[istr];alphass=alphas[istr];xss=xs[istr]
    yss*=alphass;
    yass=yss #WOW this vector is made of only 3 numbers: a,0,-a
    b=yaKx(yass,kernel,xss,xss[0])-yss[0] #take first istar
    #b is an integer!!
    dec=np.empty_like(ys,dtype=bool)
    for i,x in enumerate(xc):
        if (yaKx(yass,kernel,xss,x)-b)>=0: dec[i]=True
        else: dec[i]=False 
    return dec

def constrains(alphas):
    say=np.dot(alphas,yxs['y'])#alphas,ys
    #cv=np.empty(len(alphas+1))
    #cv[0]=say #for equality contrain
    #cv[1:]=alphas #for inequality
    return np.array([say])#cv


def pyoptof(alphas,**kwrags):
    f=-of(alphas)
    g=constrains(alphas)
    return f,g,0
    


