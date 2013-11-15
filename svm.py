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
#lis=lijs.T[0].copy()
#ljs=lijs.T[1].copy()

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

yKs=yKcalc(yxs['x'],yxs['y'],Kradial)


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
    #tia=tia.T[0].copy()
#    for ij in li[:5]: print ij
#    yks=ykc.setdefault(hash(  (str(ys),kernel)  )
#           , np.array([ys[ij[0]]*ys[ij[1]]*kernel(xs[ij[0]],xs[ij[1]])\
#                       for ij in ijs],dtype=mf)
#         )

    
    
#    #@jit(restype=numba.float_,argtypes=[typeof(ijs),typeof(ijs)])    
#    def ayk(i,j): return alphas[i]*alphas[j]*yks[(i,j)]#*ys[i]*ys[j]*kernel(xs[i],xs[j])    
#    
#    yks=ykc.setdefault(hash(  (str(ys),kernel)  )
#           ,np.array(list(comboswr(xrange(len(alphas)),2)))
#           # ,dict([  [ij, ys[ij[0]]*ys[ij[1]]*kernel(xs[ij[0]],xs[ij[1]])] \
#           #  for ij in np.array(list(comboswr(xrange(len(alphas)),2)) )     ])
#         )
#    sayk=0
#    
#    return yks
#    @jit(float_(int_,int_))
#    def ayk(i,j): return alphas[i]*alphas[j]*yks[(i,j)]#*ys[i]*ys[j]*kernel(xs[i],xs[j])    
#    for ij in ijs: sayk+=ayk(ij[0],ij[1])
#    sayk*=2
#    for i in xrange(len(alphas)): sayk+=ayk(i,i)
#    return sa-.5*sayk



def pyoptof(alphas,*args,**kwrags):
    alphas=args[0]
    f=of(alphas,*args[1:])
    g=np.dot(alphas,args[2])#alphas,ys
    return f,g,0
    
    