import rd
import numpy as np
from numpy.linalg import norm

def Kradial(xi,xj,**kwargs):
    gamma=kwargs.setdefault('gamma',.0521)
    return np.exp(-gamma*norm(xi-xj)**2)


def genyk(dataset,yp,ym):
    """yp, ym refer to a digit"""
    ypm={yp:+1,ym:-1}
    for ay in [yp,ym]:
        for ax in dataset[ay]:
            yield ypm[ay], ax
def listyk(*args):
    return np.array(list(genyk(*args))
                   ,dtype=[('y','int32'),('x','f32',len(rd.train['0'][0]))])


from numba import autojit,jit,float_,int_,function
from itertools import combinations as combos
from itertools import combinations_with_replacement as comboswr
ic={}#index cache
ykc={}#yi*yj*K cache
@jit(float_(float_[:],float_[:,:],float_[:],function))
def of(alphas,xs,ys,kernel):
    sa= np.sum(alphas)
    ijs=ic.setdefault(len(alphas)
        ,np.array(list(combos(xrange(len(alphas)),2)),dtype='uint32')   )
    return 3.0
#    yks=ykc.setdefault(hash(  (len(alphas),kernel)  )
#            ,dict([  [ij, ys[ij[0]]*ys[ij[1]]*kernel(xs[ij[0]],xs[ij[1]])] \
#             for ij in comboswr(xrange(len(alphas)),2)     ])
#         )
#    sayk=0
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
    
    