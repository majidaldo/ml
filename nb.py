import numpy as np
import rd

train=rd.train
test=rd.test

#train: first index accesses the digit. the 2nd the line. the 3rd the vector
#.T is for transpose

#known probabilities
Ps=dict(
    (ad,
        
       dict((
       (i,np.bincount(train[ad].T[i],minlength=2) #counting 0s and 1s
                       /float(len(train[ad])) #div by 125 normally
       )
        for i in xrange(train[ad].shape[1])  #should be 0 to 63
        ))
                                                   
    ) for ad in train #the digits
    )
#data struct Ps is [ [digit0,[(x0,[n0s,n1s]),(x1,[n0s,n1s]),...]]
#                 ,digit1 [....]
#                 ,digit2..
    #           ]

#to be strict.. count the freq of the digits
Pd=dict( (ad,(len(train[ad]))) for ad in train )
Pds=sum(Pd.values());
for ad in Pd: Pd[ad]=Pd[ad]/float(Pds)  #all =.1
del Pds

def Pc(X,C):
    """probability of X given C"""
    mul=np.array([Ps[C][i][X[i]] #(don't like X[i])
    for i in xrange(len(X))],dtype=float)
    #vector of probabilities of a vector X
    #mul=(mul)/(max(mul)) #to not overflow the datatype
    return np.prod((mul))

def h(X): return rd.digits[np.argmax([Pc(X,C)*(Pd[C])
                                    for C in (rd.digits)])]

def results():
    return [ (ad,
    sum(h(x)==ad for x in test[ad])/float(len(test[ad]))
              )
    for ad in test]#digits in test



if __name__=='__main__': print results()
