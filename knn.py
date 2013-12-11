#an experiment in python generators

import numpy as np

import rd
from rd import digits

#from numba import autojit

#@autojit
def w1(d,*extra): return 1
#@autojit
def we(d,eps): return (d**2+eps)**-1
#@autojit
def d2n(x1,x2): return np.linalg.norm(x1-x2) #2norm


from itertools import combinations
def pairs(test,train):
    for atest in rd.getdata(test):
        yield atest, rd.getdata(train)


#could have just used scipy.spatial.distance.pdist!
from itertools import imap
def distances(pairs, dfunc):
    for atest,trains in pairs:
        yield atest\
        , imap( lambda (atst,atrn): (atrn[1], dfunc(atst[0],atrn[0]))#the return
               , ((atest,atrain) for atrain in trains)   )


from random import choice
def classify(k,dists,wfunc,*wfuncargs):
    """use k=None for all pts """
    for atest,distsfromtest in dists:
        trndists=np.array( [((clss),adist) for (clss,adist) in distsfromtest]
                            ,dtype=[('c','S1'),('d',float)] ) #one char array
          #          ,dtype=[('c',str),('d',float)] ) #lesson: str does not work here
        trndists.sort(order='d')
        tc=atest[1]#test class
        #def voting(k):
        knn=trndists[:k]
        ws=( wfunc(ad,*wfuncargs) for (ac,ad) in knn )#weights
        votes=vote(knn['c'],ws)
       #     return votes
       # votes=voting(k)
        if len(set(votes.values()))!=len(votes.values()):#ties possible
            #raise ValueError("tie",votes)
            winc=choice(votes.keys())
            #winc='t'#t for tie
        else: winc=votes.keys()[np.argmax(votes.values())]
        yield tc,winc#,votes#test classification,winner


        
from itertools import izip
def vote(classes,weights):
    votes={}
    for c,w in izip(classes,weights):
        if c not in votes: votes[c]=w
        else: votes[c]+=w
    return votes




#ds=distances(pairs(rd.test,rd.train),d2n)
#tc=classify(1,distances(pairs(rd.test,rd.train),d2n),w1)
def tally(classit):
    nc=0;ln=0;
    for ac,apc in classit:
        ln+=1
        if ac==apc: nc+=1
    return nc,ln
    

def main(distfunc,k,wfunc,*wfargs):
    ds=(distances(pairs(rd.test,rd.train),distfunc))
    cs=classify(k,distances(pairs(rd.test,dict([(ak,rd.train[ak]) for ak in rd.digits]) ),distfunc)
                ,wfunc,*wfargs)
    return tally(cs)

from itertools import product
def hw():
    cases=list((product(range(1,8),[w1,we])  ))
    cases=cases+[(None,we)]
    wn={ hash(w1):'1'
        ,hash(we):'wi'}
    print 'w\tk\tpct'
    for ak,awf in cases:
        t=main(d2n,ak,awf,1)
        print wn[hash(awf)],'\t', ak,'\t',str(float(t[0])/t[1])[:4]


if __name__=='__main__':hw()
