import numpy as np

import rd
from rd import digits

def w1(d): return 1
def we(d,eps): return (d**2+eps)**-1
def d2n(x1,x2): return np.linalg.norm(x1-x2) #2norm


from itertools import combinations
def pairs(test,train):
    for atest in rd.getdata(test):
        yield atest, rd.getdata(train)

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
            winc='t'#t for tie
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
