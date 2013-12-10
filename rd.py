import numpy as np
import os
dfs=os.listdir('d')



dfs=dict([(adf,np.loadtxt( os.path.join(os.path.curdir,'d',adf)
                      ,dtype='bool')
           ) for adf in dfs]);

test={}
train={}
for adf,nums in dfs.items(): #trainX.txt Xi=5
    if 'train' in adf: train[adf[-5]]=nums
    else:               test[adf[-5]]=nums
#note the data is accessed like  data[digit][avectorindex]
del adf;del nums;


digits=np.sort(train.keys()) #list of the digits

from itertools import chain
def genbinaryds(adgt,dataset):#a digit
    """generate a data set that only has digit and not digit"""
    notdigits=set(digits)-set(adgt)
    bds={};bds[adgt]=dataset[adgt]
    bds['!'+adgt]=list(  chain(*[dataset[antdgt] for antdgt in notdigits])  )
    return bds
bintrain={}
for ad in digits:
    #test.update(genbinaryds(ad,test))
    train.update((genbinaryds(ad,train)))
    #bintrain[ad]=genbinaryds(ad,train)

def getdata(dataset):
    ds=dataset #test or train
    for ad in ds:#adigit in ds
        for avec in ds[ad]:
            yield avec, ad

def getlen(data):
    n=0
    for i in data: n+=1
    return n

#just a personal note
#maybe input data in numpy record array
#ddt=[('c',str,1),('x',train['0'].shape[1])] #i dont want to type 64
    
