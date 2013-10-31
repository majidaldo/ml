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

def getdata(dataset):
    ds=dataset #test or train
    for ad in ds:#adigit in ds
        for avec in ds[ad]:
            yield avec, ad




#just a personal note
#maybe input data in numpy record array
#ddt=[('c',str,1),('x',train['0'].shape[1])] #i dont want to type 64
    