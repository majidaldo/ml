import numpy as np
import os
dfs=os.listdir('d')

dfs=dict([(adf,np.loadtxt( os.path.join(os.path.curdir,'d',adf)
,dtype='bool')
) 
for adf in dfs])

test={}
train={}
for adf,nums in dfs.items():
    if 'train' in adf: train.update({adf[-5]:nums})
    else:               test.update({adf[-5]:nums})
#not the data is accessed like  data[digit][anarray],[arrayindex]

##put data in numpy record array
#ddt=[('c',str,1),('x',train['0'].shape[1])] #i dont want to type 64
#train=(for )
digits=np.sort(train.keys())