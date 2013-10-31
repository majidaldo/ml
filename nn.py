"""code to recognize handwritten digits using feedforward neural network
weights are adjusted according to backpropagation algorithm with momentum
"""

#code correctness verified with sample calcs
#frm http://ccg.doc.gold.ac.uk
#/teaching/artificial_intelligence/lecture13.html
#
#
#tn=createnet(3,[2],2,bias=False)
#tw=createllv(tn)
#tw['v']=[.2,-.1,.4,.7,-1.2,1.2,1.1,.1,3.1,1.17]
#/ forward propagation
#fwdp(tn,tw,[10,30,20])
#Out[9]: array([ 0.75019771,  0.95709878], dtype=float32)
#/ error calculation
#errp(tn,tw,[1,0])
#Out[23]: 
#{1: array([ -6.40188227e-05,  -2.74555990e-04], dtype=float32),
# 2: array([ 0.04681322, -0.03929915])}
#te=Out[23]
#dw['v']=tw.copy()['v']=fDw(tn,tw,te,eta=.1)
#dw
#Out[36]: 
#array([(0L, 0L, 1L, 0L, -6.401882274076343e-05),
#       (0L, 1L, 1L, 0L, -0.00019205646822229028),
#       (0L, 2L, 1L, 0L, -0.00012803764548152685),
#       (0L, 0L, 1L, 1L, -0.00027455599047243595),
#       (0L, 1L, 1L, 1L, -0.0008236679714173079),
#       (0L, 2L, 1L, 1L, -0.0005491119809448719),
#       (1L, 0L, 2L, 0L, 0.0046770572662353516),
#       (1L, 1L, 2L, 0L, 3.1331393984146416e-05),
#       (1L, 0L, 2L, 1L, -0.003926334902644157),
#       (1L, 1L, 2L, 1L, -2.6302335754735395e-05)], 
#      dtype=[('la', '<u4'), ('ia', '<u4'), ('lb', '<u4'), ('ib', '<u4'), ('v', '<f4')])

import rd # data reading module
import numpy as np #array operations

from numbapro import autojit #partially compile code

netdtype='f32'
def createnet(nin,nhdn_list,nout,**kwargs):#,bias=True):
    """
    creates a network of nodes.
        -nin number of input nodes
        -nhdn_list list number of nodes in the hidden layers
           eg [3,4,9]. first,second,third hidden layer have 3,4,9 nodes
        -nout number of output nodes
        -keyword argument bias=True/False can be set to inlude a bias term
        which is inserted in all layers except the ouput layer
    the output is a dictionary of 1-D arrays representing each layer. so 
    a node is identified by two numbers: layernumber, number(inlayer)
    the parameters specified exclude the bias 'node'
    """
    bias=kwargs.setdefault('bias',True)
    if bias!=True: bias=0
    else: bias=1
    #network    
    net=[np.empty(nin+bias,dtype=netdtype)] #don't see a need for init vals for input
    for ahn in nhdn_list:
        a=np.empty(ahn+bias,dtype=netdtype);#
        net.append( a )
    net.append(np.empty(nout,dtype=netdtype)) #dont see a need for init vals for output
    if bias==True: 
        for anl in net[:-1]: anl[0]=1 #att. 1 for all node layers xcpt last 1
    net=dict(zip(range(len(net)),net))
    return net



def neti(net,direction):#fwd and bwd
    """iteration mechanism through the network.
    beginning with the first layer, it gives the layer index and the nodes
    for the next layer for all layers of the network until the one before 
    the last layer.
    by not specifiying 'fwd' for the direction arg. the iteration can be
    reversed.
    """
    if direction=='fwd': direction=False
    else: direction=True
    lk=sorted(net.keys(),reverse=direction); lki=0
    for alk in lk[:-1]:
        yield alk, ((lk[lki+1],ani) for ani in xrange(len(net[lk[lki+1]])))
        #key of layer feeding into a node, node keys
        lki+=1 


#FUNCTIONS FOR CONNECTIVITY---------------------------------------

def createll(net):
    """a connectivity generator for the given network.
    gives a tuple (nodeA,nodeB) for all connections between two nodes"""
    for alli,nis in neti(net,'fwd'): #left layerindex, right node keys
        for arni in nis: #right node index
            for alni in xrange(len(net[alli])): #left node index in layer
                yield (alli,alni) , arni #left index, right index

def createllv(net,**kwargs):#,initv=.1):
    """a data structure that creates an association "table"
    b/w links and a value..for weights
    default initial values are uniformly distributed (-.1,1) """
    initv=kwargs.setdefault('initv',.1)
    return np.array([(   a[0],a[1],b[0],b[1]
                         ,(initv--initv)*np.random.rand()+-initv  )#rnd[+a,-a]
                      for a,b in createll(net)]
    ,dtype=[('la','uint'),('ia','uint'),('lb','uint'),('ib','uint')
            ,('v','f32')]) #layerA, indexA for a node (in a layer),...,'v'alue

vintonode_cache={}#lesson: should have split the connectivity table from the
#values for more efficiency
def vintonode(anodei,llv,direction):
    """returns indexes of nodes pointing to specified node in a given direciton.
    (a helper function)"""
    #(anodei),hash(str(llv)),(direction)
    if ((anodei),(direction)) in vintonode_cache:
        return vintonode_cache[(anodei,direction)]
    if direction=='fwd': la=['lb','ib']
    else: la=['la','ia']
    # llv[llv[llv[la[0]]==anodei[0]][la[1]]==anodei[1]] #comprende??
    #T/F array that satisfies conditions
    vinto= np.where(np.multiply(llv[la[0]]==anodei[0] #multiply two 
                    ,llv[la[1]]==anodei[1])) #conditions to get AND
    vintonode_cache[(anodei,direction)]=vinto
    return vinto
    #returns (array,) for some reason                   
#------------------------------------------------------------


#NODE FUNCTIONS---------------------------------------

@autojit
def nf(x): return 1/(1+2.7182818284590451**(-x)) #sigmoid
@autojit
def no(netl,weights):
    """nueron output"""
    dp=np.dot(netl,weights)
    return nf(dp)
#-------------------------------------------------------



#PROPAGATION FUNCTIONS-----------------------------------

def fwdp(net,weights,inputv):
    """forward propagate the input through the network given the weights 
    and the input vector"""
    x=inputv
    if type(x) is not np.ndarray: x=np.array(x,dtype=netdtype)
    if len(inputv)+1==len(net[0]): il=1 #there is a bias node
    else: il=0
    #assert(len(net[0][il:])==len(x))
    net[0][il:]=np.array(x,dtype=net[0].dtype) #assign input to network
    #go fwd thru net
    for alinis in neti(net,'fwd'): #layerindex, node keys pointed to
        ali=alinis[0];nis=alinis[1]
        #fwdpcalc(nis,net,ali,weights)        
        for an in nis: #a node in node keys
            #assert(len(net[ali])==len(weights[vintonode(an,weights,'fwd')]['v']))
            net[an[0]][an[1]]=\
            no(net[ali],weights[vintonode(an,weights,'fwd')]['v'])
    return net[max(net.keys())].copy() #careful
#def fwdpcalc(nis,net,ali,weights):
#    for an in nis:
#        net[an[0]][an[1]]=\
#            no(net[ali],weights[vintonode(an,weights,'fwd')]['v'])

from copy import deepcopy
def errp(net,weights,targetout):
    """backpropagate the errors through the network
    given weights and target output"""
    d=deepcopy(net);#use same data structure for errors
    d.pop(0) #don't need the first (input) layer
    t=targetout;w=weights
    li=sorted(net.keys())[-1] #(last) output layer
    #for each net output o_k unit calc err term:
    ok=net[li];
    #assert(len(ok)==len(t))
    d[li]=ok*(1-ok)*(t-ok); del ok; del t;#dk=d[li];dk=ok*(1-ok)*(t-ok) #NOOOO!
    #calc errors for hidden nodes
    for alinis in neti(d,'bwd'): #for alayer, nodes in next layer (backward)
        ali=alinis[0];nis=alinis[1] 
        for an in nis: #for anode in 
            #assert(len(d[ali])==len(w[vintonode(an,w,'bwd')]['v'])  )
            #print net[an[0]][an[1]],np.dot(d[ali],w[vintonode(an,w,'bwd')]['v'])
            d[an[0]][an[1]]=net[an[0]][an[1]]*(1-net[an[0]][an[1]])\
            *np.dot(d[ali],w[vintonode(an,w,'bwd')]['v'])
            #errpcalc(an,nis,w,net,d,ali)
    return d
#def errpcalc(an,nis,w,net,d,ali):
#    d[an[0]][an[1]]=net[an[0]][an[1]]*(1-net[an[0]][an[1]])\
#    *np.dot(d[ali],w[vintonode(an,w,'bwd')]['v'])

#import numexpr
@autojit
def fDw(net,weights,errs,eta):#,**kwargs):#eta=.05):
    """calculates Delta weights given: errors and learning rate eta"""
    #eta=.1#kwargs.setdefault('eta',.1)
    w=weights; d=errs;
#    for aw in w:
#        print eta,d[aw['lb']][aw['ib']],net[aw['la']][aw['ia']]
#    dws=[eta*d[aw['lb']][aw['ib']]*net[aw['la']][aw['ia']]
#                   for aw in w]
    dws=np.empty(len(w))
    for ai2w in xrange(len(w)):#numbapro.prange(len(w)):#doesn't work
    #for index to weight
        dws[ai2w]=eta*d[w[ai2w]['lb']][w[ai2w]['ib']]\
                   *net[w[ai2w]['la']][w[ai2w]['ia']]
    return dws
#    dj=np.fromiter([d[aw['lb']][aw['ib']] for aw in w],netdtype
#    ,count=len(w))
#    #,dtype=netdtype)
#    xji=np.fromiter([net[aw['la']][aw['ia']] for aw in w],netdtype
#    ,count=len(w))
    #,dtype=netdtype)
#    return eta#*dj*xji
    #numexpr to the rescue!
    #return numexpr.evaluate('eta')#eta*dj*xji')
#-----------------------------------------------------------------
    


#TRAINING FUNCTIONS-----------------------------------------------

#functions and data structures related to classification-
digit2target={}
i2digit={} # ['0','1','2',..] indexed as 0,1,2...
hashstrtarget2digit={}
for ad in rd.digits:
    #np.array([int(ad)],dtype=netdtype) #1 unit for output [0...9]
    tda=np.zeros(len(rd.digits),dtype=netdtype) #10 units 2:[0,0,1,0,0,0....]
    i2digit[np.where(rd.digits==ad)[0][0]]=ad
    tda[np.where(rd.digits==ad)[0][0]]=1
    digit2target[ad]=tda
    hashstrtarget2digit[hash(str(tda))]=ad
del ad
def classof(vector): return i2digit[np.argmax(vector)]
#---
    
@autojit
def trainex(inputvec_targetout,net,weights,eta):#,**kwargs):
    """gives Delta weight for a (training) example"""
    x=inputvec_targetout[0];
    t=inputvec_targetout[1];
    w=weights;
    fwdp(  net,w,x)#,**kwargs)
    d=errp(net,w,t)#,**kwargs)
    Dw=fDw(net,w,d,eta)#,**kwargs)
    return Dw

from collections import deque
def trainexs(net,weights,**kwargs):
    """main training routine:operates in-place on the given weights.
    a part of the training set is set for validation. the convergence is 
    assessed against the validation set after each epoch.
    """
    global vintonode_cache;vintonode_cache={} #global...ewww
    maxcloops=kwargs.setdefault('maxcloops',100) #max epoch loops
    mincloops=kwargs.setdefault('mincloops',10) #min epoch loops
    alpha=kwargs.setdefault('alpha',.5) #momentum
    eta=kwargs.setdefault('lrate',.1)   #learning rate
    ccritpct=float(kwargs.setdefault('ccritpct',5))#asympotic stopping criteria
    validpct=float(kwargs.setdefault('validpct',20))#pct of traindata for validation
    trainds=kwargs.setdefault('trainds',rd.train) #training dataset
    trainds=list(genrandexs(trainds)) 
    vn=int((validpct*.01)*len(trainds))
    tn=len(trainds)-vn
    ts=trainds[:tn]; #training set
    vs=trainds[tn:]  #validation set

    #initializations
    wl=[] #container for history of weights
    Dw=np.zeros_like(weights['v']) #space for Delta weights
    vfc=validate(net,weights,vs) 
    tfc=validate(net,weights,ts)
    maxv=0.0001
    tchanges=deque(maxlen=mincloops) #keeping last mincloops of test...
    vchanges=deque(maxlen=mincloops) #...and validation fraction correct
    print 'epoch | % correct: \tvalidation \ttrain'
    for i in xrange(maxcloops):
        for example in ts:
            Dwp=Dw #previous Dw
            Dw=trainex(example,net,weights,eta)+Dwp*alpha
            weights['v']+=Dw
            wl.append(weights['v'].copy())
            Dwp=Dw
        oldtfc=tfc
        oldvfc=vfc
        vfc=validate(net,weights,vs);
        tfc=validate(net,weights,ts)
        maxv=max(maxv,vfc)
        print i+1\
             ,'\t\t\t%(pc)g' % {'pc':vfc*100}\
               ,'\t\t%(pc)g' % {'pc':tfc*100}
        tchange=(tfc-oldtfc)/oldtfc
        vchange=(vfc-oldvfc)/oldvfc
        tchanges.append(tchange)        
        vchanges.append(vchange);
#        print '\t\t\t',np.average(vchanges)\
#             ,'\t\t',np.average(tchanges)
        #the expected trend is asymptotic. stop when training going up
        #if np.average(vchanges)<-ccrit\
        if (vfc-maxv)/maxv<-ccritpct*.01\
                    and i>=mincloops-1:
    #and  np.average(tchanges)> ccrit\#(/*while validation going down)*/
                   break
    if i==maxcloops-1: print "didn't converge in allowed time"
    
    errs=1-vfc;
    n=len(trainds)
    pm=1.96*(errs*(1-errs)/n)**.5
    return {'w':weights
            ,'vfc':vfc,'tfc':tfc 
            ,'whist':np.array(wl)
            ,'95conf':(errs-pm,errs+pm)
           }

def committee(net,weights,**kwargs):
    """an attempt to minimize (stabilize?) error by passing weights
    from one committee  member to the next"""
    n=kwargs.setdefault('n',10)
    #NOTE the weights are passed from one training to the next
    results=([trainexs(net,weights,**kwargs) for i in xrange(n)])
    return results#([r for r in results])

def validate(net,weights,examples):
    """returns fraction of correctly classified examples"""
    evals=np.array([
    classof(fwdp(net,weights,ex[0]))
                        ==hashstrtarget2digit[hash(str(ex[1]))]
    for ex in examples],dtype=bool)
    return np.sum(evals)/float(len(evals))
#------------------------------------------------------------------



#DATA RECALL FUNCTIONS----------------------------------------------
def genexs(ds,**kwargs):
    """gives examples from the dataset ds: (input vector,target vector)""" 
    for avec,adigit in rd.getdata(ds):
        yield avec, digit2target[adigit]


def genrandexs(ds,**kwargs):
    """gives random examples selected from the dataset ds
    specify a number to be given with keyword argument n=123. otherwise
    it will give all examples from the ds"""
    n=kwargs.setdefault('n','all')
    exs=list(genexs(ds,**kwargs))
    if n=='all': n=None
    i2exs=np.arange(len(exs),dtype='uint32')
    np.random.shuffle(i2exs)
    for i in i2exs[:n]: yield exs[i]
#---------------------------------------------------------------






#"main" functions

def inittraindigits(nhdn_list,**kwargs):
    """creates the data structures to be processed with to  recognize digits"""
    nt=len(digit2target[rd.digits[0]])#=number of target nodes
    ni=len(rd.train[rd.digits[0]][0])#=64 num of input nodes
    net=createnet(ni,nhdn_list,nt,**kwargs)
    weights=createllv(net,**kwargs)
    return {'net':net,'w':weights}#,'d2t':d2t}


def digitsprogram(hiddenN,bias=False,initweight=.1
    ,eta=.1,alpha=.5,validpct=20.0):
    nnc=inittraindigits([hiddenN],bias=bias,initv=initweight)
    return trainexs(nnc['net'],nnc['w'],eta=eta,alpha=alpha)

if '__main__'==__name__:
    import argparse
    parser = argparse.ArgumentParser(description='recognize digits'
    ,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('nhdn', metavar='nH', type=int,help='number of hidden units')
    parser.add_argument('--eta',default=.1,type=float,help='learning rate')
    parser.add_argument('--alpha',default=.5,type=float,help='momentum')
    parser.add_argument('--bias',default=False,type=bool,help='bias terms?')    
    parser.add_argument('--initweights',default=.1,type=float
    ,help='weights with uniform distribution (+w,-w)')    
    parser.add_argument('--validpct',default=20,type=float
    ,help='percentage of training data set used for validation')
    args=parser.parse_args()
    results=digitsprogram(args.nhdn,bias=args.bias,initweight=args.initweights
                  ,eta=args.eta,alpha=args.alpha,validpct=args.validpct)
    for aw in results['w']: print aw['la'],aw['ia'],aw['lb'],aw['ia'],aw['v']
    print '(layer,index)(layer,index)(weight)'
    errs=1-results['vfc'];
    n=int((1-args.validpct*.01)*len(list(genexs(rd.train))))
    pm=1.96*(errs*(1-errs)/n)**.5
    print '95% confidence interval: ', errs-pm,',',errs+pm
    
    
    