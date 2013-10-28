

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

import rd
import numpy as np

import numbapro
from numbapro import autojit

netdtype='f32'
def createnet(nin,nhdn_list,nout,**kwargs):#,bias=True):
    """the parameters exclude the bias 'node'"""
    bias=kwargs.setdefault('bias',True)
#    if bias!=True: b=0
#    else: b=1
#    n=np.empty(sum([nin+b]+[sum(nhdn_list)+b*len(nhdn_list)]+[nout+b])
#    ,dtype='f32') #nodes
#    n.fill(initv)
    if bias!=True: bias=0
    else: bias=1    
    net=[np.empty(nin+bias,dtype=netdtype)] #don't see a need for init vals for input
    for ahn in nhdn_list:
        a=np.empty(ahn+bias,dtype=netdtype);# a.fill(initv)
        net.append( a )
    net.append(np.empty(nout,dtype=netdtype)) #dont see a need for init vals for output
    if bias==True: 
        for anl in net[:-1]: anl[0]=1 #att. 1 for all node layers xcpt last 1
    net=dict(zip(range(len(net)),net))
    
    #ll=[] #link list defining connectivity
#    li=0
#    for alayer in (net[:-1]):#connecting layers up to output layer 
#        for anodei in xrange(len(alayer)):
#            for afwdnodei in xrange(len(net[li+1])):
#                ll.append(  (li,anodei,   li+1,afwdnodei) )
#        li+=1
   # ll=[ni for ni in neti(net)]
#    ll=np.array(ll #record array. this allows sorting for fwd and bwd 
#    ,dtype=[('la','uint'),('ia','uint'),('lb','uint'),('ib','uint'),])
             #layer a, index of a node in a
    #ws= np.empty(len(ll),dtype='f32')
    #ws.fill(initw)
    return net#{'net':net, 'll':ll, 'ws':ws}
    
def createll(net):
    """connectivity of net"""
    for alli,nis in neti(net,'fwd'): #left layerindex, right node keys
        for arni in nis: #right node index
            for alni in xrange(len(net[alli])): #left node index in layer
                yield (alli,alni) , arni #left index, right index

def createllv(net,**kwargs):#,initv=.1):
    """a data structure that creates an assoc b/w 
    links and a value..for weights """
    initv=kwargs.setdefault('initv',.1)
    return np.array([(   a[0],a[1],b[0],b[1]
                         ,(initv--initv)*np.random.rand()+-initv  )#rnd[+a,-a]
                      for a,b in createll(net)]
    ,dtype=[('la','uint'),('ia','uint'),('lb','uint'),('ib','uint')
            ,('v','f32')]) #layerA, indexA for a node (in a layer),...,'v'alue


vintonode_cache={}
def vintonode(anodei,llv,direction):
    """returns indexes to values assoc with 
    nodes pointing to specified node index"""
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
                    



#nn traversal: get keys for a layer going into a node
    #and /that/ node's keys

def neti(net,direction):#fwd and bwd
    """iteration mechanism through the network"""
    if direction=='fwd': direction=False
    else: direction=True
    lk=sorted(net.keys(),reverse=direction); lki=0
    for alk in lk[:-1]:
        yield alk, ((lk[lki+1],ani) for ani in xrange(len(net[lk[lki+1]])))
        #key of layer feeding into a node, node keys
        lki+=1 


#node functions
@autojit
def nf(x): return 1/(1+2.7182818284590451**(-x))
@autojit
def no(netl,weights):
    """nueron output"""
    dp=np.dot(netl,weights)
    return nf(dp)


def fwdp(net,weights,inputv):
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
def errp(net,weights,targetout):#"backpropagation"
    d=deepcopy(net); d.pop(0) #don't need the first (input) layer
    t=targetout;w=weights
    #for each net output o_k unit calc err term:
    li=sorted(net.keys())[-1] #(last) output layer
    #fi=sorted(net.keys())[0] #1st
    ok=net[li];
    #assert(len(ok)==len(t))
    d[li]=ok*(1-ok)*(t-ok); del ok; del t;#dk=d[li];dk=ok*(1-ok)*(t-ok) #NOOOO!
    #calc errors for hidden
    for alinis in neti(d,'bwd'):
        ali=alinis[0];nis=alinis[1]
        #if ali==fi+1: break #dont want to update the input layer
        for an in nis:
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
def fDw(net,weights,errs):#,**kwargs):#eta=.05):
    eta=.1#kwargs.setdefault('eta',.1)
    w=weights; d=errs;
#    for aw in w:
#        print eta,d[aw['lb']][aw['ib']],net[aw['la']][aw['ia']]
#    dws=[eta*d[aw['lb']][aw['ib']]*net[aw['la']][aw['ia']]
#                   for aw in w]
    dws=np.empty(len(w))
    for ai2w in xrange(len(w)):#numbapro.prange(len(w)):#doesn't work
        dws[ai2w]=eta*d[w[ai2w]['lb']][w[ai2w]['ib']]*net[w[ai2w]['la']][w[ai2w]['ia']]
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

def shouldstop(neww,oldw,**kwargs):#crit=.001):#assuming same order!
    crit=kwargs.setdefault('crit',.001)    
    fc=np.abs(neww-oldw)/oldw
    return np.all(fc<crit)

@autojit
def trainex(inputvec_targetout,net,weights):#,**kwargs):
    x=inputvec_targetout[0];
    t=inputvec_targetout[1];
    w=weights;
    fwdp(  net,w,x)#,**kwargs)
    d=errp(net,w,t)#,**kwargs)
    Dw=fDw(net,w,d)#,**kwargs)
    return Dw


digit2target={} #digit2target
for ad in rd.digits:
    tda=np.zeros(len(rd.digits),dtype=netdtype)
    tda[np.where(rd.digits==ad)[0]]=1
    digit2target[ad]=tda #10 units
    #digit2target[ad]=np.array([int(ad)],dtype=netdtype) #1 unit
del ad

def inittraindigits(nhdn_list,**kwargs):
    nt=len(digit2target[rd.digits[0]])#=number of target nodes
    ni=len(rd.train[rd.digits[0]][0])#=64 num of input nodes
    net=createnet(ni,nhdn_list,nt,**kwargs)
    weights=createllv(net,**kwargs)
    return {'net':net,'w':weights}#,'d2t':d2t}


wl=[]
def trainexs(net,weights,**kwargs):
    #must go thru all examples once
    alpha=kwargs.setdefault('alpha',.5)     
    for example in genrandtrainingexs(**kwargs):
        Dw=trainex(example,net,weights)#,**kwargs)
        weights['v']+=Dw
    #return weights
    #convergence loop
    for i in xrange(10):
        print 'convergence pass',1+i
        for example in genrandtrainingexs(**kwargs):
            #weight=weights.copy()
            Dwp=Dw
            Dw=trainex(example,net,weights)+Dwp*alpha
            weights['v']+=Dw
            wl.append(weights['v'].copy())
            Dwp=Dw
#            if True==shouldstop(weights2['v'],weights['v'],**kwargs):
#                return weights2
#            #print 'weight diff abs sum=',sum(np.abs(weights2['v']-weights['v']))
#            weights=weights2
    #print "didn't converge"
    return weights




def gentrainingexs(**kwargs):#ts=rd.train):
    ts=kwargs.setdefault('ts',rd.train) #training set
    for avec,adigit in rd.getdata(ts):
        yield avec, digit2target[adigit]    

def genrandtrainingexs(**kwargs):
    tex=list(gentrainingexs(**kwargs))
    i2tex=np.arange(len(tex),dtype='uint32')
    np.random.shuffle(i2tex)
    for i in i2tex: yield tex[i]
    




#    
#    