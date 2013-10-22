from rd import digits
import numpy as np



def createnet(nin,nhdn_list,nout,initw=.1,initv=10,bias=True):
#    if bias!=True: b=0
#    else: b=1
#    n=np.empty(sum([nin+b]+[sum(nhdn_list)+b*len(nhdn_list)]+[nout+b])
#    ,dtype='f32') #nodes
#    n.fill(initv)
    if bias!=True: bias=0
    else: bias=1    
    net=[np.empty(nin+bias,dtype='f32')] #don't see a need for init vals for input
    for ahn in nhdn_list:
        a=np.empty(ahn+bias,dtype='f32'); a.fill(initv)
        net.append( a )
    net.append(np.empty(nout,dtype='f32')) #dont see a need for init vals for output
    if bias==True: 
        for anl in net: anl[0]=1 #att. 1 for all node layers
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
    ws= np.empty(len(ll),dtype='f32')
    ws.fill(initw)
    return {'net':net}#, 'll':ll, 'ws':ws}
    
#nn traversal: a node and what branches from it
    #or a layer going into a node  ahead <--

def neti(net):#
    li=0
    for alayer in (net[:-1]):#connecting layers up to output layer 
        for anodei in xrange(len(alayer)):
            for afwdnodei in xrange(len(net[li+1])):
                yield  (li,anodei,   li+1,afwdnodei)
        li+=1

#def fwdp(net,ll,ws):
#    ll.sort(order=['la','lb']) #fwd
#    for ali in xrange(len())

    
    

def pf(x): return 1/(1+2.7182818284590451**(-x))
def no(netl,weights):
    dp=np.dot(netl,weights)
    return pf(dp)


#def createnet(nin,nhdn_list,nout,initw=.1,initv=10,bias=True):
#    if bias!=True: bias=0
#    net=[np.empty(nin+bias)] #don't see a need for init vals for input
#    for ahn in nhdn_list:
#        a=np.empty(ahn+bias); a.fill(initv)
#        net.append( a )
#    net.append(np.empty(nout)) #dont see a need for init vals for output
#    w=[];
#    i=0;
#    for al in [ahln for ahln in nhdn_list]+[nout]:#ahl in nhdn_list:
#        a=np.empty( [al,len(net[i])] )#+1 for the bias
#        i+=1
#        a.fill(initw)
#        w.append(a) 
#    w=np.array(w);
#    net=np.array(net)
#    if bias==True: 
#        for anl in net: anl[0]=1 #att. 1 for all node layers
#    return {'n': (net) 
#    ,'w': (w )} #weights correspond to  the middle layers
#    #this return will be my data structuure
#
#def wi2n(nodei,net,weights):
#    """est relation b/w inner nodes and their weights.
#    node index"""
#    return net[1+nodei[0]][nodei[1]]

#def neti(net,weights):#[::-1] reverses
##    li=0; #nl=len(weights); 
##    for alayer in weights:
##        ni=0;
##        for anodeweights in alayer:
##            try: yield net[1+li][ni+1],anodeweights,net[li] #values
##                 # node, its weights , preceding layer i       
##            #, wi2n((li,ni),net,weights)
##            except IndexError: raise StopIteration
##            ni+=1
##        li+=1
#    li=0; nwl=len(weights)#;nnl=len(net) 
#    for alayeri in xrange(nwl):
#        ni=0;
#        for anodeweightsi in xrange(len(weights[alayeri])):
#            if (len(net[1+li])-ni)<2:break#print ni,li,len(weights[alayeri])
#            try:
#                #yield net[1+li][ni+1],weights[alayeri][anodeweightsi],net[li]#values
#                yield (1+li,ni+1),(alayeri,anodeweightsi),li                  
#                 # node, its weights , preceding layer i       
#            #, wi2n((li,ni),net,weights)
#            except IndexError: raise StopIteration #shouldn't come here
#            ni+=1
#        li+=1


#def fwdp(net,weights):
#    for ani, awi, pli in neti(net,weights):
#        net[ani[0]][ani[1]]=no(net[pli],weights[awi[0]][awi[1]])
#        
#def bwdp(net,weights):
#    bl= [(ani, awi, pli) for (ani, awi, pli) in neti(net,weights)]
#    
#    