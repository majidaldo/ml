import svm
import numpy as np

import pyOpt


# def testprob():
#     def ft(x): return (3+x[0]**2,np.array([0]),0)
#     optprob=pyOpt.Optimization('test',ft)
#     optprob.addObj('of')
#     optprob.addVarGroup('a',1,lower=-10,upper=10)
#     optprob.addCon('saye0',type='e',equal=0)
#     return optprob



def constrains(alphas):
    say=np.dot(alphas,svm.yxs['y'])#alphas,ys
    #cv=np.empty(len(alphas+1))
    #cv[0]=say #for equality contrain
    #cv[1:]=alphas #for inequality
    return np.array([say])#cv


def pyoptof(alphas,**kwrags):
    f=-svm.of(alphas)
    g=constrains(alphas)
    return f,g,0



def svmprobi():
    optprob=pyOpt.Optimization('test',pyoptof)
    optprob.addObj('of')
    optprob.addVarGroup('a',len(svm.yxs['x']),value=svm.C/2.0
          ,lower=0,upper=svm.C)
    #optprob.addCon('saye0',type='e',equal=0)
    optprob.addCon('saye0',type='i',lower=-1e-7,upper=1e-7)
    return optprob
def svmprobe():
    optprob=svmprobi()
    optprob.delCon(0)  
    optprob.addCon('saye0',type='e',equal=0)
    return optprob


def saveopt(class1,class2,alphas):
    of=open(fnhash(class1,class2),'w')
    for aa in alphas: of.write(str(aa)+'\n')
    of.close()
def fnhash(class1,class2): return str(abs(hash((class1,class2))))+'.a'
# 1,2,4,5,7,9
#from ipyton.paralllel import Client
#client=Client()
#%px execfile(svmmain)
#dv=client[:]
#dv.map_sync(trainnots,  ['1','2','4','5','7','9'])

def trainnots(digits):
    o=[]
    for ad in digits:
        nd='!'+ad
        svm.init(ad,nd)
        nsga=pyOpt.pyNSGA2.pyNSGA2.NSGA2()
        nsga.setOption('PrintOut',0)
        opto=nsga(svmprobi())
        saveopt(ad,nd,opto[1])
        o.append(opto)
    return o
    
def loadopt(class1,class2): return np.loadtxt(fnhash(class1,class2),dtype=svm.mf)

def train(class1,class2):
    svm.init(class1,class2)
    nsga=pyOpt.pyNSGA2.pyNSGA2.NSGA2()
    nsga.setOption('PrintOut',0)
    opto=nsga(svmprobi())
    saveopt(ad,nd,opto[1])
    return opto


test= zip(*[(ak,av) for ak in svm.rd.digits for av in svm.rd.test[ak]])
test={'y':np.array(test[0],dtype=svm.mui),'x':np.array(test[1],dtype=bool)}

def classify10(xc):
    sums=np.empty((len(svm.rd.digits),len(xc)),dtype=svm.mf)
    for ad in svm.rd.digits:
        svm.init(ad,'!'+ad)
        sums[int(ad)]=svm.evalsums(xc,loadopt(ad,'!'+ad))
    return np.argmax(sums,axis=0)
def evalclassify10(ys,xc):
    correct=(ys==xc)
    return float(sum(correct))/len(correct)
    

def classify210():
    br=[]
    for ad in svm.rd.digits:
        svm.init(ad,'!'+ad)
        sp=svm.evalsep(svm.yxstest,loadopt(ad,'!'+ad))
        br.append(sum(sp)/float(len(sp)))
    return br

#fsqp=pyOpt.pyFSQP.FSQP() #optimizer did not compile!
#psqp=pyOpt.pyPSQP.PSQP() #the most appropriate but..obj func blows up
#algencan=pyOpt.pyALGENCAN.ALGENCAN() #did not compile!
#slsqp=pyOpt.pySLSQP.SLSQP() #fast but have to retry
nsga=pyOpt.pyNSGA2.pyNSGA2.NSGA2()#equality constraint not supported
#coblya=pyOpt.pyCOBYLA.pyCOBYLA.COBYLA()#equality not supported
#sdpen=pyOpt.pySDPEN.SDPEN() #cannot handle equality


#change inputs in svm.py
#svm.init(class1,class2) #includes !class
#optimize: slsqp(svmprob())
#classify: svm.separate(svm.classi)

