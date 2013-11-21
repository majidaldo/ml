import svm
import numpy as np

import pyOpt



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



def testprob():
    def ft(x): return (3+x[0]**2,np.array([0]),0)
    optprob=pyOpt.Optimization('test',ft)
    optprob.addObj('of')
    optprob.addVarGroup('a',1,lower=-10,upper=10)
    optprob.addCon('saye0',type='e',equal=0)
    return optprob
    


#fsqp=pyOpt.pyFSQP.FSQP() #optimizer did not compile!
psqp=pyOpt.pyPSQP.PSQP() #the most appropriate but..obj func blows up
#algencan=pyOpt.pyALGENCAN.ALGENCAN() #did not compile!
slsqp=pyOpt.pySLSQP.SLSQP() #fast but have to retry
nsga=pyOpt.pyNSGA2.pyNSGA2.NSGA2()#equality constraint not supported
coblya=pyOpt.pyCOBYLA.pyCOBYLA.COBYLA()#equality not supported
sdpen=pyOpt.pySDPEN.SDPEN() #cannot handle equality


#change inputs in svm.py
#reload(svm)
#optimize: slsqp(svmprob())
#classify: svm.separate(svm.classi)
#
