import svm
import numpy as np

import pyOpt

def svmprob():
    optprob=pyOpt.Optimization('test',svm.pyoptof)
    optprob.addObj('of')
    optprob.addVarGroup('a',len(svm.yxs['x']),lower=0,upper=svm.C)
    optprob.addCon('saye0',type='e',equal=0)
    #opt=pyOpt.pyPSQP.PSQP()
    return optprob


def testprob():
    def ft(x): return (3+x[0]**2,np.array([0]),0)
    optprob=pyOpt.Optimization('test',ft)
    optprob.addObj('of')
    optprob.addVarGroup('a',1,lower=-10,upper=10)
    optprob.addCon('saye0',type='e',equal=0)
    #opt=pyOpt.pyPSQP.PSQP()
    return optprob
    


#psqp=pyOpt.pyPSQP.PSQP()
#psqp.options['MIT']=[int,1]
slsqp=pyOpt.pySLSQP.SLSQP()
#optr=opt(optprob)