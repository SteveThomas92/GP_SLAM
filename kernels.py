import numpy as np
import numpy.random as rnd
import scipy.linalg as la

def confidenceInterval(cov):
    return 1.96 * np.diag(cov)
# This will be used to modify for use with non-stationary Kernels
def multVarDiffMat(x,y):
    p,n = x.shape
    q = len(y)

    X = np.zeros((p,q,n))
    Y = np.zeros((p,q,n))
    for i in range(p):
        for j in range(q):
            X[i,j] = x[i]
            Y[i,j] = y[j]
            
    return X-Y

def diffMat(t1,t2):
    """
    Scalar diffmat
    Use multivar diffmat for multivariable case
    """
    if len(t1.shape) > 1:
        T = multVarDiffMat(t1,t2)
    else:
        T = np.outer(t1,np.ones(len(t2))) - np.outer(np.ones(len(t1)),t2)
    return T

class gaussianProcess:
    def __init__(self,x=None,y=None):
        self.set_data(x,y)
        
    def k(self,x1,x2):
        return np.zeros((len(x1),len(x2)))

    def set_data(self,x=None,y=None):
        if x is None:
            self.x = None
        else:
            self.x = x
            self.K = self.k(x,x)
            self.Kchol = la.cholesky(self.K,lower=True)
            self.scaled_data = la.solve(self.K,y)

    def prediction(self,x):
        """
        Assuming that x is a numpy array of prediction values
        """
        if self.x is None:
            mean = np.zeros(len(x))
            cov = self.k(x,x)
        else:
            kVec = self.k(self.x,x)
            mean = np.dot(kVec.T,self.scaled_data)
            schurFactor = la.solve(self.Kchol,kVec)
            cov = self.k(x,x) - np.dot(schurFactor.T,schurFactor)
        return mean,cov

    def sample(self,x,NumSamples=1):
        mean,cov = self.prediction(x)
        gain = la.cholesky(cov,lower=True)

        if NumSamples == 1:
            WShape = (len(x),)
            meanMat = mean
        else:
            WShape = (len(x),NumSamples)
            meanMat = np.outer(mean,np.ones(NumSamples))
        W = rnd.randn(*WShape)
        return meanMat + np.dot(gain,W)
    
# This will ONLY work for stationary kernels
# Fixing above to work on both
class gaussian(gaussianProcess):
    def __init__(self,theta=1.,maxVar=1.):
        self.theta = theta
        self.maxVar = maxVar
        gaussianProcess.__init__(self)

    def k(self,t1,t2=None):
        if t2 is None:
            t = t1
        else:
            t = diffMat(t1,t2)
        return self.maxVar * np.exp(-self.theta * t**2.)

    def dk(self,t1,t2=None):
        if t2 is None:
            t = t1
        else:
            t = diffMat(t1,t2)
        return -2. * self.theta * t * self.k(t)

    def ddk(self,t1,t2=None):
        if t2 is None:
            t = t1
        else:
            t = diffMat(t1,t2)
        return ((2*self.theta*t)**2. - 2*self.theta) * self.k(t)
        

class inverseMultiquadric(gaussianProcess):
    def __init__(self,theta=1.,maxVar=1.):
        self.theta = theta
        self.maxVar = maxVar

    def kNormed(self,t1,t2=None):
        if t2 is None:
            t = t1
        else:
            t = diffMat(t1,t2)

        return 1./(1.+self.theta * t**2.)
    
    def k(self,t1,t2=None):
        if t2 is None:
            t = t1
        else:
            t = diffMat(t1,t2)

        return self.maxVar * self.kNormed(t)

    def dk(self,t1,t2=None):
        if t2 is None:
            t = t1
        else:
            t = diffMat(t1,t2)

        return -2. * self.maxVar * self.theta * t * (self.kNormed(t)**2.)

    def ddk(self,t1,t2=None):
        if t2 is None:
            t = t1
        else:
            t = diffMat(t1,t2)

        val1 = -2 * self.maxVar * self.theta * (self.kNormed(t)**2.)
        val2 = 2 * self.maxVar * \
               ((2.*self.theta*t)**2.) * (self.kNormed(t)**3.)
        return val1 + val2

class matern1(gaussianProcess):
    def __init__(self,theta=1.,maxVar=1.):
        self.theta = theta
        self.maxVar = maxVar

    def eFun(self,t):
        return self.maxVar * np.exp(-self.theta * np.abs(t))

    def k(self,t1,t2=None):
        if t2 is None:
            t = t1
        else:
            t = diffMat(t1,t2)

        
        return (1.+self.theta*np.abs(t)) * self.eFun(t)

    def dk(self,t1,t2=None):
        if t2 is None:
            t = t1
        else:
            t = diffMat(t1,t2)

        return -self.theta**2. * np.abs(t) * self.eFun(t)

    # Not Twice Differentiable at 0.


class matern2(gaussianProcess):
    def __init__(self,theta=1.,maxVar=1.):
        self.theta = theta
        self.maxVar = maxVar
        p0 = np.array([theta**2.,3.*theta,3.])
        p1 = np.polyadd(np.polyder(p0),-theta*p0)
        p2 = np.polyadd(np.polyder(p1),-theta*p1)
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2

    def eFun(self,t):
        return self.maxVar * np.exp(-self.theta * np.abs(t))
    

    def k(self,t1,t2=None):
        if t2 is None:
            t = t1
        else:
            t = diffMat(t1,t2)

        pVal = np.polyval(self.p0,np.abs(t))
        return pVal * self.eFun(t)

    def dk(self,t1,t2=None):
        if t2 is None:
            t = t1
        else:
            t = diffMat(t1,t2)

        pVal = np.polyval(self.p1,np.abs(t))
        return pVal * self.eFun(t)

    # Also not twice differentiable.


#### Multivariate Gaussian code

def mpolysort(p):
    if len(p) == 0:
        return []
    DList = []
    CList = []
    
    for m in p:
        c,d = m
        CList.append(c)
        DList.append(d)
    
    CArr = np.array(CList)
    DArr = np.array(DList,dtype=int)
    
    
    
    DTup = tuple([DL for DL in DArr.T])
    ind = np.lexsort(DTup[::-1])
    
    DSorted = DArr[ind]
    CSorted = CArr[ind]
    
    cCur = None
    pSorted = []
    for c,d in zip(CSorted,DSorted):
        if cCur is None:
            cCur = c
            dCur = d
        else:
            if np.max(np.abs(d-dCur)) == 0:
                # Same degree. Just add the coefficients
                cCur += c
            else:
                # New degree. Put in the value and increment.
                pSorted.append((cCur,dCur))
                cCur = c
                dCur = d
                
                
    # Put in the final value
    pSorted.append((cCur,dCur))
                
    return pSorted
    
def monomialval(x,m):
    V = np.ones(x.shape[:-1])
    for i in range(len(m)):
        V = V * (x[...,i] ** m[i])
        
    return V
    
def mpolyval(x,p):
    V = np.zeros(x.shape[:-1])
    for m in p:
        c,d = m
        V = V + c * monomialval(x,d)
        
    return V

def mpolyderind(p,ind):
    pDer = []
    for m in p:
        c,d = m
        if d[ind] > 0:
            cDer = d[ind] * c
            dDer = np.array(d,copy=True)
            dDer[ind] = d[ind] - 1
            pDer.append((cDer,dDer))
            
    return mpolysort(pDer)

def mpolyder(p,indexArray):
    curIndArray = np.array(indexArray,copy=True)
    
    pDer = None
    for i in range(len(indexArray)):
        while curIndArray[i] > 0:
            if pDer is None:
                pDer = mpolyderind(p,i)
            else:
                pDer = mpolyderind(pDer,i)
            curIndArray[i] -= 1
    return pDer

def mpolysmul(p,s):
    """
    multiply a polynomial by a scalar
    """
    sp = []
    for c,d in p:
        sp.append((s*c,d))
        
    return sp

def mpolyadd(p1,p2):
    pBoth = p1 + p2
    return mpolysort(pBoth)
    
def mpolymul(p1,p2):
    pTot = []
    for c1,d1 in p1:
        for c2,d2 in p2:
            pTot.append((c1*c2,np.array(d1)+np.array(d2)))
            
    return mpolysort(pTot)


normSquareMat = lambda M : np.sum(M**2.,axis=2)

class mvGaussian:
    def __init__(self,theta=1.,maxVar=1.):
        self.theta = theta
        self.maxVar = theta
        
    def k0(self,x,y):
        D = multVarDiffMat(x,y)
        M = normSquareMat(D)
        return self.maxVar * np.exp(-self.theta*M)
    
    def k(self,x,y,XDerivativeIndex = None,YDerivativeIndex = None):
        if (XDerivativeIndex is None) and (YDerivativeIndex is None):
            # No Derivatives
            return self.k0(x,y)
        else:
            n = x.shape[-1]
            p = [(1.,np.zeros(n,dtype=int))]
            TotalDerivativeIndex = np.zeros(n,dtype=int)
            psign = 1.
            if XDerivativeIndex is not None:
                TotalDerivativeIndex += XDerivativeIndex
            if YDerivativeIndex is not None:
                TotalDerivativeIndex += YDerivativeIndex
                psign = (-1.)**(np.sum(YDerivativeIndex))
                
            p = [(psign,np.zeros(n,dtype=int))]
            
            for i in range(n):
                eVec = np.zeros(n,dtype=int)
                eVec[i] = 1
                for j in range(TotalDerivativeIndex[i]):
                    pDer = mpolyderind(p,i)
                    pMul = mpolymul(p,[(-2.*self.theta,eVec)])
                    p = mpolyadd(pDer,pMul)
                    
            D = multVarDiffMat(x,y)
            pVal = mpolyval(D,p)
            return pVal * self.k0(x,y)
