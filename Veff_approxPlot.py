import numpy as np
import scipy
import matplotlib.pyplot as plt

class Veff_approx():

    def setSM(self, mW, mZ, mh, v, mt):
      self.mW = mW
      self.mZ = mZ
      self.mh = mh
      self.v = v
      self.mt = mt

    def setParam(self, lambda_h, lambda_s, lambda_hs):
        self.lambda_s = lambda_s
        self.lambda_h = lambda_h
        self.lambda_hs = lambda_hs

    def setParamSM(self, g1, g2, yt):
        self.g1 = g1
        self.g2 = g2
        self.yt = yt

    def setDOF(self, n_h, n_GB, n_W, n_Z, n_t, n_s):
        self.n_h = n_h
        self.n_GB = n_GB
        self.n_W = n_W
        self.n_Z = n_Z
        self.n_t = n_t
        self.n_s = n_s


    def setVEVs(self, T, TC):
        self.v_h = (1 - (2*self.c_h/(self.m_h**2))*T**2)*self.v**2
        self.v_s = (self.m_h/(2*self.v)*self.v_h**2*np.sqrt(2*self.lambda_s) + self.c_s*TC**2 - self.c_s*T**2)/self.lambda_s

    def setMu(self):
        self.mu_h = self.lambda_h*self.v_h
        self.mu_s = -1*self.lambda_s*self.v_s

    def setTermCoeff(self):
        self.c_h = (2*self.mW**2 + self.mZ**2 + 2*self.mt**2)/(4*self.v**2) + self.lambda_hs/24
        self.c_s = (2*self.lambda_hs + 3*self.lambda_s)/12

    def info(self):
        print ('Parameters:')

    def init(self, lambda_s, lambda_hs, TC):
        """
        """
        self.TC = TC
        
        # Masses (and vev) at 0T [GeV]
        mW = 80.385   #W boson mass
        mZ = 91.1876  #Z boson mass
        mh = 125.1    #Higgs mass 
        v = 246.22    #Higgs VEV
        mt = 173.34   #Top quark mass

        self.v = v

        # Couplings at EW h = v
        g1 = 4*mW**2/v**2               #g coupling squared
        g2 = 2*(mZ**2 - mW**2)/v**2     #g' coupling squared
        yt = 2*mt**2/v**2               #top coupling squared
        
        lambda_h = mh**2/(2*v**2)
        self.mu_h = mh**2/2              #squared

        self.setParam(lambda_h, lambda_s, lambda_hs)
        self.setParamSM(g1, g2, yt)
        self.setTermCoeff()

        self.info()

    def V0(self, X):
        X = np.asanyarray(X)
        h,s = X[...,0], X[...,1]
        y = -0.5*self.mu_h*h**2 + 0.25*self.lambda_h*h**4 + 0.5*self.mu_s*s**2 + 0.25*self.lambda_s*s**4
        y += 0.25*self.lambda_hs*h**2*s**2
        return y
    
    def VTapprox(self, X, T):
        X = np.asanyarray(X)
        T = np.asanyarray(T)
        h,s = X[...,0], X[...,1]
        y = 0.5*self.c_h*h**2*T**2 + 0.5*self.c_s*s**2*T**2
        return y

    # def VTerm(self, X, T):
    #     T = np.asanyarray(T, dtype=float)
    #     X = np.asanyarray(X, dtype=float)
    #     h,s = X[...,0], X[...,1]
    #     term_h = (3*self.g1/16 + self.g2/16 + self.lambda_h/2 + self.yt/4 + self.lambda_s/12)*T**2
    #     term_s = (self.lambda_hs/3 + self.lambda_s/4)*T**2
    #     y = term_h*h**2/2 + term_s*s**2/2
    #     return y

    def Vtot(self, X, T):
        T = np.asanyarray(T, dtype=float)
        X = np.asanyarray(X, dtype=float)
        h,s = X[...,0], X[...,1]
        y = self.V0(X)
        y += self.VTapprox(X, T)
        return y

    def makePlots():
        m = Veff_approx()
        

