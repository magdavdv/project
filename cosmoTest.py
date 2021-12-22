import numpy as np
import scipy
import matplotlib.pyplot as plt
from cosmoTransitions import generic_potential

class sm_singlet(generic_potential.generic_potential):

    def setSM(self, mW, mZ, mh, v, mt):
      self.mW = mW
      self.mZ = mZ
      self.mh = mh
      self.v = v
      self.mt = mt

    def setParam(self, lambda_h, lambda_s, lambda_hs, mu_h, mass_s, v):
        self.lambda_s = lambda_s
        self.lambda_h = lambda_h
        self.lambda_hs = lambda_hs
        self.mu_s = mass_s**2 - lambda_hs*v**2
        self.mu_h = mu_h

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


    # def setVEVs(self, T, TC):
    #     self.v_h = (1 - (2*self.c_h/(self.m_h**2))*T**2)*self.v**2
    #     self.v_s = (self.m_h/(2*self.v)*self.v_h**2*np.sqrt(2*self.lambda_s) + self.c_s*TC**2 - self.c_s*T**2)/self.lambda_s

    # def setMu(self):
    #     self.mu_h = self.lambda_h*self.v_h
    #     self.mu_s = -1*self.lambda_s*self.v_s

    # def setTermCoeff(self):
    #     self.c_h = (2*self.mW**2 + self.mZ**2 + 2*self.mt**2)/(4*self.v**2) + self.lambda_hs/24
    #     self.c_s = (2*self.lambda_hs + 3*self.lambda_s)/12

    def info(self):
        print ('Parameters:')
        print ('lamba_s=',self.lambda_s,',','lambda_hs=',self.lambda_hs)

    def init(self, lambda_s=1, lambda_hs=0.44, mass_s=80):
        """
        """
        self.Ndim = 2
        
        # Masses (and vev) at 0T [GeV]
        mW = 80.385   #W boson mass
        mZ = 91.1876  #Z boson mass
        mh = 125.1    #Higgs mass 
        v = 246.22    #Higgs VEV
        mt = 173.34   #Top quark mass

        # Couplings at EW h = v
        g1 = 4*mW**2/v**2               #g coupling squared
        g2 = 2*(mZ**2 - mW**2)/v**2     #g' coupling squared
        yt = 2*mt**2/v**2               #top coupling squared

        # DOF
        n_h = 1
        n_GB = 3
        n_W = 6
        n_Z = 3
        n_t = 12
        n_s = 1
        
        lambda_h = mh**2/(2*v**2)
        mu_h = mh**2/2              #squared

        self.setParam(lambda_h, lambda_s, lambda_hs, mu_h, mass_s, v)
        self.setParamSM(g1, g2, yt)
        self.setDOF(n_h, n_GB, n_W, n_Z, n_t, n_s)
        self.info()

    def V0(self, X):
        X = np.asanyarray(X)
        h,s = X[...,0], X[...,1]
        y = -0.5*self.mu_h*h**2 + 0.25*self.lambda_h*h**4 + 0.5*self.mu_s*s**2 + 0.25*self.lambda_s*s**4
        y += 0.25*self.lambda_hs*h**2*s**2
        return y
    
    # def VTapprox(self, X, T):
    #     X = np.asanyarray(X)
    #     T = np.asanyarray(T)
    #     h,s = X[...,0], X[...,1]
    #     y = 0.5*self.c_h*h**2*T**2 + 0.5*self.c_s*s**2*T**2
    #     return y

    def boson_massSq(self, X, T):
        X = np.asanyarray(X)
        h,s = X[...,0], X[...,1]

        # Termal corrections to the masses
        term_h = (3*self.g1/16 + self.g2/16 + self.lambda_h/2 + self.yt/4 + self.lambda_s/12)*T**2
        term_s = (self.lambda_hs/3 + self.lambda_s/4)*T**2
        term_W = 11*self.g1*T**2/6
        term_GB = term_h
        term_Z1 = 11*self.g1*T**2/6
        term_Z2 = 11*self.g2*T**2/6

        # Mass matrix for scalars h and s, with termal corrections
        a = -self.mu_h + 3*self.lambda_h*h**2 + self.lambda_hs*s**2/2 + term_h
        b = self.mu_s + 3*self.lambda_s*s**2 + self.lambda_hs*h**2/2 + term_s
        c = self.lambda_hs*s*h
        A = (a + b)/2
        B = np.sqrt((a - b)**2/4 + c**2)

        # Mass matrix for Z, photon
        a_Z = self.g1*h**2/4 + term_Z1
        b_Z = self.g2*h**2/4 + term_Z2
        c_Z = self.g1*self.g2*h**2/4
        A_Z = (a_Z + b_Z)/2
        B_Z = np.sqrt((a_Z - b_Z)**2/4 + c_Z**2)

        # Boson masses
        mW = self.g1*h**2/4 + term_W
        mZ = A_Z + B_Z
        mGB = -self.mu_h + self.lambda_h*h**2 + self.lambda_hs*s**2 + term_GB

        M = np.array([A+B, A-B, mW, mZ, mGB])
        M = np.rollaxis(M, 0, len(M.shape))

        # DOF
        dof = np.array([self.n_h, self.n_s, self.n_W, self.n_Z, self.n_GB])

        # Coeff for CW
        c = np.array([1.5, 1.5, 5/6, 5/6, 1.5])

        return M, dof, c

    def fermion_massSq(self, X):
        Nfermions = 1
        h = X[...,0]
        mt = self.yt*h**2/2
        M = np.empty(mt.shape + (Nfermions,))
        M[...,0] = mt
        dof = np.array([self.n_t])
        return M, dof

    # def Vtot(self, X, T):
    #     T = np.asanyarray(T, dtype=float)
    #     X = np.asanyarray(X, dtype=float)
    #     h,s = X[...,0], X[...,1]
    #     bosons = self.boson_massSq(X,T)
    #     y = self.V0(X)
    #     y += self.VTerm(X, T)
    #     return y

    # def VTerm(self, X, T):
    #     T = np.asanyarray(T, dtype=float)
    #     X = np.asanyarray(X, dtype=float)
    #     h,s = X[...,0], X[...,1]
    #     term_h = (3*self.g1/16 + self.g2/16 + self.lambda_h/2 + self.yt/4 + self.lambda_s/12)*T**2
    #     term_s = (self.lambda_hs/3 + self.lambda_s/4)*T**2
    #     y = term_h*h**2/2 + term_s*s**2/2
    #     return y

    def approxZeroTMin(self):
        """
        Returns approximate values of the zero-temperature minima.

        This should be overridden by subclasses, although it is not strictly
        necessary if there is only one minimum at tree level. The precise values
        of the minima will later be found using :func:`scipy.optimize.fmin`.

        Returns
        -------
        minima : list
            A list of points of the approximate minima.
        """
        # This should be overridden.
        w = np.sqrt(2*(-2*self.lambda_h*self.mu_s - self.lambda_hs*self.mu_h)/(4*self.lambda_h*self.lambda_s - self.lambda_hs**2))
        print(w)
        return [np.array([246., 100])]


def makePlots(m=None):
    import matplotlib.pyplot as plt
    if m is None:
        m = sm_singlet()
        m.findAllTransitions()
    # --
    plt.figure()
    m.plotPhasesPhi()
    plt.axis([0,300,-50,550])
    plt.title("Minima as a function of temperature")
    plt.show()
    # --
    plt.figure(figsize=(8,3))
    ax = plt.subplot(131)
    T = 0
    m.plot2d((-450,450,-450,450), T=T, cfrac=.4,clevs=65,n=100,lw=.5)
    ax.set_aspect('equal')
    ax.set_title("$T = %0.2f$" % T)
    ax.set_xlabel(R"$\phi_1$")
    ax.set_ylabel(R"$\phi_2$")
    ax = plt.subplot(132)
    T = m.TnTrans[1]['Tnuc']
    print(T)
    instanton = m.TnTrans[1]['instanton']
    phi = instanton.Phi
    m.plot2d((-450,450,-450,450), T=T, cfrac=.4,clevs=65,n=100,lw=.5)
    ax.plot(phi[:,0], phi[:,1], 'k')
    ax.set_aspect('equal')
    ax.set_title("$T = %0.2f$" % T)
    ax.set_yticklabels([])
    ax.set_xlabel(R"$\phi_1$")
    ax = plt.subplot(133)
    T = m.TnTrans[0]['Tnuc']
    m.plot2d((-450,450,-450,450), T=T, cfrac=.4,clevs=65,n=100,lw=.5)
    ax.set_aspect('equal')
    ax.set_title("$T = %0.2f$" % T)
    ax.set_yticklabels([])
    ax.set_xlabel(R"$\phi_1$")
    # --
    plt.figure()
    plt.plot(instanton.profile1D.R, instanton.profile1D.Phi)
    plt.xlabel("radius")
    plt.ylabel(R"$\phi-\phi_{min}$ (along the path)")
    plt.title("Tunneling profile")

def getAction(m=None):
    if m is None:
        m = sm_singlet()
        transition = m.calcTcTrans()
        action = transition[0]['action']
        print('Action: ', action)
        print(transition)

if __name__ == "__main__":
    m = sm_singlet()
    m.findAllTransitions()
    m.prettyPrintTnTrans()
    #makePlots()
