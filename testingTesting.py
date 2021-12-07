import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import fmin
from scipy.optimize import fmin_tnc
from scipy import optimize
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

delta = 0.2

def rescaledPotential(phi):
    return phi**4/4 - phi**3 + delta*phi**2/2

def equations(r, y):
    phi = y[0]
    v = y[1]        #dphi/dr
    
    dphidr = v
    dvdr = - (3/r)*v + phi**3 -3*phi**2 + delta*phi        #ändra, - (3/r)*v + dVdphi(phi)
    
    return np.array([dphidr, dvdr])

phi_span = np.linspace(-2, 4, 100)
potential = rescaledPotential(phi_span)

def makePlotsPotential(phi_span, potential):

    plt.figure()
    ax = plt.subplot(121)
    ax.plot(phi_span, potential)
    ax.set_xlabel('$\phi$')
    ax.set_ylabel('$V(\phi)$')             

    ax = plt.subplot(122)
    ax.plot(phi_span, -potential)
    ax.set_xlabel('$\phi$')
    ax.set_ylabel('$- V(\phi)$')
    plt.show()

#makePlotsPotential(phi_span, potential)

r = np.linspace(0.001, 10, 100)
xspan = (r[0], r[-1])
dphidr0 = 0
f0 = [2.8, dphidr0]

sol = solve_ivp(equations, xspan, f0, t_eval=r)           # atol=1e-20, rtol=1e-10, max_step=1e-7
print(sol.message)
y_num = sol.y[0, :]

for solution in sol.y:
    y = [y for y in solution if y >= 0]
    t = sol.t[:len(y)]
    plt.plot(t, y)
plt.show()

plt.plot(r, y_num, 'b.')            #undershoot w f0 = phi0, 0, skumt bör bli overshoot
plt.show()


def objective(phi0):
    sol = solve_ivp(equations, xspan, \
            [phi0, 0], t_eval = r)
    y = sol.y[0]
    print(y[-1])
    return abs(y[-1] - phiFalseVacuum)                    #ändra, som det står nu y[-1] - 0 jämför vi sista värdet på y givet sista värdet på r --- ger oscillationer runt 0

def root(phi0, f0):
    phi0, = fsolve(objective, phi0)
    print(phi0)
    f0 = [phi0, 0]
    sol = solve_ivp(equations, xspan, f0, t_eval = r)
    y_num = sol.y[0, :]
    return y_num

plt.plot(r, y_num, 'b.')            #undershoot w f0 = phi0, 0, skumt bör bli overshoot
plt.xlabel('r')
plt.ylabel('$\phi$')
plt.plot(r, sol.y[1, :], 'r.')
plt.show()
