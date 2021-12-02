import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import fmin
from scipy.optimize import fmin_tnc
from scipy import optimize
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
plt.style.use('seaborn')

a = 125**2
b = 8
c = 125**2/(2*246**2)

def V(phi):
    return -a*phi**2 -b*phi**3 + c*phi**4

phi_span = np.linspace(-400, 400, 100)

V_false_minimum = fmin_tnc(V, 0, approx_grad=True, bounds=[(phi_span[0], 0)])
constant = - V(V_false_minimum[0])
minimum = fmin(V, 0)
phi0 = minimum[0]
print(phi0)

pot = V(phi_span) + constant

plt.figure()
plt.plot(phi_span, pot)
plt.xlabel('$\phi$')
plt.ylabel('$V(\phi)$')
plt.show(block=False)                      #block=False

plt.figure()
plt.plot(phi_span, -pot)
plt.xlabel('$\phi$')
plt.ylabel('$- V(\phi)$')
plt.show()

#------
# Bounce
#  d2phi/dr2 = -3/r dphi/dr + dV/dphi
# BV: dphi/dr = 0 at r = 0, phi(r) = phi_F = 0 (false vacua) at r = inf
# --> IC: dphi/dr = 0 at r = 0, phi(r) = phi0, intital guess that gets updated (need to be close to phi_B, the right solution)

def dVdphi(phi):
    return -2*a*phi -3*b*phi**2 + 4*c*phi**3

def equations(r, y):
    phi = y[0]
    v = y[1]        #dphi/dr
    
    dphidr = v
    dvdr = (-2/(r))*v - dVdphi(phi)        #ändra
    
    return np.array([dphidr, dvdr])

r = np.linspace(0.0001, 0.1, 100)
xspan = (r[0], r[-1])

f0 = [phi0+3000, 0]          #phi0 - first guess, dphi/dr = 0

# plt.plot(phi_span, dVdphi(phi_span))
# plt.show()

sol = solve_ivp(equations, xspan, f0, t_eval = r)
y_num = sol.y[0, :]

plt.plot(r, y_num, 'b.')            #undershoot w f0 = phi0, 0, skumt bör bli overshoot
plt.show()

# fixa shooting method
    
# def objective(phi0):
#     xspan = (x[0], x[-1])
#     sol = solve_ivp(equations, xspan, \
#             [phi0, 0], t_eval = x)
#     t = sol.t[0]
#     return t[-1]

# def root():
#     xspan = (x[0], x[-1])
#     v0, = fsolve(objective, phi0)
#     f0 = [phi0, 0]
#     sol = solve_ivp(equations, xspan, f0, t_eval = x)
#     y_num = sol.y[0, :]
#     return y_num

# f_num_root = root()