import numpy as np
import scipy
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from scipy.signal import argrelextrema

"""
Test scrpit for the bounce solver. First part constructs and plots the 1D effective potential
V(phi) = -a*phi^2 -b*phi^3 + c*phi^4 + constant. Then we have the bounce solver, where we solve the eom 
for the bounce dphi^2/dr^2 - (d-1)/r*dphi/dr = dV/dphi with a shooting method. Not done yet!

"""
#----Potential----
# Define the constants for the potential and then plot it. Two plots are shown, both V(phi) 
# and -V(phi) (to visualize the path for the point particle later for bounce). We also get
# the value of the true vacuum, called phi0.

# Set parameters for the potential, V(phi) = -a*phi^2 -b*phi^3 + c*phi^4 + constant
a = 40
b = 12
c = 1

def V(phi):
    """
    Function defining the potential, V(phi) = a*phi^2 -b*phi^3 + c*phi^4 
    The parameters a, b, c should be definied s.t we get a 1st order PT, i.e a bump
    Important to only have one bump, and hence only 2 minimas.

    @param phi: our scalar field, should be array
    @return pot: The effective potential
            phiTrueVacuum: The value of phi at the true vacuum
            phiFalseVacuum: The value of phi at the false vacuum
    """
    pot = +a*phi**2 -b*phi**3 + c*phi**4
    indexMinimas = argrelextrema(pot, np.less)

    # indexMaxima = argrelextrema(pot, np.greater)
    # print(indexMaxima)
    phiFalseVacuum = phi[indexMinimas[0]][1]
    phiTrueVacuum = phi[indexMinimas[0]][0]

    # print(phiFalseVacuum, phiTrueVacuum)
    # constant = - pot[indexMaxima[0]][0]
    # print(constant)
    return pot, phiTrueVacuum, phiFalseVacuum


def makePlotsPotential(phi_span, potential):
    """
    Function to make plots for the potential. One of the potential and one of 
    the inverted potential to visualize the potential for the point particle 
    that we use later on when calculating the bounce. 
     

    @param phi_span: The span of phi, i.e x-axis
           potential: The potential V(phi), i.e y-axis, for every point of phi_span
    @return Two plots in one figure, one of V(phi) and the other -V(phi)
    """
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

# Get plots of potential V(phi) = 40*phi^2 -12*phi^3 + phi^4
phi_span = np.linspace(-1, 6, 100)
pot, trueVacuum, phiFalseVacuum = V(phi_span)
makePlotsPotential(phi_span, pot)

#------
# Bounce

def dVdphi(phi):
    """
    Function defining the derivative of the potential, given it's on the 
    form V(phi) = -a*phi^2 -b*phi^3 + c*phi^4 + constant

    @param phi: our scalar field
    @return The derivative of the potential
    """
    return 2*a*phi -3*b*phi**2 + 4*c*phi**3

def equations(r, y):
    """
    Function setting our equation system from the EOM of the bounce
    Turning equation of 2nd order ODE, to system of two 1st order ODEs
        From: dphi^2/dr^2 - (3/r)*dphi/dr = dV/dphi
        To:   dy/dr = ......                                                    #fortsätt kommentera

    @param phi: our scalar field
           phi_span: The span of phi given as - some value to + the same value
    @return The effective potential
    """
    phi = y[0]
    v = y[1]        #dphi/dr
    
    dphidr = v
    dvdr = - (3/r)*v + dVdphi(phi)
    
    return np.array([dphidr, dvdr])

# shooting method

r = np.linspace(0.00001, 4, 100)

def shooting(r, trueVacuum):
    tol = 1e-3
    max_iters = 200
    low = trueVacuum
    high = trueVacuum + 2
    dphidr0 = 0
    count = 0
    xspan = (r[0], r[-1])

    print('phi_T = ', trueVacuum, 'phi_F = ', phiFalseVacuum)

    while count <= max_iters:
        count = count + 1              
        phi0 = np.mean([low, high])
        f0 = [phi0, dphidr0]

        # Solve the system using our guess
        sol = solve_ivp(equations, xspan, f0, t_eval = r)       # # atol=1e-20, rtol=1e-10, max_step=1e-7
        #print(sol.message)

        y_num = sol.y[0, :]  #the numerical solution

        # Felsök
        print('low =', low, 'high = ', high)
        print('count: ', count, 'Value of last point in numerical: ', y_num[-1], 'Initial guess: ', phi0)

        if np.abs(y_num[-1] - phiFalseVacuum) <= tol:        #check if last point is within tolerance
            break
        
        # Adjust our bounds if we are not within tolerance, bisection method

        if y_num[-1] < 0 or y_num[-1] > (phiFalseVacuum + tol):                     #overshoot                 
            low = phi0                  
        else:                           
            high = phi0                       #undershoot
            
        #print(count, y_num[-1], phi0)

    return y_num

y_num = shooting(r, trueVacuum)
plt.plot(r, y_num, 'b.')
plt.xlabel('r')
plt.ylabel('$\phi_B$(r)')             
plt.show()
