import numpy as np
from scipy.integrate import solve_ivp
from scipy import optimize
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
plt.style.use('seaborn')

#y'' = -3y, 2nd order ODE

x = np.linspace(0, 2 * np.pi, 100)
y_exact = 7 * np.cos(np.sqrt(3) * x) - 7 * np.cos(2 * np.pi * np.sqrt(3) ) / np.sin(2 * np.pi * np.sqrt(3) )* np.sin(np.sqrt(3)*x)
y0 = 7
v0 = 0

def equations(x, y):
    yprime = np.zeros(2)                #2nd order ODE to system of 1st order ODE
    
    yprime[0] = y[1]                    #yprime = [y,v], v=dy/dt, dv/dt=d2y/dt2 = -3y       
    yprime[1] = -3 * y[0]
    
    return yprime

def bisection():
    tol = 1e-6
    max_iters = 100
    low = -10
    high = 10
    count = 0

    while count <= max_iters:
        count = count + 1
        xspan = (x[0], x[-1])
        
        v0 = np.mean([low, high])
        
        f0 = [y0, v0]                   #y(0) = 7, v(0) = y' = guess

        # Solve the system using our guess
        sol = solve_ivp(equations, xspan, f0, t_eval = x)

        y_num = sol.y[0, :]  #the numerical solution

        if np.abs(y_num[-1]) <= tol:        #check if last point (where we land) is within tolerance
            break
        
        #  Adjust our bounds if we are not within tolerance, bisection method

        if y_num[-1] < 0:               #undershoot, need to decrease y', i.e velocity, to raise the endpoint --> set y'0 to high then in next iteration                       
            high = v0                  # y'0 = mean(high,low) will be lower 
        else:                           #overshoot, same as above but opposite
            low = v0
            
        print(count, y_num[-1])

    return y_num


def objective(v0):
    xspan = (x[0], x[-1])
    sol = solve_ivp(equations, xspan, \
            [y0, v0], t_eval = x)           
    y = sol.y[0]
    return y[-1] - 0        #last point vs BV = 0                      

def root():
    xspan = (x[0], x[-1])

    # for v0_guess in range(1, 100, 10):
    #     v0, = fsolve(objective, v0_guess)
    #     print('First guess: %d, Result: %.1f' \
    #             %(v0_guess, v0))

    v0, = fsolve(objective, 10)                     # find the root of the objective function (diff num sol vs bv) given an initial guess
    #print(v0) 
    f0 = [y0, v0]
    sol = solve_ivp(equations, xspan, f0, t_eval = x)
    y_num = sol.y[0, :]
    return y_num

f_num_bisect = bisection()
f_num_root = root()

# f0 = [7, 0]
# xspan = (x[0], x[-1])
# sol = solve_ivp(equations, xspan, f0, t_eval = x)
# y_num = sol.y[0, :]
# #print(y_num)

# plt.plot(x, y_num, 'b.')
# plt.show()

# #  Plot the solution and compare it to the analytical form defined above
plt.plot(x, y_exact, 'k', label='Exact')
plt.plot(x, f_num_root, 'b.', label='Numeric, root')
plt.plot(x, f_num_bisect, 'r.', label='Numeric, bisect')
plt.plot([0, 2*np.pi], [y0,0], 'bo')
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()