import numpy as np
import scipy.constants as sc
import scipy.integrate as si
import matplotlib.pyplot as plt

plt.style.use("ggplot")

def odef (r, y):
    p_der = -alpha*y[0]**(1/gamma)*y[1]/(r**2)
    M_der = beta*r**2*y[0]**(1/gamma)
    return np.array([p_der,M_der])

M_sun = 1.98851e30
c = 299792.458
G = 6.6743e-20

# Integrating the ODEs, units: km and M_sun*c^2, case: Non Relativistic

gamma = 5/3
K_nonrel = ((sc.hbar**2/(15*np.pi**2*sc.m_e))*(3*np.pi**2/(2.15*sc.m_n*sc.c**2))**(gamma))*3.1728144285153128e+25
R0 = G*M_sun/(c**2)
alpha = 0.05
eps0 = ((1/K_nonrel)*(R0/alpha)**(gamma))**(1/(gamma-1))
beta = (4*np.pi*eps0)/((K_nonrel*eps0**(gamma-1))**(1/gamma))

m0 = 0.0
p0 = 1e-16
ci = np.array([p0, m0])

y = np.array([[p0],[m0]])
r = np.array([0])
r_end = 0.01
r_step = 10

while y[0,-1] > p0*1e-10:
    sol = si.solve_ivp(odef, (r_end,r_end + r_step), y[:,-1])
    aux = np.array([sol.y[0],sol.y[1]])
    y = np.hstack((y, aux))
    r = np.concatenate((r,sol.t))
    r_end += r_step

plt.plot(r,y[0],'-k')
plt.show()
plt.plot(r,y[1],'-c')
plt.show()

print ("Maximum Non-Relativistic Mass =", y[1,-1])

# Maximum mass and Radius vs. p0

p = np.linspace(0.5e-18, 1e-14, 200)
M = []
R = []

def solve (p):
    m0 = 0.0
    p0 = p
    y = np.array([[p0],[m0]])
    r = np.array([0])
    r_end = 0.01
    r_step = 10
    while y[0,-1].real > p0*1e-10:
        sol = si.solve_ivp(odef, (r_end,r_end + r_step), y[:,-1])
        aux = np.array([sol.y[0],sol.y[1]])
        y = np.hstack((y, aux))
        r = np.concatenate((r,sol.t))
        r_end += r_step
    return r_end, y[1][-1]

for x in p:
    r, m = solve(x)
    R.append(r)
    M.append(m)
    
plt.plot(p, np.array(M))
plt.show()
plt.plot(p, np.array(R))
plt.show()

