import numpy as np
import scipy.constants as sc
import scipy.optimize as so
import scipy.integrate as si
import matplotlib.pyplot as plt

plt.style.use("ggplot")

# Initial graph p - eps | Units: erg/cm^3

epsilon0 = 1.42e24
m_N = 1.67493*1e-24
m_e = 9.10938*1e-28
h_ = 1.054572*1e-27
c = 2.9979*1e10;
A = (m_N*2.15*c**2)/(3*(np.pi**2)*h_**3)


def ε (x):
    return (epsilon0/8)*((2*x**3 + x)*np.sqrt(1+x**2) - np.arcsinh(x)) + A*(x*m_e*c)**3
def p (x):
    return (epsilon0/24)*((2*x**3 - 3*x)*np.sqrt(1+x**2) + 3*np.arcsinh(x))

kF = np.linspace(0,2*m_e*c,5000)
x = kF/(m_e*c)

# Finding X_NR, X_R such that ε = X_NR*p^(5/3) + X_R*p^(4/3) 

eps0 = 1e37

def ε_gen (p, X_NR, X_R):
    return X_NR*p**(3/5)+ X_R*p**(3/4)

x_data = p(x)/eps0
y_data = ε(x)/eps0

param_opt, param_cov = so.curve_fit(ε_gen, x_data, y_data)

X_NR = param_opt[0]
X_R = param_opt[1]

print("X_NR =", X_NR, "& X_R =", X_R)
print('')


y_aprox = ε_gen(x_data,X_NR,X_R)

plt.plot(x_data*eps0,y_data*eps0,'-k',label = 'Analytical ε_-p_ curve')
plt.plot(x_data*eps0,y_aprox*eps0,'-c', label = 'Fitted solution')
plt.legend()
plt.show()

# Integrating the ODEs | Units: km & M_sun*c^2 | Case: General

M_sun = 1.98851e30
c = 299792.458
G = 6.6743e-20

R0 = G*M_sun/(c**2)
eps0 = eps0*0.55954e-39
beta = 4*np.pi*eps0

def odef (r, y):
    p_der = (-R0*ε_gen(y[0], X_NR, X_R)*y[1]/(r**2))
    M_der = beta*r**2*ε_gen(y[0], X_NR, X_R)
    return np.array([p_der,M_der])

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

print ("Maximum Mass =", y[1,-1])

# Maximum mass and Radius vs. p0

p = np.linspace(0.5e-16, 1e-10, 200)
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

# Maximum mass:
print(M[-1])









