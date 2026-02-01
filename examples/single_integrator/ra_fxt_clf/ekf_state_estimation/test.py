import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# Define the Van der Pol equations
def van_der_pol(t, z):
    x, y = z
    mu = 2.0

    V = x**2 + y**2
    fV = -((V) ** 0.5) - (V) ** 1.5
    ux2 = (fV + 2 * x * y) / 2 + mu * (1 - x**2) * y**2 - x * y
    dxdt = -y
    dydt = -mu * (1 - x**2) * y + x + ux2
    return [dxdt, dydt]


# Initial condition
initial_condition = [3.0, 3.0]

# Time span for integration
t_span = (0, 10)

# Solve the system of ODEs
sol = solve_ivp(van_der_pol, t_span, initial_condition, t_eval=np.linspace(0, 10, 10000))
sol_fe = [initial_condition]
for ii, tt in enumerate(np.linspace(0, 10, 10000)):
    [dxdt, dydt] = van_der_pol(tt, sol_fe[-1])
    sol_fe.append([sol_fe[ii][0] + dxdt * 1e-3, sol_fe[ii][1] + dydt * 1e-3])
# Plot the trajectory in phase space
plt.plot(sol.y[0], sol.y[1])
plt.plot(sol_fe[:][0], sol.y[1])
plt.xlabel("x")
plt.ylabel("y")
plt.title("Van der Pol Oscillator Trajectory")
plt.grid(True)
plt.show()
