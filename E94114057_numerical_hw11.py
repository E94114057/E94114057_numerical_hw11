import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from scipy.linalg import solve
import pandas as pd

# === Problem setup ===
h = 0.1
x_vals = np.arange(0, 1 + h, h)
n = len(x_vals)

#a.
def ode_system(x, y):
    dy1 = y[1]
    dy2 = -(x + 1) * y[1] + 2 * y[0] + (1 - x ** 2) * np.exp(-x)
    return [dy1, dy2]

def shoot(s):
    sol = solve_ivp(ode_system, [0, 1], [1, s], t_eval=[1])
    return sol.y[0][-1] - 2

# Find correct initial slope
shooting_sol = root_scalar(shoot, bracket=[0, 10], method='bisect')
s_val = shooting_sol.root
shooting_full_sol = solve_ivp(ode_system, [0, 1], [1, s_val], t_eval=x_vals)

#b.
A = np.zeros((n, n))
b = np.zeros(n)
A[0, 0] = 1
b[0] = 1
A[-1, -1] = 1
b[-1] = 2

for i in range(1, n - 1):
    x = x_vals[i]
    A[i, i - 1] = 1 / h**2 - (x + 1) / (2 * h)
    A[i, i] = -2 / h**2 + 2
    A[i, i + 1] = 1 / h**2 + (x + 1) / (2 * h)
    b[i] = (1 - x**2) * np.exp(-x)

y_fd = solve(A, b)

#c. 
def phi(i, x):
    return np.sin((i + 1) * np.pi * x) * x * (1 - x)

m = 3
A_var = np.zeros((m, m))
b_var = np.zeros(m)

for i in range(m):
    for j in range(m):
        phi_i = phi(i, x_vals)
        phi_j = phi(j, x_vals)
        dphi_j = np.gradient(phi_j, h)
        integrand = (
            np.gradient(np.gradient(phi_j, h), h) * phi_i
            + (x_vals + 1) * dphi_j * phi_i
            - 2 * phi_j * phi_i
        )
        A_var[i, j] = np.trapz(integrand, x_vals)
    rhs = (1 - x_vals**2) * np.exp(-x_vals) * phi(i, x_vals)
    b_var[i] = np.trapz(rhs, x_vals)

c_var = np.linalg.solve(A_var, b_var)

def y_var(x):
    base = (1 - x) * 1 + x * 2
    sum_phi = np.zeros_like(x)
    for i in range(m):
        sum_phi += c_var[i] * phi(i, x)
    return base + sum_phi

y_variational = y_var(x_vals)

# 輸出數值 
results_df = pd.DataFrame({
    'x': x_vals,
    'Shooting Method': shooting_full_sol.y[0],
    'Finite Difference': y_fd,
    'Variation Method': y_variational
})
print(results_df.round(6).to_string(index=False))


plt.figure(figsize=(10, 6))
plt.plot(x_vals, shooting_full_sol.y[0], label='Shooting Method', marker='o')
plt.plot(x_vals, y_fd, label='Finite Difference Method', marker='s')
plt.plot(x_vals, y_variational, label='Variation Method', marker='^')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Comparison of Numerical Methods')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

