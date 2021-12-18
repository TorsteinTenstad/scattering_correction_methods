import numpy as np
from matplotlib import pyplot as plt

def a(n_s, n_b=1):
    integral_n = 1000
    theta_c = np.arcsin(n_b/n_s)
    theta = np.linspace(0, np.pi/2, integral_n)
    helper = lambda theta_1, theta_2 : (1/2)*((n_s*np.cos(theta_1)-n_b*np.cos(theta_2))/(n_s*np.cos(theta_1)+n_b*np.cos(theta_2)))**2
    R_F = np.zeros(len(theta))
    for i, t in enumerate(theta):
        if t < theta_c:
            t_ = np.arcsin((n_s/n_b)*np.sin(t))
            R_F[i] = helper(t_, t) + helper(t, t_)
        else:
            R_F[i] = 1
    R_phi_integrand = 2*np.sin(theta)*np.cos(theta)*R_F
    R_j_integrand = 3*np.sin(theta)*np.cos(theta)**2*R_F
    R_phi = np.trapz(R_phi_integrand, theta)
    R_j = np.trapz(R_j_integrand, theta)
    R_eff = (R_phi+R_j)/(2-R_phi+R_j)
    return (1-R_eff)/(2*(1+R_eff))