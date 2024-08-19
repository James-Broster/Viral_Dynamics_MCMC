import numpy as np
from scipy.integrate import odeint

class VirusModel:
    @staticmethod
    def ode(y, t, alpha_f, beta, delta_f, gamma, epsilon, t_star):
        f1, f2, V = y
        H_t = 0 if t < t_star else 1
        df1_dt = alpha_f * f2 * V - (1 - epsilon * H_t) * beta * f1 * V
        df2_dt = -alpha_f * f2 * V
        dV_dt = (1 - epsilon * H_t) * gamma * f1 * V - delta_f * V
        return [df1_dt, df2_dt, dV_dt]

    @staticmethod
    def solve(parameters, time, epsilon=0, t_star=0):
        alpha_f, beta, delta_f, gamma, f2_0, V_0 = parameters
        f1_0 = 1 - f2_0  # Ensure f1 + f2 = 1
        initial_conditions = [f1_0, f2_0, V_0]

        solution = odeint(VirusModel.ode, initial_conditions, time, args=(alpha_f, beta, delta_f, gamma, epsilon, t_star))
        V = solution[:, 2]
        log_V = np.log10(np.maximum(V, 1e-300))
        result = np.column_stack((solution[:, :2], log_V))
        return result