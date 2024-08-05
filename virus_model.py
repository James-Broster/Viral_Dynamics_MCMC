import numpy as np
from scipy.integrate import odeint
import os

class VirusModel:
    @staticmethod
    def ode(y, t, alpha_f, beta, delta_f, gamma, epsilon, t_star):
        f1, f2, V = y
        H_t = 0 if t < t_star else 1
        df1_dt = alpha_f * f2 * V - (1 - epsilon * H_t) * beta * f1 * V
        df2_dt = -alpha_f * f2 * V
        dV_dt = (1 - epsilon * H_t) * gamma * f1 * V - delta_f * V
        
        # Save debug info to file
        debug_dir = 'debug_output'
        os.makedirs(debug_dir, exist_ok=True)
        with open(os.path.join(debug_dir, 'ode_debug.txt'), 'a') as f:
            f.write(f"Inputs: y={y}, t={t}, params={[alpha_f, beta, delta_f, gamma, epsilon, t_star]}\n")
            f.write(f"Outputs: df1_dt={df1_dt}, df2_dt={df2_dt}, dV_dt={dV_dt}\n\n")
        
        return [df1_dt, df2_dt, dV_dt]

    @staticmethod
    def solve(parameters, time, epsilon=0, t_star=0):
        alpha_f, beta, delta_f, gamma, f1_0, f2_0, V_0 = parameters
        initial_conditions = [f1_0, f2_0, V_0]
        solution = odeint(VirusModel.ode, initial_conditions, time, args=(alpha_f, beta, delta_f, gamma, epsilon, t_star))
        V = solution[:, 2]
        log_V = np.log10(np.maximum(V, 1e-300))
        result = np.column_stack((solution[:, :2], log_V))
        print(f"Solve summary - Shape: {result.shape}, Initial: {result[0]}, Final: {result[-1]}")
        return result

    @staticmethod
    def calculate_time_to_threshold(parameters, epsilon, t_star, threshold=4):
        time = np.linspace(0, 30, 301)  # Extend time to 30 days, with 301 points for better resolution
        solution = VirusModel.solve(parameters, time, epsilon, t_star)
        log_V = solution[:, 2]
        below_threshold = np.where(log_V <= threshold)[0]
        if len(below_threshold) == 0:
            result = 30
            print(f"Time to threshold: Viral load never dropped below {threshold}")
        else:
            result = time[below_threshold[0]]
            print(f"Time to threshold {threshold}: {result}")
        return result