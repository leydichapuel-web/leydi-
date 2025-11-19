"""
Simulación de Columna de Destilación con Modelo UNIFAC
Universidad de Antioquia - Termodinámica Química
Separación de Benceno, Etilbenceno y 1,4-Dietilbenceno

Desarrollado por: David
Curso: Termodinámica Química
"""

import numpy as np
from scipy.optimize import brentq

# ============================================================================
# CONSTANTES Y PROPIEDADES TERMODINÁMICAS
# ============================================================================

# Constantes de Antoine (P en mmHg, T en °C)
# log10(P) = A + B/(C+T)
Antoine_params = {
    "Benzene": {"A": 31.772, "B": -2725.4, "C": -8.4443},
    "Ethylbenzene": {"A": 36.2, "B": -3340.2, "C": -9.797},
    "1,4-Diethylbenzene": {"A": -2.4793, "B": -2894.2, "C": 6.7988}
}

# Parámetros UNIFAC
UNIFAC = {
    "groups": ["ACH", "ACCH2", "CH3"],
    
    # Parámetros de volumen (Rk) y superficie (Qk)
    "Rk": {
        "ACH": 0.5313,
        "ACCH2": 1.0396,
        "CH3": 0.9011
    },
    "Qk": {
        "ACH": 0.400,
        "ACCH2": 0.660,
        "CH3": 0.848
    },
    
    # Composición de grupos por componente
    "comp_groups": [
        {"ACH": 6},                          # Benceno: 6 ACH
        {"ACH": 5, "ACCH2": 1, "CH3": 1},   # Etilbenceno: 5 ACH + 1 ACCH2 + 1 CH3
        {"ACH": 4, "ACCH2": 2, "CH3": 2}    # 1,4-Dietilbenceno: 4 ACH + 2 ACCH2 + 2 CH3
    ],
    
    # Parámetros de interacción binaria a_ij
    "a_ij": {
        "ACH": {"ACH": 0.0, "ACCH2": 61.13, "CH3": 76.50},
        "ACCH2": {"ACH": -11.12, "ACCH2": 0.0, "CH3": 167.0},
        "CH3": {"ACH": 13.46, "ACCH2": 25.77, "CH3": 0.0}
    }
}

# Nombres de componentes
components = ["Benzene", "Ethylbenzene", "1,4-Diethylbenzene"]
n_comp = len(components)

# ============================================================================
# FUNCIONES TERMODINÁMICAS
# ============================================================================

def antoine_pressure(T_celsius, component):
    """Calcula la presión de vapor usando ecuación de Antoine"""
    params = Antoine_params[component]
    A, B, C = params["A"], params["B"], params["C"]
    
    log_P = A + B / (C + T_celsius)
    P_mmHg = 10 ** log_P
    P_kPa = P_mmHg * 0.133322
    
    return max(P_kPa, 1e-10)


def calculate_UNIFAC_gamma(T_K, x):
    """Calcula coeficientes de actividad usando modelo UNIFAC"""
    
    # Normalizar composiciones y evitar ceros
    x = np.array(x)
    x = np.maximum(x, 1e-10)
    x = x / np.sum(x)
    
    # Extraer parámetros UNIFAC
    groups = UNIFAC["groups"]
    Rk = UNIFAC["Rk"]
    Qk = UNIFAC["Qk"]
    comp_groups = UNIFAC["comp_groups"]
    a_ij = UNIFAC["a_ij"]
    
    n_groups = len(groups)
    
    # Calcular parámetros moleculares r_i y q_i
    r = np.zeros(n_comp)
    q = np.zeros(n_comp)
    
    for i in range(n_comp):
        for group, count in comp_groups[i].items():
            r[i] += count * Rk[group]
            q[i] += count * Qk[group]
    
    # PARTE COMBINATORIA
    phi = r * x / np.sum(r * x)
    theta = q * x / np.sum(q * x)
    l = 5 * (r - q) - (r - 1)
    
    phi_safe = np.maximum(phi, 1e-10)
    x_safe = np.maximum(x, 1e-10)
    theta_safe = np.maximum(theta, 1e-10)
    
    ln_gamma_C = np.log(phi_safe / x_safe) + 5 * q * np.log(theta_safe / phi_safe) + l - (phi_safe / x_safe) * np.sum(x * l)
    
    # PARTE RESIDUAL
    psi = np.zeros((n_groups, n_groups))
    for i, group_i in enumerate(groups):
        for j, group_j in enumerate(groups):
            psi[i, j] = np.exp(-a_ij[group_i][group_j] / T_K)
    
    e_km = np.zeros((n_comp, n_groups))
    for k in range(n_comp):
        for m, group in enumerate(groups):
            if group in comp_groups[k]:
                e_km[k][m] = comp_groups[k][group]
    
    X_m = np.zeros(n_groups)
    for m in range(n_groups):
        X_m[m] = np.sum(x * e_km[:, m])
    
    X_m_sum = np.sum(X_m)
    if X_m_sum > 1e-10:
        X_m = X_m / X_m_sum
    else:
        X_m = np.ones(n_groups) / n_groups
    
    Qk_array = np.array([Qk[g] for g in groups])
    Theta_m = X_m * Qk_array
    Theta_m_sum = np.sum(Theta_m)
    if Theta_m_sum > 1e-10:
        Theta_m = Theta_m / Theta_m_sum
    else:
        Theta_m = np.ones(n_groups) / n_groups
    
    ln_Gamma_k = np.zeros(n_groups)
    for k in range(n_groups):
        sum1 = np.sum(Theta_m * psi[:, k])
        sum1 = max(sum1, 1e-10)
        
        sum2 = 0.0
        for m in range(n_groups):
            denominator = np.sum(Theta_m * psi[:, m])
            denominator = max(denominator, 1e-10)
            sum2 += Theta_m[m] * psi[k, m] / denominator
        
        ln_Gamma_k[k] = Qk_array[k] * (1 - np.log(sum1) - sum2)
    
    ln_Gamma_k_pure = np.zeros((n_comp, n_groups))
    
    for i in range(n_comp):
        X_m_i = np.zeros(n_groups)
        for m in range(n_groups):
            if groups[m] in comp_groups[i]:
                X_m_i[m] = comp_groups[i][groups[m]]
        
        X_m_i_sum = np.sum(X_m_i)
        if X_m_i_sum > 1e-10:
            X_m_i = X_m_i / X_m_i_sum
        
        Theta_m_i = X_m_i * Qk_array
        Theta_m_i_sum = np.sum(Theta_m_i)
        if Theta_m_i_sum > 1e-10:
            Theta_m_i = Theta_m_i / Theta_m_i_sum
        
        for k in range(n_groups):
            sum1 = np.sum(Theta_m_i * psi[:, k])
            sum1 = max(sum1, 1e-10)
            
            sum2 = 0.0
            for m in range(n_groups):
                denominator = np.sum(Theta_m_i * psi[:, m])
                denominator = max(denominator, 1e-10)
                sum2 += Theta_m_i[m] * psi[k, m] / denominator
            
            ln_Gamma_k_pure[i, k] = Qk_array[k] * (1 - np.log(sum1) - sum2)
    
    ln_gamma_R = np.zeros(n_comp)
    for i in range(n_comp):
        for k in range(n_groups):
            if groups[k] in comp_groups[i]:
                v_k_i = comp_groups[i][groups[k]]
                ln_gamma_R[i] += v_k_i * (ln_Gamma_k[k] - ln_Gamma_k_pure[i, k])
    
    ln_gamma = ln_gamma_C + ln_gamma_R
    gamma = np.exp(ln_gamma)
    gamma = np.clip(gamma, 0.01, 100.0)
    
    return gamma


def bubble_temperature(P_kPa, x, T_init=None):
    """Calcula la temperatura de burbuja usando método de Newton-Raphson"""
    
    x = np.array(x)
    x = np.maximum(x, 1e-10)
    x = x / np.sum(x)
    
    if T_init is None:
        T_init = 350.0
    
    T = T_init
    max_iter = 100
    tolerance = 1e-6
    
    for iteration in range(max_iter):
        T_C = T - 273.15
        
        gamma = calculate_UNIFAC_gamma(T, x)
        P_sat = np.array([antoine_pressure(T_C, comp) for comp in components])
        P_sat = np.maximum(P_sat, 1e-10)
        
        sum_y = np.sum(gamma * x * P_sat / P_kPa)
        
        if abs(sum_y - 1.0) < tolerance:
            break
        
        dT = 0.1
        T_C_plus = (T + dT) - 273.15
        gamma_plus = calculate_UNIFAC_gamma(T + dT, x)
        P_sat_plus = np.array([antoine_pressure(T_C_plus, comp) for comp in components])
        P_sat_plus = np.maximum(P_sat_plus, 1e-10)
        sum_y_plus = np.sum(gamma_plus * x * P_sat_plus / P_kPa)
        
        d_sum_y_dT = (sum_y_plus - sum_y) / dT
        
        if abs(d_sum_y_dT) > 1e-10:
            T_new = T - (sum_y - 1.0) / d_sum_y_dT
            T_new = np.clip(T_new, T - 10, T + 10)
            T_new = np.clip(T_new, 250, 650)
            T = T_new
        else:
            break
    
    T_C = T - 273.15
    gamma = calculate_UNIFAC_gamma(T, x)
    P_sat = np.array([antoine_pressure(T_C, comp) for comp in components])
    P_sat = np.maximum(P_sat, 1e-10)
    
    y = gamma * x * P_sat / P_kPa
    y = np.maximum(y, 1e-10)
    y = y / np.sum(y)
    
    return T, y


def flash_isotermico(F_total, z, T_K, P_kPa):
    """Realiza un flash isotérmico"""
    
    z = np.array(z)
    z = np.maximum(z, 1e-10)
    z = z / np.sum(z)
    
    T_C = T_K - 273.15
    gamma = calculate_UNIFAC_gamma(T_K, z)
    P_sat = np.array([antoine_pressure(T_C, comp) for comp in components])
    P_sat = np.maximum(P_sat, 1e-10)
    K = gamma * P_sat / P_kPa
    
    def rachford_rice(beta):
        if beta <= 0 or beta >= 1:
            return 1e10
        return np.sum(z * (K - 1) / (1 + beta * (K - 1)))
    
    beta_min, beta_max = 0.001, 0.999
    
    try:
        beta = brentq(rachford_rice, beta_min, beta_max)
    except:
        beta = 0.5
    
    x = z / (1 + beta * (K - 1))
    x = np.maximum(x, 1e-10)
    x = x / np.sum(x)
    
    y = K * x
    y = np.maximum(y, 1e-10)
    y = y / np.sum(y)
    
    L = F_total * (1 - beta)
    V = F_total * beta
    
    return L, x, V, y, beta


# ============================================================================
# ALGORITMO DE SIMULACIÓN DE LA COLUMNA
# ============================================================================

def simular_columna_destilacion():
    """Simula la columna de destilación siguiendo el algoritmo del PDF"""
    
    print("=" * 80)
    print("SIMULACIÓN DE COLUMNA DE DESTILACIÓN - MÉTODO UNIFAC")
    print("Sistema: Benceno - Etilbenceno - 1,4-Dietilbenceno")
    print("=" * 80)
    
    # Alimentación (kmol/h)
    F = np.array([169.46, 91.54, 10.35])
    F_total = np.sum(F)
    z_F = F / F_total
    T_F_C = 73.6
    T_F = T_F_C + 273.15
    P_F = 110.0
    
    # Configuración de la columna
    n_stages = 20
    feed_stage = 8
    reflux_ratio = 0.3874
    
    # Presiones
    P_condenser = 105.0
    P_reboiler = 120.0
    P_stages = np.linspace(P_condenser, P_reboiler, n_stages)
    
    # Recuperaciones objetivo
    recovery_benzene = 0.9986
    recovery_ethylbenzene = 0.01
    
    print(f"\nDATOS DE ENTRADA:")
    print(f"  Alimentación total: {F_total:.2f} kmol/h")
    print(f"  Benceno: {F[0]:.2f} kmol/h")
    print(f"  Etilbenceno: {F[1]:.2f} kmol/h")
    print(f"  1,4-Dietilbenceno: {F[2]:.2f} kmol/h")
    print(f"  Número de etapas: {n_stages}")
    print(f"  Etapa de alimentación: {feed_stage}")
    print(f"  Relación de reflujo: {reflux_ratio}")
    
    # PASO 1: Suposición inicial
    D_benzene = recovery_benzene * F[0]
    D_ethylbenzene = recovery_ethylbenzene * F[1]
    D_diethylbenzene = 0.0
    
    D_total = D_benzene + D_ethylbenzene + D_diethylbenzene
    x_D = np.array([D_benzene, D_ethylbenzene, D_diethylbenzene]) / D_total
    
    B_total = F_total - D_total
    B = F - np.array([D_benzene, D_ethylbenzene, D_diethylbenzene])
    x_B = B / B_total
    
    # PASOS 2-3: Balance de masa en condensador
    V2 = D_total * (1 + reflux_ratio)
    L1 = D_total * reflux_ratio
    U1 = D_total
    
    x_L1 = x_D.copy()
    y_V2 = x_D.copy()
    
    # PASO 4-5: Temperaturas iniciales
    T1, y1 = bubble_temperature(P_stages[0], x_L1, T_init=350)
    T_reboiler, y_reboiler = bubble_temperature(P_reboiler, x_B, T_init=400)
    
    # PASO 6: Gradiente de temperaturas
    T_stages = np.linspace(T1, T_reboiler, n_stages)
    
    # PASO 7-8: Flujos y composiciones iniciales
    V_stages = np.ones(n_stages) * V2
    
    y_stages = np.zeros((n_stages, n_comp))
    for i in range(n_stages):
        alpha = i / (n_stages - 1) if n_stages > 1 else 0
        y_stages[i] = (1 - alpha) * y_V2 + alpha * y_reboiler
        y_stages[i] = y_stages[i] / np.sum(y_stages[i])
    
    L_stages = np.zeros(n_stages)
    x_stages = np.zeros((n_stages, n_comp))
    
    L_stages[0] = L1
    x_stages[0] = x_L1
    
    # ITERACIÓN PRINCIPAL
    print("\nINICIANDO ITERACIONES...")
    
    max_iterations = 50
    tolerance = 1.0
    
    for iteration in range(max_iterations):
        
        T_stages_old = T_stages.copy()
        
        # PASOS 9-11: Etapas antes de alimentación
        for stage in range(1, feed_stage - 1):
            L_in = L_stages[stage - 1]
            x_in = x_stages[stage - 1]
            V_in = V_stages[min(stage + 1, n_stages - 1)]
            y_in = y_stages[min(stage + 1, n_stages - 1)]
            
            F_total_stage = L_in + V_in
            z_stage = (L_in * x_in + V_in * y_in) / F_total_stage if F_total_stage > 0 else x_in
            
            L_out, x_out, V_out, y_out, beta = flash_isotermico(
                F_total_stage, z_stage, T_stages[stage], P_stages[stage]
            )
            
            L_stages[stage] = L_out
            x_stages[stage] = x_out
            V_stages[stage] = V_out
            y_stages[stage] = y_out
        
        # PASO 12: Etapa de alimentación
        stage = feed_stage - 1
        L_in = L_stages[stage - 1]
        x_in = x_stages[stage - 1]
        V_in = V_stages[min(stage + 1, n_stages - 1)]
        y_in = y_stages[min(stage + 1, n_stages - 1)]
        
        F_total_stage = L_in + V_in + F_total
        z_stage = (L_in * x_in + V_in * y_in + F_total * z_F) / F_total_stage
        
        L_out, x_out, V_out, y_out, beta = flash_isotermico(
            F_total_stage, z_stage, T_stages[stage], P_stages[stage]
        )
        
        L_stages[stage] = L_out
        x_stages[stage] = x_out
        V_stages[stage] = V_out
        y_stages[stage] = y_out
        
        # PASO 13: Etapas después de alimentación
        for stage in range(feed_stage, n_stages - 1):
            L_in = L_stages[stage - 1]
            x_in = x_stages[stage - 1]
            V_in = V_stages[min(stage + 1, n_stages - 1)] if stage < n_stages - 1 else 0
            y_in = y_stages[min(stage + 1, n_stages - 1)] if stage < n_stages - 1 else y_reboiler
            
            F_total_stage = L_in + V_in
            z_stage = (L_in * x_in + V_in * y_in) / F_total_stage if F_total_stage > 0 else x_in
            
            L_out, x_out, V_out, y_out, beta = flash_isotermico(
                F_total_stage, z_stage, T_stages[stage], P_stages[stage]
            )
            
            L_stages[stage] = L_out
            x_stages[stage] = x_out
            V_stages[stage] = V_out
            y_stages[stage] = y_out
        
        # PASO 14: Rehervidor
        stage = n_stages - 1
        L_in = L_stages[stage - 1]
        x_in = x_stages[stage - 1]
        
        T_reboiler_new, y_out = bubble_temperature(P_reboiler, x_in, T_init=T_reboiler)
        
        L_stages[stage] = L_in
        x_stages[stage] = x_in
        V_stages[stage] = V2
        y_stages[stage] = y_out
        T_stages[stage] = T_reboiler_new
        
        T_reboiler = T_reboiler_new
        
        # PASO 15: Recalcular T1
        T1_new, _ = bubble_temperature(P_stages[0], x_stages[0], T_init=T1)
        T_stages[0] = T1_new
        T1 = T1_new
        
        # PASO 16: Evaluar convergencia
        convergence = np.sum((T_stages - T_stages_old) ** 2)
        
        if (iteration + 1) % 10 == 0:
            print(f"  Iteración {iteration+1}: Convergencia = {convergence:.6f} K²")
        
        if convergence < tolerance:
            print(f"\n¡CONVERGENCIA ALCANZADA en iteración {iteration+1}!")
            num_iteraciones = iteration + 1
            break
    else:
        num_iteraciones = max_iterations
    
# ========================================================================
    # RESULTADOS FINALES
    # ========================================================================
    
    # Calcular recuperaciones reales
    D_benzene_real = U1 * x_stages[0][0]
    D_ethylbenzene_real = U1 * x_stages[0][1]
    D_diethylbenzene_real = U1 * x_stages[0][2]
    
    B_benzene_real = L_stages[-1] * x_stages[-1][0]
    B_ethylbenzene_real = L_stages[-1] * x_stages[-1][1]
    B_diethylbenzene_real = L_stages[-1] * x_stages[-1][2]
    
    recovery_benzene_real = (D_benzene_real / F[0]) * 100
    recovery_ethylbenzene_real = (D_ethylbenzene_real / F[1]) * 100
    
    # Mostrar número de iteraciones
    print(f"\nNúmero de iteraciones: {num_iteraciones}")
    
    # Mostrar resultados en el formato solicitado
    print(f"\nD (kmol/h) = {U1:.6f}")
    print(f"Composición destilado y_D:")
    print(f"Bz = {x_stages[0][0]:.8f}")
    print(f"Et = {x_stages[0][1]:.8f}")
    print(f"DiEt = {x_stages[0][2]:.8f}")
    
    print(f"\nB (kmol/h) = {L_stages[-1]:.6f}")
    print(f"Composición fondos x_B:")
    print(f"Bz = {x_stages[-1][0]:.8f}")
    print(f"Et = {x_stages[-1][1]:.8f}")
    print(f"DiEt = {x_stages[-1][2]:.8f}")
    
    print(f"\nRecuperación benceno = {recovery_benzene_real:.4f}%")
    print(f"Recuperación Etilbenceno = {recovery_ethylbenzene_real:.4f}%")
    
    # Balance de materia
    print(f"\nBalance de materia:")
    for i, comp_name in enumerate(["Benceno", "Etilbenceno", "1,4-Dietilbenceno"]):
        F_comp = F[i]
        D_comp = U1 * x_stages[0][i]
        B_comp = L_stages[-1] * x_stages[-1][i]
        error = F_comp - (D_comp + B_comp)
        print(f"{comp_name}: Entrada={F_comp:.6f}, Salida={D_comp+B_comp:.6f}, Error={error:.8f}")
    
    print("\nSimulación completada.")


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    simular_columna_destilacion()