"""
Simulación de Columna de Destilación con Modelo UNIFAC
Universidad de Antioquia - Termodinámica Química
Separación de Benceno, Etilbenceno y 1,4-Dietilbenceno

Desarrollado por: [Tu nombre]
Curso: Termodinámica Química
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, minimize_scalar

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

# Capacidades caloríficas líquido (J/mol-K)
Cp_liquid = {
    "Benzene": 134.63,
    "Ethylbenzene": 185.572,
    "1,4-Diethylbenzene": 239.10
}

# Capacidades caloríficas vapor (J/mol-K)
Cp_vapor = {
    "Benzene": 82.44,
    "Ethylbenzene": 127.40,
    "1,4-Diethylbenzene": 176.15
}

# Calores de vaporización a Tref (J/mol)
DHvap = {
    "Benzene": 33900.0,
    "Ethylbenzene": 41000.0,
    "1,4-Diethylbenzene": 52500.0
}

# Temperatura de referencia (K)
T_ref = 298.15  # 25°C

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
    
    # Parámetros de interacción binaria a_ij (de la imagen proporcionada)
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
    """
    Calcula la presión de vapor usando ecuación de Antoine
    
    Parámetros:
    -----------
    T_celsius: Temperatura en °C
    component: Nombre del componente
    
    Retorna:
    --------
    P_kPa: Presión de vapor en kPa
    """
    params = Antoine_params[component]
    A, B, C = params["A"], params["B"], params["C"]
    
    # log10(P) = A + B/(C+T)
    log_P = A + B / (C + T_celsius)
    P_mmHg = 10 ** log_P
    P_kPa = P_mmHg * 0.133322  # Conversión mmHg a kPa
    
    return max(P_kPa, 1e-10)  # Evitar valores negativos


def calculate_UNIFAC_gamma(T_K, x):
    """
    Calcula coeficientes de actividad usando modelo UNIFAC
    
    Parámetros:
    -----------
    T_K: Temperatura en Kelvin
    x: Fracciones molares (array de tamaño n_comp)
    
    Retorna:
    --------
    gamma: Array de coeficientes de actividad
    """
    
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
    
    # ========================================================================
    # 1. Calcular parámetros moleculares r_i y q_i para cada componente
    # ========================================================================
    r = np.zeros(n_comp)
    q = np.zeros(n_comp)
    
    for i in range(n_comp):
        for group, count in comp_groups[i].items():
            r[i] += count * Rk[group]
            q[i] += count * Qk[group]
    
    # ========================================================================
    # 2. PARTE COMBINATORIA
    # ========================================================================
    
    # Fracciones de segmento y área
    phi = r * x / np.sum(r * x)
    theta = q * x / np.sum(q * x)
    
    # Parámetro l_i
    l = 5 * (r - q) - (r - 1)
    
    # Evitar log(0) y divisiones por cero
    phi_safe = np.maximum(phi, 1e-10)
    x_safe = np.maximum(x, 1e-10)
    theta_safe = np.maximum(theta, 1e-10)
    
    # ln(gamma_C)
    ln_gamma_C = np.log(phi_safe / x_safe) + 5 * q * np.log(theta_safe / phi_safe) + l - (phi_safe / x_safe) * np.sum(x * l)
    
    # ========================================================================
    # 3. PARTE RESIDUAL
    # ========================================================================
    
    # Parámetros de interacción: psi_nm = exp(-a_nm/T)
    psi = np.zeros((n_groups, n_groups))
    for i, group_i in enumerate(groups):
        for j, group_j in enumerate(groups):
            psi[i, j] = np.exp(-a_ij[group_i][group_j] / T_K)
    
    # Fracción de área de grupo m en la mezcla: Theta_m
    e_km = np.zeros((n_comp, n_groups))  # e_km[k][m] = número de grupos m en molécula k
    for k in range(n_comp):
        for m, group in enumerate(groups):
            if group in comp_groups[k]:
                e_km[k][m] = comp_groups[k][group]
    
    # X_m: fracción molar de grupo m en la mezcla
    X_m = np.zeros(n_groups)
    for m in range(n_groups):
        X_m[m] = np.sum(x * e_km[:, m])
    
    X_m_sum = np.sum(X_m)
    if X_m_sum > 1e-10:
        X_m = X_m / X_m_sum
    else:
        X_m = np.ones(n_groups) / n_groups
    
    # Theta_m: fracción de área de grupo m
    Qk_array = np.array([Qk[g] for g in groups])
    Theta_m = X_m * Qk_array
    Theta_m_sum = np.sum(Theta_m)
    if Theta_m_sum > 1e-10:
        Theta_m = Theta_m / Theta_m_sum
    else:
        Theta_m = np.ones(n_groups) / n_groups
    
    # ln(Gamma_k) para cada grupo k en la mezcla
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
    
    # ln(Gamma_k^(i)) para cada grupo k en componente puro i
    ln_Gamma_k_pure = np.zeros((n_comp, n_groups))
    
    for i in range(n_comp):
        # X_m para componente puro i
        X_m_i = np.zeros(n_groups)
        for m in range(n_groups):
            if groups[m] in comp_groups[i]:
                X_m_i[m] = comp_groups[i][groups[m]]
        
        X_m_i_sum = np.sum(X_m_i)
        if X_m_i_sum > 1e-10:
            X_m_i = X_m_i / X_m_i_sum
        
        # Theta_m para componente puro i
        Theta_m_i = X_m_i * Qk_array
        Theta_m_i_sum = np.sum(Theta_m_i)
        if Theta_m_i_sum > 1e-10:
            Theta_m_i = Theta_m_i / Theta_m_i_sum
        
        # ln(Gamma_k^(i))
        for k in range(n_groups):
            sum1 = np.sum(Theta_m_i * psi[:, k])
            sum1 = max(sum1, 1e-10)
            
            sum2 = 0.0
            for m in range(n_groups):
                denominator = np.sum(Theta_m_i * psi[:, m])
                denominator = max(denominator, 1e-10)
                sum2 += Theta_m_i[m] * psi[k, m] / denominator
            
            ln_Gamma_k_pure[i, k] = Qk_array[k] * (1 - np.log(sum1) - sum2)
    
    # ln(gamma_R) para cada componente
    ln_gamma_R = np.zeros(n_comp)
    for i in range(n_comp):
        for k in range(n_groups):
            if groups[k] in comp_groups[i]:
                v_k_i = comp_groups[i][groups[k]]
                ln_gamma_R[i] += v_k_i * (ln_Gamma_k[k] - ln_Gamma_k_pure[i, k])
    
    # ========================================================================
    # 4. COEFICIENTE DE ACTIVIDAD TOTAL
    # ========================================================================
    
    ln_gamma = ln_gamma_C + ln_gamma_R
    gamma = np.exp(ln_gamma)
    
    # Limitar valores extremos
    gamma = np.clip(gamma, 0.01, 100.0)
    
    return gamma


def bubble_temperature(P_kPa, x, T_init=None):
    """
    Calcula la temperatura de burbuja usando método de Newton-Raphson
    
    Parámetros:
    -----------
    P_kPa: Presión en kPa
    x: Composición líquida (fracciones molares)
    T_init: Temperatura inicial de búsqueda en K
    
    Retorna:
    --------
    T_bubble: Temperatura de burbuja en K
    y: Composición vapor en equilibrio
    """
    
    # Normalizar composición
    x = np.array(x)
    x = np.maximum(x, 1e-10)
    x = x / np.sum(x)
    
    # Temperatura inicial
    if T_init is None:
        T_init = 350.0  # K
    
    T = T_init
    max_iter = 100
    tolerance = 1e-6
    
    for iteration in range(max_iter):
        T_C = T - 273.15
        
        # Calcular gamma con UNIFAC
        gamma = calculate_UNIFAC_gamma(T, x)
        
        # Calcular presiones de saturación
        P_sat = np.array([antoine_pressure(T_C, comp) for comp in components])
        P_sat = np.maximum(P_sat, 1e-10)
        
        # Suma de y_i
        sum_y = np.sum(gamma * x * P_sat / P_kPa)
        
        # Verificar convergencia
        if abs(sum_y - 1.0) < tolerance:
            break
        
        # Derivada numérica para Newton-Raphson
        dT = 0.1
        T_C_plus = (T + dT) - 273.15
        gamma_plus = calculate_UNIFAC_gamma(T + dT, x)
        P_sat_plus = np.array([antoine_pressure(T_C_plus, comp) for comp in components])
        P_sat_plus = np.maximum(P_sat_plus, 1e-10)
        sum_y_plus = np.sum(gamma_plus * x * P_sat_plus / P_kPa)
        
        d_sum_y_dT = (sum_y_plus - sum_y) / dT
        
        # Actualizar temperatura
        if abs(d_sum_y_dT) > 1e-10:
            T_new = T - (sum_y - 1.0) / d_sum_y_dT
            T_new = np.clip(T_new, T - 10, T + 10)  # Limitar cambios
            T_new = np.clip(T_new, 250, 650)  # Límites físicos
            T = T_new
        else:
            break
    
    # Calcular composición vapor final
    T_C = T - 273.15
    gamma = calculate_UNIFAC_gamma(T, x)
    P_sat = np.array([antoine_pressure(T_C, comp) for comp in components])
    P_sat = np.maximum(P_sat, 1e-10)
    
    y = gamma * x * P_sat / P_kPa
    y = np.maximum(y, 1e-10)
    y = y / np.sum(y)
    
    return T, y


def flash_isotermico(F_total, z, T_K, P_kPa):
    """
    Realiza un flash isotérmico
    
    Parámetros:
    -----------
    F_total: Flujo total de alimentación (kmol/h)
    z: Composición de alimentación
    T_K: Temperatura en K
    P_kPa: Presión en kPa
    
    Retorna:
    --------
    L, x, V, y, beta
    """
    
    # Normalizar composición
    z = np.array(z)
    z = np.maximum(z, 1e-10)
    z = z / np.sum(z)
    
    # Calcular K-values
    T_C = T_K - 273.15
    gamma = calculate_UNIFAC_gamma(T_K, z)
    P_sat = np.array([antoine_pressure(T_C, comp) for comp in components])
    P_sat = np.maximum(P_sat, 1e-10)
    K = gamma * P_sat / P_kPa
    
    # Resolver ecuación de Rachford-Rice para encontrar beta
    def rachford_rice(beta):
        if beta <= 0 or beta >= 1:
            return 1e10
        return np.sum(z * (K - 1) / (1 + beta * (K - 1)))
    
    # Buscar beta entre 0 y 1
    beta_min, beta_max = 0.001, 0.999
    
    try:
        from scipy.optimize import brentq
        beta = brentq(rachford_rice, beta_min, beta_max)
    except:
        beta = 0.5  # Valor por defecto
    
    # Calcular composiciones
    x = z / (1 + beta * (K - 1))
    x = np.maximum(x, 1e-10)
    x = x / np.sum(x)
    
    y = K * x
    y = np.maximum(y, 1e-10)
    y = y / np.sum(y)
    
    # Flujos
    L = F_total * (1 - beta)
    V = F_total * beta
    
    return L, x, V, y, beta


# ============================================================================
# ALGORITMO DE SIMULACIÓN DE LA COLUMNA
# ============================================================================

def simular_columna_destilacion():
    """
    Simula la columna de destilación siguiendo el algoritmo del PDF
    """
    
    print("=" * 80)
    print("SIMULACIÓN DE COLUMNA DE DESTILACIÓN - MÉTODO UNIFAC")
    print("Sistema: Benceno - Etilbenceno - 1,4-Dietilbenceno")
    print("=" * 80)
    
    # ========================================================================
    # DATOS DE ENTRADA
    # ========================================================================
    
    # Alimentación (kmol/h)
    F = np.array([169.46, 91.54, 10.35])
    F_total = np.sum(F)
    z_F = F / F_total
    T_F_C = 73.6  # °C
    T_F = T_F_C + 273.15  # K
    P_F = 110.0  # kPa
    
    # Configuración de la columna
    n_stages = 20
    feed_stage = 8
    reflux_ratio = 0.3874
    
    # Presiones
    P_condenser = 105.0  # kPa
    P_reboiler = 120.0  # kPa
    P_stages = np.linspace(P_condenser, P_reboiler, n_stages)
    
    # Recuperaciones objetivo
    recovery_benzene = 0.9986  # 99.86% del benceno en destilado
    recovery_ethylbenzene = 0.01  # 1% del etilbenceno en destilado
    
    print(f"\n{'DATOS DE ENTRADA':-^80}")
    print(f"Alimentación total: {F_total:.2f} kmol/h")
    print(f"  Benceno: {F[0]:.2f} kmol/h ({z_F[0]*100:.2f}%)")
    print(f"  Etilbenceno: {F[1]:.2f} kmol/h ({z_F[1]*100:.2f}%)")
    print(f"  1,4-Dietilbenceno: {F[2]:.2f} kmol/h ({z_F[2]*100:.2f}%)")
    print(f"Temperatura de alimentación: {T_F_C:.2f} °C")
    print(f"Presión de alimentación: {P_F:.2f} kPa")
    print(f"\nNúmero de etapas: {n_stages}")
    print(f"Etapa de alimentación: {feed_stage}")
    print(f"Relación de reflujo: {reflux_ratio}")
    print(f"Presión condensador: {P_condenser:.2f} kPa")
    print(f"Presión rehervidor: {P_reboiler:.2f} kPa")
    
    # ========================================================================
    # PASO 1: SUPONER SEPARACIÓN INICIAL
    # ========================================================================
    
    print(f"\n{'PASO 1: Suposición Inicial de Separación':-^80}")
    
    # Destilado
    D_benzene = recovery_benzene * F[0]
    D_ethylbenzene = recovery_ethylbenzene * F[1]
    D_diethylbenzene = 0.0  # Todo a fondos
    
    D_total = D_benzene + D_ethylbenzene + D_diethylbenzene
    x_D = np.array([D_benzene, D_ethylbenzene, D_diethylbenzene]) / D_total
    
    # Fondos
    B_total = F_total - D_total
    B = F - np.array([D_benzene, D_ethylbenzene, D_diethylbenzene])
    x_B = B / B_total
    
    print(f"\nDestilado estimado: {D_total:.2f} kmol/h")
    print(f"  Composición: Benceno={x_D[0]:.6f}, Etilbenceno={x_D[1]:.6f}, 1,4-Dietilbenceno={x_D[2]:.6f}")
    print(f"\nFondos estimados: {B_total:.2f} kmol/h")
    print(f"  Composición: Benceno={x_B[0]:.6f}, Etilbenceno={x_B[1]:.6f}, 1,4-Dietilbenceno={x_B[2]:.6f}")
    
    # Verificar volatilidades
    print(f"\nVerificación de volatilidades relativas...")
    T_check, y_check = bubble_temperature(P_condenser, x_D, T_init=350)
    print(f"Temperatura de burbuja del destilado: {T_check-273.15:.2f} °C")
    
    # ========================================================================
    # PASOS 2-3: BALANCE DE MASA EN CONDENSADOR
    # ========================================================================
    
    print(f"\n{'PASO 2-3: Balance de Masa en Condensador':-^80}")
    
    V2 = D_total * (1 + reflux_ratio)
    L1 = D_total * reflux_ratio
    U1 = D_total
    
    x_L1 = x_D.copy()
    y_V2 = x_D.copy()  # Condensador total
    
    print(f"Flujo de vapor desde etapa 2: V2 = {V2:.2f} kmol/h")
    print(f"Flujo de reflujo: L1 = {L1:.2f} kmol/h")
    print(f"Flujo de destilado: U1 = {U1:.2f} kmol/h")
    
    # ========================================================================
    # PASO 4: TEMPERATURA EN ETAPA 1
    # ========================================================================
    
    T1, y1 = bubble_temperature(P_stages[0], x_L1, T_init=350)
    print(f"\n{'PASO 4: Temperatura Etapa 1':-^80}")
    print(f"T1 = {T1-273.15:.2f} °C")
    
    # ========================================================================
    # PASO 5: TEMPERATURA DEL REHERVIDOR
    # ========================================================================
    
    T_reboiler, y_reboiler = bubble_temperature(P_reboiler, x_B, T_init=400)
    print(f"\n{'PASO 5: Temperatura Rehervidor':-^80}")
    print(f"T_reboiler = {T_reboiler-273.15:.2f} °C")
    
    # ========================================================================
    # PASO 6: GRADIENTE DE TEMPERATURAS
    # ========================================================================
    
    print(f"\n{'PASO 6: Gradiente Inicial de Temperaturas':-^80}")
    T_stages = np.linspace(T1, T_reboiler, n_stages)
    
    print(f"Perfil inicial de temperaturas:")
    for i in [0, 4, 7, 12, 19]:
        print(f"  Etapa {i+1}: {T_stages[i]-273.15:.2f} °C")
    
    # ========================================================================
    # PASO 7-8: FLUJOS Y COMPOSICIONES INICIALES
    # ========================================================================
    
    V_stages = np.ones(n_stages) * V2
    
    y_stages = np.zeros((n_stages, n_comp))
    for i in range(n_stages):
        alpha = i / (n_stages - 1) if n_stages > 1 else 0
        y_stages[i] = (1 - alpha) * y_V2 + alpha * y_reboiler
        y_stages[i] = y_stages[i] / np.sum(y_stages[i])
    
    # Inicializar arrays para almacenar resultados
    L_stages = np.zeros(n_stages)
    x_stages = np.zeros((n_stages, n_comp))
    
    L_stages[0] = L1
    x_stages[0] = x_L1
    
    # ========================================================================
    # ITERACIÓN PRINCIPAL
    # ========================================================================
    
    print(f"\n{'INICIANDO ITERACIONES':-^80}")
    
    max_iterations = 50
    tolerance = 1.0  # K²
    
    for iteration in range(max_iterations):
        
        T_stages_old = T_stages.copy()
        
        # ====================================================================
        # PASOS 9-11: ETAPAS ANTES DE LA ALIMENTACIÓN (Etapas 2-7)
        # ====================================================================
        
        for stage in range(1, feed_stage - 1):
            L_in = L_stages[stage - 1]
            x_in = x_stages[stage - 1]
            V_in = V_stages[min(stage + 1, n_stages - 1)]
            y_in = y_stages[min(stage + 1, n_stages - 1)]
            
            F_total_stage = L_in + V_in
            z_stage = (L_in * x_in + V_in * y_in) / F_total_stage if F_total_stage > 0 else x_in
            
            # Flash isotérmico
            L_out, x_out, V_out, y_out, beta = flash_isotermico(
                F_total_stage, z_stage, T_stages[stage], P_stages[stage]
            )
            
            L_stages[stage] = L_out
            x_stages[stage] = x_out
            V_stages[stage] = V_out
            y_stages[stage] = y_out
        
        # ====================================================================
        # PASO 12: ETAPA DE ALIMENTACIÓN (Etapa 8)
        # ====================================================================
        
        stage = feed_stage - 1
        L_in = L_stages[stage - 1]
        x_in = x_stages[stage - 1]
        V_in = V_stages[min(stage + 1, n_stages - 1)]
        y_in = y_stages[min(stage + 1, n_stages - 1)]
        
        # Agregar alimentación
        F_total_stage = L_in + V_in + F_total
        z_stage = (L_in * x_in + V_in * y_in + F_total * z_F) / F_total_stage
        
        L_out, x_out, V_out, y_out, beta = flash_isotermico(
            F_total_stage, z_stage, T_stages[stage], P_stages[stage]
        )
        
        L_stages[stage] = L_out
        x_stages[stage] = x_out
        V_stages[stage] = V_out
        y_stages[stage] = y_out
        
        # ====================================================================
        # PASO 13: ETAPAS DESPUÉS DE LA ALIMENTACIÓN (Etapas 9-19)
        # ====================================================================
        
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
        
        # ====================================================================
        # PASO 14: REHERVIDOR (Etapa 20)
        # ====================================================================
        
        stage = n_stages - 1
        L_in = L_stages[stage - 1]
        x_in = x_stages[stage - 1]
        
        # Temperatura de burbuja para fondos
        T_reboiler_new, y_out = bubble_temperature(P_reboiler, x_in, T_init=T_reboiler)
        
        L_stages[stage] = L_in
        x_stages[stage] = x_in
        V_stages[stage] = V2  # Aproximación
        y_stages[stage] = y_out
        T_stages[stage] = T_reboiler_new
        
        T_reboiler = T_reboiler_new
        
        # ====================================================================
        # PASO 15: RECALCULAR T1
        # ====================================================================
        
        T1_new, _ = bubble_temperature(P_stages[0], x_stages[0], T_init=T1)
        T_stages[0] = T1_new
        T1 = T1_new
        
        # ====================================================================
        # PASO 16: EVALUAR CONVERGENCIA
        # ====================================================================
        
        convergence = np.sum((T_stages - T_stages_old) ** 2)
        
        if (iteration + 1) % 5 == 0 or iteration == 0:
            print(f"Iteración {iteration+1:3d}: Convergencia = {convergence:10.4f} K²")
        
        if convergence < tolerance:
            print(f"\n¡CONVERGENCIA ALCANZADA en iteración {iteration+1}!")
            print(f"Criterio de convergencia: {convergence:.6f} K² < {tolerance} K²")
            break
    
# ========================================================================
    # RESULTADOS FINALES
    # ========================================================================
    
    print(f"\n{'RESULTADOS FINALES':-^80}")
    
    # Calcular recuperaciones reales
    # Flujos molares en destilado
    D_benzene_real = U1 * x_stages[0][0]
    D_ethylbenzene_real = U1 * x_stages[0][1]
    D_diethylbenzene_real = U1 * x_stages[0][2]
    
    # Flujos molares en fondos
    B_benzene_real = L_stages[-1] * x_stages[-1][0]
    B_ethylbenzene_real = L_stages[-1] * x_stages[-1][1]
    B_diethylbenzene_real = L_stages[-1] * x_stages[-1][2]
    
    # Recuperaciones (% del componente en alimentación que sale en destilado)
    recovery_benzene_real = (D_benzene_real / F[0]) * 100
    recovery_ethylbenzene_real = (D_ethylbenzene_real / F[1]) * 100
    recovery_diethylbenzene_real = (D_diethylbenzene_real / F[2]) * 100
    
    # Purezas (% molar en la corriente)
    purity_benzene_destilado = x_stages[0][0] * 100
    purity_ethylbenzene_destilado = x_stages[0][1] * 100
    purity_diethylbenzene_destilado = x_stages[0][2] * 100
    
    purity_benzene_fondos = x_stages[-1][0] * 100
    purity_ethylbenzene_fondos = x_stages[-1][1] * 100
    purity_diethylbenzene_fondos = x_stages[-1][2] * 100
    
    print(f"\n{'RECUPERACIONES Y PUREZAS':-^80}")
    print(f"\n{'DESTILADO':-^80}")
    print(f"Flujo total: {U1:.2f} kmol/h")
    print(f"\nComponente         Flujo (kmol/h)  Recuperación (%)  Pureza (%)")
    print("-" * 70)
    print(f"{'Benceno':<18} {D_benzene_real:>14.4f}  {recovery_benzene_real:>15.2f}  {purity_benzene_destilado:>10.4f}")
    print(f"{'Etilbenceno':<18} {D_ethylbenzene_real:>14.4f}  {recovery_ethylbenzene_real:>15.2f}  {purity_ethylbenzene_destilado:>10.4f}")
    print(f"{'1,4-Dietilbenceno':<18} {D_diethylbenzene_real:>14.4f}  {recovery_diethylbenzene_real:>15.2f}  {purity_diethylbenzene_destilado:>10.4f}")
    
    print(f"\n{'FONDOS':-^80}")
    print(f"Flujo total: {L_stages[-1]:.2f} kmol/h")
    print(f"\nComponente         Flujo (kmol/h)  Recuperación (%)  Pureza (%)")
    print("-" * 70)
    print(f"{'Benceno':<18} {B_benzene_real:>14.4f}  {(B_benzene_real/F[0])*100:>15.2f}  {purity_benzene_fondos:>10.4f}")
    print(f"{'Etilbenceno':<18} {B_ethylbenzene_real:>14.4f}  {(B_ethylbenzene_real/F[1])*100:>15.2f}  {purity_ethylbenzene_fondos:>10.4f}")
    print(f"{'1,4-Dietilbenceno':<18} {B_diethylbenzene_real:>14.4f}  {(B_diethylbenzene_real/F[2])*100:>15.2f}  {purity_diethylbenzene_fondos:>10.4f}")
    
    print(f"\n{'COMPARACIÓN CON OBJETIVOS':-^80}")
    print(f"\n{'Especificación':<50} {'Objetivo':<15} {'Obtenido':<15} {'Cumple':<10}")
    print("-" * 90)
    
    # Objetivo de recuperación de benceno: 99.86%
    objetivo_rec_benzene = 99.86
    cumple_benzene = "✓ SÍ" if recovery_benzene_real >= objetivo_rec_benzene else "✗ NO"
    print(f"{'Recuperación de Benceno en destilado (%)':<50} {objetivo_rec_benzene:>14.2f}% {recovery_benzene_real:>14.2f}% {cumple_benzene:>10}")
    
    # Objetivo de recuperación de etilbenceno: 1%
    objetivo_rec_ethylbenzene = 1.0
    cumple_ethylbenzene = "✓ SÍ" if abs(recovery_ethylbenzene_real - objetivo_rec_ethylbenzene) <= 0.5 else "✗ NO"
    print(f"{'Recuperación de Etilbenceno en destilado (%)':<50} {objetivo_rec_ethylbenzene:>14.2f}% {recovery_ethylbenzene_real:>14.2f}% {cumple_ethylbenzene:>10}")
    
    print(f"\n{'BALANCE DE MATERIA':-^80}")
    print(f"\n{'Componente':<20} {'Alimentación':<15} {'Destilado':<15} {'Fondos':<15} {'Balance':<15}")
    print("-" * 80)
    
    for i, comp_name in enumerate(["Benceno", "Etilbenceno", "1,4-Dietilbenceno"]):
        F_comp = F[i]
        D_comp = U1 * x_stages[0][i]
        B_comp = L_stages[-1] * x_stages[-1][i]
        balance = F_comp - (D_comp + B_comp)
        print(f"{comp_name:<20} {F_comp:>14.4f}  {D_comp:>14.4f}  {B_comp:>14.4f}  {balance:>14.6f}")
    
    F_total_check = np.sum(F)
    D_total_check = U1
    B_total_check = L_stages[-1]
    balance_total = F_total_check - (D_total_check + B_total_check)
    print("-" * 80)
    print(f"{'TOTAL':<20} {F_total_check:>14.4f}  {D_total_check:>14.4f}  {B_total_check:>14.4f}  {balance_total:>14.6f}")
    
    print(f"\nPERFIL DE TEMPERATURAS Y PRESIONES:")
    print(f"{'Etapa':<8} {'T (°C)':<12} {'P (kPa)':<12}")
    print("-" * 32)
    for i in range(0, n_stages, 2):  # Mostrar cada 2 etapas para no saturar
        print(f"{i+1:<8} {T_stages[i]-273.15:<12.2f} {P_stages[i]:<12.2f}")
    if n_stages % 2 == 0:  # Asegurar que se muestra la última etapa
        print(f"{n_stages:<8} {T_stages[-1]-273.15:<12.2f} {P_stages[-1]:<12.2f}")
    
    print(f"\nCOMPOSICIÓN DEL DESTILADO (Etapa 1):")
    print(f"  Benceno: {x_stages[0][0]:.6f} ({x_stages[0][0]*100:.4f}%)")
    print(f"  Etilbenceno: {x_stages[0][1]:.6f} ({x_stages[0][1]*100:.4f}%)")
    print(f"  1,4-Dietilbenceno: {x_stages[0][2]:.6f} ({x_stages[0][2]*100:.4f}%)")
    
    print(f"\nCOMPOSICIÓN DE FONDOS (Etapa {n_stages}):")
    print(f"  Benceno: {x_stages[-1][0]:.6f} ({x_stages[-1][0]*100:.4f}%)")
    print(f"  Etilbenceno: {x_stages[-1][1]:.6f} ({x_stages[-1][1]*100:.4f}%)")
    print(f"  1,4-Dietilbenceno: {x_stages[-1][2]:.6f} ({x_stages[-1][2]*100:.4f}%)")
    
    print(f"\nFLUJOS PRINCIPALES:")
    print(f"  Destilado (U1): {U1:.2f} kmol/h")
    print(f"  Reflujo (L1): {L1:.2f} kmol/h")
    print(f"  Vapor (V2): {V2:.2f} kmol/h")
    print(f"  Fondos: {L_stages[-1]:.2f} kmol/h")
    print(f"  Relación de reflujo efectiva: {L1/U1:.4f}")
    
    # Guardar resultados
    resultados = {
        'T_stages': T_stages,
        'P_stages': P_stages,
        'x_stages': x_stages,
        'y_stages': y_stages,
        'L_stages': L_stages,
        'V_stages': V_stages,
        'destilado': {
            'flujo': U1, 
            'composicion': x_stages[0],
            'flujos_componentes': np.array([D_benzene_real, D_ethylbenzene_real, D_diethylbenzene_real]),
            'recuperaciones': np.array([recovery_benzene_real, recovery_ethylbenzene_real, recovery_diethylbenzene_real])
        },
        'fondos': {
            'flujo': L_stages[-1], 
            'composicion': x_stages[-1],
            'flujos_componentes': np.array([B_benzene_real, B_ethylbenzene_real, B_diethylbenzene_real])
        },
        'objetivos': {
            'recovery_benzene_objetivo': objetivo_rec_benzene,
            'recovery_benzene_obtenido': recovery_benzene_real,
            'recovery_ethylbenzene_objetivo': objetivo_rec_ethylbenzene,
            'recovery_ethylbenzene_obtenido': recovery_ethylbenzene_real,
            'cumple_benzene': cumple_benzene,
            'cumple_ethylbenzene': cumple_ethylbenzene
        }
    }
    
    
    return resultados
