"""
Simulación de Columna de Destilación con Modelo UNIFAC
Universidad de Antioquia - Termodinámica Química
Separación de Benceno, Etilbenceno y 1,4-Dietilbenceno

INCLUYE: Flash Adiabático con Balance de Energía Completo
Desarrollado por: David
Curso: Termodinámica Química
"""

import numpy as np
from scipy.optimize import brentq, fsolve

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


# ============================================================================
# FUNCIONES DE ENTALPÍA (NUEVAS)
# ============================================================================

def enthalpy_liquid(T_K, composition):
    """
    Calcula la entalpía específica de una mezcla líquida (J/mol)
    Estado de referencia: líquido puro a T_ref = 298.15 K
    
    H_L = Σ(x_i * Cp_liquid_i * (T - T_ref))
    """
    H = 0.0
    for i, comp in enumerate(components):
        H += composition[i] * Cp_liquid[comp] * (T_K - T_ref)
    
    return H


def enthalpy_vapor(T_K, composition):
    """
    Calcula la entalpía específica de una mezcla vapor (J/mol)
    Estado de referencia: líquido puro a T_ref = 298.15 K
    
    H_V = Σ(y_i * [DHvap_i + Cp_vapor_i * (T - T_ref)])
    """
    H = 0.0
    for i, comp in enumerate(components):
        # Calor de vaporización + calor sensible del vapor
        H += composition[i] * (DHvap[comp] + Cp_vapor[comp] * (T_K - T_ref))
    
    return H


def enthalpy_stream(n_mol, composition, T_K, phase):
    """
    Calcula la entalpía total de una corriente (J/h)
    
    Parámetros:
    -----------
    n_mol: Flujo molar (kmol/h)
    composition: Composición (fracción molar)
    T_K: Temperatura (K)
    phase: "liquid" o "vapor"
    
    Retorna:
    --------
    H_total: Entalpía total en J/h
    """
    if n_mol <= 0:
        return 0.0
    
    # Convertir kmol/h a mol/h
    n_mol_h = n_mol * 1000.0
    
    if phase == "liquid":
        h_spec = enthalpy_liquid(T_K, composition)
    elif phase == "vapor":
        h_spec = enthalpy_vapor(T_K, composition)
    else:
        raise ValueError("phase debe ser 'liquid' o 'vapor'")
    
    H_total = n_mol_h * h_spec
    
    return H_total


# ============================================================================
# FLASH ADIABÁTICO (NUEVA IMPLEMENTACIÓN)
# ============================================================================

def flash_adiabatico(L_in, x_in, T_L_in, V_in, y_in, T_V_in, P_kPa):
    """
    Realiza un flash adiabático completo con balance de materia y energía
    
    Parámetros:
    -----------
    L_in: Flujo molar líquido entrante (kmol/h)
    x_in: Composición líquida entrante
    T_L_in: Temperatura líquido entrante (K)
    V_in: Flujo molar vapor entrante (kmol/h)
    y_in: Composición vapor entrante
    T_V_in: Temperatura vapor entrante (K)
    P_kPa: Presión de operación (kPa)
    
    Retorna:
    --------
    L_out, x_out, V_out, y_out, T_out
    """
    
    # Normalizar composiciones
    x_in = np.array(x_in)
    x_in = np.maximum(x_in, 1e-10)
    x_in = x_in / np.sum(x_in)
    
    y_in = np.array(y_in)
    y_in = np.maximum(y_in, 1e-10)
    y_in = y_in / np.sum(y_in)
    
    # Balance de masa total y por componente
    F_total = L_in + V_in
    
    if F_total < 1e-6:
        return 0.0, x_in, 0.0, y_in, T_L_in
    
    z = (L_in * x_in + V_in * y_in) / F_total
    z = z / np.sum(z)
    
    # Calcular entalpía de entrada (J/h)
    H_in = enthalpy_stream(L_in, x_in, T_L_in, "liquid")
    H_in += enthalpy_stream(V_in, y_in, T_V_in, "vapor")
    
    # Temperatura inicial de salida (promedio ponderado)
    if L_in + V_in > 0:
        T_out_guess = (L_in * T_L_in + V_in * T_V_in) / (L_in + V_in)
    else:
        T_out_guess = T_L_in
    
    # Función objetivo: balance de energía
    def energy_balance(T_out):
        """
        Resuelve el equilibrio a T_out y calcula error en balance de energía
        """
        if T_out < 250 or T_out > 650:
            return 1e10
        
        T_C = T_out - 273.15
        
        # Calcular K-values
        try:
            gamma = calculate_UNIFAC_gamma(T_out, z)
            P_sat = np.array([antoine_pressure(T_C, comp) for comp in components])
            P_sat = np.maximum(P_sat, 1e-10)
            K = gamma * P_sat / P_kPa
        except:
            return 1e10
        
        # Resolver Rachford-Rice para beta
        def rachford_rice(beta):
            if beta <= 0 or beta >= 1:
                return 1e10
            return np.sum(z * (K - 1) / (1 + beta * (K - 1)))
        
        try:
            beta = brentq(rachford_rice, 0.001, 0.999)
        except:
            beta = 0.5
        
        # Composiciones de salida
        x_out = z / (1 + beta * (K - 1))
        x_out = np.maximum(x_out, 1e-10)
        x_out = x_out / np.sum(x_out)
        
        y_out = K * x_out
        y_out = np.maximum(y_out, 1e-10)
        y_out = y_out / np.sum(y_out)
        
        # Flujos de salida
        L_out = F_total * (1 - beta)
        V_out = F_total * beta
        
        # Entalpía de salida
        H_out = enthalpy_stream(L_out, x_out, T_out, "liquid")
        H_out += enthalpy_stream(V_out, y_out, T_out, "vapor")
        
        # Error en balance de energía
        error = H_in - H_out
        
        return error
    
    # Resolver para T_out que satisface balance de energía
    try:
        # Buscar temperatura usando método de bisección/Newton
        T_out = fsolve(energy_balance, T_out_guess, full_output=False)[0]
        
        # Limitar temperatura a rangos razonables
        T_out = np.clip(T_out, 250, 650)
        
    except:
        # Si falla, usar temperatura estimada
        T_out = T_out_guess
    
    # Calcular composiciones finales a T_out
    T_C = T_out - 273.15
    gamma = calculate_UNIFAC_gamma(T_out, z)
    P_sat = np.array([antoine_pressure(T_C, comp) for comp in components])
    P_sat = np.maximum(P_sat, 1e-10)
    K = gamma * P_sat / P_kPa
    
    def rachford_rice(beta):
        if beta <= 0 or beta >= 1:
            return 1e10
        return np.sum(z * (K - 1) / (1 + beta * (K - 1)))
    
    try:
        beta = brentq(rachford_rice, 0.001, 0.999)
    except:
        beta = 0.5
    
    x_out = z / (1 + beta * (K - 1))
    x_out = np.maximum(x_out, 1e-10)
    x_out = x_out / np.sum(x_out)
    
    y_out = K * x_out
    y_out = np.maximum(y_out, 1e-10)
    y_out = y_out / np.sum(y_out)
    
    L_out = F_total * (1 - beta)
    V_out = F_total * beta
    
    return L_out, x_out, V_out, y_out, T_out


# ============================================================================
# ALGORITMO DE SIMULACIÓN DE LA COLUMNA
# ============================================================================

def simular_columna_destilacion():
    """Simula la columna de destilación siguiendo el algoritmo del PDF"""
    
    print("=" * 80)
    print("SIMULACIÓN DE COLUMNA DE DESTILACIÓN - MÉTODO UNIFAC")
    print("CON FLASH ADIABÁTICO Y BALANCE DE ENERGÍA COMPLETO")
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
    print("\nINICIANDO ITERACIONES (con flash adiabático)...")
    
    max_iterations = 50
    tolerance = 1.0
    
    for iteration in range(max_iterations):
        
        T_stages_old = T_stages.copy()
        
        # PASOS 9-11: Etapas antes de alimentación (CON FLASH ADIABÁTICO)
        for stage in range(1, feed_stage - 1):
            L_in = L_stages[stage - 1]
            x_in = x_stages[stage - 1]
            V_in = V_stages[min(stage + 1, n_stages - 1)]
            y_in = y_stages[min(stage + 1, n_stages - 1)]
            
            T_L_in = T_stages[stage - 1]
            T_V_in = T_stages[min(stage + 1, n_stages - 1)]
            
            # FLASH ADIABÁTICO con balance de energía
            L_out, x_out, V_out, y_out, T_out = flash_adiabatico(
                L_in, x_in, T_L_in, V_in, y_in, T_V_in, P_stages[stage]
            )
            
            L_stages[stage] = L_out
            x_stages[stage] = x_out
            V_stages[stage] = V_out
            y_stages[stage] = y_out
            T_stages[stage] = T_out
        
        # PASO 12: Etapa de alimentación (CON FLASH ADIABÁTICO)
        stage = feed_stage - 1
        L_in = L_stages[stage - 1]
        x_in = x_stages[stage - 1]
        V_in = V_stages[min(stage + 1, n_stages - 1)]
        y_in = y_stages[min(stage + 1, n_stages - 1)]
        
        T_L_in = T_stages[stage - 1]
        T_V_in = T_stages[min(stage + 1, n_stages - 1)]
        
        # Agregar alimentación mezclando corrientes
        L_total_mixed = L_in + F_total  # Asumiendo alimentación líquida
        x_mixed = (L_in * x_in + F_total * z_F) / L_total_mixed
        T_L_mixed = (L_in * T_L_in + F_total * T_F) / L_total_mixed
        
        # FLASH ADIABÁTICO
        L_out, x_out, V_out, y_out, T_out = flash_adiabatico(
            L_total_mixed, x_mixed, T_L_mixed, V_in, y_in, T_V_in, P_stages[stage]
        )
        
        L_stages[stage] = L_out
        x_stages[stage] = x_out
        V_stages[stage] = V_out
        y_stages[stage] = y_out
        T_stages[stage] = T_out
        
        # PASO 13: Etapas después de alimentación (CON FLASH ADIABÁTICO)
        for stage in range(feed_stage, n_stages - 1):
            L_in = L_stages[stage - 1]
            x_in = x_stages[stage - 1]
            V_in = V_stages[min(stage + 1, n_stages - 1)] if stage < n_stages - 1 else 0
            y_in = y_stages[min(stage + 1, n_stages - 1)] if stage < n_stages - 1 else y_reboiler
            
            T_L_in = T_stages[stage - 1]
            T_V_in = T_stages[min(stage + 1, n_stages - 1)] if stage < n_stages - 1 else T_reboiler
            
            # FLASH ADIABÁTICO
            L_out, x_out, V_out, y_out, T_out = flash_adiabatico(
                L_in, x_in, T_L_in, V_in, y_in, T_V_in, P_stages[stage]
            )
            
            L_stages[stage] = L_out
            x_stages[stage] = x_out
            V_stages[stage] = V_out
            y_stages[stage] = y_out
            T_stages[stage] = T_out
        
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
    
    # Imprimir resultados en formato de columnas
    print("\n" + "=" * 80)
    print("RESULTADOS DE LA SIMULACIÓN")
    print("=" * 80)
    
    print(f"\nNúmero de iteraciones: {num_iteraciones}")
    
    # FLUJOS
    print(f"\n{'FLUJOS TOTALES':-^80}")
    print(f"\n{'Corriente':<30} {'Valor (kmol/h)':<20}")
    print("-" * 50)
    print(f"{'D (Destilado)':<30} {U1:>19.6f}")
    print(f"{'B (Fondos)':<30} {L_stages[-1]:>19.6f}")
    
    # COMPOSICIONES
    print(f"\n{'COMPOSICIONES (FRACCIÓN MOLAR)':-^80}")
    print(f"\n{'Componente':<25} {'Destilado (y_D)':<25} {'Fondos (x_B)':<25}")
    print("-" * 75)
    print(f"{'Bz (Benceno)':<25} {x_stages[0][0]:>24.8f} {x_stages[-1][0]:>24.8f}")
    print(f"{'Et (Etilbenceno)':<25} {x_stages[0][1]:>24.8f} {x_stages[-1][1]:>24.8f}")
    print(f"{'DiEt (1,4-Dietilbenceno)':<25} {x_stages[0][2]:>24.8f} {x_stages[-1][2]:>24.8f}")
    print("-" * 75)
    print(f"{'SUMA':<25} {np.sum(x_stages[0]):>24.8f} {np.sum(x_stages[-1]):>24.8f}")
    
    # RECUPERACIONES
    print(f"\n{'RECUPERACIONES':-^80}")
    print(f"\n{'Componente':<40} {'Recuperación (%)':<20}")
    print("-" * 60)
    print(f"{'Benceno en destilado':<40} {recovery_benzene_real:>19.4f}")
    print(f"{'Etilbenceno en destilado':<40} {recovery_ethylbenzene_real:>19.4f}")
    
    # BALANCE DE MATERIA
    print(f"\n{'BALANCE DE MATERIA':-^80}")
    print(f"\n{'Componente':<20} {'Entrada':<15} {'Destilado':<15} {'Fondos':<15} {'Error':<15}")
    print("-" * 80)
    
    for i, comp_name in enumerate(["Benceno", "Etilbenceno", "1,4-Dietilbenceno"]):
        F_comp = F[i]
        D_comp = U1 * x_stages[0][i]
        B_comp = L_stages[-1] * x_stages[-1][i]
        error = F_comp - (D_comp + B_comp)
        print(f"{comp_name:<20} {F_comp:>14.6f} {D_comp:>14.6f} {B_comp:>14.6f} {error:>14.8f}")
    
    F_total_check = np.sum(F)
    D_total_check = U1
    B_total_check = L_stages[-1]
    error_total = F_total_check - (D_total_check + B_total_check)
    print("-" * 80)
    print(f"{'TOTAL':<20} {F_total_check:>14.6f} {D_total_check:>14.6f} {B_total_check:>14.6f} {error_total:>14.8f}")
    
    error_pct_total = abs(error_total / F_total_check) * 100
    print(f"\nError relativo total: {error_pct_total:.6f}%")
    
    print("\n" + "=" * 80)
    print("SIMULACIÓN COMPLETADA CON BALANCE DE ENERGÍA")
    print("=" * 80)


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    simular_columna_destilacion()