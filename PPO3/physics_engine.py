import numpy as np
import math


class SystemConfig:
    # --- Costanti Fisiche Globali ---
    C = 299792458
    WAVELENGTH = 1550e-9
    TX_APERTURE = 0.3
    RX_APERTURE = 1.0

    # Efficienze Hardware
    ETA_TERM = 0.5
    ETA_DET = 0.6
    ETA_TURB = 0.8
    ETA_PT = 0.9

    # Atmosfera e Rumore
    ALPHA_0 = 0.1
    F0 = 0.99
    N_BAR = 0.005
    F_THRESHOLD = 0.90

    # Memoria & Protocollo
    T_COH = 1.0
    ETA_RET = 0.9
    P_BSM = 0.5
    F_PULSE = 10e6

    # Parametri RL e Ambiente
    ALPHA = 0.05  # Peso Entanglement
    BETA = 0.05  # Peso Fedeltà
    DELTA = 0.9  # Penalità Scadenze
    TTL_STEPS = 225  # Tempo di vita di una richiesta


class PhysicsEngine:
    def __init__(self, config=SystemConfig()):
        self.cfg = config

    def get_link_metrics(self, distance_km, elevation_deg):
        dist_m = distance_km * 1000.0
        elev_rad = np.radians(max(elevation_deg, 1.0))  # Evita divisione per zero

        denom = (4 * self.cfg.WAVELENGTH * dist_m)
        if denom == 0: return 0.0, 0.0

        eta_fs = ((math.pi * self.cfg.TX_APERTURE * self.cfg.RX_APERTURE) / denom) ** 2
        eta_fs = min(1.0, eta_fs)

        sin_theta = max(np.sin(elev_rad), 1e-4)
        t_atm = np.exp(-self.cfg.ALPHA_0 / sin_theta)
        eta_sg = eta_fs * t_atm

        s = (self.cfg.ETA_TERM * self.cfg.ETA_DET * self.cfg.ETA_TURB * self.cfg.ETA_PT * eta_sg)
        return s, eta_sg

    def calculate_fidelity(self, eta_sg):
        if eta_sg <= 1e-9: return 0.25
        noise_factor = (1 + self.cfg.N_BAR / eta_sg) ** 2
        term = (4 * self.cfg.F0 - 1) / noise_factor
        return 0.25 * (1 + term)

    def purify(self, f_current, f_raw, rounds=1):
        """Purificazione ricorsiva (Equazione 3.8 della tesi)"""
        f_new = f_current
        for _ in range(rounds):
            num = f_new * f_raw
            den = num + (1 - f_new) * (1 - f_raw)
            if den == 0: return 0.0
            f_new = num / den
        return f_new

    def get_flip_probabilities(self, f_val):
        if f_val < 0.5: return 0.5
        arg = max(0.0, 2 * f_val - 1)
        return 0.5 + 0.5 * math.sqrt(arg)

    def get_end_to_end_prob(self, s_a, s_b, dist_a, dist_b, f_final):
        max_dist_m = max(dist_a, dist_b) * 1000
        t_rt = (2 * max_dist_m) / self.cfg.C
        eta_mem = self.cfg.ETA_RET * np.exp(-t_rt / self.cfg.T_COH)

        p_raw = s_a * s_b * self.cfg.P_BSM * eta_mem
        p_nf = self.get_flip_probabilities(f_final)
        p_state_survival = (p_nf ** 2) + ((1 - p_nf) ** 2)

        return p_raw * p_state_survival