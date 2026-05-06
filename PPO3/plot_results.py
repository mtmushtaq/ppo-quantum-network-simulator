import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from physics_engine import SystemConfig
import os

# --- STILE IEEE ---
IEEE_WIDTH = 3.4
IEEE_HEIGHT = 2.1

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "TeX Gyre Termes"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "pdf.use14corefonts": True,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.8,
    "axes.titlesize": 9,
    "axes.linewidth": 0.9,
    "lines.linewidth": 1.4,  # Default globale
    "grid.linewidth": 0.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})


def chunk_sum(data, chunk_size):
    """Somma i dati in blocchi."""
    return [sum(data[i:i + chunk_size]) for i in range(0, len(data), chunk_size)]


def chunk_mean(data, chunk_size):
    """Media dei dati in blocchi."""
    return [np.mean(data[i:i + chunk_size]) for i in range(0, len(data), chunk_size)]


def moving_average(data, window_size):
    """Media mobile per smussare le curve troppo rumorose."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def save_ieee_plot(filename):
    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"[OK] Grafico salvato: {filename}")


def main():
    print("=== GENERAZIONE GRAFICI IEEE: PPO vs GREEDY ===")

    try:
        with open("results_ppo.json", "r") as f:
            d_ppo = json.load(f)
        with open("results_greedy.json", "r") as f:
            d_greedy = json.load(f)
    except FileNotFoundError:
        print("Errore: File JSON non trovati. Esegui prima i test PPO e Greedy.")
        return

    CHUNK_SIZE = 1000
    STEPS_PER_SEC = SystemConfig.TTL_STEPS

    # ==========================================
    # 1. ANDAMENTO REWARD (Media mobile)
    # ==========================================
    plt.figure(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    window = 2000

    if len(d_ppo["rewards"]) >= window:
        # MODIFICA QUI: linewidth=0.8 per linee più sottili
        plt.plot(moving_average(d_ppo["rewards"], window), label="PPO Reward", color="seagreen", linewidth=0.8)
        plt.plot(moving_average(d_greedy["rewards"], window), label="Greedy Reward", color="darkorchid", linewidth=0.8)

    plt.xlabel("Simulation Steps")
    plt.ylabel("Smoothed Reward")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best', frameon=True, edgecolor='black')
    save_ieee_plot("ieee_01_rewards.png")

    # ==========================================
    # 2. PAIR RATIO (Cumulative Pairs / Cumulative Req)
    # ==========================================
    plt.figure(figsize=(IEEE_WIDTH, IEEE_HEIGHT))

    ppo_ratio = np.cumsum(d_ppo["pairs_generated"]) / (np.array(d_ppo["total_requests"]) * 3)
    greedy_ratio = np.cumsum(d_greedy["pairs_generated"]) / (np.array(d_greedy["total_requests"]) * 3)

    sample_rate = 100
    plt.plot(ppo_ratio[::sample_rate], label="PPO Success Rate", color="seagreen")
    plt.plot(greedy_ratio[::sample_rate], label="Greedy Success Rate", color="darkorchid")

    plt.ylim(0, 1.05)
    plt.xlabel(f"Simulation Steps (x{sample_rate})")
    plt.ylabel("Pair Generation Ratio")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', frameon=True, edgecolor='black')
    save_ieee_plot("ieee_02_pair_ratio.png")

    # ==========================================
    # 3. THROUGHPUT (Soddisfatte al Secondo)
    # ==========================================
    plt.figure(figsize=(IEEE_WIDTH, IEEE_HEIGHT))

    ppo_thr = chunk_sum(d_ppo["served"], STEPS_PER_SEC)
    greedy_thr = chunk_sum(d_greedy["served"], STEPS_PER_SEC)

    window_thr = 5
    if len(ppo_thr) > window_thr:
        plt.plot(moving_average(ppo_thr, window_thr), label="PPO", color="seagreen")
        plt.plot(moving_average(greedy_thr, window_thr), label="Greedy", color="darkorchid")

    plt.xlabel("Time (Seconds)")
    plt.ylabel("Served Requests / sec")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best', frameon=True, edgecolor='black')
    save_ieee_plot("ieee_03_throughput.png")

    # ==========================================
    # 4. SUCCESSI TOTALI (Ogni 1000 timestep)
    # ==========================================
    plt.figure(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    x_chunks = np.arange(1, len(chunk_sum(d_ppo["served"], CHUNK_SIZE)) + 1)

    plt.plot(x_chunks, chunk_sum(d_ppo["served"], CHUNK_SIZE), label="PPO Served", color="seagreen", marker='o',
             markersize=3)
    plt.plot(x_chunks, chunk_sum(d_greedy["served"], CHUNK_SIZE), label="Greedy Served", color="darkorchid", marker='s',
             markersize=3)

    plt.xlabel(f"Time Intervals ({CHUNK_SIZE} steps)")
    plt.ylabel("Completed Requests (Sum)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best', frameon=True, edgecolor='black')
    save_ieee_plot("ieee_04_successes_sum.png")

    # ==========================================
    # 5. FALLIMENTI TOTALI (Scadute ogni 1000 timestep)
    # ==========================================
    plt.figure(figsize=(IEEE_WIDTH, IEEE_HEIGHT))

    plt.plot(x_chunks, chunk_sum(d_ppo["expired"], CHUNK_SIZE), label="PPO Expired", color="darkorange", marker='o',
             markersize=3)
    plt.plot(x_chunks, chunk_sum(d_greedy["expired"], CHUNK_SIZE), label="Greedy Expired", color="red", marker='s',
             markersize=3)

    plt.xlabel(f"Time Intervals ({CHUNK_SIZE} steps)")
    plt.ylabel("Expired Requests (Sum)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best', frameon=True, edgecolor='black')
    save_ieee_plot("ieee_05_failures_sum.png")

    # ==========================================
    # 6. FALLIMENTI MEDI (Richieste scadute MEDIE per 1000 timestep)
    # ==========================================
    plt.figure(figsize=(IEEE_WIDTH, IEEE_HEIGHT))

    plt.plot(x_chunks, chunk_mean(d_ppo["expired"], CHUNK_SIZE), label="PPO Avg Expired", color="darkorange",
             marker='o', markersize=3)
    plt.plot(x_chunks, chunk_mean(d_greedy["expired"], CHUNK_SIZE), label="Greedy Avg Expired", color="red", marker='s',
             markersize=3)

    plt.xlabel(f"Time Intervals ({CHUNK_SIZE} steps)")
    plt.ylabel("Avg Expired Requests / Step")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best', frameon=True, edgecolor='black')
    save_ieee_plot("ieee_06_failures_average.png")

    print("\n[FINITO] Tutti i grafici sono stati generati con successo!")

    # # ==========================================
    # # 7. PAIRS GENERATED PER STEP (High-Frequency Fluctuations)
    # # ==========================================
    # plt.figure(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    #
    # # Usiamo una finestra piccola per mostrare le fluttuazioni (come chiesto dal prof)
    # # senza che diventi una macchia di colore incomprensibile
    # fluctuation_window = 1
    #
    # if len(d_ppo["pairs_generated"]) > fluctuation_window:
    #     ppo_fluctuations = moving_average(d_ppo["pairs_generated"], fluctuation_window)
    #     greedy_fluctuations = moving_average(d_greedy["pairs_generated"], fluctuation_window)
    #
    #     plt.plot(ppo_fluctuations, label="PPO (Pairs/Step)", color="seagreen", linewidth=0.8)
    #     plt.plot(greedy_fluctuations, label="Greedy (Pairs/Step)", color="darkorchid", linewidth=0.8, alpha=0.7)
    #
    # plt.xlabel(f"Simulation Steps (Moving Avg Window={fluctuation_window})")
    # plt.ylabel("Pairs Generated per 22ms Step")
    # plt.grid(True, linestyle='--', alpha=0.6)
    # plt.legend(loc='best', frameon=True, edgecolor='black')
    # save_ieee_plot("ieee_07_pairs_per_step_fluctuations.png")

    # ==========================================
    # 7. RAW DISCRETE PAIRS GENERATED PER STEP (Max 9)
    # ==========================================
    plt.figure(figsize=(IEEE_WIDTH, IEEE_HEIGHT))

    # Zoom sui primi 150 step per rendere i dati raw leggibili
    zoom_steps = 150

    if len(d_ppo["pairs_generated"]) > zoom_steps:
        raw_ppo = d_ppo["pairs_generated"][:zoom_steps]
        raw_greedy = d_greedy["pairs_generated"][:zoom_steps]

        # Plot dei dati PURI, senza media
        plt.plot(raw_ppo, label="PPO (Raw Pairs)", color="seagreen", drawstyle="steps-mid", linewidth=1.2)
        plt.plot(raw_greedy, label="Greedy (Raw Pairs)", color="darkorchid", drawstyle="steps-mid", linewidth=1.2,
                 alpha=0.7)

    # L'asse Y mostrerà esplicitamente solo 0, 3, 6, 9
    plt.yticks([0, 3, 6, 9])
    plt.ylim(-0.5, 9.5)

    plt.xlabel(f"Simulation Steps (Raw view: first {zoom_steps} steps)")
    plt.ylabel("Exact Pairs Generated (Discrete)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best', frameon=True, edgecolor='black')
    save_ieee_plot("ieee_07_raw_pairs_max9.png")

    # ==========================================
    # 8. RAW PAIRS GENERATED (All Simulation Steps)
    # ==========================================
    plt.figure(figsize=(IEEE_WIDTH, IEEE_HEIGHT))

    # Plottiamo tutto l'episodio. Usiamo una linea sottile per non fare un muro nero.
    plt.plot(d_ppo["pairs_generated"], label="PPO", color="seagreen", linewidth=0.2)
    plt.plot(d_greedy["pairs_generated"], label="Greedy", color="darkorchid", linewidth=0.2, alpha=0.5)

    plt.yticks([0, 3, 6, 9])
    plt.ylim(-0.5, 9.5)
    plt.xlabel("Simulation Steps (Entire Episode)")
    plt.ylabel("Raw Pairs Generated")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best', frameon=True, edgecolor='black')
    save_ieee_plot("ieee_08_raw_pairs_all_steps.png")

    # ==========================================
    # 9. REWARD COMPONENTS BREAKDOWN (PPO Only)
    # ==========================================
    plt.figure(figsize=(IEEE_WIDTH, IEEE_HEIGHT))

    # Pesi normalizzati
    alpha_w = SystemConfig.ALPHA
    beta_w = SystemConfig.BETA
    delta_w = SystemConfig.DELTA

    served_arr = np.array(d_ppo["served"])
    fid_arr = np.array(d_ppo["avg_fidelity"])
    exp_arr = np.array(d_ppo["expired"])

    # Calcolo delle singole componenti step-by-step
    c1_entanglement = alpha_w * (served_arr / 3.0)
    c2_fidelity = beta_w * fid_arr
    c3_penalty = - delta_w * exp_arr
    total_reward = c1_entanglement + c2_fidelity + c3_penalty

    # Usiamo una media mobile per rendere leggibile la correlazione dei crolli
    window_breakdown = 100

    plt.plot(moving_average(c1_entanglement, window_breakdown), label="Entanglement (+)", color="seagreen",
             linewidth=1.0)
    plt.plot(moving_average(c2_fidelity, window_breakdown), label="Fidelity (+)", color="dodgerblue", linewidth=1.0)
    plt.plot(moving_average(c3_penalty, window_breakdown), label="Expiration Penalty (-)", color="firebrick",
             linewidth=1.0)
    plt.plot(moving_average(total_reward, window_breakdown), label="Total Reward (Sum)", color="black", linestyle="--",
             linewidth=1.2)

    plt.xlabel(f"Simulation Steps (Moving Avg Window={window_breakdown})")
    plt.ylabel("Reward Value")
    plt.grid(True, linestyle='--', alpha=0.6)
    # Riduciamo il font della legenda per farci stare 4 voci
    plt.legend(loc='lower center', ncol=2, frameon=True, edgecolor='black', fontsize=6)
    save_ieee_plot("ieee_09_reward_breakdown.png")


if __name__ == "__main__":
    main()

