import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
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
    "lines.linewidth": 1.4,
    "grid.linewidth": 0.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})


def chunk_sum(data, chunk_size):
    return [sum(data[i:i + chunk_size]) for i in range(0, len(data), chunk_size)]


def chunk_mean(data, chunk_size):
    return [np.mean(data[i:i + chunk_size]) for i in range(0, len(data), chunk_size)]


def moving_average(data, window_size):
    if len(data) < window_size:
        return np.array(data)
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def save_ieee_plot(filename):
    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"[OK] Grafico salvato: {filename}")


def main():
    print("=== GENERAZIONE GRAFICI IEEE: PPO vs GREEDY (CODE DINAMICHE) ===")

    try:
        with open("results_ppo.json", "r") as f:
            d_ppo = json.load(f)
        with open("results_greedy.json", "r") as f:
            d_greedy = json.load(f)
    except FileNotFoundError:
        print("Errore: File JSON non trovati. Esegui prima i test PPO e Greedy.")
        return

    CHUNK_SIZE = 1000
    STEPS_PER_SEC = 225  # SystemConfig.TTL_STEPS

    # ==========================================
    # 1. ANDAMENTO REWARD (Media mobile)
    # ==========================================
    plt.figure(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    window = 2000

    if len(d_ppo["rewards"]) >= window:
        plt.plot(moving_average(d_ppo["rewards"], window), label="PPO Reward", color="seagreen", linewidth=0.8)
        plt.plot(moving_average(d_greedy["rewards"], window), label="Greedy Reward", color="darkorchid", linewidth=0.8)

    plt.xlabel("Simulation Steps")
    plt.ylabel("Smoothed Reward")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best', frameon=True, edgecolor='black')
    save_ieee_plot("ieee_01_rewards.png")

    # ==========================================
    # 2. PAIR RATIO (Cumulative Pairs / Max Possible Pairs from dynamically generated reqs)
    # ==========================================
    plt.figure(figsize=(IEEE_WIDTH, IEEE_HEIGHT))

    # np.maximum per evitare divisioni per zero se all'inizio total_requests è 0
    ppo_total_possible = np.maximum(np.array(d_ppo["total_requests"]) * 3, 1)
    greedy_total_possible = np.maximum(np.array(d_greedy["total_requests"]) * 3, 1)

    ppo_ratio = np.cumsum(d_ppo["pairs_generated"]) / ppo_total_possible
    greedy_ratio = np.cumsum(d_greedy["pairs_generated"]) / greedy_total_possible

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
    # 6. RAW DISCRETE PAIRS GENERATED (Zoom)
    # ==========================================
    plt.figure(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    zoom_steps = 150

    if len(d_ppo["pairs_generated"]) > zoom_steps:
        raw_ppo = d_ppo["pairs_generated"][:zoom_steps]
        raw_greedy = d_greedy["pairs_generated"][:zoom_steps]

        plt.plot(raw_ppo, label="PPO (Raw Pairs)", color="seagreen", drawstyle="steps-mid", linewidth=1.2)
        plt.plot(raw_greedy, label="Greedy (Raw Pairs)", color="darkorchid", drawstyle="steps-mid", linewidth=1.2,
                 alpha=0.7)

    plt.yticks([0, 3, 6, 9, 12, 15])  # Alzato il limite visivo in caso di code lunghe attive
    plt.ylim(-0.5, 15.5)
    plt.xlabel(f"Simulation Steps (Raw view: first {zoom_steps} steps)")
    plt.ylabel("Exact Pairs Generated")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best', frameon=True, edgecolor='black')
    save_ieee_plot("ieee_06_raw_pairs_zoom.png")

    # ==========================================
    # 7. GENERATED VS SERVED (Cumulative Load Analysis) - NUOVO GRAFICO
    # ==========================================
    plt.figure(figsize=(IEEE_WIDTH, IEEE_HEIGHT))

    # Plottiamo l'andamento del carico (quante richieste entrano vs quante vengono evase)
    plt.plot(d_ppo["total_requests"][::sample_rate], label="Total Generated (Load)", color="black", linestyle='--',
             linewidth=1.0)
    plt.plot(np.cumsum(d_ppo["served"])[::sample_rate], label="PPO Total Served", color="seagreen", linewidth=1.2)
    plt.plot(np.cumsum(d_greedy["served"])[::sample_rate], label="Greedy Total Served", color="darkorchid",
             linewidth=1.2)

    plt.xlabel(f"Simulation Steps (x{sample_rate})")
    plt.ylabel("Cumulative Requests")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left', frameon=True, edgecolor='black')
    save_ieee_plot("ieee_07_cumulative_load_vs_served.png")

    print("\n[FINITO] Tutti i grafici sono stati generati con successo!")


if __name__ == "__main__":
    main()