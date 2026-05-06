from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from quantum_env import QuantumSatelliteEnv
import gymnasium as gym
import numpy as np
import random
import os
import json

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# Funzione ponte per ActionMasker
def mask_fn(env: gym.Env) -> np.ndarray:
    return env.unwrapped.valid_action_mask()


def main():
    print("=== TEST DELLA SIMULAZIONE MASKABLE PPO (MICRO-STEP) ===")

    random.seed(42)
    np.random.seed(42)
    env = QuantumSatelliteEnv(csv_path="../data/simultaneous_visibility_3_stations_20-21_test.csv")

    # Applichiamo il wrapper per l'Action Masking
    env = ActionMasker(env, mask_fn)

    test_start = "2026-02-19 20:05:00"
    test_end = "2026-02-19 20:17:00"

    model_path = "models/ppo_trained_maskable"
    try:
        model = MaskablePPO.load(model_path)
    except Exception as e:
        print(f"Errore: Nessun modello trovato in {model_path}. Avvia prima train_ppo.py!")
        return

    print(f"Avvio test ininterrotto dal {test_start} al {test_end}")

    obs, info = env.reset(seed=42, options={
        'start_time': test_start,
        'end_time': test_end
    })

    req_counter = 0
    # Accediamo alla queue tramite env.unwrapped
    for req in env.unwrapped.queue:
        req['id'] = req_counter
        req_counter += 1

    total_generated = 3
    total_served = 0

    fidelities = []

    # Liste storiche per il salvataggio nel JSON
    ppo_rewards = []
    pairs_generated_history = []
    total_requests_history = []
    served_history = []
    expired_history = []
    avg_fidelity_history = []

    print(f"\nTopologia dell'Episodio Continuo:")
    print(f"Source GS: {env.unwrapped.sources} | Dest GS: {env.unwrapped.destinations}")
    print(f"Step Fisici (Macro-Step) totali previsti: {env.unwrapped.max_steps}\n")
    print("-" * 50)

    terminated = False
    physical_steps_passed = 0

    while not terminated:
        # Recuperiamo le maschere logiche
        action_masks = env.action_masks()

        # L'agente sceglie l'azione rispettando i vincoli fisici
        action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        # Salviamo le metriche SOLO alla fine di un ciclo completo (Macro-Step 22ms)
        if info.get("macro_step_complete", False):
            physical_steps_passed += 1
            served_this_step = info.get('served', 0)
            total_served += served_this_step
            ppo_rewards.append(reward)

            avg_fidelity_history.append(info.get('avg_fidelity', 0.0))

            if served_this_step > 0:
                fidelities.append(info['avg_fidelity'])

            pairs_generated_history.append(info.get('pairs_generated', 0))
            served_history.append(served_this_step)
            expired_history.append(info.get('expired', 0))

            for req in env.unwrapped.queue:
                if 'id' not in req:
                    req['id'] = req_counter
                    req_counter += 1
                    total_generated += 1

            total_requests_history.append(total_generated)

            if physical_steps_passed % 45 == 0:
                print(f"Macro-Step {physical_steps_passed:05d}/{env.unwrapped.max_steps} | "
                      f"Servite: {served_this_step}/3 | "
                      f"Tempo: {info['current_time'].split()[-1][:12]}")

    print("-" * 50)
    print("=== RISULTATI FINALI SIMULAZIONE CONTINUA ===")
    print(f"Richieste Totali Generate: {total_generated}")
    print(f"Richieste Servite (3/3): {total_served}")
    print(f"Richieste Scadute (Timeout): {sum(expired_history)}")

    success_rate = (total_served / total_generated) * 100 if total_generated > 0 else 0.0
    print(f"Success Rate Globale: {success_rate:.2f}%")

    overall_avg_fid = np.mean(fidelities) if fidelities else 0.0
    print(f"Fedeltà Media: {overall_avg_fid:.2%}")

    # --- SALVATAGGIO DEI DATI FORMATTATI PER PLOT_RESULTS.PY ---
    export_data = {
        "rewards": ppo_rewards,
        "pairs_generated": pairs_generated_history,
        "total_requests": total_requests_history,
        "served": served_history,
        "expired": expired_history,
        "avg_fidelity": avg_fidelity_history
    }

    with open("results_ppo.json", "w") as f:
        json.dump(export_data, f)
    print("\n[OK] Dati storici PPO salvati con successo in 'results_ppo.json'.")
    print("Ora puoi lanciare SimulatorGreedy_alignedToPPO2.py, e infine plot_results.py!")


if __name__ == "__main__":
    main()