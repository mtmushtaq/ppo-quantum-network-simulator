from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from quantum_env import QuantumSatelliteEnv
import gymnasium as gym
import numpy as np
import random
import os
import json

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.unwrapped.valid_action_mask()


def main():
    print("=== TEST DELLA SIMULAZIONE MASKABLE PPO (CODE DINAMICHE) ===")
    random.seed(42)
    np.random.seed(42)

    env = QuantumSatelliteEnv(csv_path="../data/simultaneous_visibility_3_stations_20-21_test.csv")
    env = ActionMasker(env, mask_fn)

    test_start = "2026-02-19 20:05:00"
    test_end = "2026-02-19 20:17:00"

    model_path = "models/ppo_trained_maskable"
    try:
        model = MaskablePPO.load(model_path)
    except Exception:
        print(f"Errore: Nessun modello in {model_path}. Avvia prima train_ppo.py!")
        return

    obs, info = env.reset(seed=42, options={'start_time': test_start, 'end_time': test_end})

    total_served = 0
    fidelities = []
    ppo_rewards = []
    pairs_generated_history = []
    total_requests_history = []
    served_history = []
    expired_history = []
    avg_fidelity_history = []

    terminated = False
    truncated = False
    physical_steps_passed = 0

    # Ora il ciclo si ferma se l'episodio è terminato O se è andato in timeout (truncated)
    while not (terminated or truncated):
        action_masks = env.action_masks()
        action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if info.get("macro_step_complete", False):
            physical_steps_passed += 1
            served_this_step = info.get('served', 0)
            n_active = info.get('active_requests_count', 1)
            total_served += served_this_step
            ppo_rewards.append(reward)

            avg_fidelity_history.append(info.get('avg_fidelity', 0.0))
            if served_this_step > 0:
                fidelities.append(info['avg_fidelity'])

            pairs_generated_history.append(info.get('pairs_generated', 0))
            served_history.append(served_this_step)
            expired_history.append(info.get('expired', 0))

            tot_gen = env.unwrapped.total_generated_counter
            total_requests_history.append(tot_gen)

            if physical_steps_passed % 45 == 0:
                print(f"Macro-Step {physical_steps_passed:05d}/{env.unwrapped.max_steps} | "
                      f"Servite: {served_this_step}/{n_active} attive | "
                      f"Tempo: {info['current_time'].split()[-1][:12]}")

    print("-" * 50)
    print("=== RISULTATI FINALI SIMULAZIONE PPO ===")
    tot_gen_final = env.unwrapped.total_generated_counter
    print(f"Richieste Totali Generate: {tot_gen_final}")
    print(f"Richieste Servite: {total_served}")
    print(f"Richieste Scadute (Timeout): {sum(expired_history)}")

    success_rate = (total_served / tot_gen_final) * 100 if tot_gen_final > 0 else 0.0
    print(f"Success Rate Globale: {success_rate:.2f}%")

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


if __name__ == "__main__":
    main()