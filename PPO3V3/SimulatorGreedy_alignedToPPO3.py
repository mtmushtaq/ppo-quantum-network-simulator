import numpy as np
import pandas as pd
import random
import json
from physics_engine import PhysicsEngine, SystemConfig


class SimulatorGreedyAligned:
    def __init__(self, csv_path, ttl_steps=SystemConfig.TTL_STEPS):
        self.physics = PhysicsEngine()
        self.step_size = pd.Timedelta(milliseconds=22)
        self.ttl_steps = ttl_steps

        print(f"Caricamento dati in RAM da: {csv_path}...")
        self.df = pd.read_csv(csv_path)
        self.df['Start_Time_UTC'] = pd.to_datetime(self.df['Start_Time_UTC'])
        self.df['End_Time_UTC'] = pd.to_datetime(self.df['End_Time_UTC'])

        gs_set = set(self.df['GS_1']).union(set(self.df['GS_2'])).union(set(self.df['GS_3']))
        self.unique_gs = sorted(list(gs_set))
        self.gs_to_id = {gs: i for i, gs in enumerate(self.unique_gs)}
        self.all_gs_ids = list(self.gs_to_id.values())

        self.sources = []
        self.destinations = []
        self.queues = {}
        self.queue_capacities = {}
        self.active_requests = []
        self.total_generated_counter = 0

    def reset(self, start_time, end_time, seed=42):
        random.seed(seed)
        np.random.seed(seed)

        self.current_time = pd.to_datetime(start_time)
        end_time_dt = pd.to_datetime(end_time)
        delta_seconds = (end_time_dt - self.current_time).total_seconds()
        self.max_steps = int(delta_seconds / self.step_size.total_seconds())
        self.current_step = 0
        self.total_generated_counter = 0

        # Divisione topologica 3 Sender / 3 Receiver
        gs_list = self.all_gs_ids.copy()
        random.shuffle(gs_list)
        half = len(gs_list) // 2
        self.sources = gs_list[:half]
        self.destinations = gs_list[half:]

        # Code solo per i Sender
        self.queue_capacities = {gs: random.randint(1, 5) for gs in self.sources}
        self.queues = {gs: [] for gs in self.sources}

        self._fill_queues()
        self._prepare_macro_step()

    def _fill_queues(self):
        existing_directed = set()
        for q in self.queues.values():
            for req in q:
                existing_directed.add((req['src'], req['dst']))

        for s in self.sources:
            cap = self.queue_capacities[s]
            while len(self.queues[s]) < cap:
                possible_dsts = [d for d in self.destinations if (s, d) not in existing_directed]
                if not possible_dsts:
                    break
                d = random.choice(possible_dsts)

                new_req = {'id': self.total_generated_counter, 'src': s, 'dst': d, 'wait_time': 0, 'pairs': 0}
                self.queues[s].append(new_req)
                self.total_generated_counter += 1
                existing_directed.add((s, d))

    def _prepare_macro_step(self):
        self.active_requests = []
        for s in self.sources:
            if len(self.queues[s]) > 0:
                self.active_requests.append(self.queues[s][0])

    def _get_satellite_pool(self):
        mask = (self.df['Start_Time_UTC'] <= self.current_time) & (self.df['End_Time_UTC'] >= self.current_time)
        active_sats = self.df[mask]

        pool = []
        for _, row in active_sats.iterrows():
            duration = (row['End_Time_UTC'] - row['Start_Time_UTC']).total_seconds()
            if duration <= 0: continue

            elapsed = (self.current_time - row['Start_Time_UTC']).total_seconds()
            frac = elapsed / duration
            angles = []
            distances = []

            for i in [1, 2, 3]:
                start_a = row[f'Start_angle_GS_{i}']
                end_a = row[f'End_angle_GS_{i}']
                angles.append(start_a + frac * (end_a - start_a))

                start_d = row[f'Start_Distance_to_{i}']
                end_d = row[f'End_Distance_to_{i}']
                distances.append(start_d + frac * (end_d - start_d))

            elev = min(angles)
            dist = max(distances)

            _, eta_a = self.physics.get_link_metrics(dist, elev)
            f_link = self.physics.calculate_fidelity(eta_a)
            f_raw = f_link * f_link

            f_final = f_raw
            rounds = 0
            while f_final < SystemConfig.F_THRESHOLD and rounds < 3:
                f_final = self.physics.purify(f_final, f_raw, 1)
                rounds += 1

            valid_gs = [self.gs_to_id[row['GS_1']], self.gs_to_id[row['GS_2']], self.gs_to_id[row['GS_3']]]
            pool.append({'id': row['Name_sat'], 'elev': max(30.0, elev), 'dist': max(500.0, dist), 'fidelity': f_final,
                         'valid_gs': valid_gs})

        pool.sort(key=lambda x: x['fidelity'], reverse=True)
        return pool

    def step_greedy(self):
        locked_satellites = set()
        completed_requests_this_step = 0
        pairs_generated_this_step = 0
        serving_fidelities = []
        requests_to_remove = []

        sat_pool = self._get_satellite_pool()
        n_active = len(self.active_requests)

        for i, req in enumerate(self.active_requests):
            best_sat_id = None
            best_fid = 0.0

            for sat in sat_pool:
                if sat['id'] not in locked_satellites:
                    if req['src'] in sat['valid_gs'] and req['dst'] in sat['valid_gs']:
                        best_sat_id = sat['id']
                        best_fid = sat['fidelity']
                        break

            if best_sat_id is not None and not str(best_sat_id).startswith(
                    'DUMMY') and best_fid >= SystemConfig.F_THRESHOLD:
                locked_satellites.add(best_sat_id)
                req['pairs'] += 3
                pairs_generated_this_step += 3
                serving_fidelities.append(best_fid)

                if req['pairs'] >= 3:
                    completed_requests_this_step += 1
                    requests_to_remove.append(i)

        for i in sorted(requests_to_remove, reverse=True):
            served_req = self.active_requests[i]
            self.queues[served_req['src']].pop(0)

        expired_this_step = 0
        for s in self.sources:
            new_q = []
            for req in self.queues[s]:
                req['wait_time'] += 1
                if req['wait_time'] >= self.ttl_steps:
                    expired_this_step += 1
                else:
                    new_q.append(req)
            self.queues[s] = new_q

        self._fill_queues()
        self._prepare_macro_step()

        self.current_time += self.step_size
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        avg_fid = np.mean(serving_fidelities) if serving_fidelities else 0.0

        return completed_requests_this_step, pairs_generated_this_step, expired_this_step, avg_fid, terminated, n_active


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    print("=== TEST DELLA SIMULAZIONE GREEDY ALLINEATA AL PPO ===")

    CSV_PATH = "../data/simultaneous_visibility_3_stations_20-21_test.csv"
    TEST_START = "2026-02-19 20:05:00"
    TEST_END = "2026-02-19 20:17:00"

    sim = SimulatorGreedyAligned(csv_path=CSV_PATH)
    sim.reset(start_time=TEST_START, end_time=TEST_END, seed=42)

    total_served = 0
    fidelities = []
    greedy_rewards = []
    pairs_generated_history = []
    total_requests_history = []
    served_history = []
    expired_history = []
    avg_fidelity_history = []

    print("-" * 50)

    for step in range(sim.max_steps):
        served, pairs_generated, expired_this_step, avg_fid, terminated, n_active = sim.step_greedy()

        total_served += served
        if pairs_generated > 0:
            fidelities.append(avg_fid)

        pairs_generated_history.append(pairs_generated)
        served_history.append(served)
        expired_history.append(expired_this_step)
        avg_fidelity_history.append(avg_fid)

        expiration_penalty = SystemConfig.DELTA * expired_this_step
        entanglement_ratio = (served / n_active) if n_active > 0 else 0.0

        step_reward = (SystemConfig.ALPHA * entanglement_ratio
                       + SystemConfig.BETA * avg_fid
                       - expiration_penalty)

        greedy_rewards.append(step_reward)
        total_requests_history.append(sim.total_generated_counter)

        if step % 45 == 0:
            print(f"Step {step:05d}/{sim.max_steps} | "
                  f"Servite: {served}/{n_active} attive | "
                  f"Tempo: {str(sim.current_time).split()[-1][:12]}")

        if terminated: break

    print("-" * 50)
    print("=== RISULTATI FINALI SIMULAZIONE GREEDY ===")
    print(f"Richieste Totali Generate: {sim.total_generated_counter}")
    print(f"Richieste Servite con Successo: {total_served}")
    print(f"Richieste Scadute (Timeout): {sum(expired_history)}")

    success_rate = (total_served / sim.total_generated_counter) * 100 if sim.total_generated_counter > 0 else 0.0
    print(f"Success Rate Globale: {success_rate:.2f}%")

    export_data = {
        "rewards": greedy_rewards,
        "pairs_generated": pairs_generated_history,
        "total_requests": total_requests_history,
        "served": served_history,
        "expired": expired_history,
        "avg_fidelity": avg_fidelity_history
    }
    with open("results_greedy.json", "w") as f:
        json.dump(export_data, f)
    print("\n[OK] Dati storici Greedy salvati in 'results_greedy.json'.")