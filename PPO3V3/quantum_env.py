import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import random
from physics_engine import PhysicsEngine, SystemConfig


class QuantumSatelliteEnv(gym.Env):
    def __init__(self, csv_path='../data/simultaneous_visibility_3_stations_19-20.csv',
                 alpha=SystemConfig.ALPHA, beta=SystemConfig.BETA, delta=SystemConfig.DELTA,
                 max_steps=1000, ttl_steps=SystemConfig.TTL_STEPS):
        super(QuantumSatelliteEnv, self).__init__()

        self.physics = PhysicsEngine()
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.max_steps = max_steps
        self.ttl_steps = ttl_steps
        self.step_size = pd.Timedelta(milliseconds=22)

        print(f"Caricamento dati in RAM da: {csv_path}...")
        self.df = pd.read_csv(csv_path)
        self.df['Start_Time_UTC'] = pd.to_datetime(self.df['Start_Time_UTC'])
        self.df['End_Time_UTC'] = pd.to_datetime(self.df['End_Time_UTC'])

        gs_set = set(self.df['GS_1']).union(set(self.df['GS_2'])).union(set(self.df['GS_3']))
        self.unique_gs = sorted(list(gs_set))
        self.gs_to_id = {gs: i for i, gs in enumerate(self.unique_gs)}
        self.all_gs_ids = list(self.gs_to_id.values())

        self.min_time = self.df['Start_Time_UTC'].min()
        self.max_time = self.df['End_Time_UTC'].max() - (self.step_size * self.max_steps)

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(59,), dtype=np.float32)

        self.current_step = 0
        self.current_time = self.min_time

        # Strutture dati per le code dinamiche
        self.sources = []
        self.destinations = []
        self.queues = {}
        self.queue_capacities = {}
        self.active_requests = []
        self.total_generated_counter = 0

        self.current_req_idx = 0
        self.locked_satellites = set()
        self._reset_macro_trackers()

    def _reset_macro_trackers(self):
        self._macro_served = 0
        self._macro_pairs = 0
        self._macro_fidelities = []
        self._requests_to_remove = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.current_step = 0
        self.locked_satellites = set()
        self.total_generated_counter = 0
        self._reset_macro_trackers()

        if options is not None and 'start_time' in options and 'end_time' in options:
            self.current_time = pd.to_datetime(options['start_time'])
            end_time = pd.to_datetime(options['end_time'])
            delta_seconds = (end_time - self.current_time).total_seconds()
            self.max_steps = int(delta_seconds / self.step_size.total_seconds())
        else:
            if self.min_time < self.max_time:
                random_offset = random.uniform(0, (self.max_time - self.min_time).total_seconds())
                self.current_time = self.min_time + pd.Timedelta(seconds=random_offset)
            else:
                self.current_time = self.min_time

        # Divisione topologica 3 Sender / 3 Receiver
        gs_list = self.all_gs_ids.copy()
        random.shuffle(gs_list)
        half = len(gs_list) // 2
        self.sources = gs_list[:half]
        self.destinations = gs_list[half:]

        # Setup capacità casuali delle code (da 1 a 5) SOLO per i Sender
        self.queue_capacities = {gs: random.randint(1, 5) for gs in self.sources}
        self.queues = {gs: [] for gs in self.sources}

        self._fill_queues()
        self._prepare_macro_step()

        return self._get_observation(), {}

    def valid_action_mask(self):
        mask = np.zeros(6, dtype=np.int8)
        if len(self.active_requests) == 0:
            mask[5] = 1  # Solo Wait se non ci sono richieste attive
            return mask

        req = self.active_requests[self.current_req_idx]
        sat_pool = self._get_satellite_pool()

        for i, sat in enumerate(sat_pool):
            if not str(sat['id']).startswith('DUMMY') and sat['id'] not in self.locked_satellites:
                if req['src'] in sat['valid_gs'] and req['dst'] in sat['valid_gs']:
                    mask[i] = 1

        mask[5] = 1
        return mask

    def step(self, action):
        sat_pool = self._get_satellite_pool()

        # Se ci sono richieste da servire nel micro-step
        if len(self.active_requests) > 0:
            req = self.active_requests[self.current_req_idx]

            if action < 5:
                sat = sat_pool[action]
                self.locked_satellites.add(sat['id'])
                req['pairs'] += 3
                self._macro_pairs += 3
                self._macro_fidelities.append(sat['fidelity'])

                if req['pairs'] >= 3:
                    self._macro_served += 1
                    self._requests_to_remove.append(self.current_req_idx)

            self.current_req_idx += 1

        # CONTROLLO SE IL MACRO-STEP E' FINITO
        if self.current_req_idx < len(self.active_requests):
            return self._get_observation(), 0.0, False, False, {}

        # --- FINE MACRO-STEP ---

        # A. Rimozione richieste servite (pop dalla testa della coda specifica)
        for idx in sorted(self._requests_to_remove, reverse=True):
            served_req = self.active_requests[idx]
            self.queues[served_req['src']].pop(0)

        # B. Invecchiamento (TTL) e Scadenze SOLO nelle code dei Sender
        expired_count = 0
        for s in self.sources:
            new_q = []
            for req in self.queues[s]:
                req['wait_time'] += 1
                if req['wait_time'] >= self.ttl_steps:
                    expired_count += 1
                else:
                    new_q.append(req)
            self.queues[s] = new_q

        # C. Refill delle code e reset puntatori
        self._fill_queues()
        self._prepare_macro_step()
        self.locked_satellites = set()

        # D. Avanzamento dell'Orologio Fisico
        self.current_time += self.step_size
        self.current_step += 1

        # --- MODIFICA PER LA GAE ---
        # Essendo una simulazione continua, finisce solo per limite di tempo (truncated).
        is_time_limit = self.current_step >= self.max_steps
        terminated = False  # Non c'è un vero "game over"
        truncated = is_time_limit

        # E. CALCOLO REWARD CON DENOMINATORE DINAMICO
        avg_fidelity = np.mean(self._macro_fidelities) if self._macro_fidelities else 0.0
        expiration_penalty = self.delta * expired_count

        n_active = len(self.active_requests)
        entanglement_ratio = (self._macro_served / n_active) if n_active > 0 else 0.0

        reward = (self.alpha * entanglement_ratio
                  + self.beta * avg_fidelity
                  - expiration_penalty)

        info = {
            "macro_step_complete": True,
            "served": self._macro_served,
            "pairs_generated": self._macro_pairs,
            "expired": expired_count,
            "avg_fidelity": avg_fidelity,
            "current_time": str(self.current_time),
            "expiration_penalty": expiration_penalty,
            "active_requests_count": n_active
        }

        self._reset_macro_trackers()

        return self._get_observation(), float(reward), terminated,truncated , info

    def _fill_queues(self):
        """Riempie le code fino alla loro capacità, pescando le dest dai receiver"""
        existing_directed = set()
        for q in self.queues.values():
            for req in q:
                existing_directed.add((req['src'], req['dst']))

        for s in self.sources:
            cap = self.queue_capacities[s]
            while len(self.queues[s]) < cap:
                # Scegli una destinazione che non sia già in coda per questo sender
                possible_dsts = [d for d in self.destinations if (s, d) not in existing_directed]
                if not possible_dsts:
                    break
                d = random.choice(possible_dsts)

                new_req = {'id': self.total_generated_counter, 'src': s, 'dst': d, 'wait_time': 0, 'pairs': 0}
                self.queues[s].append(new_req)
                self.total_generated_counter += 1
                existing_directed.add((s, d))

    def _prepare_macro_step(self):
        """Prepara le richieste in testa alle code dei Sender per essere valutate dall'agente"""
        self.active_requests = []
        for s in self.sources:
            if len(self.queues[s]) > 0:
                self.active_requests.append(self.queues[s][0])
        self.current_req_idx = 0

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
            s_a, eta_a = self.physics.get_link_metrics(dist, elev)
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
        unique_pool = []
        seen_ids = set()
        for sat in pool:
            if sat['id'] not in seen_ids:
                unique_pool.append(sat)
                seen_ids.add(sat['id'])
                if len(unique_pool) == 5: break

        top_5 = unique_pool
        while len(top_5) < 5:
            top_5.append({'id': f'DUMMY_{len(top_5)}', 'elev': 30.0, 'dist': 2000.0, 'fidelity': 0.0, 'valid_gs': []})
        return top_5

    def _get_observation(self):
        obs = []
        num_gs = len(self.all_gs_ids) if len(self.all_gs_ids) > 0 else 6

        if len(self.active_requests) == 0:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        req = self.active_requests[self.current_req_idx]

        src_vec = [0.0] * num_gs
        dst_vec = [0.0] * num_gs
        if req['src'] < num_gs: src_vec[req['src']] = 1.0
        if req['dst'] < num_gs: dst_vec[req['dst']] = 1.0

        obs.extend(src_vec)
        obs.extend(dst_vec)
        obs.append(min(req['wait_time'] / self.ttl_steps, 1.0))
        obs.append(min(req['pairs'] / 3.0, 1.0))

        sat_pool = self._get_satellite_pool()
        for sat in sat_pool:
            obs.append(sat['elev'] / 90.0)
            obs.append(sat['dist'] / 2000.0)
            obs.append(sat['fidelity'])
            cov_vec = [0.0] * num_gs
            for gs_id in sat['valid_gs']:
                if gs_id < num_gs: cov_vec[gs_id] = 1.0
            obs.extend(cov_vec)

        return np.array(obs, dtype=np.float32)