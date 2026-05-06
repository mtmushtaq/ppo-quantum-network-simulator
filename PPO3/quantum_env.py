import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import random
from physics_engine import PhysicsEngine, SystemConfig


class QuantumSatelliteEnv(gym.Env):
    """
    Ambiente Custom Maskable PPO per Quantum Network Satellitare.
    Architettura a Micro-Step: valuta UNA richiesta alla volta.
    """

    def __init__(self,
                 csv_path='../data/simultaneous_visibility_3_stations_19-20.csv',
                 alpha=SystemConfig.ALPHA,
                 beta=SystemConfig.BETA,
                 delta=SystemConfig.DELTA,
                 max_steps=1000,
                 ttl_steps=SystemConfig.TTL_STEPS):

        super(QuantumSatelliteEnv, self).__init__()

        self.physics = PhysicsEngine()
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.max_steps = max_steps
        self.ttl_steps = ttl_steps
        self.step_size = pd.Timedelta(milliseconds=22)

        # --- CARICAMENTO DATI ---
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

        # --- AZIONI E OSSERVAZIONI ---
        # Azioni: 0,1,2,3,4 (Seleziona Satellite i) | 5 (Wait/Defer)
        self.action_space = spaces.Discrete(6)

        # OSSERVAZIONE (Singola Richiesta!):
        # 1 Richiesta x (6 bit Src + 6 bit Dst + 1 Wait Time + 1 Pairs Count) = 14
        # 5 Satelliti x (1 elev + 1 dist + 1 fidelity + 6 bit Copertura) = 45
        # Totale = 59 dimensioni
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(59,), dtype=np.float32)

        self.current_step = 0
        self.current_time = self.min_time
        self.queue = []
        self.sources = []
        self.destinations = []

        # Variabili di gestione Micro-Step
        self.current_req_idx = 0
        self.locked_satellites = set()
        self._reset_macro_trackers()

    def _reset_macro_trackers(self):
        """Resetta i contatori all'inizio di ogni tick fisico di 22ms"""
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
        self.current_req_idx = 0
        self.locked_satellites = set()
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

        gs_list = self.all_gs_ids.copy()
        random.shuffle(gs_list)
        half = len(gs_list) // 2
        self.sources = gs_list[:half]
        self.destinations = gs_list[half:]

        self.queue = []
        self._fill_queue()

        return self._get_observation(), {}

    def valid_action_mask(self):
        """
        ACTION MASKING: Genera un array booleano [M0, M1, M2, M3, M4, MWait].
        True = Azione valida. False = Azione bloccata (l'agente non può sceglierla).
        """
        mask = np.zeros(6, dtype=np.int8)
        req = self.queue[self.current_req_idx]
        sat_pool = self._get_satellite_pool()

        for i, sat in enumerate(sat_pool):
            # Se non è un DUMMY, non è già bloccato e ha visibilità simultanea
            if not str(sat['id']).startswith('DUMMY'):
                if sat['id'] not in self.locked_satellites:
                    if req['src'] in sat['valid_gs'] and req['dst'] in sat['valid_gs']:
                        mask[i] = 1

        # L'azione 5 (Wait) è SEMPRE valida
        mask[5] = 1
        return mask

    def step(self, action):
        sat_pool = self._get_satellite_pool()
        req = self.queue[self.current_req_idx]

        # 1. ESECUZIONE AZIONE MICRO-STEP
        if action < 5:  # L'agente ha scelto un satellite specifico
            sat = sat_pool[action]
            # Sappiamo che è valido grazie all'Action Masking
            self.locked_satellites.add(sat['id'])
            req['pairs'] += 3  #PRIMA 1
            self._macro_pairs += 3 #PRIMA 1
            self._macro_fidelities.append(sat['fidelity'])

            if req['pairs'] >= 3:
                self._macro_served += 1
                self._requests_to_remove.append(self.current_req_idx)

        # Passiamo alla prossima richiesta
        self.current_req_idx += 1

        # 2. CONTROLLO SE IL MACRO-STEP (22ms) E' FINITO
        if self.current_req_idx < 3:
            # Siamo ancora dentro lo stesso 22ms. Non diamo reward parziali.
            return self._get_observation(), 0.0, False, False, {}

        # --- FINE MACRO-STEP (Tutte e 3 le richieste sono state valutate) ---

        # A. Rimozione richieste servite
        for idx in sorted(self._requests_to_remove, reverse=True):
            self.queue.pop(idx)

        # B. Invecchiamento (TTL) e Scadenze
        expired_count = 0
        expired_indices = []
        for i, r in enumerate(self.queue):
            r['wait_time'] += 1
            if r['wait_time'] >= self.ttl_steps:
                expired_indices.append(i)
                expired_count += 1

        for i in sorted(expired_indices, reverse=True):
            self.queue.pop(i)

        # C. Refill della coda e Reset puntatori
        self._fill_queue()
        self.current_req_idx = 0
        self.locked_satellites = set()

        # D. Avanzamento dell'Orologio Fisico
        self.current_time += self.step_size
        self.current_step += 1
        terminated = self.current_step >= self.max_steps

        # E. CALCOLO REWARD NORMALIZZATA (Senza doppia penalità)
        avg_fidelity = np.mean(self._macro_fidelities) if self._macro_fidelities else 0.0
        expiration_penalty = self.delta * expired_count

        reward = (self.alpha * (self._macro_served / 3.0)
                  + self.beta * avg_fidelity
                  - expiration_penalty)

        # Info finali da passare al logger
        info = {
            "macro_step_complete": True,  # Flag utile per i test
            "served": self._macro_served,
            "pairs_generated": self._macro_pairs,
            "expired": expired_count,
            "avg_fidelity": avg_fidelity,
            "current_time": str(self.current_time),
            "expiration_penalty": expiration_penalty
        }

        self._reset_macro_trackers()

        return self._get_observation(), float(reward), terminated, False, info

    def _fill_queue(self):
        used_dsts = [req['dst'] for req in self.queue]
        active_sats = self._get_satellite_pool()
        available_pairs = []

        for sat in active_sats:
            if str(sat['id']).startswith('DUMMY'):
                continue
            sat_sources = [gs for gs in sat['valid_gs'] if gs in self.sources]
            sat_destinations = [gs for gs in sat['valid_gs'] if gs in self.destinations]

            for s in sat_sources:
                for d in sat_destinations:
                    if d not in used_dsts:
                        available_pairs.append((s, d))

        while len(self.queue) < 3:
            if available_pairs:
                s, d = random.choice(available_pairs)
                self.queue.append({'src': s, 'dst': d, 'wait_time': 0, 'pairs': 0})
                used_dsts.append(d)
                available_pairs = [p for p in available_pairs if p[1] != d]
            else:
                available_srcs = self.sources
                available_dsts = [d for d in self.destinations if d not in used_dsts]

                if not available_srcs or not available_dsts:
                    break

                s = random.choice(available_srcs)
                d = random.choice(available_dsts)
                self.queue.append({'src': s, 'dst': d, 'wait_time': 0, 'pairs': 0})
                used_dsts.append(d)

    def _get_satellite_pool(self):
        # Questo metodo rimane identico alla tua versione: estrae i 5 satelliti
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

            valid_gs = [
                self.gs_to_id[row['GS_1']],
                self.gs_to_id[row['GS_2']],
                self.gs_to_id[row['GS_3']]
            ]

            pool.append({
                'id': row['Name_sat'],
                'elev': max(30.0, elev),
                'dist': max(500.0, dist),
                'fidelity': f_final,
                'valid_gs': valid_gs
            })

        pool.sort(key=lambda x: x['fidelity'], reverse=True)

        unique_pool = []
        seen_ids = set()
        for sat in pool:
            if sat['id'] not in seen_ids:
                unique_pool.append(sat)
                seen_ids.add(sat['id'])
                if len(unique_pool) == 5:
                    break

        top_5 = unique_pool

        while len(top_5) < 5:
            top_5.append({
                'id': f'DUMMY_{len(top_5)}',
                'elev': 30.0,
                'dist': 2000.0,
                'fidelity': 0.0,
                'valid_gs': []
            })

        return top_5

    def _get_observation(self):
        """Costruisce lo stato per UNA SOLA RICHIESTA (quella corrente)"""
        obs = []
        num_gs = len(self.all_gs_ids) if len(self.all_gs_ids) > 0 else 6

        # Dati della richiesta che stiamo valutando ORA
        req = self.queue[self.current_req_idx]

        src_vec = [0.0] * num_gs
        dst_vec = [0.0] * num_gs

        if req['src'] < num_gs: src_vec[req['src']] = 1.0
        if req['dst'] < num_gs: dst_vec[req['dst']] = 1.0

        obs.extend(src_vec)
        obs.extend(dst_vec)
        obs.append(min(req['wait_time'] / self.ttl_steps, 1.0))
        obs.append(min(req['pairs'] / 3.0, 1.0))

        # Dati della costellazione
        sat_pool = self._get_satellite_pool()
        for sat in sat_pool:
            obs.append(sat['elev'] / 90.0)
            obs.append(sat['dist'] / 2000.0)
            obs.append(sat['fidelity'])

            cov_vec = [0.0] * num_gs
            for gs_id in sat['valid_gs']:
                if gs_id < num_gs:
                    cov_vec[gs_id] = 1.0
            obs.extend(cov_vec)

        return np.array(obs, dtype=np.float32)