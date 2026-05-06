from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from quantum_env import QuantumSatelliteEnv
import gymnasium as gym
import os
import numpy as np
import random


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.unwrapped.valid_action_mask()


class RewardComponentsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        rewards = self.locals.get("rewards")

        if infos is not None and len(infos) > 0:
            info = infos[0]

            if info.get("macro_step_complete", False):
                served = info.get('served', 0)
                pairs_generated = info.get('pairs_generated', 0)
                avg_fid = info.get('avg_fidelity', 0.0)
                expiration_penalty = info.get('expiration_penalty', 0.0)
                n_active = info.get('active_requests_count', 1)

                entanglement_ratio = (served / n_active) if n_active > 0 else 0.0
                entanglement_reward = 0.45 * entanglement_ratio

                self.logger.record("custom_reward/0_entanglement_reward_part", entanglement_reward)
                self.logger.record("custom_metrics/pairs_generated_per_macrostep", pairs_generated)
                self.logger.record("custom_reward/1_penalty_expired", expiration_penalty)
                self.logger.record("custom_reward/2_macro_step_reward", rewards[0])

                if served > 0:
                    self.logger.record("custom_reward/3_avg_fidelity_on_success", avg_fid)

        return True


def main():
    print("=== TRAINING MASKABLE PPO (MICRO-STEP DINAMICO) ===")
    random.seed(42)
    np.random.seed(42)

    log_dir = "./ppo_tensorboard/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    env = QuantumSatelliteEnv(csv_path="../data/simultaneous_visibility_3_stations_19-20_train.csv",
                              max_steps=500)
    env = ActionMasker(env, mask_fn)
    env = Monitor(env)

    model = MaskablePPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir,
                        device="cuda", seed=42, learning_rate=0.003, n_steps=2048,batch_size=256)

    reward_cb = RewardComponentsCallback()
    model.learn(total_timesteps=100000, tb_log_name="MaskablePPO_DynamicQueues", callback=reward_cb)
    model.save("models/ppo_trained_maskable")
    print("Modello salvato con successo.")


if __name__ == "__main__":
    main()