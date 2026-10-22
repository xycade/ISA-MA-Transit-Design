import os
import sys
import time
import json
import pandas as pd
import numpy as np
from multiprocessing import freeze_support

# Ensure the parent directory is in the system path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.default_config import CONFIG, NETWORK_CONFIGS
from utils.seed_manager import seed_everything
from envs.network_env import TransitNetwork
from envs.passenger_sim import PassengerSimulator
from core.ga_engine import GAEngine


# [Define a custom JSON encoder for NumPy]
class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def run_batch_experiment():
    networks = ['mandl', 'mumford0', 'mumford1']
    modes = ['GA', 'MA-Random', 'MA-Fixed', 'ISA-MA']
    num_seeds = 30
    max_gens = 1000
    patience = 150

    global_config = CONFIG.copy()
    global_config['MAX_GENERATIONS'] = max_gens
    global_config['PATIENCE'] = patience

    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output', 'exp1_2_data'))
    xai_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output', 'exp4_6_xai_data'))
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(xai_dir, exist_ok=True)

    perf_csv = os.path.join(output_dir, "Table_Performance_Summary_Raw.csv")
    time_csv = os.path.join(output_dir, "Table_Computational_Overhead_Raw.csv")
    traj_csv = os.path.join(output_dir, "Fitness_Trajectory_Raw.csv")

    # Initialize Headers
    for path, header in [
        (perf_csv, "Network,Mode,Seed,Final_Fitness,Unmet_Demand,Total_Travel_Time,Direct_Rate,Converged_Generation\n"),
        (time_csv, "Network,Mode,Seed,Total_Time_Sec,Eval_Time_Sec,Operator_Time_Sec,RL_Time_Sec\n"),
        (traj_csv, "Network,Mode,Seed," + ",".join([f"Gen_{i}" for i in range(max_gens)]) + "\n")
    ]:
        if not os.path.exists(path):
            with open(path, 'w') as f: f.write(header)

    print("=" * 70)
    print(f"🚀 BATCH EXP 1 & 2 LAUNCHED | MAX GENS: {max_gens} | PATIENCE: {patience} | SEEDS: {num_seeds}")
    print("=" * 70)

    for network in networks:
        net_config = NETWORK_CONFIGS[network]
        global_config.update({
            'BASE_PATH': net_config['path'], 'NUM_ROUTES': net_config['num_routes'],
            'MAX_TOTAL_FLEET': net_config['max_fleet'], 'MAX_STOPS_PER_ROUTE': net_config['max_stops'],
            'MIN_STOPS_PER_ROUTE': net_config['min_stops']
        })

        print(f"\n[Batch Runner] Instantiating Environment for {network.upper()}...")
        network_env = TransitNetwork(base_path=global_config['BASE_PATH'])
        total_demand = network_env.total_demand

        for mode in modes:
            for seed in range(1, num_seeds + 1):
                print(f"  >>> Running [ {network.upper()} | {mode} | Seed: {seed}/30 ]")

                current_seed = 42 + seed
                global_config['GLOBAL_SEED'] = current_seed
                seed_everything(current_seed)

                simulator = PassengerSimulator(network_env=network_env, config=global_config)
                engine = GAEngine(config=global_config, network_env=network_env, simulator=simulator, mode=mode)

                t_total_start = time.perf_counter()

                best_ind, fit_hist, op_hist, q_snaps, timings, actual_gens = engine.run()

                total_time = time.perf_counter() - t_total_start

                # Metric Calculation
                final_fitness, unmet, travel_time, direct_rate = -1, -1, -1, -1
                if best_ind:
                    best_ind['frequencies'] = simulator.calculate_demand_responsive_frequencies(best_ind)

                    metrics = simulator.calculate_performance_metrics(simulator._prepare_network_data(best_ind))
                    final_fitness = metrics['fitness']
                    unmet = metrics['unmet_passengers']
                    travel_time = metrics['total_passenger_time_cost']
                    direct_rate = metrics['passengers_0_transfer'] / total_demand

                # Data Persistence
                with open(perf_csv, 'a') as f:
                    f.write(
                        f"{network},{mode},{current_seed},{final_fitness},{unmet},{travel_time},{direct_rate:.4f},{actual_gens}\n")

                with open(time_csv, 'a') as f:
                    f.write(
                        f"{network},{mode},{current_seed},{total_time:.2f},{timings['eval_time']:.2f},{timings['op_time']:.2f},{timings['rl_time']:.2f}\n")

                # Dynamic Padding
                if len(fit_hist) < max_gens:
                    padding_length = max_gens - len(fit_hist)
                    fit_hist.extend([fit_hist[-1]] * padding_length)

                with open(traj_csv, 'a') as f:
                    traj_str = ",".join([f"{val:.2f}" for val in fit_hist[:max_gens]])
                    f.write(f"{network},{mode},{current_seed},{traj_str}\n")

                # JSON Archiving (Using the custom NumpyEncoder)
                if best_ind:
                    route_log_path = os.path.join(output_dir, f"Routes_{network}_{mode}_Seed_{current_seed}.json")
                    with open(route_log_path, 'w') as jf:
                        json.dump(best_ind, jf, cls=NumpyEncoder)  # [🚨 关键修改点 🚨]

                # XAI Data Capture
                if mode == 'ISA-MA' and q_snaps:
                    final_gen_key = max(q_snaps.keys()) if q_snaps else 0
                    final_q = q_snaps.get(final_gen_key, np.zeros((4, 4)))
                    np.save(os.path.join(xai_dir, f"QTable_{network}_Seed_{current_seed}.npy"), final_q)
                    pd.DataFrame(op_hist).to_csv(os.path.join(xai_dir, f"OpHistory_{network}_Seed_{current_seed}.csv"),
                                                 index=False)

                print(
                    f"      -> OK! Fit: {final_fitness:,.0f} | Converged Gen: {actual_gens} | Time: {total_time:.1f}s")

    print("\n✅ EXP 1 & 2 EXECUTION COMPLETE. DATA SECURED.")


if __name__ == '__main__':
    freeze_support()
    run_batch_experiment()