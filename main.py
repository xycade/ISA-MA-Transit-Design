import os
import argparse
import copy
from multiprocessing import freeze_support

# --- Import Internal Modules ---
from configs.default_config import CONFIG, NETWORK_CONFIGS
from utils.seed_manager import seed_everything
from utils.visualizer import Visualizer
from envs.network_env import TransitNetwork
from envs.passenger_sim import PassengerSimulator
from core.ga_engine import GAEngine


def main():
    """
    Main entry point for the ISA-MA optimization framework.
    Orchestrates the environment setup, evolutionary execution, and result visualization.
    """
    # 1. Parse Command Line Arguments
    parser = argparse.ArgumentParser(description="ISA-MA Framework for Transit Network Design")
    parser.add_argument('--network', type=str, default='mandl', choices=['mandl', 'mumford0', 'mumford1'],
                        help='Benchmark network to optimize (default: mandl)')
    parser.add_argument('--mode', type=str, default='ISA-MA', choices=['GA', 'MA-Random', 'MA-Fixed', 'ISA-MA'],
                        help='Optimization mode (default: ISA-MA)')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save visualization results (default: ./output)')
    args = parser.parse_args()

    # 2. Merge Network-Specific Configs into the Global Configuration
    net_config = NETWORK_CONFIGS[args.network]
    global_config = copy.deepcopy(CONFIG)

    global_config['BASE_PATH'] = net_config['path']
    global_config['NUM_ROUTES'] = net_config['num_routes']
    global_config['MAX_TOTAL_FLEET'] = net_config['max_fleet']
    global_config['MAX_STOPS_PER_ROUTE'] = net_config['max_stops']
    global_config['MIN_STOPS_PER_ROUTE'] = net_config['min_stops']

    # 3. Lock Global Seed for Reproducibility
    seed_everything(global_config['GLOBAL_SEED'])

    output_path = os.path.join(args.output_dir, args.network, args.mode)
    print("=" * 60)
    print(f"= ISA-MA Project Execution Started")
    print(f"= Network: {args.network.upper()} | Mode: {args.mode}")
    print(f"= Global Seed: {global_config['GLOBAL_SEED']}")
    print("=" * 60 + "\n")

    try:
        # 4. Initialize Environment and Simulator
        network_env = TransitNetwork(base_path=global_config['BASE_PATH'])
        simulator = PassengerSimulator(network_env=network_env, config=global_config)

        # 5. Initialize and Run the Evolutionary Engine
        engine = GAEngine(config=global_config, network_env=network_env, simulator=simulator, mode=args.mode)

        # [修复点] 解包接收全部 6 个返回值
        best_individual, fitness_history, operator_history, q_snapshots, timings, actual_gens = engine.run()

        # 6. Terminal Reporting & Visualization
        if best_individual:
            print("\n" + "=" * 60)
            print("🎯 FINAL OPTIMIZATION REPORT")
            print("=" * 60)

            # Formally calculate and assign the final demand-responsive frequencies
            best_individual['frequencies'] = simulator.calculate_demand_responsive_frequencies(best_individual)

            # Recalculate detailed metrics for the best solution
            precomputed = simulator._prepare_network_data(best_individual)
            metrics = simulator.calculate_performance_metrics(precomputed)

            total_demand = network_env.total_demand
            unmet = metrics['unmet_passengers']
            served = total_demand - unmet

            # Print Global Metrics
            print(f"➤ Final Fitness Score:       {metrics['fitness']:,.2f}")
            print(
                f"➤ Converged Generation:      {actual_gens} / {global_config.get('MAX_GENERATIONS', 1000)} (Patience: {global_config.get('PATIENCE', 150)})")
            print(f"➤ Total Passenger Time Cost: {metrics['total_passenger_time_cost']:,.2f} min")
            print(f"➤ Demand Satisfaction Rate:  {served} / {total_demand} ({(served / total_demand) * 100:.2f}%)")

            # Print Transfer Analysis
            p0 = metrics['passengers_0_transfer']
            p1 = metrics['passengers_1_transfer']
            p2 = metrics['passengers_2_transfer']
            print("\n📊 Passenger Transfer Breakdown:")
            print(f"  - 0 Transfers (Direct):    {p0} pax ({(p0 / total_demand) * 100:.2f}%)")
            print(f"  - 1 Transfer:              {p1} pax ({(p1 / total_demand) * 100:.2f}%)")
            print(f"  - 2 Transfers:             {p2} pax ({(p2 / total_demand) * 100:.2f}%)")

            # Print Network Topology (Routes & Frequencies)
            print("\n🚌 Optimized Bus Routes Topology:")
            for i, route in enumerate(best_individual['routes']):
                freq = best_individual['frequencies'][i]
                route_str = " -> ".join(map(str, route))
                print(f"  [Route {i + 1}] (Freq: {freq} buses/hr): {route_str}")

            # Print Computational Overhead
            print("\n⏱️ Computational Overhead:")
            print(f"  - Evaluation Time:  {timings['eval_time']:,.2f} s")
            print(f"  - Memetic Ops Time: {timings['op_time']:,.2f} s")
            if args.mode == 'ISA-MA':
                print(f"  - RL Agent Time:    {timings['rl_time']:,.2f} s")
            print("=" * 60 + "\n")

            print(f"[Main] Generating visualizations in: {output_path}")
            viz = Visualizer(output_dir=output_path)

            # Plot the Dual-View Fitness Evolution Curve
            viz.plot_fitness_evolution(fitness_history, filename=f"fitness_{args.network}_{args.mode}.png")

            if args.mode == 'ISA-MA':
                action_names = ['Action 0: Exploiter', 'Action 1: Recombiner', 'Action 2: Reformer',
                                'Action 3: Explorer']
                state_names = ['Stagnant & Div-H', 'Stagnant & Div-L', 'Converged & Div-H', 'Converged & Div-L']

                # Extract and plot the final Q-Table heatmap based on the actual stopped generation
                final_gen_key = max(q_snapshots.keys()) if q_snapshots else -1
                if final_gen_key != -1:
                    final_q_table = q_snapshots[final_gen_key]
                    viz.plot_q_table_heatmap(final_q_table, action_names, state_names,
                                             filename=f"q_table_{args.network}.png")

            print(f"[Main] All tasks completed successfully.")
        else:
            print("[Main] Warning: Evolution failed to find a valid solution.")

    except Exception as e:
        import traceback
        print(f"\n[Error] Execution aborted due to: {str(e)}")
        traceback.print_exc()


if __name__ == '__main__':
    freeze_support()
    main()