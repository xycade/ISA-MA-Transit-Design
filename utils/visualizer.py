import os
import numpy as np
from typing import List, Dict
import matplotlib

matplotlib.use('Agg')  # Headless mode for multiprocessing and server environments
import matplotlib.pyplot as plt
import seaborn as sns


class Visualizer:
    """
    Visualization engine designed to generate publication-ready plots
    for XAI and evolutionary performance metrics.
    """

    def __init__(self, output_dir: str):
        """
        Initializes the visualizer and configures global plotting styles.
        """
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Global publication-ready configurations
        plt.rcParams['axes.unicode_minus'] = False
        sns.set_theme(style="whitegrid", context="paper")

    def plot_fitness_evolution(self, fitness_history: List[float], filename: str = "fitness_evolution.png"):
        """
        Plots the convergence curve of the best fitness over generations.
        Uses a dual-view approach (Log Scale + Zoomed-in Linear Scale) to reveal late-stage optimization details.
        """
        if len(fitness_history) < 50:
            # Fallback for very short runs
            plt.figure(figsize=(10, 6))
            plt.plot(fitness_history, linestyle='-', linewidth=2.5, color='#2c3e50')
            plt.title("Fitness Evolution Curve", fontsize=18, fontweight='bold', pad=15)
            plt.yscale('log')  # Prevent massive initial drops from flattening the curve
            plt.xlabel("Generation", fontsize=14)
            plt.ylabel("Best Fitness (Log Scale)", fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, filename), dpi=400)
            plt.close()
            return

        # Publication-ready Dual-View Layout
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # --- View 1: Global Evolution (Log Scale) ---
        ax1.plot(fitness_history, linestyle='-', linewidth=2.5, color='#2c3e50')
        ax1.set_yscale('log')
        ax1.set_title("Global Evolution (Log Scale)", fontsize=16, fontweight='bold', pad=15)
        ax1.set_xlabel("Generation", fontsize=14)
        ax1.set_ylabel("Best Fitness", fontsize=14)
        ax1.tick_params(axis='both', which='major', labelsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)

        # --- View 2: Late-Stage Fine-Tuning (Linear Scale, Zoomed In) ---
        # Skip the first 10% of generations to bypass the initial cliff
        zoom_start = int(len(fitness_history) * 0.1)
        zoomed_history = fitness_history[zoom_start:]

        ax2.plot(range(zoom_start, len(fitness_history)), zoomed_history,
                 linestyle='-', linewidth=2.5, color='#27ae60')

        # Dynamically adjust y-limits to make subtle steps visible
        y_min, y_max = min(zoomed_history), max(zoomed_history)
        padding = (y_max - y_min) * 0.1 if y_max > y_min else 1.0
        ax2.set_ylim(y_min - padding, y_max + padding)

        ax2.set_title(f"Late-Stage Dynamics (Gen {zoom_start}-{len(fitness_history)})",
                      fontsize=16, fontweight='bold', pad=15)
        ax2.set_xlabel("Generation", fontsize=14)
        ax2.set_ylabel("Best Fitness (Linear)", fontsize=14)
        ax2.tick_params(axis='both', which='major', labelsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=500)
        plt.close()

    def plot_q_table_heatmap(self, q_table: np.ndarray, action_names: List[str], state_names: List[str],
                             filename: str = "xai_q_table_heatmap.png"):
        """
        Generates an Interpretable Q-Table Heatmap to audit the agent's decision logic.
        """
        plt.figure(figsize=(10, 8))

        # Use a high-contrast colormap for clear policy distinction
        heatmap = sns.heatmap(
            q_table,
            annot=True,
            fmt=".3f",
            cmap="YlGnBu",
            xticklabels=action_names,
            yticklabels=state_names,
            linewidths=1.0,
            linecolor='white',
            cbar_kws={"shrink": .8},
            annot_kws={"size": 13, "weight": "bold"}
        )

        heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=15, ha='right', fontsize=12)
        heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=12)

        plt.title("Interpretable XRL Policy: Q-Table Heatmap", fontsize=18, fontweight='bold', pad=20)
        plt.xlabel("Memetic Operators (Actions)", fontsize=14, labelpad=10)
        plt.ylabel("Population Dynamics (States)", fontsize=14, labelpad=10)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=500)
        plt.close()