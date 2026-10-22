import os
import random
import copy
import time
import numpy as np
from typing import List, Tuple, Dict
from functools import partial
from multiprocessing import Pool, Manager
from tqdm import tqdm

from operators.memetic_tools import MemeticOperators
from core.isa_agent import ISAAgent


def worker_init_fn(base_seed):
    np.random.seed(base_seed)
    random.seed(base_seed)


def _parallel_eval_wrapper(individual, simulator_instance, shared_cache):
    return simulator_instance.evaluate_fitness(individual, shared_cache)


class GAEngine:
    """
    Multi-mode Evolutionary Engine integrating GA, MA-Random, MA-Fixed, and ISA-MA.
    """

    def __init__(self, config: dict, network_env, simulator, mode: str = 'ISA-MA'):
        self.config = config
        self.net = network_env
        self.sim = simulator
        self.mode = mode

        self.pop_size = config.get('POP_SIZE', 30)
        self.max_generations = config.get('MAX_GENERATIONS', 1000)
        self.patience = config.get('PATIENCE', 150)

        self.elitism_rate = config.get('ELITISM_RATE', 0.1)
        self.crossover_rate = config.get('CROSSOVER_RATE', 0.1)
        self.mutation_rate = config.get('MUTATION_RATE', 0.1)
        self.base_seed = config.get('GLOBAL_SEED', 42)

        self.memetic_ops = MemeticOperators(
            config=config, sp=self.net.sp, sp_len=self.net.sp_len,
            stop_nodes_set=self.net.stop_nodes_set, od_df=self.net.od_df
        )

        self.agent = ISAAgent(config=config) if self.mode == 'ISA-MA' else None

        self.fitness_history = []
        self.operator_history = []
        self.q_snapshots = {}
        self.timings = {'eval_time': 0.0, 'op_time': 0.0, 'rl_time': 0.0}

    def _initialize_population(self) -> List[dict]:
        population = []
        num_routes = self.config.get('NUM_ROUTES', 4)
        for _ in range(self.pop_size):
            individual = {'routes': [self.memetic_ops._generate_valid_route() for _ in range(num_routes)]}
            population.append(self.memetic_ops.repair_individual(individual))
        return population

    def _calculate_diversity(self, population: List[dict]) -> float:
        unique_edges = set()
        total_edges = 0
        for ind in population:
            for r in ind.get('routes', []):
                if not r: continue
                edges = [(r[i], r[i + 1]) for i in range(len(r) - 1)]
                unique_edges.update(edges)
                total_edges += len(edges)
        return len(unique_edges) / (total_edges + 1e-6)

    def _select(self, population: List[dict], fitnesses: List[float], k: int = 3) -> dict:
        selected_indices = random.sample(range(len(population)), k)
        best_idx = min(selected_indices, key=lambda idx: fitnesses[idx])
        return copy.deepcopy(population[best_idx])

    def _crossover(self, p1: dict, p2: dict) -> Tuple[dict, dict]:
        if random.random() > self.crossover_rate:
            return copy.deepcopy(p1), copy.deepcopy(p2)
        c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
        r1, r2 = c1['routes'], c2['routes']
        if len(r1) > 1 and len(r2) > 1:
            pt = random.randint(1, min(len(r1), len(r2)) - 1)
            c1['routes'] = r1[:pt] + r2[pt:]
            c2['routes'] = r2[:pt] + r1[pt:]
        return c1, c2

    def run(self) -> Tuple[dict, List[float], List[dict], Dict, Dict, int]:
        t_op_start = time.perf_counter()
        population = self._initialize_population()
        self.timings['op_time'] += time.perf_counter() - t_op_start

        manager = Manager()
        fitness_cache = manager.dict()
        eval_func = partial(_parallel_eval_wrapper, simulator_instance=self.sim, shared_cache=fitness_cache)

        max_workers = self.config.get('MAX_CPU_WORKERS', 6)
        num_cores = min(os.cpu_count() or 4, max_workers, self.pop_size)
        tasks_per_child = self.config.get('MAX_TASKS_PER_CHILD', 150)
        chunk_size = self.config.get('EVAL_CHUNKSIZE', 3)

        best_individual = None
        current_best_fitness = float('inf')

        stagnation_count = 0
        reference_best_fitness = float('inf')
        actual_generations = self.max_generations

        with Pool(processes=num_cores, initializer=worker_init_fn, initargs=(self.base_seed,),
                  maxtasksperchild=tasks_per_child) as pool:
            t_eval_start = time.perf_counter()
            fitnesses = pool.map(eval_func, population, chunksize=chunk_size)
            self.timings['eval_time'] += time.perf_counter() - t_eval_start

            with tqdm(total=self.max_generations, desc=f"Running {self.mode}", leave=False) as pbar:
                transitions = []

                for gen in range(self.max_generations):

                    current_best_idx = np.argmin(fitnesses)
                    if fitnesses[current_best_idx] < current_best_fitness:
                        current_best_fitness = fitnesses[current_best_idx]
                        best_individual = copy.deepcopy(population[current_best_idx])

                    self.fitness_history.append(current_best_fitness)

                    # --- Early Stopping Tracker ---
                    if reference_best_fitness != float('inf'):
                        rel_improvement = (reference_best_fitness - current_best_fitness) / (
                                    reference_best_fitness + 1e-9)
                        if rel_improvement > 1e-5:
                            reference_best_fitness = current_best_fitness
                            stagnation_count = 0
                        else:
                            stagnation_count += 1
                    else:
                        reference_best_fitness = current_best_fitness

                    if stagnation_count >= self.patience:
                        actual_generations = gen + 1
                        pbar.update(self.max_generations - gen)
                        break

                        # --- RL State Perception ---
                    current_state = None
                    if self.mode == 'ISA-MA' and self.agent:
                        t_rl_start = time.perf_counter()
                        diversity = self._calculate_diversity(population)

                        # [UNIFIED LOGIC]: If flatlined for 10+ gens, force Stagnant state
                        is_stagnant = (stagnation_count >= 10)

                        current_state = self.agent.get_state(is_stagnant, diversity)
                        self.timings['rl_time'] += time.perf_counter() - t_rl_start

                    # --- Selection & Crossover ---
                    t_op_start = time.perf_counter()
                    offspring_pool = []
                    while len(offspring_pool) < self.pop_size:
                        p1 = self._select(population, fitnesses)
                        p2 = self._select(population, fitnesses)
                        c1, c2 = self._crossover(p1, p2)
                        offspring_pool.extend([c1, c2])
                    offspring_pool = offspring_pool[:self.pop_size]
                    self.timings['op_time'] += time.perf_counter() - t_op_start

                    # --- Mutation / Local Search Routing ---
                    next_population = []
                    if self.mode in ['ISA-MA', 'MA-Random', 'MA-Fixed']:
                        t_eval_start = time.perf_counter()
                        fitnesses_before = pool.map(eval_func, offspring_pool, chunksize=chunk_size)
                        self.timings['eval_time'] += time.perf_counter() - t_eval_start

                        refined_offspring = []
                        actions_chosen = []

                        for child in offspring_pool:
                            if self.mode == 'ISA-MA':
                                t_rl_start = time.perf_counter()
                                action = self.agent.choose_action(current_state)
                                self.timings['rl_time'] += time.perf_counter() - t_rl_start
                            elif self.mode == 'MA-Random':
                                t_op_start = time.perf_counter()
                                action = random.randint(0, 3)
                                self.timings['op_time'] += time.perf_counter() - t_op_start
                            elif self.mode == 'MA-Fixed':
                                t_op_start = time.perf_counter()
                                action = np.random.choice(4, p=[0.4, 0.3, 0.2, 0.1])
                                self.timings['op_time'] += time.perf_counter() - t_op_start

                            t_op_start = time.perf_counter()
                            mutated = self.memetic_ops.apply_action(action, child)
                            refined_offspring.append(mutated)
                            self.timings['op_time'] += time.perf_counter() - t_op_start

                            actions_chosen.append(action)
                            self.operator_history.append(
                                {'Generation': gen, 'Action': action, 'CV': getattr(self, '_last_cv', 0.0)})

                        t_eval_start = time.perf_counter()
                        fitnesses_after = pool.map(eval_func, refined_offspring, chunksize=chunk_size)
                        self.timings['eval_time'] += time.perf_counter() - t_eval_start

                        if self.mode == 'ISA-MA':
                            t_rl_start = time.perf_counter()
                            for i in range(len(refined_offspring)):
                                rel_imp = (fitnesses_before[i] - fitnesses_after[i]) / (fitnesses_before[i] + 1e-6)
                                reward = np.tanh(rel_imp * 100.0)

                                if current_state in [0, 1]:
                                    reward += 0.2 if actions_chosen[i] != 0 else -0.2

                                transitions.append((current_state, actions_chosen[i], reward))
                            self.timings['rl_time'] += time.perf_counter() - t_rl_start

                        next_population.extend(refined_offspring)

                    elif self.mode == 'GA':
                        t_op_start = time.perf_counter()
                        for child in offspring_pool:
                            if random.random() < self.mutation_rate:
                                child = self.memetic_ops.apply_action(random.randint(0, 3), child)
                            next_population.append(child)
                        self.timings['op_time'] += time.perf_counter() - t_op_start

                    # --- Elitism ---
                    t_op_start = time.perf_counter()
                    num_elites = max(1, int(self.pop_size * self.elitism_rate))
                    elite_indices = np.argsort(fitnesses)[:num_elites]
                    elites = [copy.deepcopy(population[idx]) for idx in elite_indices]
                    next_population = next_population[:-num_elites] + elites
                    population = next_population
                    self.timings['op_time'] += time.perf_counter() - t_op_start

                    t_eval_start = time.perf_counter()
                    fitnesses = pool.map(eval_func, population, chunksize=chunk_size)
                    self.timings['eval_time'] += time.perf_counter() - t_eval_start

                    self._last_cv = np.std(fitnesses) / (np.mean(fitnesses) + 1e-6)
                    # --- TRUE DELAYED RL UPDATE ---
                    if self.mode == 'ISA-MA' and self.agent:
                        t_rl_start = time.perf_counter()
                        next_diversity = self._calculate_diversity(population)
                        next_is_stagnant = (stagnation_count >= 10)
                        next_state = self.agent.get_state(next_is_stagnant, next_diversity)

                        for s, a, r in transitions:
                            self.agent.learn(s, a, r, next_state)

                        self.agent.step_decay(next_state)

                        if gen % 50 == 0 or gen == self.max_generations - 1:
                            self.q_snapshots[gen] = copy.deepcopy(self.agent.q_table)
                        self.timings['rl_time'] += time.perf_counter() - t_rl_start

                        transitions.clear()

                    pbar.set_postfix({'Best': f"{current_best_fitness:.1e}", 'Stagnant': stagnation_count})
                    pbar.update(1)

                if self.mode == 'ISA-MA' and self.agent:
                    self.q_snapshots[actual_generations - 1] = copy.deepcopy(self.agent.q_table)

                return best_individual, self.fitness_history, self.operator_history, self.q_snapshots, self.timings, actual_generations

