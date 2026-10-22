import random
import copy
from typing import Tuple


class CrossoverOperator:
    """
    Handles the genetic recombination (crossover) of two parent solutions
    in the Memetic Algorithm.
    """

    def __init__(self, config: dict):
        """
        Initializes the crossover operator.

        Args:
            config (dict): Global configurations containing the crossover rate.
        """
        self.crossover_rate = config.get('CROSSOVER_RATE', 0.1)

    def apply(self, parent1: dict, parent2: dict) -> Tuple[dict, dict]:
        """
        Executes a single-point crossover on the routes of two parent individuals.

        Args:
            parent1 (dict): The first parent individual.
            parent2 (dict): The second parent individual.

        Returns:
            Tuple[dict, dict]: A tuple containing two generated offspring individuals.
        """
        # Return deep copies if crossover is not triggered
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)

        p1_routes = parent1.get('routes', [])
        p2_routes = parent2.get('routes', [])

        # Ensure both parents have enough routes to perform crossover
        if len(p1_routes) < 2 or len(p2_routes) < 2:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)

        min_len = min(len(p1_routes), len(p2_routes))
        if min_len < 2:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)

        # Select a random single-point for crossover
        crossover_point = random.randint(1, min_len - 1)

        # Recombine routes
        c1_routes = p1_routes[:crossover_point] + p2_routes[crossover_point:]
        c2_routes = p2_routes[:crossover_point] + p1_routes[crossover_point:]

        return {'routes': c1_routes}, {'routes': c2_routes}