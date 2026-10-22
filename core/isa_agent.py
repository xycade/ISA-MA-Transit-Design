import numpy as np
import random


class ISAAgent:
    """
    Inherently Interpretable State-Aware (ISA) Meta-controller based on Tabular Q-Learning.
    """

    def __init__(self, config: dict):
        self.config = config

        self.lr = config.get('RL_LEARNING_RATE', 0.2)
        self.gamma = config.get('RL_DISCOUNT_FACTOR', 0.9)
        self.epsilon = config.get('RL_EPSILON_START', 1.0)
        self.epsilon_end = config.get('RL_EPSILON_END', 0.01)
        self.epsilon_decay = config.get('RL_EPSILON_DECAY', 0.985)
        self.temp = config.get('RL_TEMPERATURE', 0.5)
        self.diversity_threshold = config.get('RL_DIVERSITY_THRESHOLD', 0.1)

        # Q-Table: 4 States x 4 Memetic Actions
        self.q_table = np.zeros((4, 4))

    def get_state(self, is_stagnant: bool, diversity: float) -> int:
        """
        [Unified Stagnation Update]
        Directly uses the boolean is_stagnant flag synchronized with the Early Stopping tracker.
        """
        is_div_high = diversity >= self.diversity_threshold

        if is_stagnant and is_div_high:
            return 0
        elif is_stagnant and not is_div_high:
            return 1
        elif not is_stagnant and is_div_high:
            return 2
        else:
            return 3

    def choose_action(self, state: int) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, 3)

        q_values = self.q_table[state]
        exp_q = np.exp((q_values - np.max(q_values)) / self.temp)
        probs = exp_q / np.sum(exp_q)
        return np.random.choice(4, p=probs)

    def learn(self, state: int, action: int, reward: float, next_state: int):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.lr * td_error

    def step_decay(self, current_state: int):
        if current_state in [0, 1]:
            # Rebound exploration to break out of stagnation
            self.epsilon = min(1.0, self.epsilon + 0.2)
        else:
            if self.epsilon > self.epsilon_end:
                self.epsilon *= self.epsilon_decay