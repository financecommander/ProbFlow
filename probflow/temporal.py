import numpy as np
from typing import List, Tuple

class MarkovChain:
    def __init__(self, transition_matrix: np.ndarray, initial_state: np.ndarray):
        self.transition_matrix = transition_matrix
        self.state = initial_state
    
    def step(self, n_steps: int = 1) -> np.ndarray:
        for _ in range(n_steps):
            self.state = np.dot(self.state, self.transition_matrix)
        return self.state
    
    def sample(self, n_steps: int) -> List[int]:
        states = []
        current = np.random.choice(len(self.state), p=self.state)
        for _ in range(n_steps):
            states.append(current)
            current = np.random.choice(len(self.state), p=self.transition_matrix[current])
        return states

class HiddenMarkovModel:
    def __init__(self, transition_matrix: np.ndarray, emission_matrix: np.ndarray, initial_state: np.ndarray):
        self.transition_matrix = transition_matrix
        self.emission_matrix = emission_matrix
        self.initial_state = initial_state
    
    def viterbi(self, observations: List[int]) -> List[int]:
        # TODO: Implement full Viterbi algorithm for most likely state sequence
        return [0] * len(observations)
    
    def baum_welch(self, observations: List[int], n_iter: int = 10):
        # TODO: Implement Baum-Welch for parameter estimation
        pass
