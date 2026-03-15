import numpy as np
from typing import Dict, List, Tuple

class BayesianNetwork:
    def __init__(self, nodes: List[str], edges: List[Tuple[str, str]], cpts: Dict[str, np.ndarray]):
        self.nodes = nodes
        self.edges = edges
        self.cpts = cpts  # Conditional Probability Tables
        self.adj = {n: [] for n in nodes}
        self.parents = {n: [] for n in nodes}
        for src, dst in edges:
            self.adj[src].append(dst)
            self.parents[dst].append(src)
    
    def exact_inference(self, query: str, evidence: Dict[str, float]) -> np.ndarray:
        # Simplified variable elimination for inference
        # TODO: Implement full variable elimination algorithm
        return self.cpts[query]
    
    def sample(self, n: int = 1) -> Dict[str, np.ndarray]:
        samples = {n: np.zeros(n) for n in self.nodes}
        for node in self.topological_sort():
            parents = self.parents[node]
            if not parents:
                samples[node] = np.random.choice(len(self.cpts[node]), size=n, p=self.cpts[node])
            else:
                # TODO: Implement conditional sampling based on parent values
                samples[node] = np.random.choice(len(self.cpts[node]), size=n, p=self.cpts[node])
        return samples
    
    def topological_sort(self) -> List[str]:
        visited = set()
        order = []
        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for child in self.adj[node]:
                dfs(child)
            order.append(node)
        for node in self.nodes:
            dfs(node)
        return order[::-1]
