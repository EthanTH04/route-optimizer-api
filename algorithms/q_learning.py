import random
import time
from pathlib import Path

import numpy as np

from algorithms.agent import Agent
from algorithms.city import City
from algorithms.graph import Graph, UNVISITED, VISITED


# Default hyperparameters (from the original Java implementation)
DEFAULT_TRIALS = 4000
DEFAULT_ALPHA = 0.125         # learning rate
DEFAULT_GAMMA = 0.35          # discount factor
DEFAULT_Q0 = 0.8              # exploration/exploitation coefficient
DEFAULT_DELTA = 1             # power for Q value in action selection
DEFAULT_BETA = 2              # power for distance in action selection
DEFAULT_RANDOM_SEED = 12345


def load_cities_from_file(file_path: str) -> list[City]:
    """
    Load cities from a space-separated file.
    Expected format per line: name x y prize
    """
    cities = []
    with open(file_path, "r") as f:
        for original_index, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            name = parts[0]
            x = float(parts[1])
            y = float(parts[2])
            prize = int(parts[3])
            cities.append(City(name, x, y, prize, original_index))
    return cities


def prepare_cities(cities: list[City], start_name: str, end_name: str) -> list[City]:
    """
    Rearrange the city list so the start city is at index 0 and end city is at the last index.
    If start == end, the same city appears at both index 0 and the last index.
    """
    name_to_index = {city.name.lower(): i for i, city in enumerate(cities)}
    start_key = start_name.lower()
    end_key = end_name.lower()

    if start_key not in name_to_index or end_key not in name_to_index:
        raise ValueError(f"City not found: start={start_name}, end={end_name}")

    start_city = cities[name_to_index[start_key]]

    if start_key == end_key:
        remaining = [c for c in cities if c.name.lower() != start_key]
        return [start_city] + remaining + [start_city]
    else:
        end_city = cities[name_to_index[end_key]]
        remaining = [
            c for c in cities
            if c.name.lower() != start_key and c.name.lower() != end_key
        ]
        return [start_city] + remaining + [end_city]


class QLearning:
    def __init__(
        self,
        cities: list[City],
        budget: float,
        trials: int = DEFAULT_TRIALS,
        alpha: float = DEFAULT_ALPHA,
        gamma: float = DEFAULT_GAMMA,
        q0: float = DEFAULT_Q0,
        delta: float = DEFAULT_DELTA,
        beta: float = DEFAULT_BETA,
        random_seed: int = DEFAULT_RANDOM_SEED,
    ):
        self.cities = cities
        self.budget = budget
        self.trials = trials
        self.alpha = alpha
        self.gamma = gamma
        self.q0 = q0
        self.delta = delta
        self.beta = beta

        self.rng = random.Random(random_seed)

        # Build the graph (Floyd-Warshall runs during construction)
        self.graph = Graph(cities)
        self.states_count = self.graph.n

        # Initialize Q-table and R-table
        self.Q = np.zeros((self.states_count, self.states_count))
        self.R = np.zeros((self.states_count, self.states_count))
        for i in range(self.states_count):
            for j in range(self.states_count):
                if self.graph.matrix[i][j] != 0:
                    self.Q[i][j] = (self.graph.get_prize(i) + self.graph.get_prize(j)) / self.graph.matrix[i][j]

        # Result variables
        self.route: list[int] = []
        self.total_prize = 0
        self.total_weight = 0.0
        self.remaining_budget = 0.0
        self.runtime_ms = 0.0

    def learn(self):
        """
        Main Q-Learning training loop.
        Runs `trials` episodes. Each episode:
        1. Create an agent
        2. Agent picks next state using exploration/exploitation rule
        3. Update Q-table with Bellman equation
        4. Repeat until agent budget is exhausted
        """
        for i in range(self.trials):
            # Fresh graph copy for this episode so marks don't persist
            episode_graph = Graph(self.cities)
            agent = Agent(self.states_count, self.budget, episode_graph)

            while not agent.is_done:
                # q0 grows over trials - more exploitation later
                current_q0 = 1 - self.q0 * (self.trials - i) / self.trials

                next_state = self.get_next_state(agent, agent.current_state, current_q0)

                if next_state == agent.last_node():
                    agent.is_done = True

                max_q = self.max_q(agent, next_state)

                # Bellman update (includes reward for Q-Learning)
                self.Q[agent.current_state][next_state] = (
                    (1 - self.alpha) * self.Q[agent.current_state][next_state]
                    + self.alpha * (self.R[agent.current_state][next_state] + self.gamma * max_q)
                )

                agent.index_path.append(next_state)
                agent.total_weight += agent.shortest_path(agent.current_state, next_state)
                agent.total_prize += agent.get_total_prize(next_state)
                agent.set_mark(next_state, VISITED)
                agent.current_state = next_state

    def get_next_state(self, agent: Agent, s: int, q0: float) -> int:
        """
        Choose the next state based on the current q0 value.
        - Exploration (roulette wheel): pick with probability weighted by Q * prize / distance^beta
        - Exploitation: pick the state with the highest Q * prize / distance^beta

        Includes the infinite-loop fix from your Java code:
        - If total is zero, pick a random feasible node
        - Bounded while loop prevents floating-point precision issues
        """
        # Build feasible set: unvisited cities where the round trip fits in remaining budget
        feasible = []
        for i in range(1, agent.last_node()):
            if (agent.get_mark(i) == UNVISITED
                    and agent.shortest_path(s, i) + agent.shortest_path(i, agent.last_node())
                        < agent.budget - agent.total_weight):
                feasible.append(i)

        if len(feasible) == 0:
            return agent.last_node()

        if self.rng.random() > q0:
            # Exploration - roulette wheel selection
            probabilities = []
            total = 0.0
            for u in feasible:
                weight = agent.weight(s, u)
                if weight == 0:
                    probabilities.append(0.0)
                    continue
                p = (self.Q[s][u] ** self.delta) * agent.get_prize(u) / (weight ** self.beta)
                probabilities.append(p)
                total += p

            # Infinite loop fix: if all probabilities are zero, choose randomly
            if total == 0:
                return self.rng.choice(feasible)

            # Normalize
            probabilities = [p / total for p in probabilities]

            # Roulette wheel with bounded iteration (prevents floating-point precision infinite loop)
            target = self.rng.random()
            idx = -1
            while target > 0 and idx < len(feasible) - 1:
                idx += 1
                target -= probabilities[idx]
            return feasible[idx]
        else:
            # Exploitation - pick highest value
            best_idx = -1
            best_value = float("-inf")
            for i, u in enumerate(feasible):
                weight = agent.weight(s, u)
                if weight == 0:
                    continue
                value = (self.Q[s][u] ** self.delta) * agent.get_prize(u) / (weight ** self.beta)
                if value > best_value:
                    best_idx = i
                    best_value = value
            return feasible[best_idx]

    def max_q(self, agent: Agent, next_state: int) -> float:
        """
        Find the maximum Q value among unvisited states reachable from next_state.
        Used in the Bellman equation update.
        """
        max_value = 0.0
        for i in range(self.states_count):
            if agent.shortest_path(next_state, i) != 0 and agent.get_mark(i) == UNVISITED:
                if self.Q[next_state][i] > max_value:
                    max_value = self.Q[next_state][i]
        return max_value

    def traverse(self):
        """
        After learning, follow the Q-table greedily to construct the final route.
        Starts at city 0 and greedily picks the highest-Q next state until reaching the last node.
        """
        # Reset for the greedy traversal
        self.route = []
        self.total_weight = 0.0
        for i in range(self.states_count):
            self.graph.set_mark(i, UNVISITED)
        self.graph.set_mark(self.graph.last_node(), UNVISITED)  # Will visit last node too

        current = 0
        self.graph.set_mark(current, VISITED)
        self.route.append(current)

        while current != self.graph.last_node():
            next_state = self.get_highest_q(current)
            if next_state == -1:
                next_state = self.graph.last_node()
                if self.total_weight + self.graph.weight(current, next_state) > self.budget:
                    break

            self.graph.set_mark(next_state, VISITED)
            # Handle detours through intermediate cities
            self._add_detour_path(current, next_state)
            self.total_weight += self.graph.shortest_path(current, next_state)
            current = next_state

        # Compute total prize based on the route
        self.total_prize = 0
        visited = set()
        for city in self.route:
            if city not in visited:
                visited.add(city)
                self.total_prize += self.graph.get_prize(city)

        self.remaining_budget = self.budget - self.total_weight

    def get_highest_q(self, v: int) -> int:
        """
        Find the unvisited state with the highest Q value from v.
        Returns -1 if no feasible unvisited state exists.
        """
        best = float("-inf")
        index = -1
        for i in range(1, self.graph.last_node()):
            if (self.Q[v][i] > best
                    and self.graph.get_mark(i) == UNVISITED
                    and self.graph.shortest_path(v, i) + self.graph.shortest_path(i, self.graph.last_node())
                        < self.budget - self.total_weight):
                best = self.Q[v][i]
                index = i
        return index

    def _add_detour_path(self, start: int, end: int):
        """
        If the shortest path from start to end goes through other cities, add them to the route.
        Otherwise just add end.
        """
        if self.graph.shortest_next[start][end] != end:
            while start != end:
                next_city = self.graph.shortest_next[start][end]
                self.route.append(next_city)
                start = next_city
        else:
            self.route.append(end)

    def run(self) -> dict:
        """
        Full Q-Learning pipeline: learn then traverse.
        Returns a dictionary of results.
        """
        start_time = time.perf_counter()
        self.learn()
        self.traverse()
        end_time = time.perf_counter()
        self.runtime_ms = (end_time - start_time) * 1000

        return {
            "algorithm": "q-learning",
            "num_cities": self.states_count,
            "budget": self.budget,
            "total_distance": self.total_weight,
            "prize_collected": self.total_prize,
            "runtime_ms": self.runtime_ms,
            "remaining_budget": self.remaining_budget,
            "route": [self.graph.get_name(c) for c in self.route],
        }