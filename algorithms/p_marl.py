import random
import time
from pathlib import Path

import numpy as np

from algorithms.agent import Agent
from algorithms.city import City
from algorithms.graph import Graph, UNVISITED, VISITED


# Default hyperparameters (from the original Java implementation)
DEFAULT_TRIALS = 4000
DEFAULT_NUM_AGENTS = 5
DEFAULT_W = 1000.0            # reinforcement constant for cooperative phase
DEFAULT_ALPHA = 0.125          # learning rate
DEFAULT_GAMMA = 0.35           # discount factor
DEFAULT_Q0 = 0.8               # exploration/exploitation coefficient
DEFAULT_DELTA = 1              # power for Q value in action selection
DEFAULT_BETA = 2               # power for distance in action selection
DEFAULT_RANDOM_SEED = 12345


class PMARL:
    def __init__(
        self,
        cities: list[City],
        budget: float,
        trials: int = DEFAULT_TRIALS,
        num_agents: int = DEFAULT_NUM_AGENTS,
        w: float = DEFAULT_W,
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
        self.num_agents = num_agents
        self.w = w
        self.alpha = alpha
        self.gamma = gamma
        self.q0 = q0
        self.delta = delta
        self.beta = beta

        self.rng = random.Random(random_seed)

        self.graph = Graph(cities)
        self.states_count = self.graph.n

        self.Q = np.zeros((self.states_count, self.states_count))
        self.R = np.zeros((self.states_count, self.states_count))
        for i in range(self.states_count):
            for j in range(self.states_count):
                if self.graph.matrix[i][j] != 0:
                    self.Q[i][j] = (self.graph.get_prize(i) + self.graph.get_prize(j)) / self.graph.matrix[i][j]

        # Track best route across all episodes
        self.prize_max = float("-inf")
        self.route_max = []
        self.route_max_iter = 0

        # Result variables
        self.route: list[int] = []
        self.total_prize = 0
        self.total_weight = 0.0
        self.remaining_budget = 0.0
        self.runtime_ms = 0.0

    def learn(self):
        """
        Main P-MARL training loop.
        Each episode runs multiple agents, then does a cooperative update
        along the best agent's route.
        """
        for i in range(self.trials):
            agents = self._create_agents()

            # Each agent explores independently
            while not self._all_done(agents):
                for agent in agents:
                    if agent.is_done:
                        continue

                    current_q0 = 1 - self.q0 * (self.trials - i) / self.trials
                    next_state = self.get_next_state(agent, agent.current_state, current_q0)

                    if next_state == agent.last_node():
                        agent.is_done = True

                    max_q_val = self.max_q(agent, next_state)

                    # P-MARL step-by-step update: NO reward term (only Q update)
                    self.Q[agent.current_state][next_state] = (
                        (1 - self.alpha) * self.Q[agent.current_state][next_state]
                        + self.alpha * self.gamma * max_q_val
                    )

                    agent.index_path.append(next_state)
                    agent.total_weight += agent.shortest_path(agent.current_state, next_state)
                    agent.total_prize += agent.get_total_prize(next_state)
                    agent.set_mark(next_state, VISITED)
                    agent.current_state = next_state

            # Cooperative learning phase: reinforce the best agent's path
            most_fit_index = self._find_highest_prize(agents)
            j_star = agents[most_fit_index]
            path = j_star.index_path
            j_star.reset_marks()

            if j_star.total_prize > 0:
                # Update R and Q along j_star's path
                for v in range(len(path) - 1):
                    src = path[v]
                    dst = path[v + 1]
                    q = self.Q[src][dst]
                    max_q_val = self.max_q(j_star, dst)
                    self.R[src][dst] += self.w / j_star.total_prize
                    self.Q[src][dst] = (1 - self.alpha) * q + self.alpha * (self.R[src][dst] + self.gamma * max_q_val)

                # Extra reinforcement if this is a new best route
                if j_star.total_prize > self.prize_max:
                    self.prize_max = j_star.total_prize
                    self.route_max = path
                    self.route_max_iter = i
                    for v in range(len(path) - 1):
                        src = path[v]
                        dst = path[v + 1]
                        q = self.Q[src][dst]
                        max_q_val = self.max_q(j_star, dst)
                        self.R[src][dst] += (i * self.w) / j_star.total_prize
                        self.Q[src][dst] = (1 - self.alpha) * q + self.alpha * (self.R[src][dst] + self.gamma * max_q_val)

    def _create_agents(self) -> list[Agent]:
        """Create fresh agents with fresh graph copies for a new episode."""
        agents = []
        for _ in range(self.num_agents):
            episode_graph = Graph(self.cities)
            agents.append(Agent(self.states_count, self.budget, episode_graph))
        return agents

    def _all_done(self, agents: list[Agent]) -> bool:
        return all(a.is_done for a in agents)

    def _find_highest_prize(self, agents: list[Agent]) -> int:
        best_index = 0
        best_prize = float("-inf")
        for j, agent in enumerate(agents):
            if agent.total_prize > best_prize:
                best_prize = agent.total_prize
                best_index = j
        return best_index

    def get_next_state(self, agent: Agent, s: int, q0: float) -> int:
        """
        Same action selection logic as Q-Learning.
        Exploration uses roulette wheel with Q * prize / distance^beta.
        Exploitation picks the maximum of that same product.
        """
        feasible = []
        for i in range(1, agent.last_node()):
            if (agent.get_mark(i) == UNVISITED
                    and agent.shortest_path(s, i) + agent.shortest_path(i, agent.last_node())
                        < agent.budget - agent.total_weight):
                feasible.append(i)

        if len(feasible) == 0:
            return agent.last_node()

        if self.rng.random() > q0:
            # Exploration
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

            if total == 0:
                return self.rng.choice(feasible)

            probabilities = [p / total for p in probabilities]

            target = self.rng.random()
            idx = -1
            while target > 0 and idx < len(feasible) - 1:
                idx += 1
                target -= probabilities[idx]
            return feasible[idx]
        else:
            # Exploitation
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
        max_value = 0.0
        for i in range(self.states_count):
            if agent.shortest_path(next_state, i) != 0 and agent.get_mark(i) == UNVISITED:
                if self.Q[next_state][i] > max_value:
                    max_value = self.Q[next_state][i]
        return max_value

    def traverse(self):
        """Follow Q-table greedily to build the final route (same as Q-Learning)."""
        self.route = []
        self.total_weight = 0.0
        for i in range(self.states_count):
            self.graph.set_mark(i, UNVISITED)

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
            self._add_detour_path(current, next_state)
            self.total_weight += self.graph.shortest_path(current, next_state)
            current = next_state

        self.total_prize = 0
        visited = set()
        for city in self.route:
            if city not in visited:
                visited.add(city)
                self.total_prize += self.graph.get_prize(city)

        self.remaining_budget = self.budget - self.total_weight

    def get_highest_q(self, v: int) -> int:
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
        if self.graph.shortest_next[start][end] != end:
            while start != end:
                next_city = self.graph.shortest_next[start][end]
                self.route.append(next_city)
                start = next_city
        else:
            self.route.append(end)

    def run(self) -> dict:
        start_time = time.perf_counter()
        self.learn()
        self.traverse()
        end_time = time.perf_counter()
        self.runtime_ms = (end_time - start_time) * 1000

        return {
            "algorithm": "p-marl",
            "num_cities": self.states_count,
            "budget": self.budget,
            "total_distance": self.total_weight,
            "prize_collected": self.total_prize,
            "runtime_ms": self.runtime_ms,
            "remaining_budget": self.remaining_budget,
            "route": [self.graph.get_name(c) for c in self.route],
        }