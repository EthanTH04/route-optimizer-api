import numpy as np
from algorithms.city import City, euclidean_distance


# Constants for the Mark array
UNVISITED = 0
VISITED = 1
LAST_VISIT = 2


class Graph:
    def __init__(self, cities: list[City]):
        n = len(cities)
        self.n = n
        self.names = [city.name for city in cities]
        self.prizes = [city.prize for city in cities]
        self.marks = [UNVISITED] * n
        self.marks[n - 1] = LAST_VISIT

        # Distance matrix
        self.matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    self.matrix[i][j] = euclidean_distance(cities[i], cities[j])

        # Shortest path data - populated by construct_shortest_paths()
        self.shortest_matrix = None
        self.shortest_next = None
        self.construct_shortest_paths()

    def construct_shortest_paths(self):
        """
        Floyd-Warshall algorithm: precompute shortest paths between all pairs of cities.
        After this runs, shortest_matrix[i][j] holds the shortest distance from i to j,
        and shortest_next[i][j] holds the next city to visit when going from i to j.
        """
        n = self.n
        self.shortest_matrix = self.matrix.copy()
        self.shortest_next = np.zeros((n, n), dtype=int)

        # Initially, next city from i to j is just j (direct edge)
        for i in range(n):
            for j in range(n):
                self.shortest_next[i][j] = j

        # Floyd-Warshall main loop
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if self.shortest_matrix[i][k] + self.shortest_matrix[k][j] < self.shortest_matrix[i][j]:
                        self.shortest_matrix[i][j] = self.shortest_matrix[i][k] + self.shortest_matrix[k][j]
                        self.shortest_next[i][j] = self.shortest_next[i][k]

    def weight(self, i: int, j: int) -> float:
        # Direct edge weight from city i to city j
        return self.matrix[i][j]

    def shortest_path(self, i: int, j: int) -> float:
        # Shortest total distance from i to j (may go through other cities)
        return self.shortest_matrix[i][j]

    def get_mark(self, v: int) -> int:
        return self.marks[v]

    def set_mark(self, v: int, val: int):
        self.marks[v] = val

    def get_prize(self, v: int) -> int:
        return self.prizes[v]

    def get_name(self, v: int) -> str:
        return self.names[v]

    def last_node(self) -> int:
        return self.n - 1