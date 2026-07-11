from algorithms.graph import Graph, UNVISITED, VISITED


class Agent:
    def __init__(self, states_count: int, budget: float, graph: Graph):
        self.states_count = states_count
        self.graph = graph
        self.budget = budget
        self.total_prize = 0
        self.total_weight = 0.0
        self.current_state = 0
        self.index_path = []
        self.is_done = False

    def set_mark(self, v: int, val: int):
        self.graph.set_mark(v, val)

    def get_mark(self, v: int) -> int:
        return self.graph.get_mark(v)

    def weight(self, i: int, v: int) -> float:
        return self.graph.matrix[i][v]

    def shortest_path(self, i: int, v: int) -> float:
        return self.graph.shortest_path(i, v)

    def get_prize(self, v: int) -> int:
        return self.graph.get_prize(v)

    def last_node(self) -> int:
        return self.graph.last_node()

    def get_total_prize(self, end: int) -> int:
        """
        When the agent moves toward `end`, it may pass through intermediate cities
        because the shortest path from current_state to end might go through other nodes.
        This function collects the prizes from all intermediate cities that haven't been visited yet.
        """
        start = self.current_state
        if self.graph.shortest_next[start][end] != end:
            # There's a detour - collect prizes from intermediate cities
            total = 0
            while start != end:
                next_city = self.graph.shortest_next[start][end]
                if self.graph.get_mark(next_city) == UNVISITED:
                    total += self.graph.get_prize(next_city)
                    self.graph.set_mark(next_city, VISITED)
                start = next_city
            return total
        else:
            return self.graph.get_prize(end)

    def reset_marks(self):
        for i in range(self.graph.n):
            self.graph.set_mark(i, UNVISITED)