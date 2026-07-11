import math


class City:
    def __init__(self, name: str, x: float, y: float, prize: int, original_index: int = 0):
        self.name = name
        self.x = x
        self.y = y
        self.prize = prize
        self.original_index = original_index

    def __repr__(self):
        return f"City({self.name}, x={self.x}, y={self.y}, prize={self.prize})"


def euclidean_distance(city1: City, city2: City) -> float:
    return math.sqrt((city1.x - city2.x) ** 2 + (city1.y - city2.y) ** 2)