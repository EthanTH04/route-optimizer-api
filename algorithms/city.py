import math


class City:
    def __init__(self, name: str, lat: float, lon: float, prize: int, original_index: int = 0):
        self.name = name
        self.lat = lat
        self.lon = lon
        self.prize = prize
        self.original_index = original_index

    def __repr__(self):
        return f"City({self.name}, lat={self.lat}, lon={self.lon}, prize={self.prize})"


def euclidean_distance(city1: City, city2: City) -> float:
    return math.sqrt((city1.lat - city2.lat) ** 2 + (city1.lon - city2.lon) ** 2)