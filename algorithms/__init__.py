from algorithms.city import City


def prepare_cities(cities: list[City], start_name: str, end_name: str) -> list[City]:
    """
    Rearrange the city list so the start city is at index 0 and end city is at the last index.
    If start == end, the same city appears at both positions (round trip).
    """
    name_to_index = {city.name.lower(): i for i, city in enumerate(cities)}
    start_key = start_name.lower()
    end_key = end_name.lower()

    if start_key not in name_to_index:
        raise ValueError(f"Start city not found: {start_name}")
    if end_key not in name_to_index:
        raise ValueError(f"End city not found: {end_name}")

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