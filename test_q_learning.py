"""
Small test of Q-Learning on a 5-city problem.
Prize values and coordinates chosen so a smart algorithm can find a good route.
"""

from algorithms.city import City
from algorithms.q_learning import QLearning


def main():
    # 5 cities laid out roughly in a pentagon
    # Start and end at city A (index 0 and last index)
    cities = [
        City("Start", 0, 0, 0),        # start city
        City("B",     3, 4, 50),       # far but high prize
        City("C",     5, 0, 30),
        City("D",     3, -4, 40),
        City("E",     1, 0, 20),       # close, moderate prize
        City("End",   0, 0, 0),        # same location as start (loop)
    ]

    budget = 30.0

    print(f"Running Q-Learning on {len(cities)} cities with budget {budget}...")
    print()

    q = QLearning(cities=cities, budget=budget, trials=1000)
    result = q.run()

    print("=== Q-Learning Results ===")
    print(f"Total distance:     {result['total_distance']:.2f}")
    print(f"Prize collected:    {result['prize_collected']}")
    print(f"Runtime:            {result['runtime_ms']:.2f} ms")
    print(f"Remaining budget:   {result['remaining_budget']:.2f}")
    print(f"Route:              {' -> '.join(result['route'])}")


if __name__ == "__main__":
    main()