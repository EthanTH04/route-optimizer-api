"""
Small test of Q-Learning and P-MARL on a 5-city problem.
"""

from algorithms.city import City
from algorithms.q_learning import QLearning
from algorithms.p_marl import PMARL


def main():
    cities = [
        City("Start", 0, 0, 0),
        City("B",     3, 4, 50),
        City("C",     5, 0, 30),
        City("D",     3, -4, 40),
        City("E",     1, 0, 20),
        City("End",   0, 0, 0),
    ]

    budget = 30.0

    print("=" * 50)
    print("Q-LEARNING")
    print("=" * 50)
    q = QLearning(cities=cities, budget=budget, trials=1000)
    q_result = q.run()
    for k, v in q_result.items():
        if k == "route":
            print(f"{k}: {' -> '.join(v)}")
        elif isinstance(v, float):
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v}")

    print()
    print("=" * 50)
    print("P-MARL")
    print("=" * 50)
    p = PMARL(cities=cities, budget=budget, trials=1000, num_agents=5)
    p_result = p.run()
    for k, v in p_result.items():
        if k == "route":
            print(f"{k}: {' -> '.join(v)}")
        elif isinstance(v, float):
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v}")

    print()
    print("=" * 50)
    print("COMPARISON")
    print("=" * 50)
    print(f"Q-Learning:  prize={q_result['prize_collected']}, runtime={q_result['runtime_ms']:.2f}ms")
    print(f"P-MARL:      prize={p_result['prize_collected']}, runtime={p_result['runtime_ms']:.2f}ms")


if __name__ == "__main__":
    main()