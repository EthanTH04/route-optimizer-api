import os

from openai import OpenAI


def get_client() -> OpenAI:
    """Create an OpenAI client using the API key from environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    return OpenAI(api_key=api_key)


def generate_explanation(q_result: dict, p_result: dict) -> str:
    """
    Generate a plain-English explanation of why one algorithm outperformed the other.
    Takes both algorithm results and asks OpenAI to compare them.
    """
    client = get_client()

    prompt = f"""You are helping explain the results of two reinforcement learning algorithms that solved a variant of the Traveling Salesman Problem with a budget constraint.

Q-Learning result:
- Total distance: {q_result['total_distance']:.2f}
- Prize collected: {q_result['prize_collected']}
- Runtime: {q_result['runtime_ms']:.2f} ms
- Remaining budget: {q_result['remaining_budget']:.2f}
- Route: {' -> '.join(q_result['route'])}

P-MARL result:
- Total distance: {p_result['total_distance']:.2f}
- Prize collected: {p_result['prize_collected']}
- Runtime: {p_result['runtime_ms']:.2f} ms
- Remaining budget: {p_result['remaining_budget']:.2f}
- Route: {' -> '.join(p_result['route'])}

In 2-3 short paragraphs, explain in plain English which algorithm performed better and why. Focus on the tradeoffs between runtime and route quality. Avoid technical jargon."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that explains algorithm performance in clear, non-technical language."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=500,
        temperature=0.7,
    )

    return response.choices[0].message.content