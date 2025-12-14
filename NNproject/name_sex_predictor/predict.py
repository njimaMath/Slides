import json
from pathlib import Path
from typing import List


from train_model import encode_name  # reuse encoding from training


def load_sigma(model_file: Path) -> List[int]:
    """
    Load the sigma weights from a JSON file.

    Args:
        model_file: Path to the JSON file containing sigma weights.

    Returns:
        A list of +1/-1 integers representing the weight vector.
    """
    with model_file.open('r', encoding='utf-8') as f:
        sigma = json.load(f)
    return [int(x) for x in sigma]


def predict(name: str, sigma: List[int]) -> str:
    """
    Predict the sex for a given name using the sigma weights.

    Args:
        name: The input first name to classify.
        sigma: List of +1/-1 weights, length should be 50.

    Returns:
        "male" if the dot product >= 0, otherwise "female".
    """
    vec = encode_name(name)
    # Convert to ±1
    v2 = [2 * bit - 1 for bit in vec]
    dot = sum(w * x for w, x in zip(sigma, v2))
    return 'male' if dot >= 0 else 'female'


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Predict the sex from a Japanese first name using precomputed sigma.")
    parser.add_argument('name', type=str, help='First name to classify (e.g., "Haruto")')
    parser.add_argument('-m', '--model', type=str, default='model.json', help='Path to the sigma weights file.')
    args = parser.parse_args()
    sigma = load_sigma(Path(args.model))
    result = predict(args.name, sigma)
    print(result)


if __name__ == '__main__':
    main()