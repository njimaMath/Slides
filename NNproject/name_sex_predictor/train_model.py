import json
import random
from pathlib import Path
from typing import List, Tuple


# Dimension constants
NAME_LENGTH = 8           # Characters for first name (increased from 5 for better accuracy)
BITS_PER_CHAR = 10        # Bits per character (A-Z encoded as 1-26)
TOTAL_BITS = NAME_LENGTH * BITS_PER_CHAR  # 80 bits total


def letter_to_bits(letter: str) -> List[int]:
    """
    Convert a single uppercase letter into its 10-bit binary representation.
    A -> 0000000001, B -> 0000000010, ..., Z -> 0000011010 (26).
    Unknown characters or empty slots return ten zeros.

    Args:
        letter: A single character.

    Returns:
        A list of 10 integers (0 or 1) representing the binary encoding.
    """
    if len(letter) != 1:
        return [0] * BITS_PER_CHAR
    # Only uppercase letters A-Z are valid. Others are encoded as zeros.
    if 'A' <= letter <= 'Z':
        # Convert A=1, B=2, ..., Z=26
        val = ord(letter) - ord('A') + 1
    else:
        val = 0
    # Convert value to 10-bit binary string
    return [int(bit) for bit in f"{val:010b}"]


def encode_name(name: str) -> List[int]:
    """
    Encode a first name into a 50-bit vector in {0,1}^{50}.

    The encoding reserves 5 characters (50 bits) for the first name.
    If the name is shorter than 5 characters, its encoding is right-aligned
    and padded on the left with groups of ten zeros. Each letter is encoded
    using the scheme from `letter_to_bits`.

    Args:
        name: A string containing the first name (e.g., "HARUTO").

    Returns:
        A list of 50 integers (0/1) representing the encoded name.
    """
    # Remove spaces and convert to uppercase
    normalized = ''.join(ch for ch in name.strip() if ch.isalpha()).upper()
    if not normalized:
        # Return a zero vector if the name is empty
        return [0] * TOTAL_BITS
    
    # Keep only the rightmost NAME_LENGTH characters if longer
    trimmed = normalized[-NAME_LENGTH:]
    # Build 10-bit codes for each character
    codes = [letter_to_bits(ch) for ch in trimmed]
    # Left-pad with [0]*10 groups to length NAME_LENGTH
    pad_len = NAME_LENGTH - len(codes)
    padding = [[0] * BITS_PER_CHAR] * pad_len
    full = padding + codes
    # Flatten list of lists to a 50-length list
    return [bit for code in full for bit in code]


def to_pm1(vec: List[int]) -> List[int]:
    """Convert a {0,1} vector to {-1,+1} via 2*v - 1."""
    return [2 * b - 1 for b in vec]


def dot_product(v1: List[int], v2: List[int]) -> int:
    """Compute dot product of two integer vectors."""
    return sum(a * b for a, b in zip(v1, v2))


def sign(x: int) -> int:
    """Sign function: returns +1 if x >= 0, else -1."""
    return 1 if x >= 0 else -1


def compute_accuracy(vectors_pm1: List[List[int]], labels: List[int], w: List[int]) -> Tuple[int, int, float]:
    """
    Compute the number of correct predictions.

    For perceptron: predict = sign(g · w) where g = 2*vec - 1
    We want predict == label.

    Args:
        vectors_pm1: List of {-1,+1} encoded name vectors.
        labels: List of labels (+1 for male, -1 for female).
        w: Weight vector.

    Returns:
        Tuple of (correct_count, total_count, accuracy_ratio).
    """
    correct = 0
    total = len(labels)
    for g, label in zip(vectors_pm1, labels):
        pred = sign(dot_product(g, w))
        if pred == label:
            correct += 1
    return correct, total, correct / total if total > 0 else 0.0


def compute_sigma_correlation(vectors: List[List[int]], labels: List[int]) -> List[int]:
    """
    Compute the weight vector σ ∈ {-1, +1}^120 using correlation method.

    The method computes the correlation between each bit of (2*v - 1) and the label (±1).
    The sign of the correlation determines the weight at that position.

    Args:
        vectors: A list of 0/1 vectors representing encoded names.
        labels: A list of labels (1 for male, -1 for female).

    Returns:
        A list of 120 integers, each either +1 or -1, representing the weight vector.
    """
    n = len(vectors[0]) if vectors else TOTAL_BITS
    # Initialize sums for each dimension
    correlation = [0] * n
    for vec, label in zip(vectors, labels):
        # Convert vector to ±1 form
        g = to_pm1(vec)
        # Accumulate correlation: sum of label * g_i
        for i in range(n):
            correlation[i] += label * g[i]
    # Convert correlation to σ by taking the sign (zero → +1)
    sigma = [sign(c) for c in correlation]
    return sigma


def perceptron_train(vectors_pm1: List[List[int]], labels: List[int],
                     max_epochs: int = 100, learning_rate: float = 1.0,
                     shuffle: bool = True, verbose: bool = False) -> Tuple[List[float], int]:
    """
    Train a perceptron using the standard perceptron algorithm.

    Goal: Find w such that sign(g · w) = label for all (g, label) pairs.
    The update rule is: if sign(g · w) != label, then w += learning_rate * label * g

    Args:
        vectors_pm1: List of {-1,+1} vectors (already converted via 2*vec - 1).
        labels: List of labels (+1 for male, -1 for female).
        max_epochs: Maximum number of passes through the data.
        learning_rate: Learning rate for weight updates.
        shuffle: Whether to shuffle data each epoch.
        verbose: Whether to print progress.

    Returns:
        Tuple of (weight_vector, num_mistakes_in_last_epoch).
    """
    n = len(vectors_pm1[0]) if vectors_pm1 else TOTAL_BITS
    # Initialize weights to zero (real-valued during training)
    w = [0.0] * n

    data = list(zip(vectors_pm1, labels))
    original_labels = labels[:]
    best_w = w[:]
    best_correct = 0

    for epoch in range(max_epochs):
        if shuffle:
            random.shuffle(data)
        mistakes = 0
        for g, label in data:
            # Compute prediction
            score = sum(w[i] * g[i] for i in range(n))
            pred = 1 if score >= 0 else -1
            # Update if wrong
            if pred != label:
                mistakes += 1
                for i in range(n):
                    w[i] += learning_rate * label * g[i]

        # Track best weights (using original order for consistency)
        correct, _, _ = compute_accuracy(vectors_pm1, original_labels, w)
        if correct > best_correct:
            best_correct = correct
            best_w = w[:]

        if verbose and (epoch % 10 == 0 or epoch == max_epochs - 1):
            print(f"Epoch {epoch + 1}: mistakes = {mistakes}, accuracy = {correct}/{len(original_labels)}")

        # Early stop if no mistakes
        if mistakes == 0:
            if verbose:
                print(f"Converged at epoch {epoch + 1}")
            break

    return best_w, mistakes


def averaged_perceptron_train(vectors_pm1: List[List[int]], labels: List[int],
                              max_epochs: int = 100, verbose: bool = False) -> List[float]:
    """
    Train using the averaged perceptron algorithm for better generalization.

    Args:
        vectors_pm1: List of {-1,+1} vectors.
        labels: List of labels (+1 for male, -1 for female).
        max_epochs: Maximum number of epochs.
        verbose: Whether to print progress.

    Returns:
        The averaged weight vector.
    """
    n = len(vectors_pm1[0]) if vectors_pm1 else TOTAL_BITS
    w = [0.0] * n       # Current weights
    w_sum = [0.0] * n   # Sum of all weight vectors
    count = 0

    data = list(zip(vectors_pm1, labels))

    for epoch in range(max_epochs):
        random.shuffle(data)
        mistakes = 0
        for g, label in data:
            score = sum(w[i] * g[i] for i in range(n))
            pred = 1 if score >= 0 else -1
            if pred != label:
                mistakes += 1
                for i in range(n):
                    w[i] += label * g[i]
            # Accumulate for averaging
            for i in range(n):
                w_sum[i] += w[i]
            count += 1

        if verbose and (epoch % 10 == 0 or epoch == max_epochs - 1):
            print(f"Epoch {epoch + 1}: mistakes = {mistakes}")

        if mistakes == 0:
            if verbose:
                print(f"Converged at epoch {epoch + 1}")
            break

    # Return averaged weights
    return [ws / count for ws in w_sum]


def quantize_to_pm1(w: List[float]) -> List[int]:
    """Quantize real-valued weights to {-1, +1}."""
    return [1 if x >= 0 else -1 for x in w]


def greedy_coordinate_descent(vectors_pm1: List[List[int]], labels: List[int],
                              init_sigma: List[int], max_iter: int = 1000,
                              verbose: bool = False) -> Tuple[List[int], int]:
    """
    Greedy coordinate descent to find best σ ∈ {-1, +1}^n.

    At each step, try flipping each coordinate and keep the flip that
    improves accuracy the most.

    Args:
        vectors_pm1: List of {-1,+1} vectors.
        labels: List of labels.
        init_sigma: Initial weight vector in {-1, +1}^n.
        max_iter: Maximum iterations.
        verbose: Print progress.

    Returns:
        Best sigma found and its accuracy (count of correct).
    """
    n = len(init_sigma)
    sigma = init_sigma[:]
    best_correct, _, _ = compute_accuracy(vectors_pm1, labels, sigma)

    for iteration in range(max_iter):
        improved = False
        best_flip = -1
        best_flip_correct = best_correct

        # Find the best single flip
        for i in range(n):
            sigma[i] *= -1
            correct, _, _ = compute_accuracy(vectors_pm1, labels, sigma)
            if correct > best_flip_correct:
                best_flip_correct = correct
                best_flip = i
            sigma[i] *= -1  # Revert

        if best_flip >= 0:
            sigma[best_flip] *= -1
            best_correct = best_flip_correct
            improved = True
            if verbose:
                print(f"  Iter {iteration + 1}, flip bit {best_flip}: {best_correct}/{len(labels)}")

        if not improved:
            if verbose:
                print(f"Converged at iteration {iteration + 1}")
            break

    return sigma, best_correct


def simulated_annealing_pm1(vectors_pm1: List[List[int]], labels: List[int],
                            init_sigma: List[int], max_iter: int = 10000,
                            temp_start: float = 10.0, temp_end: float = 0.1,
                            verbose: bool = False) -> Tuple[List[int], int]:
    """
    Simulated annealing to find good σ ∈ {-1, +1}^n.

    Args:
        vectors_pm1: List of {-1,+1} vectors.
        labels: List of labels.
        init_sigma: Initial weight vector.
        max_iter: Maximum iterations.
        temp_start: Starting temperature.
        temp_end: Ending temperature.
        verbose: Print progress.

    Returns:
        Best sigma found and its accuracy (count of correct).
    """
    import math

    n = len(init_sigma)
    sigma = init_sigma[:]
    current_correct, _, _ = compute_accuracy(vectors_pm1, labels, sigma)
    best_sigma = sigma[:]
    best_correct = current_correct

    for iteration in range(max_iter):
        # Temperature schedule (exponential decay)
        t = temp_start * (temp_end / temp_start) ** (iteration / max_iter)

        # Pick a random coordinate to flip
        i = random.randint(0, n - 1)
        sigma[i] *= -1

        new_correct, _, _ = compute_accuracy(vectors_pm1, labels, sigma)
        delta = new_correct - current_correct

        # Accept or reject
        if delta > 0 or random.random() < math.exp(delta / t):
            current_correct = new_correct
            if current_correct > best_correct:
                best_correct = current_correct
                best_sigma = sigma[:]
                if verbose and iteration % 1000 == 0:
                    print(f"  Iter {iteration}: new best {best_correct}/{len(labels)}, T={t:.2f}")
        else:
            sigma[i] *= -1  # Revert

        # Early stop if perfect
        if best_correct == len(labels):
            break

    return best_sigma, best_correct


def random_search_pm1(vectors_pm1: List[List[int]], labels: List[int],
                      num_trials: int = 1000, verbose: bool = False) -> Tuple[List[int], int]:
    """
    Random search for good σ ∈ {-1, +1}^n with local optimization.

    Generate random sigma vectors and apply greedy optimization.

    Args:
        vectors_pm1: List of {-1,+1} vectors.
        labels: List of labels.
        num_trials: Number of random starting points.
        verbose: Print progress.

    Returns:
        Best sigma found and its accuracy (count of correct).
    """
    n = len(vectors_pm1[0]) if vectors_pm1 else TOTAL_BITS
    best_sigma = [1] * n
    best_correct = 0

    for trial in range(num_trials):
        # Generate random sigma
        sigma = [random.choice([-1, 1]) for _ in range(n)]

        # Local optimization
        sigma, correct = greedy_coordinate_descent(vectors_pm1, labels, sigma,
                                                    max_iter=100, verbose=False)

        if correct > best_correct:
            best_correct = correct
            best_sigma = sigma[:]
            if verbose:
                print(f"Trial {trial + 1}: new best {best_correct}/{len(labels)}")

            # Early stop if perfect
            if best_correct == len(labels):
                if verbose:
                    print(f"Found perfect solution at trial {trial + 1}")
                break

    return best_sigma, best_correct


def train(data_file: Path, output_file: Path, method: str = 'perceptron',
          max_epochs: int = 100, verbose: bool = True, quantize: bool = True) -> None:
    """
    Train the model by reading a JSON dataset and writing the sigma weights.

    The dataset should be a list of objects with fields "name" and "sex".
    The output is written as a JSON list of weights to the specified output file.

    Args:
        data_file: Path to the JSON file containing training data.
        output_file: Path where the weights will be saved.
        method: Training method - 'correlation', 'perceptron', or 'averaged'.
        max_epochs: Maximum epochs for perceptron methods.
        verbose: Whether to print training progress.
        quantize: Whether to quantize weights to {-1, +1} (default True).
    """
    # Load dataset
    with data_file.open('r', encoding='utf-8') as f:
        data = json.load(f)

    # Prepare vectors and labels
    vectors = []
    labels = []
    for entry in data:
        name = entry.get('name', '')
        sex = entry.get('sex', '').strip().lower()
        vec = encode_name(name)
        vectors.append(vec)
        if sex == 'male':
            labels.append(1)
        elif sex == 'female':
            labels.append(-1)
        else:
            raise ValueError(f"Unknown sex label: {sex} in entry: {entry}")

    # Convert to {-1, +1} for perceptron training
    vectors_pm1 = [to_pm1(v) for v in vectors]

    if verbose:
        print(f"Loaded {len(vectors)} training examples")
        print(f"  Males: {labels.count(1)}, Females: {labels.count(-1)}")
        print(f"  Vector dimension: {TOTAL_BITS} bits ({NAME_LENGTH} chars × {BITS_PER_CHAR} bits)")
        print(f"  Training method: {method}")
        print(f"  Quantize to {{-1,+1}}: {quantize}")
        print()

    # Train based on method
    if method == 'correlation':
        sigma = compute_sigma_correlation(vectors, labels)
        w_float = [float(s) for s in sigma]  # Already {-1,+1}
    elif method == 'perceptron':
        w_float, mistakes = perceptron_train(vectors_pm1, labels, max_epochs=max_epochs, verbose=verbose)
    elif method == 'averaged':
        w_float = averaged_perceptron_train(vectors_pm1, labels, max_epochs=max_epochs, verbose=verbose)
    elif method == 'search':
        # Start with correlation solution, then optimize in {-1,+1}
        init_sigma = compute_sigma_correlation(vectors, labels)
        if verbose:
            init_correct, _, _ = compute_accuracy(vectors_pm1, labels, init_sigma)
            print(f"Initial (correlation): {init_correct}/{len(labels)}")

        # Also try perceptron initialization
        w_float, _ = perceptron_train(vectors_pm1, labels, max_epochs=100, verbose=False)
        perceptron_sigma = quantize_to_pm1(w_float)
        perceptron_correct, _, _ = compute_accuracy(vectors_pm1, labels, perceptron_sigma)
        if verbose:
            print(f"Initial (perceptron):  {perceptron_correct}/{len(labels)}")

        # Use the better starting point
        if perceptron_correct > init_correct:
            init_sigma = perceptron_sigma
            init_correct = perceptron_correct

        if verbose:
            print("Optimizing σ ∈ {-1,+1}^120 via coordinate descent...")

        sigma, best_correct = greedy_coordinate_descent(vectors_pm1, labels, init_sigma,
                                                         max_iter=TOTAL_BITS * 2, verbose=verbose)
        w_float = [float(s) for s in sigma]
        quantize = True  # Force quantize since we're already in {-1,+1}
    elif method == 'random':
        # Random search with local optimization
        if verbose:
            print(f"Random search with {max_epochs} trials...")
        sigma, best_correct = random_search_pm1(vectors_pm1, labels, num_trials=max_epochs, verbose=verbose)
        w_float = [float(s) for s in sigma]
        quantize = True
    elif method == 'annealing':
        # Simulated annealing
        if verbose:
            print(f"Simulated annealing ({max_epochs * 100} iterations)...")
        # Start from correlation-based sigma
        init_sigma = compute_sigma_correlation(vectors, labels)
        sigma, best_correct = simulated_annealing_pm1(vectors_pm1, labels, init_sigma,
                                                       max_iter=max_epochs * 100, verbose=verbose)
        # Polish with greedy descent
        sigma, best_correct = greedy_coordinate_descent(vectors_pm1, labels, sigma,
                                                         max_iter=TOTAL_BITS, verbose=False)
        w_float = [float(s) for s in sigma]
        quantize = True
    else:
        raise ValueError(f"Unknown method: {method}. Use 'correlation', 'perceptron', 'averaged', 'search', 'random', or 'annealing'.")

    # Optionally quantize
    if quantize:
        sigma = quantize_to_pm1(w_float)
        weights_for_eval = sigma
    else:
        sigma = w_float
        weights_for_eval = w_float

    # Evaluate final accuracy with the weights being saved
    correct, total, acc = compute_accuracy(vectors_pm1, labels, weights_for_eval)
    if verbose:
        print()
        print(f"Final model accuracy: {correct}/{total} = {acc:.4f} ({acc*100:.2f}%)")
        if quantize:
            print(f"  Weight vector σ ∈ {{-1, +1}}^{TOTAL_BITS}")
            num_pos = sum(1 for s in sigma if s == 1)
            num_neg = sum(1 for s in sigma if s == -1)
            print(f"  Positive weights (+1): {num_pos}, Negative weights (-1): {num_neg}")
        else:
            print(f"  Weight vector w ∈ ℝ^{TOTAL_BITS} (real-valued)")
            max_w = max(abs(x) for x in w_float) if w_float else 0
            print(f"  Max |w_i|: {max_w:.2f}")

    # Save to JSON file
    with output_file.open('w', encoding='utf-8') as f:
        json.dump(sigma, f)

    if verbose:
        print(f"\nModel saved to {output_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Train a sex classifier from Japanese names using perceptron.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python train_model.py -i data.json -o model.json --method perceptron --epochs 100
  python train_model.py -i data.json -o model.json --method correlation
  python train_model.py -i data.json -o model.json --method averaged
  python train_model.py -i data.json -o model.json --method perceptron --no-quantize

Methods:
  correlation : Simple sign of correlation between features and labels (fast)
  perceptron  : Standard perceptron algorithm (iterative, may converge)
  averaged    : Averaged perceptron for better generalization
  search      : Perceptron + greedy coordinate descent for best σ ∈ {-1,+1}^120
  random      : Random search with local optimization
  annealing   : Simulated annealing for finding good σ ∈ {-1,+1}^120

The model learns a weight vector σ ∈ {-1, +1}^120 (or w ∈ ℝ^120 if --no-quantize) such that:
  sign((2g - 1) · σ) ≈ label
where g ∈ {0,1}^120 is the binary encoding of the name.
"""
    )
    parser.add_argument('-i', '--input', type=str, default='data.json',
                        help='Path to the JSON dataset file (default: data.json)')
    parser.add_argument('-o', '--output', type=str, default='model.json',
                        help='Path to write the sigma weights (default: model.json)')
    parser.add_argument('-m', '--method', type=str, default='perceptron',
                        choices=['correlation', 'perceptron', 'averaged', 'search', 'random', 'annealing'],
                        help='Training method (default: perceptron)')
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='Maximum epochs for perceptron training (default: 100)')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress verbose output')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--no-quantize', action='store_true',
                        help='Keep real-valued weights instead of quantizing to {-1,+1}')

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    data_path = Path(args.input)
    out_path = Path(args.output)

    if not data_path.exists():
        # Also try data.json if data.jason not found (typo handling)
        alt = data_path.with_suffix('.json')
        if alt.exists():
            data_path = alt
        else:
            raise FileNotFoundError(f"Dataset file {args.input} not found.")

    train(data_path, out_path, method=args.method, max_epochs=args.epochs,
          verbose=not args.quiet, quantize=not args.no_quantize)


if __name__ == '__main__':
    main()