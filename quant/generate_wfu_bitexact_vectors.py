"""Generate deterministic WeightFinalizeUnit bit-exact verification vectors."""

import argparse
import random
from pathlib import Path

try:
    from .kdpoint_geometry import wfu_weights_exact
except ImportError:
    from kdpoint_geometry import wfu_weights_exact


DIRECTED_VECTORS = (
    (100, 100, 100),
    (4, 16, 64),
    (0, 5, 9),
    (5, 0, 9),
    (5, 9, 0),
    (0, 0, 9),
    (5, 0, 0),
    (0, 0, 0),
    (1, 2, 3),
    (1, 1, 1),
    (1, 1, 195075),
    (1, 195074, 195075),
    (195075, 195075, 195075),
)


def generate_vectors(random_count: int, seed: int):
    if random_count < 0:
        raise ValueError("random_count must be nonnegative")
    rng = random.Random(seed)
    distances = list(DIRECTED_VECTORS)
    for _ in range(random_count):
        distances.append(tuple(sorted(rng.randint(1, 195075) for _ in range(3))))
    return [values + wfu_weights_exact(values) for values in distances]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--random-count", type=int, default=1000)
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=0x4B445057)
    args = parser.parse_args()

    vectors = generate_vectors(args.random_count, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="ascii", newline="\n") as stream:
        for vector in vectors:
            stream.write(" ".join(str(value) for value in vector) + "\n")
    print(f"wrote {len(vectors)} WFU vectors to {args.output}")


if __name__ == "__main__":
    main()
