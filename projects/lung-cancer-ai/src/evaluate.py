from __future__ import annotations

from src.train import train


def evaluate_saved_model() -> dict[str, object]:
    return train()


if __name__ == "__main__":
    print(evaluate_saved_model())

