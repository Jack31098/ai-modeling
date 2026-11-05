"""Finish the 28-case comparison, then the predeclared 72-template expansion."""
import json

from .run import BASE, frozen_protocol, run_model


def main():
    frozen_protocol()
    models = json.loads((BASE / "assets.lock.json").read_text(encoding="utf-8"))["models"]
    docs = [json.loads(line) for line in (BASE / "inputs/all100.jsonl").read_text(encoding="utf-8").splitlines()]
    for count in (28, 100):
        for model in models:
            run_model(model, docs[:count], 18471)


if __name__ == "__main__":
    main()
