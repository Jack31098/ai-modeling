"""Show the model-input whitelist separately from currently consumable labels."""

import argparse
import json
from pathlib import Path

from .events import consume_labels, model_input
from .io_utils import read_jsonl


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--doc-id", required=True)
    parser.add_argument("--cutoff", type=int, required=True)
    parser.add_argument("--mode", choices=["READ_FULL", "READ_STREAM"], default="READ_STREAM")
    args = parser.parse_args()
    doc = next(d for d in read_jsonl(args.release / "documents.jsonl") if d["doc_id"] == args.doc_id)
    observations = read_jsonl(args.release / doc["observation_file"])
    labels = read_jsonl(args.release / doc["supervision_file"])
    print(json.dumps({"model_input": model_input(observations, args.cutoff), "loss_targets_separate_from_input": consume_labels(labels, args.cutoff, args.mode)}, ensure_ascii=False, indent=2))
