from __future__ import annotations

import argparse
import json
import sys

from .config import load_config
from .pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate prompts and trajectories with vLLM."
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to config.json (default: ./config.json)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    result = run_pipeline(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Fatal error: {exc}", file=sys.stderr)
        raise
