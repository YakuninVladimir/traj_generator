from __future__ import annotations

"""
Backwards-compatible wrapper.

The new, config-driven generator is implemented in `scripts/generator_cli.py`.
"""

from generator_cli import main


if __name__ == "__main__":
    main()

