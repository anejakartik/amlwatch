"""Train the triage model and write the artifact the API loads.

    python -m amlwatch.train [path]
"""

from __future__ import annotations

import sys
from pathlib import Path

from .evaluate import train_triage

MODEL_PATH = Path(__file__).with_name("triage_model.json")


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else MODEL_PATH
    triage = train_triage()
    triage.save(path)
    print(f"wrote {path} ({triage.version}, margin {triage.margin:.2f})")


if __name__ == "__main__":
    main()
