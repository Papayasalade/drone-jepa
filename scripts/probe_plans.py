"""Thin shim — the plan probe lives in the package now.

  .venv/bin/python -m drone_jepa.eval.probe stem1 stem2 ...
"""
import sys

sys.path.insert(0, ".")
from drone_jepa.eval.probe import main  # noqa: E402

if __name__ == "__main__":
    main()
