"""Summarize rotor_fly logs into a compact table.

    python scripts/summarize_rotor_fly.py artifacts/*/rotor_fly.log
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


SUMMARY = re.compile(
    r"== (?P<wins>\d+)/(?P<trials>\d+) WON, (?P<crashes>\d+) hard-crashed, "
    r"(?P<respawns>\d+) respawns, (?P<gates>[0-9.]+) gates/race, "
    r"(?P<flipped>[0-9.]+)% steps flipped =="
)


def experiment_name(path: Path) -> str:
    if path.parent.name == "artifacts":
        return path.name.removesuffix(".rotor_fly.log")
    return path.parent.name


def main() -> None:
    print(f"{'experiment':<42} {'wins':>7} {'gates':>7} {'crash':>7} {'resp':>7} {'flip%':>7}")
    for arg in sys.argv[1:]:
        path = Path(arg)
        name = experiment_name(path)
        if not path.exists():
            print(f"{name:<42} {'MISSING':>7}")
            continue
        text = path.read_text()
        match = SUMMARY.search(text)
        if not match:
            print(f"{name:<42} {'NO SUMMARY':>7}")
            continue
        d = match.groupdict()
        wins = f"{d['wins']}/{d['trials']}"
        print(
            f"{name:<42} {wins:>7} {d['gates']:>7} "
            f"{d['crashes']:>7} {d['respawns']:>7} {d['flipped']:>7}"
        )


if __name__ == "__main__":
    main()
