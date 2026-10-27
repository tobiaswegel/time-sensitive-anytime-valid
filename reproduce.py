"""Reproduce all six paper figures with the original settings and seeds."""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

FIGURES = {
    "1": [
        "figure_1.py",
        "--horizon",
        "150",
        "--mu0",
        "0",
        "--mu1",
        "0.6",
        "--sigma",
        "1",
        "--no-linear",
        "--max-t-plot",
        "50",
        "--time-scale",
        "30",
        "--exp-time-scale",
        "15",
        "--beta",
        "5",
        "--cdf-mode",
        "normal",
        "--color-limits",
        "panel",
        "--quantity",
        "lambda",
        "--seed",
        "0",
        "--no-plot",
        "--output",
        "figure_1.png",
        "--results-path",
        "figure_1.json",
    ],
    "2": ["figure_2.py"],
    "3": ["figure_3.py"],
    "4": ["figure_4.py"],
    "5": [
        "figure_5.py",
        "--seed",
        "0",
        "--no-plot",
        "--output",
        "figure_5.png",
        "--results-path",
        "figure_5.json",
    ],
    "6": [
        "figure_6.py",
        "--seed",
        "0",
        "--no-show",
        "--output",
        "figure_6.png",
    ],
}


def main():
    """Run selected figures headlessly and record commands, versions, and runtimes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=FIGURES,
        default=list(FIGURES),
        help="Figure numbers to generate (default: all six).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Where to save figures, data, and logs (default: results).",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MPLBACKEND="Agg", MPLCONFIGDIR=str(output / ".matplotlib"))
    manifest = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: version(name) for name in ("numpy", "matplotlib", "pandas")},
        "figures": {},
    }
    for figure in args.figures:
        command = FIGURES[figure]
        print(f"Generating Figure {figure}...", flush=True)
        start = time.monotonic()
        log_path = output / f"figure_{figure}.log"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                [sys.executable, str(root / command[0]), *command[1:]],
                cwd=output,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        manifest["figures"][figure] = {
            "command": command,
            "seconds": round(time.monotonic() - start, 2),
            "exit_code": completed.returncode,
        }
        (output / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        if completed.returncode:
            raise SystemExit(f"Figure {figure} failed; see {log_path}")
        print(
            f"Figure {figure} saved ({manifest['figures'][figure]['seconds']} s).",
            flush=True,
        )
    print(f"Outputs: {output}")


if __name__ == "__main__":
    main()
