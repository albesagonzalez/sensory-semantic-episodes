import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time

from pathlib import Path


DEFAULT_NOTEBOOKS = [
    "Figure_2.ipynb",
    "Figure_3.ipynb",
    "Figure_4.ipynb",
    "Figure_5.ipynb",
    "Figure_6A.ipynb",
    "Figure_6B.ipynb",
    "Figure_7_recall_blocked.ipynb",
    "Figure_7_anterograde.ipynb",
    "Figure_7_retrograde.ipynb",
    "parameter_sweeps.ipynb",
    "network_plotter.ipynb",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Execute repository notebooks in order using nbconvert.",
    )
    parser.add_argument(
        "notebooks",
        nargs="*",
        help="Optional explicit notebook list. Defaults to the repo execution order.",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/notebook-execution",
        help="Directory where executed notebooks and logs are written.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=21600,
        help="Per-cell timeout in seconds (default: 21600; -1 disables the limit).",
    )
    parser.add_argument(
        "--kernel-name",
        default=None,
        help="Use a registered kernel instead of the current Python environment.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running later notebooks after a failure.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the default execution order and exit.",
    )
    return parser.parse_args()


def build_command(notebook_path, output_dir, timeout, kernel_name=None):
    command = [
        sys.executable,
        "-m",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(notebook_path),
        "--output",
        f"{notebook_path.stem}.executed.ipynb",
        "--output-dir",
        str(output_dir),
        f"--ExecutePreprocessor.timeout={int(timeout)}",
    ]
    if kernel_name:
        command.append(f"--ExecutePreprocessor.kernel_name={kernel_name}")
    return command


def run_notebook(notebook_path, output_dir, timeout, kernel_name=None):
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="sensory-semantic-kernel-") as kernel_dir:
        env = os.environ.copy()
        if kernel_name is None:
            # A global/user python3 kernelspec may point outside this environment.
            # Register a private, temporary kernel without changing user settings.
            kernel_name = Path(kernel_dir).name
            spec_dir = Path(kernel_dir) / "kernels" / kernel_name
            spec_dir.mkdir(parents=True)
            (spec_dir / "kernel.json").write_text(json.dumps({
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "Sensory-semantic episodes (current environment)",
                "language": "python",
            }))
            env["JUPYTER_PATH"] = os.pathsep.join(
                path for path in [kernel_dir, env.get("JUPYTER_PATH")] if path
            )
        command = build_command(
            notebook_path=notebook_path,
            output_dir=output_dir,
            timeout=timeout,
            kernel_name=kernel_name,
        )
        completed = subprocess.run(
            command,
            cwd=notebook_path.parent,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    duration = time.time() - started
    log_path = output_dir / f"{notebook_path.stem}.log"
    log_path.write_text(completed.stdout)
    return {
        "notebook": notebook_path,
        "returncode": int(completed.returncode),
        "duration_seconds": float(duration),
        "log_path": log_path,
        "executed_notebook_path": output_dir / f"{notebook_path.stem}.executed.ipynb",
    }


def resolve_notebooks(root, notebook_args):
    if notebook_args:
        notebook_names = notebook_args
    else:
        notebook_names = DEFAULT_NOTEBOOKS

    notebook_paths = []
    for notebook_name in notebook_names:
        notebook_path = root / notebook_name
        if not notebook_path.exists():
            raise FileNotFoundError(f"Notebook not found: {notebook_name}")
        notebook_paths.append(notebook_path)
    return notebook_paths


def print_summary(results):
    print("\nSummary")
    for result in results:
        status = "ok" if result["returncode"] == 0 else "failed"
        print(
            f"- {result['notebook'].name}: {status} "
            f"({result['duration_seconds']:.1f}s) "
            f"log={result['log_path']}"
        )


def main():
    args = parse_args()

    if args.list:
        for notebook_name in DEFAULT_NOTEBOOKS:
            print(notebook_name)
        return 0

    required_modules = {
        "matplotlib": "matplotlib", "numpy": "numpy", "pandas": "pandas",
        "scipy": "scipy", "sklearn": "scikit-learn", "seaborn": "seaborn",
        "torch": "torch", "nbconvert": "nbconvert", "ipykernel": "ipykernel",
    }
    missing = [package for module, package in required_modules.items()
               if importlib.util.find_spec(module) is None]
    if missing:
        print("Missing dependencies in this Python environment: " + ", ".join(missing), file=sys.stderr)
        print("Install them with: python -m pip install -r requirements.txt", file=sys.stderr)
        return 2

    root = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        notebook_paths = resolve_notebooks(root=root, notebook_args=args.notebooks)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    (root / "Data" / "checkpoints").mkdir(parents=True, exist_ok=True)
    (root / "Figures" / "SVG").mkdir(parents=True, exist_ok=True)

    results = []
    for notebook_path in notebook_paths:
        print(f"Running {notebook_path.name}...")
        result = run_notebook(
            notebook_path=notebook_path,
            output_dir=output_dir,
            timeout=args.timeout,
            kernel_name=args.kernel_name,
        )
        results.append(result)
        if result["returncode"] != 0:
            print(
                f"Failed: {notebook_path.name}. See {result['log_path']}",
                file=sys.stderr,
            )
            if not args.continue_on_error:
                print_summary(results)
                return result["returncode"]

    print_summary(results)
    return 1 if any(result["returncode"] != 0 for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
