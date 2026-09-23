"""Unified training entrypoint: dispatches to each training pipeline by task.

Usage: ``python train.py [task ...]``, where ``task`` is one of ``benchmark`` /
``transfer`` / ``generation`` / ``twostage``; all tasks run by default.
"""
import argparse

from analysis import generation, generation_twostage
from benchmarks import transfer
from benchmarks import run as benchmark_run

TASKS = {
    "benchmark": benchmark_run.run,
    "transfer": transfer.run,
    "generation": generation.run,
    "twostage": generation_twostage.run,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified training entrypoint")
    parser.add_argument("tasks", nargs="*", choices=sorted(TASKS),
                        help="training tasks to run; all tasks run by default")
    args = parser.parse_args()
    tasks = args.tasks or list(TASKS)
    for task in tasks:
        print(f"\n===== {task} =====", flush=True)
        TASKS[task]()


if __name__ == "__main__":
    main()
