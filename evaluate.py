"""Evaluation pipeline entrypoint: dispatches to evaluation/analysis tasks.

Usage: ``python evaluate.py [task ...]``, where ``task`` is one of ``benchmark`` /
``transfer`` / ``generation`` / ``twostage``; all tasks run by default.
"""
import argparse

from analysis import generation, generation_twostage
from benchmarks import transfer
from benchmarks import run as benchmark_run

EVAL_TASKS = {
    "benchmark": benchmark_run.run,
    "transfer": transfer.run,
    "generation": generation.run,
    "twostage": generation_twostage.run,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation pipeline entrypoint")
    parser.add_argument("tasks", nargs="*", choices=sorted(EVAL_TASKS),
                        help="evaluation tasks to run; all tasks run by default")
    args = parser.parse_args()
    tasks = args.tasks or list(EVAL_TASKS)
    for task in tasks:
        print(f"\n===== {task} =====", flush=True)
        EVAL_TASKS[task]()


if __name__ == "__main__":
    main()
