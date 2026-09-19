"""评估流程入口：分发到评估/分析类任务。

用法：``python evaluate.py [task ...]``，``task`` 取 ``benchmark`` / ``transfer`` /
``generation`` / ``twostage``；缺省运行全部。
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
    parser = argparse.ArgumentParser(description="评估流程入口")
    parser.add_argument("tasks", nargs="*", choices=sorted(EVAL_TASKS),
                        help="要运行的评估任务；缺省运行全部")
    args = parser.parse_args()
    tasks = args.tasks or list(EVAL_TASKS)
    for task in tasks:
        print(f"\n===== {task} =====", flush=True)
        EVAL_TASKS[task]()


if __name__ == "__main__":
    main()
