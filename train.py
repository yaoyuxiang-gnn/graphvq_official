"""统一训练入口：按任务分发到各训练流程。

用法：``python train.py [task ...]``，``task`` 取 ``benchmark`` / ``transfer`` /
``generation`` / ``twostage``；缺省运行全部。
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
    parser = argparse.ArgumentParser(description="统一训练入口")
    parser.add_argument("tasks", nargs="*", choices=sorted(TASKS),
                        help="要运行的训练任务；缺省运行全部")
    args = parser.parse_args()
    tasks = args.tasks or list(TASKS)
    for task in tasks:
        print(f"\n===== {task} =====", flush=True)
        TASKS[task]()


if __name__ == "__main__":
    main()
