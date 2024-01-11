from src.task import BaseTask
import importlib
import os

TASK_DIR = "src/tasks"  # will make a dir called 'tasks' when I add more tasks
GPT_4_VERSION = "gpt-4-0613"
GPT_3_VERSION = "gpt3-0613"


def get_task(task_name: str):
    task_to_classname = {
        "scan": "ScanTask",
        "cogs": "CogsTask",
        "colours": "ColoursTask",
        "cherokee": "CherokeeTask",
        "arc": "ArcTask",
        "naclo": "NacloTask",
        "functions": "FunctionsTask",
    }
    task_cls = BaseTask.tasks.get(task_to_classname[task_name])
    if task_cls is None:
        raise ValueError(f"Task {task_name} not registered")

    return task_cls


def import_tasks(namespace: str = "src.tasks"):
    for file in os.listdir(TASK_DIR):
        if file.endswith(".py") and not file.startswith("__"):
            print(f"Importing {file}")
            importlib.import_module(f"{namespace}.{file[:-3]}")


import_tasks()
