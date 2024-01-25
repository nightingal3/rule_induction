from src.task import BaseTask
import importlib
import os
from src.tasks.scan_task import ScanTask
from src.tasks.cogs_task import CogsTask
from src.tasks.colours_task import ColoursTask
from src.tasks.cherokee_task import CherokeeTask
from src.tasks.arc_task import ArcTask
from src.tasks.naclo_task import NacloTask
from src.tasks.functions_task import FunctionsTask


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
