from .task import BaseTask

def get_task(task_name: str):
    task_cls = BaseTask.tasks.get(task_name)
    if task_cls is None:
        raise ValueError(f"Task {task_name} not registered")
    
    return task_cls()