class BaseTask:
    tasks = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        BaseTask.tasks.append(cls)

    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
    
    def get_input(self, idx: int) -> str:
        raise NotImplementedError
    
    def validate(self, idx: int, output: str) -> bool:
        raise NotImplementedError
    

def get_task(task_name: str):
    task_cls = BaseTask.tasks.get(task_name)
    if task_cls is None:
        raise ValueError(f"Task {task_name} not found")
    
    return task_cls()