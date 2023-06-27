class BaseTask:
    tasks = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        BaseTask.tasks[cls.__name__] = cls

    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
    
    def get_input(self, idx: int) -> str:
        raise NotImplementedError
    
    def validate(self, idx: int, output: str) -> bool:
        raise NotImplementedError
