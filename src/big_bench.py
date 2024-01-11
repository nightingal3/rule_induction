import datasets

relevant_tasks = [
    "formal_fallacies_syllogisms_negation",
    "logical_fallacy_detection",
]
class TaskFactory:
    def create_task(self, task_type, data):
        raise NotImplementedError


if __name__ == "__main__":
    for task_name in relevant_tasks:
        task = datasets.load_dataset("bigbench", task_name)
        import pdb; pdb.set_trace()