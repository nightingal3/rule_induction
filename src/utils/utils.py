import pandas as pd
import subprocess
import re
from typing import List

def find_examples_with_pattern(data: pd.DataFrame, pattern: str) -> pd.DataFrame:
    return data[data["patterns"].apply(lambda x: pattern in x)]

def find_examples_with_all_patterns(data, patterns: List[str]) -> pd.DataFrame:
    return data[data["patterns"].apply(lambda x: set(patterns).issubset(set(x)))]

def get_job_info_by_name(job_name: str) -> str:
    cmd = "squeue --format=%j,%i,%N"
    proc = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, err = proc.communicate()

    if err:
        raise Exception(err)

    for line in output.decode('utf-8').split('\n'):
        if line.startswith(job_name + ','):
            _, job_id, node = line.split(',')
            return job_id, node

    print(f'Job {job_name} not found')
    return None
    