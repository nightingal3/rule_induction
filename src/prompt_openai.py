import argparse
import openai
import os
from tqdm import tqdm

from .task import get_task

def prompt_model(task, model_name, prompt_type, temp):
    correct = 0
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt OpenAI models with task specs")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", choices=["gpt-4", "gpt-3.5-turbo", "text-davinci-003"], help="OpenAI model to use")
    parser.add_argument("--temp", default=0.0, type=float, help="Temperature for sampling")
    parser.add_argument("--dataset", default="scan_simple", choices=["scan_simple", "scan_length", "scan_jump"])
    parser.add_argument("--prompt_type", default="base", choices=["base", "full_grammar", "grammar_induction"])
    parser.add_argument("--output", type=str)

    args = parser.parse_args()

    openai.api_key = os.environ["OPENAI_API_KEY"]
    try:
        task = get_task(args.dataset)
    except ValueError:
        raise ValueError(f"Dataset {args.dataset} not registered")
    


    

