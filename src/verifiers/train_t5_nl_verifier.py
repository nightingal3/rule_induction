import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from datasets import load_dataset
import pandas as pd
from src.utils.utils import find_examples_with_pattern, find_examples_with_all_patterns
from src.tasks.scan_task import ScanTask
from typing import List
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

def corrupt_examples(examples: List, pattern: str) -> pd.DataFrame:
    # randomly corrupt the examples to produce negative examples
    other_actions = ["I_WALK", "I_LOOK", "I_RUN", "I_TURN_LEFT", "I_TURN_RIGHT"]
    new_examples = []
    for example in examples:
        example_split = example.split(" ")
        if len(example_split) == 0 or pattern not in example_split:
            continue
        pattern_ind = example_split.index(pattern)
        example_split[pattern_ind] = other_actions[np.random.randint(0, len(other_actions))]
        example = " ".join(example_split)
        new_examples.append(example)
    return new_examples

def get_acc(preds: List, actual: List) -> float:
    return sum([1 if preds[i] == actual[i] else 0 for i in range(len(preds))]) / len(preds)

# Define the device
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
# Load the tokenizer and the model
# TODO: change this to t5-base or large after
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")
model.to(device)


rule_str = "Rule: [[jump]] -> I_JUMP"
data_df = pd.read_csv("./data/scan/scan_simple_train.csv")
data_df_test = pd.read_csv("./data/scan/scan_simple_test.csv")
examples_with_jump = find_examples_with_pattern(data_df, "jump")
test_examples_with_jump = find_examples_with_pattern(data_df_test, "jump")

scan_task_dummy = ScanTask("./data/scan/scan_simple_train.csv", "./data/scan/scan_simple_test.csv", prompt_style="base")
scan_inputs = scan_task_dummy.few_shot_examples_wrap(list(examples_with_jump["commands"]), list(examples_with_jump["actions"]))
scan_inputs = scan_inputs.split("\n\n")
scan_test_inputs = scan_task_dummy.few_shot_examples_wrap(list(test_examples_with_jump["commands"]), list(test_examples_with_jump["actions"]))
scan_test_inputs = scan_test_inputs.split("\n\n")

data = [
    ("Rule: [[jump]] -> I_JUMP", inp, "True") for inp in scan_inputs[:1000]
]
corrupted_data = [
    ("Rule: [[jump]] -> I_JUMP", inp, "False") for inp in corrupt_examples(scan_inputs[1000:2000], "I_JUMP")
]
train_data = data + corrupted_data

test_data = [   
    ("Rule: [[jump]] -> I_JUMP", inp, "True") for inp in scan_test_inputs[:1000]
]
test_data_corrupted = [
    ("Rule: [[jump]] -> I_JUMP", inp, "False") for inp in corrupt_examples(scan_test_inputs[1000:2000], "I_JUMP")
]
test_data = test_data + test_data_corrupted

#train_dataset = load_dataset("json", data_files="./data/cognac/wordnet/train.jsonl")
# Define a Dataset
class StatementDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        premise, hypothesis, label = self.data[idx]
        encoding = self.tokenizer(f"verify that the rule is followed every time in the example: {premise}\nExample: {hypothesis}", return_tensors="pt", max_length=self.max_length, padding="max_length", truncation=True)
        labels = torch.tensor([1 if label == "True" else 0])
        return encoding.input_ids.squeeze(), labels

# Create a DataLoader
dataset = StatementDataset(data, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Define an optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
model.train()
for epoch in range(1):  # for example, train for 10 epochs
    for i, (input_ids, labels) in enumerate(tqdm(dataloader)):
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, labels=labels)
        logits = outputs.logits

        # Convert labels to one-hot encoding
        batch_size = labels.size(0)
        vocab_size = logits.size(-1)

        # Get the positions of "0" and "1" tokens in the vocabulary
        zero_token_id = tokenizer.encode("0", add_special_tokens=False)[0]
        one_token_id = tokenizer.encode("1", add_special_tokens=False)[0]

        # Create the one-hot encoding based on token positions
        labels_one_hot = torch.zeros(batch_size, vocab_size).to(device)
        labels_one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        labels_one_hot[:, zero_token_id] = 1.0
        labels_one_hot[:, one_token_id] = 1.0

        # Calculate loss
        loss = F.binary_cross_entropy_with_logits(logits.view(-1, vocab_size), labels_one_hot.view(-1, vocab_size))

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 100 == 0:  # print loss every 100 steps
            print(f"Epoch: {epoch}, Step: {i}, Loss: {loss.item()}")

            # Validation on the test set
            model.eval()
            test_dataset = StatementDataset(test_data, tokenizer)
            test_dataloader = DataLoader(test_dataset, batch_size=8)
            acc_sum = 0
            total_batches = 0
            for j, (input_ids, labels) in enumerate(tqdm(test_dataloader)):
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                outputs = model(input_ids=input_ids, labels=labels)
                logits = outputs.logits
                predictions = logits.argmax(dim=-1)
                accuracy = (predictions == labels).float().mean()
                loss = F.binary_cross_entropy_with_logits(logits.view(-1, vocab_size), labels_one_hot.view(-1, vocab_size))
                acc_sum += accuracy
                total_batches += 1
            print(f"Test accuracy: {acc_sum / total_batches}")
            model.train()

# Try trained model on examples with jump
model.eval()
final_test_data = [(rule_str, inp, "True") for inp in scan_test_inputs[-10:]]
test_dataset = StatementDataset(final_test_data, tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=1)

for i, (input_ids, labels) in enumerate(tqdm(test_dataloader)):
    print(f"Input: {final_test_data[i][1]}")
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    outputs = model(input_ids=input_ids, labels=labels)
    output_text = tokenizer.decode(input_ids.squeeze())
    print(f"Output: {output_text}")
    logits = outputs.logits
    predictions = logits.argmax(dim=-1)
    print(f"Prediction: {predictions.item()}")
    print(f"Label: {labels.item()}")
    
