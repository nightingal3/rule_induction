import argparse
import jsonlines
import csv

def extract_field(field: str, jsonl_file: str, csv_filename: str):
    with open(jsonl_file, 'r', encoding='utf-8') as input_file, open(csv_filename, 'w', encoding='utf-8', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerow([field, 'annotation'])
        
        for item in jsonlines.Reader(input_file):
            field1_value = item.get(field, '')
            model_value = item.get('model', '')
            annotation_value = ''
            writer.writerow([field1_value, model_value, annotation_value])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["scan", "colours"], required=True)
    parser.parse_args()
    args = parser.parse_args()
    # Usage example
    jsonl_file_path = f"./data/{args.dataset}/gpt_4_induced_grammars.jsonl"
    csv_file_path = f"./data/{args.dataset}/grammars_for_annotation.csv"

    extract_field("grammar", jsonl_file_path, csv_file_path)