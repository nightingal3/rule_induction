import json
import pandas as pd
import evaluate

from src.utils.parsing_utils import parse_production_rule

direction = "ek"
model = "meta-llama/Llama-2-70b-chat-hf"
df_hyps_path = f"/data/tir/projects/tir5/users/mengyan3/rule_induction/mtob/baselines/logs/kalamang/{direction}/{model}/grammar_induction/p_data_given_hyp_logprobs/temp_0.05/n_hyps_5/0/results_test_tgi-env_Llama-2-70b-chat-hf_temp_0.05_reference_sentences_2_induced_5_p_data_given_hyp_logprobs_induced_grammar_p_data_given_hyp_logprobs_HYPSFILE.csv"
df_hyps = pd.read_csv(df_hyps_path)
base_path = f"/data/tir/projects/tir5/users/mengyan3/rule_induction/mtob/baselines/logs/kalamang/{direction}/{model}/base/0/results_test_"

with open(
    "/data/tir/projects/tir5/users/mengyan3/rule_induction/mtob/resources/wordlist.json",
    "r",
) as f:
    wordlist = json.load(f)

wordlist = wordlist[direction]
exclude_morphology_settings = [False]

chrf_metric = evaluate.load("chrf")
num_correct = 0
total_chrf_score = 0
total = 0
pred_is_correct = []
for exclude_morphology in exclude_morphology_settings:
    print(f"Exclude morphology: {exclude_morphology}")
    for row in df_hyps.to_dict(orient="records"):
        hyp = row["hypothesis"]
        try:
            parsed_rule = parse_production_rule(hyp)
            word = list(parsed_rule.keys())[0].lower()
            pred = list(parsed_rule.values())[0].lower()
            true_trans = wordlist.get(word, None)
            if true_trans is None:
                continue

            if direction == "ke":
                true_trans = true_trans[1]  # first is pos for ke

            true_trans_lst = true_trans.split(";")

            if exclude_morphology:
                if len(true_trans_lst) > 1:
                    continue
                special_chars = ["-", "=", "*"]
                if any([x in pred for x in special_chars]):
                    continue

            true_trans_lst = [
                x.replace("=", "").replace("-", "").replace("*", "").strip()
                for x in true_trans_lst
            ]
            if pred in true_trans_lst:
                num_correct += 1
                if row["rank"] == 0:
                    pred_is_correct.append(True)
            else:
                if row["rank"] == 0:
                    pred_is_correct.append(False)

            scores = [
                chrf_metric.compute(predictions=[pred], references=[[t]])["score"]
                for t in true_trans_lst
            ]
            best_score = max(scores) if scores else 0

            total_chrf_score += best_score

            total += 1
        except:
            # non-parseable is usually wrong
            total += 1
            continue

    print(f"Accuracy: {num_correct / total}")
    print(f"CHRF: {total_chrf_score / total}")
