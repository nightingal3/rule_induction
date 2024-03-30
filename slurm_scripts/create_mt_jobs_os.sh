#!/bin/bash
set -e 
output_folder="./slurm_scripts/jobs"
mkdir -p $output_folder

base_script="main.py"
jobs_filename="/data/tir/projects/tir5/users/mengyan3/rule_induction/mtob_runs_os"
prompt_types=("base" "full_grammar" "grammar_induction" "zs-cot")
models=("meta-llama/Llama-2-70b-chat-hf" "meta-llama/Llama-2-7b-chat-hf")
temps=("0.05")
run_num=0
directions=("ek" "ke")

sep="\t"

# Delete the existing output file if it exists
if [ -f "$output_folder/${jobs_filename}_${run_num}.txt" ]; then
    rm "$output_folder/${jobs_filename}_${run_num}.txt"
fi

for direction in ${directions[@]}; do
    for model in ${models[@]}; do
        for prompt_type in ${prompt_types[@]}; do
            for temp in ${temps[@]}; do
                if [ $prompt_type == "grammar_induction" ]; then
                    rerank_by=("p_data_given_hyp_guess" "p_data_given_hyp_logprobs" "p_answer_given_hyp_logprobs")
                    for method in ${rerank_by[@]}; do
                        out_path="/data/tir/projects/tir5/users/mengyan3/rule_induction/mtob/baselines/logs/kalamang/${direction}/${model}/${prompt_type}/${method}/temp_${temp}/n_hyps_$nhyps/$run_num"
                        nhyps=5
                        
                        if ls "${out_path}"* 1> /dev/null 2>&1; then
                            echo "Skipping $out_path since it's already done"
                            continue
                        fi

                        script_args="--model_type tgi-env --use_reference_sentences --induce_wordlist --use_induced_grammar --grammar_sketch_path ../resources/kalamang_grammar_sketch_${model}.txt  --n_hyps 5 --rerank_hyps_method ${method}"

                        echo "python $base_script $script_args --model_name $model --output_dir $out_path --temperature $temp --direction ${direction}" >> $output_folder/${jobs_filename}_${run_num}.txt
                    done
                else
                    out_path="/data/tir/projects/tir5/users/mengyan3/rule_induction/mtob/baselines/logs/kalamang/${direction}/${model}/${prompt_type}/temp_${temp}/$run_num"
                
                    if ls "${out_path}"* 1> /dev/null 2>&1; then
                        echo "Skipping $out_path since it's already done"
                        continue
                    fi

                    if [ $prompt_type == "base" ]; then
                        script_args="--use_reference_sentences"
                    elif [ $prompt_type == "full_grammar" ]; then
                        script_args="--use_reference_sentences --use_reference_wordlist --use_reference_grammar_sketch"
                    elif [ $prompt_type == "zs-cot" ]; then
                        script_args="--use_reference_sentences --use_zs_cot"
                    fi

                    echo "python $base_script $script_args  --model_type tgi-env --model_name $model --output_dir $out_path --temperature $temp --direction $direction" >> $output_folder/${jobs_filename}_${run_num}.txt
                fi
                
            done
        done
    done

done

echo "Created jobs file at $output_folder/${jobs_filename}_${run_num}.txt"