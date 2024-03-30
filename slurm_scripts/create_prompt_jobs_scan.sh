#!/bin/bash
set -e 
output_folder="./slurm_scripts/jobs"
mkdir -p $output_folder

base_script="src/prompt_open_llms.py"
prompt_types=("base" "full_grammar" "grammar_induction")
models=("gpt-3.5-turbo" "gpt-4")
sep="\t"
for model in ${models[@]}; do
    for prompt_type in ${prompt_types[@]}; do
        for sind in {0..900..100}; do
            eind=$((sind+100))
            echo "python3 $base_script --use_min_cover --start_ind $sind --end_ind $eind --prompt_type $prompt_type --model $model --dataset scan --no_few_shot_examples" >> $output_folder/jobs_${model}_${prompt_type}_no_icl.txt
        done
    done
done

