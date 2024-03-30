#!/bin/bash
set -e 
output_folder="./slurm_scripts/jobs"
mkdir -p $output_folder

base_script="src/prompt_open_llms.py"
jobs_filename="jobs_synthetic_os_try"
domains=("functions" "colours")
splits=("simple" "miniscan")
prompt_types=("base" "full_grammar" "grammar_induction" "zs-cot")
models=("meta-llama/Llama-2-70b-chat-hf" "meta-llama/Llama-2-7b-chat-hf")
temps=("0.1" "1.0")
run_num=2

sep="\t"

# Delete the existing output file if it exists
if [ -f "$output_folder/${jobs_filename}_${run_num}.txt" ]; then
    rm "$output_folder/${jobs_filename}_${run_num}.txt"
fi

# Delete the existing output file if it exists
if [ -f "$output_folder/${jobs_filename}_${run_num}.txt" ]; then
    rm "$output_folder/${jobs_filename}_${run_num}.txt"
fi

for domain in ${domains[@]}; do
    if [ "$domain" == "colours" ]; then
        script="${base_script} --use_min_cover"
    else
        script=$base_script
    fi
    for model in ${models[@]}; do
        for prompt_type in ${prompt_types[@]}; do
            if [ "$prompt_type" == "grammar_induction" ]; then
                rerank_by=("ground_truth" "p_data_given_hyp_guess" "p_answer_given_hyp_logprobs" "p_data_given_hyp_logprobs")
                for method in ${rerank_by[@]}; do
                    for temp in ${temps[@]}; do
                        nhyps=5 # vary this as well
                        out_path="logs/${domain}/${model}/${prompt_type}/${method}/temp_${temp}/n_hyps_$nhyps/$run_num.csv"
                        out_path_prefix="logs/${domain}/${model}/${prompt_type}/${method}/temp_${temp}/n_hyps_$nhyps/$run_num"

                        if ls "${out_path_prefix}"* 1> /dev/null 2>&1; then
                            echo "Skipping $out_path since it's already done"
                            continue
                        fi
                        echo "python3 $script --temp ${temp} --prompt_type $prompt_type --model $model --dataset ${domain} --hyp_reranking_method $method --num_hyps $nhyps --output $out_path" >> $output_folder/${jobs_filename}_${run_num}.txt
                    done
                done
            
            else
                for temp in ${temps[@]}; do
                        out_path="logs/${domain}/${model}/${prompt_type}/temp_${temp}/$run_num.csv"
                        out_path_prefix="logs/${domain}/${model}/${prompt_type}/temp_${temp}/$run_num"
                        
                        if ls "${out_path_prefix}"* 1> /dev/null 2>&1; then
                            echo "Skipping $out_path since it's already done"
                            continue
                        fi

                        echo "python3 $script --temp ${temp} --prompt_type $prompt_type --model $model --dataset ${domain} --output $out_path" >> $output_folder/${jobs_filename}_${run_num}.txt
                done
            fi
        done
    done
done

echo "Output commands to $output_folder/${jobs_filename}_${run_num}.txt"