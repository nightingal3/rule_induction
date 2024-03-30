#!/bin/bash
#SBATCH --job-name=llama2_remaining
#SBATCH --output=llama2_remaining_induced_grammar_%A.out
#SBATCH --error=llama2_remaining_induced_grammar_%A.err
#SBATCH --time=1-00:00:00
#SBATCH --mem=10G

# Load the required modules (if any)
export OPENAI_API_KEY=XXXXXX
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tgi-env
export PYTHONPATH=/projects/tir5/users/mengyan3/rule_induction/

# Read the jobs.txt file into an array
mapfile -t commands < /data/tir/projects/tir5/users/mengyan3/rule_induction/slurm_scripts/jobs/remaining_jobs_os.txt

# Run each job one by one
for command in "${commands[@]}"; do
    echo "Executing \"$command\""
  eval "$command"
done