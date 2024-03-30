#!/bin/bash
#SBATCH --job-name=llama-synthetic_rest
#SBATCH --output=llama-synthetic_rest_%A-%a.out
#SBATCH --time=7-00:00:00
#SBATCH --mem=10G
#SBATCH --array=0-25%6
#SBATCH --mail-user=emmy@cmu.edu
#SBATCH --mail-type=END
#SBATCH --partition=long

# Load the required modules (if any)
input_file=${1:-"slurm_scripts/jobs/jobs_synthetic_final_run_0_colours_gtruth_excluded.txt"}
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tgi-env

export PYTHONPATH=/data/tir/projects/tir5/users/mengyan3/rule_induction/
#export OPENAI_API_KEY=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
export OPENAI_API_KEY=XXXXXX

# Read the jobs.txt file into an array
mapfile -t commands < $input_file

# Run the job corresponding to this array index
eval "${commands[$SLURM_ARRAY_TASK_ID]}"