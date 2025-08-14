#!/bin/bash

# TODO: insert your tokens here!!!
export HF_TOKEN=
export OPENAI_API_KEY=

if [ -z "${HF_TOKEN}" ]; then
  echo "❌ Error: API keys not provided. Please insert them at the top of this script!"
  exit -1
fi

if [ -z "${OPENAI_API_KEY}" ]; then
  echo "❌ Error: API keys not provided. Please insert them at the top of this script!"
  exit -1
fi


get_time() {
    date -Iminutes
}

mkdir -p reproduction
mkdir -p loras
mkdir -p results
mkdir -p logs
source .venv/bin/activate
# initialize a git repo if not already in one
git init
git add .

# Step 0: data generation
python data_generation/gen_descriptive_data.py
cd data && python gen_bio_only_test.py 
cd ..
git diff 2>&1 | tee reproduction/data_generation_diff.txt

# Step 1: Few shot scan
if [ "$1" = "manual" ]; then
  echo "skipping experiment, just generating output."
else
  (unbuffer time python exp_few_shot_scan.py 2>&1) | tee -a "logs/log-few-shot-scan-$(get_time)".txt
fi
python plot_few_shot_scan.py
cp exp_few_shot_scan.pdf reproduction/figure2.pdf
cp exp_few_shot_scan_embedded.pdf reproduction/figure5_top.pdf

# Step 2: Lora Hyperparameter scan
if [ "$1" = "manual" ]; then
  echo "skipping experiment, just generating output."
else
  (unbuffer time python exp_lora_scan.py 2>&1) | tee -a "logs/log-lora-scan-$(get_time)".txt
fi
cp results/lora_scan/summary.csv reproduction/table2.csv
python find_best_lora.py 2>&1 | tee reproduction/table2_best_lora.txt

# Step 3: temperature scan
if [ "$1" = "manual" ]; then
  echo "skipping experiment, just generating output."
else
  (unbuffer time python exp_temp_scan.py 2>&1) | tee -a "logs/log-temp-scan-$(get_time)".txt
fi
python plot_comparison_table.py 2>&1 | tee reproduction/table1.txt
python plot_temp_scan.py
python plot_temp_scan_chatgpt.py
cp exp_temp_scan.pdf reproduction/figure3.pdf
cp exp_temp_scan_gpt.pdf reproduction/figure4.pdf
cp exp_temp_scan_extra.pdf reproduction/figure5_bottom.pdf

# Step 4: INDRA and Kinmodgpt comparison
if [ "$1" = "manual" ]; then
  echo "skipping experiment, just generating output."
else
  # apply patch to avoid differences between GPU architectures (even if not necessary)
  # apply only here as otherwise it drastically increases execution time
  patch -p1 TransformersAPI.py < precision_patch.txt
  (unbuffer time python exp_test_kinmodgpt.py 2>&1) | tee -a "logs/log-kinmodgpt-$(get_time)".txt
  patch -R -p1 TransformersAPI.py < precision_patch.txt
  (unbuffer time python exp_compare_indra.py 2>&1) | tee -a "logs/log-indra-$(get_time)".txt
fi
cp results/kinmodgpt_scan/their_examples/log.txt reproduction/tables_3_4_5_6.txt
python extract_table.py

