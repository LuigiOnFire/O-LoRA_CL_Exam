#!/bin/bash
set -e

port=$((29500 + ${SLURM_JOB_ID:-0} % 10000))
OUTPUT_DIR="logs_and_outputs/order_minimal/outputs"
mkdir -p "$OUTPUT_DIR"

# --- STEP 1: Train on Task 1 (DBpedia) ---
echo "=== Step 1: Training Task 1 (DBpedia) ==="
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
    --do_train \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path initial_model/t5-large \
    --data_dir CL_Benchmark \
    --task_config_dir configs/order_minimal_configs/task1_dbpedia \
    --instruction_file configs/instruction_config.json \
    --instruction_strategy single \
    --output_dir "$OUTPUT_DIR/1-dbpedia" \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 128 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-03 \
    --num_train_epochs 1 \
    --deepspeed configs/ds_configs/stage2.config \
    --run_name order_min_r1 \
    --max_source_length 512 \
    --max_target_length 50 \
    --generation_max_length 50 \
    --add_task_name True \
    --add_dataset_name True \
    --overwrite_output_dir \
    --overwrite_cache \
    --lr_scheduler_type constant \
    --warmup_steps 0 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy no \
    --save_strategy no \
    --lamda_1 0.5 \
    --lamda_2 0

# --- STEP 2: Train on Task 2 (Financial PhraseBank) & Re-evaluate Both ---
echo "=== Step 2: Training Task 2 (Financial PhraseBank) & Evaluating Forgetting ==="
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
    --do_train \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path "$OUTPUT_DIR/1-dbpedia/adapter" \
    --data_dir CL_Benchmark \
    --task_config_dir configs/order_minimal_configs/task2_fpb \
    --instruction_file configs/instruction_config.json \
    --instruction_strategy single \
    --output_dir "$OUTPUT_DIR/2-financial_phrasebank" \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 128 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-03 \
    --num_train_epochs 3 \
    --deepspeed configs/ds_configs/stage2.config \
    --run_name order_min_r2 \
    --max_source_length 512 \
    --max_target_length 50 \
    --generation_max_length 50 \
    --add_task_name True \
    --add_dataset_name True \
    --overwrite_output_dir \
    --overwrite_cache \
    --lr_scheduler_type constant \
    --warmup_steps 0 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy no \
    --save_strategy no \
    --lamda_1 0.5 \
    --lamda_2 0

echo "✓ Minimal O-LoRA Sequence Executed Successfully!"