#!/bin/bash


model=${MODEL:-meta-llama/Llama-2-7b-chat-hf}
k_shot=${K_SHOT:-0}
prompt_name=${PROMPT_NAME:-short}
is_sts=${IS_STS:-False}
seed=${SEED:-42}
train_file=${TRAIN_FILE:-data/csts_train.csv} # Use for ICL
validation_file=${EVAL_FILE:-data/csts_validation.csv}
test_file=${TEST_FILE:-None} # Warning, default is None
output_dir=${OUTPUT_DIR:-output}
output_dir_prefix=${OUTPUT_DIR_PREFIX:-}
batch_size=${BATCH_SIZE:-1}
max_eval_samples=${MAX_EVAL_SAMPLES:-0}
dataset_on_the_fly=${DATASET_ON_THE_FLY:-True}
multistep_reasoning_zero_shot_cot=${MULTISTEP_REASONING_ZERO_SHOT_COT:-False}
multistep_reasoning_zero_shot_cot2=${MULTISTEP_REASONING_ZERO_SHOT_COT2:-False}
multistep_reasoning_zero_shot_cot3=${MULTISTEP_REASONING_ZERO_SHOT_COT3:-False}
multistep_reasoning_csts_cot=${MULTISTEP_REASONING_CSTS_COT:-True}
vllm=${VLLM:-False}

config=k_shot_${k_shot}__prompt_name_${prompt_name}__s_${seed}
if [[ ${is_sts} = True ]]; then
  config=${config}__is_sts
fi

if [[ ${multistep_reasoning_zero_shot_cot} = True ]]; then
  config=${config}__zero
elif [[ ${multistep_reasoning_zero_shot_cot2} = True ]]; then
  config=${config}__zero2
elif [[ ${multistep_reasoning_zero_shot_cot3} = True ]]; then
  config=${config}__zero3
elif [[ ${multistep_reasoning_csts_cot} = True ]]; then
  config=${config}__cstscot
fi

if [[ ${vllm} = True ]]; then
  config=${config}__vllm
fi

if [[ ${max_eval_samples} != 0 ]]; then
  config=${config}__max_eval_samples_${max_eval_samples}
fi

final_output_dir=${output_dir}/${model//\//__}/${config}
if [[ ! -z ${output_dir_prefix} ]]; then
  final_output_dir=${final_output_dir}__${output_dir_prefix}
fi

mkdir -p "${final_output_dir}"

echo ${prompt_name}
python -u run_sts_fewshot.py \
  --model_name_or_path ${model} \
  --k_shot ${k_shot} \
  --prompt_name ${prompt_name} \
  --seed ${seed} \
  --is_sts ${is_sts} \
  --train_file ${train_file} \
  --validation_file ${validation_file} \
  --test_file ${test_file} \
  --output_dir "${final_output_dir}" \
  --overwrite_output_dir True \
  --dtype bf16 \
  --batch_size ${batch_size} \
  --max_eval_samples ${max_eval_samples} \
  --dataset_on_the_fly ${dataset_on_the_fly} \
  --multistep_reasoning_csts_cot ${multistep_reasoning_csts_cot} \

  >"${final_output_dir}/run.log" 2>&1
