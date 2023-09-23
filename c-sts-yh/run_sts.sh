#!/bin/bash
model=${MODEL:-princeton-nlp/sup-simcse-roberta-large} # pre-trained model
encoding=${ENCODER_TYPE:-bi_encoder}                   # cross_encoder, bi_encoder, tri_encoder
finetune=${FINE_TUNE:-True}
[[ ${finetune} = True ]] && do_train=True || do_train=False
[[ ${finetune} = True ]] && freeze_encoder=False || freeze_encoder=True
lr=${LR:-1e-5}                               # learning rate
wd=${WD:-0.1}                                # weight decay
transform=${TRANSFORM:-False}                # whether to use an additional linear layer after the encoder
objective=${OBJECTIVE:-mse}                  # mse, triplet, triplet_mse
triencoder_head=${TRIENCODER_HEAD:-hadamard} # hadamard, concat (set for tri_encoder)
use_prompt=${USE_PROMPT:-False}
use_prompt2=${USE_PROMPT2:-False}
use_prompt3=${USE_PROMPT3:-False}
use_prompt4=${USE_PROMPT4:-False}
seed=${SEED:-42}
train_file=${TRAIN_FILE:-data/csts_train.csv}
eval_file=${EVAL_FILE:-data/csts_validation.csv}
test_file=${TEST_FILE:-data/csts_test.csv}
val_as_test=${VAL_AS_TEST:-True} # whether to use validation set as test set
output_dir=${OUTPUT_DIR:-output}

config=enc_${encoding}__lr_${lr}__wd_${wd}__trans_${transform}__obj_${objective}__tri_${triencoder_head}__s_${seed}__finetune_${finetune}__val_as_test_${val_as_test}

if [[ ${use_prompt} = True ]]; then
  config=${config}__prompt
elif [[ ${use_prompt2} = True ]]; then
  config=${config}__prompt2
elif [[ ${use_prompt3} = True ]]; then
  config=${config}__prompt3
elif [[ ${use_prompt4} = True ]]; then
  config=${config}__prompt4
fi

pooler_type=${POOLER_TYPE:-cls_before_pooler}

if [[ "${pooler_type}" = "hypernet" ]]; then
  config=hypernet__${config}
elif [[ "${pooler_type}" = "hypernet2" ]]; then
  config=hypernet2__${config}
elif [[ "${pooler_type}" = "hypernet3" ]]; then
  config=hypernet3__${config}
fi

mkdir -p "${output_dir}/${model//\//__}/${config}"

python run_sts.py \
  --output_dir "${output_dir}/${model//\//__}/${config}" \
  --model_name_or_path ${model} \
  --objective ${objective} \
  --encoding_type ${encoding} \
  --pooler_type ${pooler_type} \
  --freeze_encoder ${freeze_encoder} \
  --transform ${transform} \
  --triencoder_head ${triencoder_head} \
  --use_prompt ${use_prompt} \
  --use_prompt2 ${use_prompt2} \
  --use_prompt3 ${use_prompt3} \
  --use_prompt4 ${use_prompt4} \
  --max_seq_length 512 \
  --train_file ${train_file} \
  --validation_file ${eval_file} \
  --test_file ${test_file} \
  --val_as_test ${val_as_test} \
  --condition_only False \
  --sentences_only False \
  --do_train ${do_train} \
  --do_eval \
  --do_predict \
  --evaluation_strategy epoch \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --learning_rate ${lr} \
  --weight_decay ${wd} \
  --max_grad_norm 0.0 \
  --num_train_epochs 3 \
  --lr_scheduler_type linear \
  --warmup_ratio 0.1 \
  --log_level info \
  --disable_tqdm True \
  --save_strategy epoch \
  --save_total_limit 1 \
  --seed ${seed} \
  --data_seed ${seed} \
  --fp16 True \
  --log_time_interval 15 >"${output_dir}/${model//\//__}/${config}/run.log" 2>&1
