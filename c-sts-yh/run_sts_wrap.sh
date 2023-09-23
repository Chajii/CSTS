#!/bin/bash

current_date_time=$(date)
echo "Start Current date and time: $current_date_time"

CUDA_VISIBLE_DEVICES=0,1 POOLER_TYPE=hypernet MODEL=princeton-nlp/sup-simcse-roberta-large bash ./run_sts.sh &
CUDA_VISIBLE_DEVICES=2,3 POOLER_TYPE=hypernet2 MODEL=princeton-nlp/sup-simcse-roberta-large bash ./run_sts.sh &
wait

CUDA_VISIBLE_DEVICES=0,1 POOLER_TYPE=hypernet3 MODEL=princeton-nlp/sup-simcse-roberta-large bash ./run_sts.sh &
wait

current_date_time=$(date)
echo "End Current date and time: $current_date_time"
