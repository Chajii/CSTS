#!/bin/bash

current_date_time=$(date)
echo "Start Current date and time: $current_date_time"

MODEL=meta-llama/Llama-2-70b-chat-hf \
  BATCH_SIZE=1 \
  MULTISTEP_REASONING_ZERO_SHOT_COT3=True \
  bash ./run_sts_fewshot.sh &
wait

current_date_time=$(date)
echo "End Current date and time: $current_date_time"
