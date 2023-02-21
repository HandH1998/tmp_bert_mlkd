MODEL_PATH=../../model/original_pretrained_model/bert-large-uncased

TASK_NAME=SST-2
python -u run_glue.py \
  --model_type bert \
  --model_name_or_path $MODEL_PATH \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir ../../data/glue_data/$TASK_NAME/ \
  --max_seq_length 64 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 4.0 \
  --save_steps 1000000 \
  --output_dir ../../model/BERT-EMD/fine-tuned_pretrained_model/$TASK_NAME/on_original_data/ \
  --evaluate_during_training \
  --overwrite_output_dir \
  --gpu_id 1\
  > log/SST-2_fine_tune.log 2>&1

TASK_NAME=QNLI
python -u run_glue.py \
  --model_type bert \
  --model_name_or_path $MODEL_PATH \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir ../../data/glue_data/$TASK_NAME/ \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 4.0 \
  --save_steps 100000000 \
  --output_dir ../../model/BERT-EMD/fine-tuned_pretrained_model/$TASK_NAME/on_original_data/ \
  --evaluate_during_training \
  --overwrite_output_dir \
  --gpu_id 1\
  > log/QNLI_fine_tune.log 2>&1

TASK_NAME=MNLI
python -u run_glue.py \
  --model_type bert \
  --model_name_or_path $MODEL_PATH \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir ../../data/glue_data/$TASK_NAME/ \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs 4.0 \
  --save_steps 100000000 \
  --output_dir ../../model/BERT-EMD/fine-tuned_pretrained_model/$TASK_NAME/on_original_data/ \
  --evaluate_during_training \
  --overwrite_output_dir \
  --gpu_id 1\
  > log/MNLI_fine_tune.log 2>&1



