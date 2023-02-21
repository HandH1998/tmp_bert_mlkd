MODEL_PATH=../../model/original_pretrained_model/bert-large-uncased

TASK_NAME=CoLA
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
  --learning_rate 1e-5 \
  --num_train_epochs 6.0 \
  --save_steps 1000000 \
  --output_dir ../../model/BERT-EMD/fine-tuned_pretrained_model/$TASK_NAME/on_original_data/ \
  --evaluate_during_training \
  --overwrite_output_dir \
  --gpu_id 0\
  > log/CoLA_fine_tune.log 2>&1

TASK_NAME=MRPC
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
  --num_train_epochs 6.0 \
  --save_steps 1000000 \
  --output_dir ../../model/BERT-EMD/fine-tuned_pretrained_model/$TASK_NAME/on_original_data/ \
  --evaluate_during_training \
  --overwrite_output_dir \
  --gpu_id 0\
  > log/MRPC_fine_tune.log 2>&1

TASK_NAME=RTE
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
  --num_train_epochs 6.0 \
  --save_steps 1000000 \
  --output_dir ../../model/BERT-EMD/fine-tuned_pretrained_model/$TASK_NAME/on_original_data/ \
  --evaluate_during_training \
  --overwrite_output_dir \
  --gpu_id 0\
  > log/RTE_fine_tune.log 2>&1

TASK_NAME=STS-B
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
  --num_train_epochs 6.0 \
  --save_steps 1000000 \
  --output_dir ../../model/BERT-EMD/fine-tuned_pretrained_model/$TASK_NAME/on_original_data/ \
  --evaluate_during_training \
  --overwrite_output_dir \
  --gpu_id 0\
  > log/STS-B_fine_tune.log 2>&1

TASK_NAME=QQP
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
  --gpu_id 0\
  > log/QQP_fine_tune.log 2>&1
# TASK_NAME=QQP
# python -u run_glue.py \
#   --model_type bert \
#   --model_name_or_path $MODEL_PATH \
#   --task_name $TASK_NAME \
#   --do_train \
#   --do_eval \
#   --do_lower_case \
#   --data_dir ../../data/glue_data/$TASK_NAME/ \
#   --max_seq_length 128 \
#   --per_gpu_train_batch_size 32 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 4.0 \
#   --save_steps 2000 \
#   --output_dir ../../model/BERT-EMD/fine_tuned_pretrained_model/$TASK_NAME/on_original_data/ \
#   --evaluate_during_training \
#   --overwrite_output_dir \
#   > QQP_fine_tune.log 2>&1

# TASK_NAME=STS-B
# python -u run_glue.py \
#   --model_type bert \
#   --model_name_or_path $MODEL_PATH \
#   --task_name $TASK_NAME \
#   --do_train \
#   --do_eval \
#   --do_lower_case \
#   --data_dir ../../data/glue_data/$TASK_NAME/ \
#   --max_seq_length 128 \
#   --per_gpu_train_batch_size 32 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 8.0 \
#   --save_steps 100 \
#   --output_dir ../../model/BERT-EMD/fine_tuned_pretrained_model/$TASK_NAME/on_original_data/ \
#   --evaluate_during_training \
#   --overwrite_output_dir \
#   > STS-B_fine_tune.log 2>&1
