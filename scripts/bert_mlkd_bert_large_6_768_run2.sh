#!/bin/bash
list_task_name=("qnli" "mnli")
list_data_name=("QNLI" "MNLI")
# list_task_name=("qqp")
# list_data_name=("QQP")
# distill the intermediate layer with resual knowledge review with enhanced respective fusion
base_dir1="model/multi_level_distillation/distilled_intermediate_model/bert_mlkd_bert_large_6_768"
base_dir2="model/multi_level_distillation/distilled_prediction_model/bert_mlkd_bert_large_6_768"
gpu_id=1
num=0
while (($num < ${#list_data_name[*]}))
do
    nohup python -u task_distill.py \
    --teacher_model "model/fine-tuned_pretrained_model/bert-large-uncased/${list_task_name[${num}]}/on_original_data" \
    --student_model "model/distilled_pretrained_model/2nd_General_TinyBERT_6L_768D" \
    --data_dir "data/glue_data/${list_data_name[${num}]}" \
    --task_name "${list_task_name[${num}]}" \
    --output_dir "${base_dir1}/${list_task_name[${num}]}/on_original_data"  \
    --do_lower_case \
    --tensorboard_log_save_dir "tensorboard_log/multi_level_distillation/distilled_intermediate_model/bert_mlkd_bert_large_6_768/${list_task_name[${num}]}/on_original_data" \
    --gradient_accumulation_steps 1 \
    --gpu_id ${gpu_id}\
    > "log/multi_level_distillation/distilled_intermediate_model/bert_mlkd_bert_large_6_768/${list_task_name[${num}]}.log" 2>&1 
    
    nohup python -u task_distill.py \
    --pred_distill \
    --teacher_model "model/fine-tuned_pretrained_model/bert-large-uncased/${list_task_name[${num}]}/on_original_data" \
    --student_model "${base_dir1}/${list_task_name[${num}]}/on_original_data" \
    --data_dir "data/glue_data/${list_data_name[${num}]}" \
    --task_name "${list_task_name[${num}]}" \
    --output_dir "${base_dir2}/${list_task_name[${num}]}/on_original_data"  \
    --do_lower_case \
    --tensorboard_log_save_dir "tensorboard_log/multi_level_distillation/distilled_prediction_model/bert_mlkd_bert_large_6_768/${list_task_name[${num}]}/on_original_data" \
    --gradient_accumulation_steps 1 \
    --gpu_id ${gpu_id}\
    > "log/multi_level_distillation/distilled_prediction_model/bert_mlkd_bert_large_6_768/${list_task_name[${num}]}.log" 2>&1 
    let "num++"
done

