#!/bin/bash
# list_task_name=("mrpc" "rte" "cola" "sst-2" "sts-b" "qnli" "mnli" "qqp")
# list_data_name=("MRPC" "RTE" "CoLA" "SST-2" "STS-B" "QNLI" "MNLI" "QQP")
list_task_name=("sst-2" "mnli" "qqp")
list_data_name=("SST-2" "MNLI" "QQP")
# distill the intermediate layer with resual knowledge review with enhanced respective fusion
base_dir1="model/multi_level_distillation/distilled_intermediate_model/bert_mlkd_rand_head_num_3"
base_dir2="model/multi_level_distillation/distilled_prediction_model/bert_mlkd_rand_head_num_3"
num=0
while (($num < ${#list_data_name[*]}))
do
    nohup python -u task_distill.py \
    --teacher_model "model/fine-tuned_pretrained_model/${list_task_name[${num}]}/on_original_data" \
    --student_model "model/distilled_pretrained_model/2nd_General_TinyBERT_4L_312D" \
    --data_dir "data/glue_data/${list_data_name[${num}]}" \
    --task_name "${list_task_name[${num}]}" \
    --output_dir "${base_dir1}/${list_task_name[${num}]}/on_original_data"  \
    --do_lower_case \
    --tensorboard_log_save_dir "tensorboard_log/multi_level_distillation/distilled_intermediate_model/bert_mlkd_rand_head_num_3/${list_task_name[${num}]}/on_original_data" \
    --gradient_accumulation_steps 1 \
    > "log/multi_level_distillation/distilled_intermediate_model/bert_mlkd_rand_head_num_3/${list_task_name[${num}]}.log" 2>&1 
    
    nohup python -u task_distill.py \
    --pred_distill \
    --teacher_model "model/fine-tuned_pretrained_model/${list_task_name[${num}]}/on_original_data" \
    --student_model "${base_dir1}/${list_task_name[${num}]}/on_original_data" \
    --data_dir "data/glue_data/${list_data_name[${num}]}" \
    --task_name "${list_task_name[${num}]}" \
    --output_dir "${base_dir2}/${list_task_name[${num}]}/on_original_data"  \
    --do_lower_case \
    --tensorboard_log_save_dir "tensorboard_log/multi_level_distillation/distilled_prediction_model/bert_mlkd_rand_head_num_3/${list_task_name[${num}]}/on_original_data" \
    --gradient_accumulation_steps 1 \
    > "log/multi_level_distillation/distilled_prediction_model/bert_mlkd_rand_head_num_3/${list_task_name[${num}]}.log" 2>&1 
    let "num++"
done

