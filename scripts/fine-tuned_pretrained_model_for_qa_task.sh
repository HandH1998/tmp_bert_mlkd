#!/bin/bash
list_task_name=("squad1" "squad2")
list_train_file=("train-v1.1.json" "train-v2.0.json")
list_predict_file=("dev-v1.1.json" "dev-v2.0.json")
list_version=(0 1)
# list_task_name=("qqp")
# list_data_name=("QQP")
# distill the intermediate layer with resual knowledge review with enhanced respective fusion
# base_dir1="model/multi_level_distillation/distilled_intermediate_model/bert_mlkd_for_qa_task_bert_base_4_312"
# base_dir2="model/multi_level_distillation/distilled_prediction_model/bert_mlkd_for_qa_task_bert_base_4_312"
# gpu_id=0
# num=0
base_dir="model/fine-tuned_pretrained_model/bert-base-uncased"
num=0
while (($num < ${#list_task_name[*]}))
do
    nohup python superbert_run_en_classifier.py \
    --model "model/original_pretrained_model/bert-base-uncased" \
    --data_dir "data/squad_data" \
    --task_name "${list_task_name[${num}]}" \
    --output_dir "${base_dir}/${list_task_name[${num}]}/on_original_data"\
    --do_lower_case \
    --gradient_accumulation_steps 1 \
    --version_2_with_negative ${list_version[${num}]} \
    --train_file ${list_train_file[${num}]} \
    --predict_file ${list_predict_file[${num}]} \
    > "log/fine-tuned_pretrained_model/bert-base-uncased/${list_task_name[${num}]}.log" 2>&1
    let "num++"
done
# while (($num < ${#list_task_name[*]}))
# do
#     nohup python -u task_distill.py \
#     --teacher_model "model/fine-tuned_pretrained_model/bert-base-uncased/${list_task_name[${num}]}/on_original_data" \
#     --student_model "model/distilled_pretrained_model/2nd_General_TinyBERT_4L_312D" \
#     --data_dir "data/squad_data" \
    # --task_name "${list_task_name[${num}]}" \
    # --output_dir "${base_dir1}/${list_task_name[${num}]}/on_original_data"  \
#     --do_lower_case \
#     --tensorboard_log_save_dir "tensorboard_log/multi_level_distillation/distilled_intermediate_model/bert_mlkd_for_qa_task_bert_base_4_312/${list_task_name[${num}]}/on_original_data" \
    # --gradient_accumulation_steps 1 \
    # --version_2_with_negative ${list_version[${num}]} \
    # --train_file ${list_train_file[${num}]} \
    # --predict_file ${list_predict_file[${num}]} \
#     --gpu_id ${gpu_id}\
#     > "log/multi_level_distillation/distilled_intermediate_model/bert_mlkd_for_qa_task_bert_base_4_312/${list_task_name[${num}]}.log" 2>&1 
    
#     nohup python -u task_distill.py \
#     --pred_distill \
#     --teacher_model "model/fine-tuned_pretrained_model/bert-base-uncased/${list_task_name[${num}]}/on_original_data" \
#     --student_model "${base_dir1}/${list_task_name[${num}]}/on_original_data" \
#     --data_dir "data/squad_data" \
#     --task_name "${list_task_name[${num}]}" \
#     --output_dir "${base_dir2}/${list_task_name[${num}]}/on_original_data"  \
#     --do_lower_case \
#     --tensorboard_log_save_dir "tensorboard_log/multi_level_distillation/distilled_prediction_model/bert_mlkd_for_qa_task_bert_base_4_312/${list_task_name[${num}]}/on_original_data" \
#     --gradient_accumulation_steps 1 \
#     --version_2_with_negative ${list_version[${num}]} \
#     --train_file ${list_train_file[${num}]} \
#     --predict_file ${list_predict_file[${num}]} \
#     --gpu_id ${gpu_id}\
#     > "log/multi_level_distillation/distilled_prediction_model/bert_mlkd_for_qa_task_bert_base_4_312/${list_task_name[${num}]}.log" 2>&1 
#     let "num++"
# done

