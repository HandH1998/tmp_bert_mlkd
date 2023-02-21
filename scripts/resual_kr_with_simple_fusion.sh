#!/bin/bash
# list_task_name=("cola" "mnli" "mrpc" "qnli" "qqp" "rte" "sst-2" "sts-b" "wnli")
# list_data_name=("CoLA" "MNLI" "MRPC" "QNLI" "QQP" "RTE" "SST-2" "STS-B" "WNLI")
list_task_name=("qqp")
list_data_name=("QQP")

# distill the intermediate layer with resual knowledge review with simple fusion
num=0
while (($num < ${#list_data_name[*]}))
do
    nohup python task_distill.py \
    --teacher_model "model\fine-tuned_pretrained_model\\${list_task_name[${num}]}\on_original_data" \
    --student_model "model\distilled_pretrained_model\2nd_General_TinyBERT_4L_312D" \
    --data_dir "data\glue_data\\${list_data_name[${num}]}" \
    --task_name "${list_task_name[${num}]}" \
    --output_dir "model\knowledge_review\distilled_intermediate_model\resual_kr_with_simple_fusion\\${list_task_name[${num}]}\on_original_data"  \
    --do_lower_case \
    --tensorboard_log_save_dir "tensorboard_log\distilled_intermediate_model\resual_kr_with_simple_fusion\\${list_task_name[${num}]}\on_original_data" \
    > "resual_kr_with_simple_fusion\\${list_task_name[${num}]}.log" 2>&1 &
    wait
    let "num++"
done

# distill the prediction layer with resual knowledge review with simple fusion
list_task_name=("mnli" "qnli" "qqp")
list_data_name=("MNLI" "QNLI" "QQP")
num=0
while (($num < ${#list_data_name[*]}))
do
    nohup python task_distill.py \
    --pred_distill \
    --teacher_model "model\fine-tuned_pretrained_model\\${list_task_name[${num}]}\on_original_data" \
    --student_model "model\knowledge_review\distilled_intermediate_model\resual_kr_with_simple_fusion\\${list_task_name[${num}]}\on_original_data" \
    --data_dir "data\glue_data\\${list_data_name[${num}]}" \
    --task_name "${list_task_name[${num}]}" \
    --output_dir "model\knowledge_review\distilled_prediction_model\resual_kr_with_simple_fusion\\${list_task_name[${num}]}\on_original_data"  \
    --do_lower_case \
    --tensorboard_log_save_dir "tensorboard_log\distilled_prediction_model\resual_kr_with_simple_fusion\\${list_task_name[${num}]}\on_original_data" \
    > "final_resual_kr_with_simple_fusion\\${list_task_name[${num}]}.log" 2>&1 &
    wait
    let "num++"
done