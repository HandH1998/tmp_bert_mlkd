#!/bin/bash
# list_task_name=("mnli" "cola" "mrpc" "qnli" "qqp" "rte" "sst-2" "sts-b")
# list_data_name=("MNLI" "CoLA" "MRPC" "QNLI" "QQP" "RTE" "SST-2" "STS-B")
list_task_name=("sts-b" "mrpc")
list_data_name=("STS-B" "MRPC")
# list_task_name=("mnli")
# list_data_name=("MNLI")
# distill the intermediate layer with resual knowledge review with enhanced respective fusion
# num=0
# while (($num < ${#list_data_name[*]}))
# do
#     nohup python task_distill.py \
#     --do_predict \
#     --student_model "model\knowledge_review\distilled_prediction_model\resual_kr_with_enhanced_respective_fusion\\${list_task_name[${num}]}\on_original_data" \
#     --data_dir "data\glue_data\\${list_data_name[${num}]}" \
#     --task_name "${list_task_name[${num}]}" \
#     --write_predict_dir "model\knowledge_review\distilled_prediction_model\resual_kr_with_enhanced_respective_fusion\test_results" \
#     --do_lower_case 
#     let "num++"
# done
base_dir="model\BERT-EMD\fine_tuned_pretrained_model"
num=0
while (($num < ${#list_data_name[*]}))
do
    nohup python task_distill.py \
    --do_predict \
    --student_model "${base_dir}\\${list_data_name[${num}]}\on_original_data" \
    --data_dir "data\glue_data\\${list_data_name[${num}]}" \
    --task_name "${list_task_name[${num}]}" \
    --write_predict_dir "${base_dir}\test_results" \
    --do_lower_case 
    let "num++"
done

