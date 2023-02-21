#!/bin/bash
# list_task_name=("cola" "mnli" "mrpc" "qnli" "qqp" "rte" "sst-2" "sts-b" "wnli")
# list_data_name=("CoLA" "MNLI" "MRPC" "QNLI" "QQP" "RTE" "SST-2" "STS-B" "WNLI")
# list_task_name=("cola" "mnli" "mrpc")
# list_data_name=("CoLA" "MNLI" "MRPC")
#distill the intermediate layer with vanilla knowledge review
# num=0
# while (($num < ${#list_data_name[*]}))
# do
#     nohup python task_distill.py --teacher_model "model\fine-tuned_pretrained_model\\${list_task_name[${num}]}\on_original_data" \
#     --student_model "model\distilled_pretrained_model\2nd_General_TinyBERT_4L_312D" --data_dir "data\glue_data\\${list_data_name[${num}]}" \
#     --task_name "${list_task_name[${num}]}" --output_dir "model\knowledge_review\distilled_intermediate_model\vanilla_knowledge_review\\${list_task_name[${num}]}\on_original_data"  \
#     --do_lower_case > "vanilla_knowledge_review\\${list_task_name[${num}]}.log" &&
#     let "num++"
# done

#distill the prediction layer with vanilla knowledge review
# num=0
# while (($num < ${#list_data_name[*]}))
# do
#     nohup python task_distill.py --pred_distill --teacher_model "model\fine-tuned_pretrained_model\\${list_task_name[${num}]}\on_original_data" \
#     --student_model "model\knowledge_review\distilled_intermediate_model\vanilla_knowledge_review\\${list_task_name[${num}]}\on_original_data" --data_dir "data\glue_data\\${list_data_name[${num}]}" \
#     --task_name "${list_task_name[${num}]}" --output_dir "model\knowledge_review\distilled_prediction_model\vanilla_knowledge_review\\${list_task_name[${num}]}\on_original_data"  \
#     --do_lower_case --learning_rate 3e-5  --num_train_epochs  5 > "final_vanilla_knowledge_review\\${list_task_name[${num}]}.log" &&
#     let "num++"
# done

#distill the intermediate layer with vanilla knowledge review
# list_task_name=("qnli" "qqp" "rte" "sst-2" "sts-b" "wnli")
# list_data_name=("QNLI" "QQP" "RTE" "SST-2" "STS-B" "WNLI")
# num=0
# while (($num < ${#list_data_name[*]}))
# do
#     nohup python task_distill.py --teacher_model "model\fine-tuned_pretrained_model\\${list_task_name[${num}]}\on_original_data" \
#     --student_model "model\distilled_pretrained_model\2nd_General_TinyBERT_4L_312D" --data_dir "data\glue_data\\${list_data_name[${num}]}" \
#     --task_name "${list_task_name[${num}]}" --output_dir "model\knowledge_review\distilled_intermediate_model\vanilla_knowledge_review\\${list_task_name[${num}]}\on_original_data"  \
#     --do_lower_case > "vanilla_knowledge_review\\${list_task_name[${num}]}.log" &&
#     let "num++"
# done

#distill the prediction layer with vanilla knowledge review
# num=0
# while (($num < ${#list_data_name[*]}))
# do
#     nohup python task_distill.py --pred_distill --teacher_model "model\fine-tuned_pretrained_model\\${list_task_name[${num}]}\on_original_data" \
#     --student_model "model\knowledge_review\distilled_intermediate_model\vanilla_knowledge_review\\${list_task_name[${num}]}\on_original_data" --data_dir "data\glue_data\\${list_data_name[${num}]}" \
#     --task_name "${list_task_name[${num}]}" --output_dir "model\knowledge_review\distilled_prediction_model\vanilla_knowledge_review\\${list_task_name[${num}]}\on_original_data"  \
#     --do_lower_case --learning_rate 3e-5  --num_train_epochs  5 > "final_vanilla_knowledge_review\\${list_task_name[${num}]}.log" &&
#     let "num++"
# done
#distill the prediction layer with simple fusion
# list_task_name=("qqp" "rte" "sst-2" "sts-b" "wnli")
# list_data_name=("QQP" "RTE" "SST-2" "STS-B" "WNLI")

# num=0
# while (($num < ${#list_data_name[*]}))
# do
#     nohup python task_distill.py --pred_distill --teacher_model "model\fine-tuned_pretrained_model\\${list_task_name[${num}]}\on_original_data" \
#     --student_model "model\knowledge_review\distilled_intermediate_model\simple_fusion\\${list_task_name[${num}]}\on_original_data" --data_dir "data\glue_data\\${list_data_name[${num}]}" \
#     --task_name "${list_task_name[${num}]}" --output_dir "model\knowledge_review\distilled_prediction_model\simple_fusion\\${list_task_name[${num}]}\on_original_data"  \
#     --do_lower_case --learning_rate 3e-5  --num_train_epochs  5 > "final_simple_fusion\\${list_task_name[${num}]}.log" &&
#     let "num++"
# done

#distill the intermediate layer with simple fusion on aug data
list_task_name=("cola")
list_data_name=("CoLA")
num=0
while (($num < ${#list_data_name[*]}))
do
    nohup python task_distill.py --teacher_model "model\fine-tuned_pretrained_model\\${list_task_name[${num}]}\on_original_data" \
    --student_model "model\distilled_pretrained_model\2nd_General_TinyBERT_4L_312D" --data_dir "data\glue_data\\${list_data_name[${num}]}" \
    --task_name "${list_task_name[${num}]}" --output_dir "model\knowledge_review\distilled_intermediate_model\simple_fusion\\${list_task_name[${num}]}\on_aug_data"  \
    --do_lower_case --aug_train> "on_aug_data\simple_fusion\\${list_task_name[${num}]}.log" &&
    let "num++"
done