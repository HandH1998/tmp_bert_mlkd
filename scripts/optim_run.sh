#!/bin/bash
# list_task_name=("cola" "mnli" "mrpc" "qnli" "qqp" "rte" "sst-2" "sts-b" "wnli")
# list_data_name=("CoLA" "MNLI" "MRPC" "QNLI" "QQP" "RTE" "SST-2" "STS-B" "WNLI")
list_task_name=("cola" "mnli" "mrpc")
list_data_name=("CoLA" "MNLI" "MRPC")
# evaluate the student after transformer distillation
# num=0
# while (($num < ${#list_data_name[*]}))
# do
#     nohup python task_distill.py --do_eval --student_model "model\distilled_fine-tuned_model\\${list_task_name[${num}]}\on_original_data" \
#     --data_dir "data\glue_data\\${list_data_name[${num}]}" --task_name "${list_task_name[${num}]}" --output_dir "model\distilled_fine-tuned_model\\${list_task_name[${num}]}\on_original_data\best_eval_result" \
#     --do_lower_case > "model\distilled_fine-tuned_model\tmp_student_eval_results\\${list_task_name[${num}]}.log" &&
#     let "num++"
# done

#evaluate the student after prediction distillation
# num=0
# while (($num < ${#list_data_name[*]}))
# do
#     nohup python task_distill.py --do_eval --student_model "model\distilled_fine-tuned_model\\${list_task_name[${num}]}\on_original_data_final" \
#     --data_dir "data\glue_data\\${list_data_name[${num}]}" --task_name "${list_task_name[${num}]}"  \
#     --do_lower_case > "model\distilled_fine-tuned_model\final_student_eval_restults\\${list_task_name[${num}]}.log" &&
#     let "num++"
# done
# nohup python task_distill.py --do_eval --student_model "model\distilled_fine-tuned_model\mnli\on_original_data_final" \
# --data_dir "data\glue_data\MNLI" --task_name "mnli-mm"  \
# --do_lower_case > "model\distilled_fine-tuned_model\final_student_eval_restults\\mnli-mm.log" &&


#distill the intermediate layer whith hidden states with attention enhanced
# num=0
# while (($num < ${#list_data_name[*]}))
# do
#     nohup python task_distill.py --teacher_model "model\fine-tuned_pretrained_model\\${list_task_name[${num}]}\on_original_data" \
#     --student_model "model\distilled_pretrained_model\2nd_General_TinyBERT_4L_312D" --data_dir "data\glue_data\\${list_data_name[${num}]}" \
#     --task_name "${list_task_name[${num}]}" --output_dir "model\knowledge_review\distilled_intermediate_model\simple_fusion\\${list_task_name[${num}]}\on_original_data"  \
#     --do_lower_case > "simple_fusion\\${list_task_name[${num}]}.log" &&
#     let "num++"
# done

#evaluate the simple fusion distillation
# num=0
# while (($num < ${#list_data_name[*]}))
# do
#     nohup python task_distill.py --do_eval --student_model "model\knowledge_review\distilled_intermediate_model\simple_fusion\\${list_task_name[${num}]}\on_original_data" \
#     --data_dir "data\glue_data\\${list_data_name[${num}]}" --task_name "${list_task_name[${num}]}" --output_dir "model\knowledge_review\distilled_intermediate_model\simple_fusion\\${list_task_name[${num}]}\on_original_data\best_eval_result" \
#     --do_lower_case > "model\knowledge_review\distilled_intermediate_model\simple_fusion\tmp_student_eval_results\\${list_task_name[${num}]}.log" &&
#     let "num++" 
# done

#distill the intermediate layer whith vanilla knowledge review
num=0
while (($num < ${#list_data_name[*]}))
do
    nohup python task_distill.py --teacher_model "model\fine-tuned_pretrained_model\\${list_task_name[${num}]}\on_original_data" \
    --student_model "model\distilled_pretrained_model\2nd_General_TinyBERT_4L_312D" --data_dir "data\glue_data\\${list_data_name[${num}]}" \
    --task_name "${list_task_name[${num}]}" --output_dir "model\knowledge_review\distilled_intermediate_model\vanilla_knowledge_review\\${list_task_name[${num}]}\on_original_data"  \
    --do_lower_case > "vanilla_knowledge_review\\${list_task_name[${num}]}.log" &&
    let "num++"
done
