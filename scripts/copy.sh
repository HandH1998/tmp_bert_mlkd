#!/bin/bash
# from 10.2.10.210
list_task_name=("resual_kr_with_enhanced_respective_fusion_and_hcl" "resual_kr_with_multi_teacher_layer_v2")
source_dir=zy@10.2.10.210:/home/zy/projects_py/tiny_bert/model/knowledge_review
local_dir=/d/zy/tiny_bert/model/knowledge_review
num=0
while (($num < ${#list_task_name[*]}))
do
    scp -rp $source_dir/distilled_intermediate_model/${list_task_name[$num]}/ $local_dir/distilled_intermediate_model/ 
    scp -rp $source_dir/distilled_prediction_model/${list_task_name[$num]}/ $local_dir/distilled_prediction_model/ 
    let "num++"
done

scp -rp zy@10.2.10.210:/home/zy/projects_py/tiny_bert/model/distilled_fine-tuned_model/6-768/ /d/zy/tiny_bert/model/distilled_fine-tuned_model/

scp -rp /d/zy/tiny_bert/model/multi_level_distillation/ zy@10.2.10.210:/home/zy/projects_py/tiny_bert/model/
scp -rp /d/zy/tiny_bert/log/multi_level_distillation/ zy@10.2.10.210:/home/zy/projects_py/tiny_bert/log/
scp -rp /d/zy/tiny_bert/tensorboard_log/multi_level_distillation/ zy@10.2.10.210:/home/zy/projects_py/tiny_bert/tensorboard_log/