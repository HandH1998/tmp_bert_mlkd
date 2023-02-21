#!/bin/bash
list_task_name=("cola" "mrpc" "mnli" "sst-2" "sts-b" "qnli" "rte" "qqp")
list_data_name=("CoLA" "MRPC" "MNLI" "SST-2" "STS-B" "QNLI" "RTE" "QQP")

base_dir="model/fine-tuned_pretrained_model/bert-large-uncased"
num=0
while (($num < ${#list_data_name[*]}))
do
    nohup python superbert_run_en_classifier.py \
    --model "model/original_pretrained_model/bert-large-uncased" \
    --data_dir "data/glue_data/${list_data_name[${num}]}" \
    --task_name "${list_task_name[${num}]}" \
    --output_dir "${base_dir}/${list_task_name[${num}]}/on_original_data"\
    --do_lower_case \
    > "log/fine-tuned_pretrained_model/bert-large-uncased/${list_task_name[${num}]}.log" 2>&1
    let "num++"
done
# fine-tune bert-base-uncased

# python superbert_run_en_classifier.py --data_dir "data/glue_data/MRPC" \
# --model "model/original_pretrained_model" --task_name "mrpc" \
# --output_dir "model/fine-tuned_pretrained_model/mrpc/on_original_data" --do_lower_case

# python superbert_run_en_classifier.py --data_dir "data/glue_data/CoLA" \
# --model "model/original_pretrained_model" --task_name "cola" \
# --output_dir "model/fine-tuned_pretrained_model/cola/on_original_data" --do_lower_case 

# python superbert_run_en_classifier.py --data_dir "data/glue_data/SST-2" \
# --model "model/original_pretrained_model" --task_name "sst-2" \
# --output_dir "model/fine-tuned_pretrained_model/sst-2/on_original_data" --do_lower_case 

# nohup python superbert_run_en_classifier.py --data_dir "data/glue_data/QQP" \
# --model "model/original_pretrained_model" --task_name "qqp" \
# --output_dir "model/fine-tuned_pretrained_model/qqp/on_original_data" --do_lower_case >qqp.out

# nohup python superbert_run_en_classifier.py --data_dir "data/glue_data/RTE" \
# --model "model/original_pretrained_model" --task_name "rte" \
# --output_dir "model/fine-tuned_pretrained_model/rte/on_original_data" --do_lower_case > rte.out

# nohup python superbert_run_en_classifier.py --data_dir "data/glue_data/STS-B" \
# --model "model/original_pretrained_model" --task_name "sts-b" \
# --output_dir "model/fine-tuned_pretrained_model/sts-b/on_original_data" --do_lower_case >sts-b.out

# nohup python superbert_run_en_classifier.py --data_dir "data/glue_data/QNLI" \
# --model "model/original_pretrained_model" --task_name "qnli" \
# --output_dir "model/fine-tuned_pretrained_model/qnli/on_original_data" --do_lower_case >qnli.out

# nohup python superbert_run_en_classifier.py --data_dir "data/glue_data/WNLI" \
# --model "model/original_pretrained_model" --task_name "wnli" \
# --output_dir "model/fine-tuned_pretrained_model/wnli/on_original_data" --do_lower_case >wnli.out

# nohup python superbert_run_en_classifier.py --data_dir "data/glue_data/MNLI" \
# --model "model/original_pretrained_model" --task_name "mnli" \
# --output_dir "model/fine-tuned_pretrained_model/mnli/on_original_data" --do_lower_case >mnli.out

# nohup python superbert_run_en_classifier.py --data_dir "data/squad_data" \
# --model "model/original_pretrained_model" --task_name "squad1" \
# --output_dir "model/fine-tuned_pretrained_model/squad1/on_original_data" \
# --gradient_accumulation_steps 2 --do_lower_case >squad1.out

# nohup python superbert_run_en_classifier.py --data_dir "data/squad_data" \
# --model "model/original_pretrained_model" --task_name "squad2" \
# --output_dir "model/fine-tuned_pretrained_model/squad2/on_original_data" \
# --gradient_accumulation_steps 2 --do_lower_case --version_2_with_negative 1 >squad2.out

# transformer distillation
# nohup python task_distill.py --teacher_model "model\fine-tuned_pretrained_model\cola\on_original_data" \
# --student_model "model\distilled_pretrained_model\2nd_General_TinyBERT_4L_312D" --data_dir "data\glue_data\CoLA" \
# --task_name "cola" --output_dir "model\distilled_fine-tuned_model\cola\on_original_data" --max_seq_length 128 \
# --do_lower_case > "model\distilled_fine-tuned_model\on_original_data_out\cola.out"

# nohup python task_distill.py --teacher_model "model\fine-tuned_pretrained_model\mrpc\on_original_data" \
# --student_model "model\distilled_pretrained_model\2nd_General_TinyBERT_4L_312D" --data_dir "data\glue_data\MRPC" \
# --task_name "mrpc" --output_dir "model\distilled_fine-tuned_model\mrpc\on_original_data" --max_seq_length 128 \
# --do_lower_case > "model\distilled_fine-tuned_model\on_original_data_out\mrpc.out"

# nohup python task_distill.py --teacher_model "model\fine-tuned_pretrained_model\mnli\on_original_data" \
# --student_model "model\distilled_pretrained_model\2nd_General_TinyBERT_4L_312D" --data_dir "data\glue_data\MNLI" \
# --task_name "mnli" --output_dir "model\distilled_fine-tuned_model\mnli\on_original_data" --max_seq_length 128 \
# --do_lower_case > "model\distilled_fine-tuned_model\on_original_data_out\mnli.out"

# nohup python task_distill.py --teacher_model "model\fine-tuned_pretrained_model\wnli\on_original_data" \
# --student_model "model\distilled_pretrained_model\2nd_General_TinyBERT_4L_312D" --data_dir "data\glue_data\WNLI" \
# --task_name "wnli" --output_dir "model\distilled_fine-tuned_model\wnli\on_original_data" --max_seq_length 128 \
# --do_lower_case > "model\distilled_fine-tuned_model\on_original_data_out\wnli.out"

# nohup python task_distill.py --teacher_model "model\fine-tuned_pretrained_model\sst-2\on_original_data" \
# --student_model "model\distilled_pretrained_model\2nd_General_TinyBERT_4L_312D" --data_dir "data\glue_data\SST-2" \
# --task_name "sst-2" --output_dir "model\distilled_fine-tuned_model\sst-2\on_original_data" --max_seq_length 128 \
# --do_lower_case > "model\distilled_fine-tuned_model\on_original_data_out\sst-2.out"

# nohup python task_distill.py --teacher_model "model\fine-tuned_pretrained_model\sts-b\on_original_data" \
# --student_model "model\distilled_pretrained_model\2nd_General_TinyBERT_4L_312D" --data_dir "data\glue_data\STS-B" \
# --task_name "sts-b" --output_dir "model\distilled_fine-tuned_model\sts-b\on_original_data" --max_seq_length 128 \
# --do_lower_case > "model\distilled_fine-tuned_model\on_original_data_out\sts-b.out"

# nohup python task_distill.py --teacher_model "model\fine-tuned_pretrained_model\qqp\on_original_data" \
# --student_model "model\distilled_pretrained_model\2nd_General_TinyBERT_4L_312D" --data_dir "data\glue_data\QQP" \
# --task_name "qqp" --output_dir "model\distilled_fine-tuned_model\qqp\on_original_data" --max_seq_length 128 \
# --do_lower_case > "model\distilled_fine-tuned_model\on_original_data_out\qqp.out"

# nohup python task_distill.py --teacher_model "model\fine-tuned_pretrained_model\qnli\on_original_data" \
# --student_model "model\distilled_pretrained_model\2nd_General_TinyBERT_4L_312D" --data_dir "data\glue_data\QNLI" \
# --task_name "qnli" --output_dir "model\distilled_fine-tuned_model\qnli\on_original_data" --max_seq_length 128 \
# --do_lower_case > "model\distilled_fine-tuned_model\on_original_data_out\qnli.out"

# nohup python task_distill.py --teacher_model "model\fine-tuned_pretrained_model\rte\on_original_data" \
# --student_model "model\distilled_pretrained_model\2nd_General_TinyBERT_4L_312D" --data_dir "data\glue_data\RTE" \
# --task_name "rte" --output_dir "model\distilled_fine-tuned_model\rte\on_original_data" --max_seq_length 128 \
# --do_lower_case > "model\distilled_fine-tuned_model\on_original_data_out\rte.out"

# prediction distillation

# nohup python task_distill.py --pred_distill --teacher_model "model\fine-tuned_pretrained_model\cola\on_original_data" \
# --student_model "model\distilled_fine-tuned_model\cola\on_original_data" --data_dir "data\glue_data\CoLA" \
# --task_name "cola" --output_dir "model\distilled_fine-tuned_model\cola\on_original_data_final" --max_seq_length 128 \
# --do_lower_case --learning_rate 3e-5  --num_train_epochs  5  > "model\distilled_fine-tuned_model\on_original_data_final_out\cola.out"

# nohup python task_distill.py --pred_distill --teacher_model "model\fine-tuned_pretrained_model\mnli\on_original_data" \
# --student_model "model\distilled_fine-tuned_model\mnli\on_original_data" --data_dir "data\glue_data\MNLI" \
# --task_name "mnli" --output_dir "model\distilled_fine-tuned_model\mnli\on_original_data_final" --max_seq_length 128 \
# --do_lower_case --learning_rate 3e-5  --num_train_epochs  5  > "model\distilled_fine-tuned_model\on_original_data_final_out\mnli.out"

# nohup python task_distill.py --pred_distill --teacher_model "model\fine-tuned_pretrained_model\mrpc\on_original_data" \
# --student_model "model\distilled_fine-tuned_model\mrpc\on_original_data" --data_dir "data\glue_data\MRPC" \
# --task_name "mrpc" --output_dir "model\distilled_fine-tuned_model\mrpc\on_original_data_final" --max_seq_length 128 \
# --do_lower_case --learning_rate 3e-5  --num_train_epochs  5  > "model\distilled_fine-tuned_model\on_original_data_final_out\mrpc.out"

# nohup python task_distill.py --pred_distill --teacher_model "model\fine-tuned_pretrained_model\qnli\on_original_data" \
# --student_model "model\distilled_fine-tuned_model\qnli\on_original_data" --data_dir "data\glue_data\QNLI" \
# --task_name "qnli" --output_dir "model\distilled_fine-tuned_model\qnli\on_original_data_final" --max_seq_length 128 \
# --do_lower_case --learning_rate 3e-5  --num_train_epochs  5  > "model\distilled_fine-tuned_model\on_original_data_final_out\qnli.out"

# nohup python task_distill.py --pred_distill --teacher_model "model\fine-tuned_pretrained_model\qqp\on_original_data" \
# --student_model "model\distilled_fine-tuned_model\qqp\on_original_data" --data_dir "data\glue_data\QQP" \
# --task_name "qqp" --output_dir "model\distilled_fine-tuned_model\qqp\on_original_data_final" --max_seq_length 128 \
# --do_lower_case --learning_rate 3e-5  --num_train_epochs  5  > "model\distilled_fine-tuned_model\on_original_data_final_out\qqp.out"

# nohup python task_distill.py --pred_distill --teacher_model "model\fine-tuned_pretrained_model\rte\on_original_data" \
# --student_model "model\distilled_fine-tuned_model\rte\on_original_data" --data_dir "data\glue_data\RTE" \
# --task_name "rte" --output_dir "model\distilled_fine-tuned_model\rte\on_original_data_final" --max_seq_length 128 \
# --do_lower_case --learning_rate 3e-5  --num_train_epochs  5  > "model\distilled_fine-tuned_model\on_original_data_final_out\rte.out"

# nohup python task_distill.py --pred_distill --teacher_model "model\fine-tuned_pretrained_model\sst-2\on_original_data" \
# --student_model "model\distilled_fine-tuned_model\sst-2\on_original_data" --data_dir "data\glue_data\SST-2" \
# --task_name "sst-2" --output_dir "model\distilled_fine-tuned_model\sst-2\on_original_data_final" --max_seq_length 128 \
# --do_lower_case --learning_rate 3e-5  --num_train_epochs  5  > "model\distilled_fine-tuned_model\on_original_data_final_out\sst-2.out"

# nohup python task_distill.py --pred_distill --teacher_model "model\fine-tuned_pretrained_model\sts-b\on_original_data" \
# --student_model "model\distilled_fine-tuned_model\sts-b\on_original_data" --data_dir "data\glue_data\STS-B" \
# --task_name "sts-b" --output_dir "model\distilled_fine-tuned_model\sts-b\on_original_data_final" --max_seq_length 128 \
# --do_lower_case --learning_rate 3e-5  --num_train_epochs  5  > "model\distilled_fine-tuned_model\on_original_data_final_out\sts-b.out"

# nohup python task_distill.py --pred_distill --teacher_model "model\fine-tuned_pretrained_model\wnli\on_original_data" \
# --student_model "model\distilled_fine-tuned_model\wnli\on_original_data" --data_dir "data\glue_data\WNLI" \
# --task_name "wnli" --output_dir "model\distilled_fine-tuned_model\wnli\on_original_data_final" --max_seq_length 128 \
# --do_lower_case --learning_rate 3e-5  --num_train_epochs  5  > "model\distilled_fine-tuned_model\on_original_data_final_out\wnli.out"

