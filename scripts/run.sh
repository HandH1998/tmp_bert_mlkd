# export TASK_NAME=mrpc

# python run_glue_no_trainer.py \
#   --model_name_or_path model/distilled_pretrained_model/2nd_General_TinyBERT_4L_312D/pytorch_model.bin \
#   --task_name $TASK_NAME \
#   --max_length 128 \
#   --per_device_train_batch_size 32 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 3 \
#   --output_dir model/fine-tuned_pretrained_model/$TASK_NAME
  
# python run_glue_no_trainer.py \
#   --model_name_or_path model/distilled_pretrained_model/2nd_General_TinyBERT_4L_312D \
#   --task_name mrpc \
#   --max_length 128 \
#   --per_device_train_batch_size 32 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 3 \
#   --output_dir model/fine-tuned_pretrained_model/mrpc

# python superbert_run_en_classifier.py --data_dir "dataset/glue/MNLI dataset/SQuAD" \
#  --model model/SuperBERT_MLM/ --task_name "mnli squad" --output_dir output/ \
#  --do_lower_case --arches_file cands/1st_generation.fast.cands 

# python data_augmentation.py --pretrained_bert_model ${BERT_BASE_DIR}$ \
#                             --glove_embs ${GLOVE_EMB}$ \
#                             --glue_dir ${GLUE_DIR}$ \  
#                             --task_name ${TASK_NAME}$

# data augmentation
# python data_augmentation.py --pretrained_bert_model model/original_pretrained_model \
#                             --glove_embs data/glove/glove.840B.300d.txt \
#                             --glue_dir data/glue_data --task_name MRPC > MRPC_aug.out
# python data_augmentation.py --pretrained_bert_model model/original_pretrained_model \
#                             --glove_embs data/glove/glove.840B.300d.txt \
#                             --glue_dir data/glue_data --task_name CoLA > CoLA_aug.out
# python data_augmentation.py --pretrained_bert_model model/original_pretrained_model \
#                             --glove_embs data/glove/glove.840B.300d.txt \
#                             --glue_dir data/glue_data --task_name  SST-2 > SST-2_aug.out
# python data_augmentation.py --pretrained_bert_model model/original_pretrained_model \
#                             --glove_embs data/glove/glove.840B.300d.txt \
#                             --glue_dir data/glue_data --task_name  STS-B > STS-B_aug.out
# python data_augmentation.py --pretrained_bert_model model/original_pretrained_model \
#                             --glove_embs data/glove/glove.840B.300d.txt \
#                             --glue_dir data/glue_data --task_name  QQP > QQP_aug.out
# python data_augmentation.py --pretrained_bert_model model/original_pretrained_model \
#                             --glove_embs data/glove/glove.840B.300d.txt \
#                             --glue_dir data/glue_data --task_name  MNLI > MNLI_aug.out
# python data_augmentation.py --pretrained_bert_model model/original_pretrained_model \
#                             --glove_embs data/glove/glove.840B.300d.txt \
#                             --glue_dir data/glue_data --task_name  QNLI > QNLI_aug.out
# python data_augmentation.py --pretrained_bert_model model/original_pretrained_model \
#                             --glove_embs data/glove/glove.840B.300d.txt \
#                             --glue_dir data/glue_data --task_name  RTE > RTE_aug.out

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

nohup python task_distill.py --pred_distill --teacher_model "model\fine-tuned_pretrained_model\sts-b\on_original_data" \
--student_model "model\distilled_fine-tuned_model\sts-b\on_original_data" --data_dir "data\glue_data\STS-B" \
--task_name "sts-b" --output_dir "model\distilled_fine-tuned_model\sts-b\on_original_data_final" --max_seq_length 128 \
--do_lower_case --learning_rate 3e-5  --num_train_epochs  5  > "model\distilled_fine-tuned_model\on_original_data_final_out\sts-b.out"

# nohup python task_distill.py --pred_distill --teacher_model "model\fine-tuned_pretrained_model\wnli\on_original_data" \
# --student_model "model\distilled_fine-tuned_model\wnli\on_original_data" --data_dir "data\glue_data\WNLI" \
# --task_name "wnli" --output_dir "model\distilled_fine-tuned_model\wnli\on_original_data_final" --max_seq_length 128 \
# --do_lower_case --learning_rate 3e-5  --num_train_epochs  5  > "model\distilled_fine-tuned_model\on_original_data_final_out\wnli.out"

