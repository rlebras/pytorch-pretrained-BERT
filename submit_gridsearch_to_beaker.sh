#!/bin/bash

name_prefix="wsc_bert"
dataset="ds_xh0l1b9lg24m"

batchsizes=( 8 )
for s in "${batchsizes[@]}"
do
    learningrates=( 2e-5 3e-5 5e-5 )
    for l in "${learningrates[@]}"
    do
        cmd="python /examples/run_classifier.py --task_name wsc --do_eval --do_train --bert_model bert-large-uncased --max_seq_length 128 --train_batch_size ${s} --learning_rate ${l} --num_train_epochs 3 --output_dir /output/batch_${s}_lr_${l}_epochs_3 --data_dir /data/ --output_file_for_pred /output/batch_${s}_lr_${l}_epochs_3valid.out.jsonl"
        ./scripts/python/cmd_with_beaker.py --gpu-count 1 --name ${name_prefix}_batch${s}_lr${l}  --source ${dataset}:/data/ --cmd="${cmd}"
    done
done
