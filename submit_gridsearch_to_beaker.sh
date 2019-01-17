#!/bin/bash

#task="anli"
task="wsc"
#name_prefix="${task}_amt_posneg_init_bert"
name_prefix="${task}_toy_bert_newsplit_t2"
#dataset="ds_xf3e7pq1o4zq"
dataset="ds_x0lvhb1lkhoa"

#batchsizes=( 2 3 4 8 )
batchsizes=( 3 )
for s in "${batchsizes[@]}"
do
    #learningrates=( 2e-5 3e-5 5e-5 )
    learningrates=( 2e-5 )
    for l in "${learningrates[@]}"
    do
        #epochs=( 3 4 )
        epochs=( 3 )
        for e in "${epochs[@]}"
        do
            cmd="python /examples/run_classifier.py --task_name ${task} --do_eval --do_train --bert_model bert-large-uncased --max_seq_length 128 --train_batch_size ${s} --learning_rate ${l} --num_train_epochs ${e} --output_dir /output/batch_${s}_lr_${l}_epochs${e} --data_dir /data/ --output_file_for_pred /output/batch_${s}_lr_${l}_epochs${e}_valid.out.jsonl"
            ./scripts/python/cmd_with_beaker.py --gpu-count 1 --name ${name_prefix}_batch${s}_lr${l}_epoch${e}  --source ${dataset}:/data/ --cmd="${cmd}"
        done
    done
done
