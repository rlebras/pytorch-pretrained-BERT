import argparse
import json
import os


def main(args):
    data_dir = args.data_dir
    output_dir = args.output_dir
    models_base_dir = args.models_base_dir
    task_name = args.task_name

    if not os.path.exists(models_base_dir):
        os.mkdir(models_base_dir)

    for batch_size in [8, 16]:
        for lr in [2e-5, 3e-5, 5e-5]:
            for num_epochs in [3, 4]:
                model_name = "batch_" + str(batch_size) + "_lr_" + str(lr) + "_epochs_" + str(num_epochs)
                cmd = [
                    "python /examples/run_classifier.py",
                    "--task_name ",
                    task_name,
                    "--do_eval",
                    "--do_train",
                    "--bert_model bert-large-uncased",
                    "--max_seq_length 128",
                    "--train_batch_size ",
                    str(batch_size),
                    "--learning_rate ",
                    str(lr),
                    "--num_train_epochs ",
                    str(num_epochs),
                    "--output_dir",
                    os.path.join(models_base_dir, model_name),
                    "--data_dir",
                    data_dir,
                    "--output_file_for_pred",
                    os.path.join(output_dir, model_name + "valid.out.jsonl"),
                ]
                print(' '.join(cmd))
                os.system(' '.join(cmd))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument('--data_dir',
                        type=str,
                        help='Data directory')
    parser.add_argument('--output_dir',
                        type=str,
                        help='Output directory')
    parser.add_argument('--models_base_dir',
                        type=str,
                        help='Models base dir')
    parser.add_argument('--task_name',
                        type=str,
                        help='Task Name')

    args = parser.parse_args()

    # Run seed selection if args valid
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=True))
    print("=======================")
    main(args)
