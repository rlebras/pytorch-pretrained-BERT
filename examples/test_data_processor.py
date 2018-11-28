from examples.run_classifier import AnliWithCSKProcessor, convert_examples_to_features_mc
from pytorch_pretrained_bert import BertTokenizer

dir = "../../abductive-nli/data/abductive_nli/one2one-correspondence/anli_with_csk/"

processor = AnliWithCSKProcessor()

examples = processor.get_train_examples(dir)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

label_list = processor.get_labels()
max_seq_length = 128
features = convert_examples_to_features_mc(examples, label_list, max_seq_length, tokenizer)


print("OK")