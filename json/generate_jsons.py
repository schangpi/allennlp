import json

def decode(data, path):
    pathlist = path.split('/')
    e = data
    for k in pathlist:
        e = e[k]
    return e

choices = {}
choices["dataset_reader/source_token_indexers/tokens/lowercase_tokens"] = [True, False]
choices["train_data_path"] = ['./dataset/seq2seq/train/upos_uni_train.txt',
                              './dataset/seq2seq/train/ner_conll03_train.txt',
                              './dataset/seq2seq/train/chunk_conll02_train.txt']
choices["validation_data_path"] = ['./dataset/seq2seq/dev/upos_uni_dev.txt',
                                   './dataset/seq2seq/dev/ner_conll03_dev.txt',
                                   './dataset/seq2seq/dev/chunk_conll02_dev.txt']
choices["test_data_path"] = ['./dataset/seq2seq/test/upos_uni_test.txt',
                             './dataset/seq2seq/test/ner_conll03_test.txt',
                             './dataset/seq2seq/test/chunk_conll02_test.txt']
choices["model/source_embedder/"] = ''
choices["model/encoder/"] = ''

with open('seq2seq_template.json', 'r') as f:
    data = json.load(f)
    print(decode(data, 'dataset_reader/source_token_indexers/tokens/lowercase_tokens'))
