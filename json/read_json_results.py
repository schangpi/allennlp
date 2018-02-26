import json
import os

def decode(data, path):
    pathlist = path.split('/')
    e = data
    for k in pathlist:
        e = e[k]
    return e

def set_data(data, path, value):
    pathlist = path.split('/')
    e = data
    for k in pathlist[:-1]:
        e = e[k]
    e[pathlist[-1]] = value
    return data

pemb = {
    'g50':'https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.50d.txt.gz',
    'g100':'https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz',
    'g300':'/data/word_vectors/glove.840B.300d.txt.gz',
    'f300':'/data/word_vectors/wiki.en.vec.gz'
}
pemb_dim = {
    'g50':50,
    'g100':100,
    'g300':300,
    'f300':300
}

pemb_gpu = {
    'g50':0,
    'g100':0,
    'g300':1,
    'f300':2
}

choices = {}
# c0
choices["model/text_field_embedder/tokens/pretrained_file"] = list(pemb.keys())
word_dim_field = "model/text_field_embedder/tokens/embedding_dim"
# c1
choices["model/text_field_embedder/token_characters/embedding/embedding_dim"] = [25]
# c2
choices["model/text_field_embedder/token_characters/encoder/input_size"] = [25]
# c3
choices["model/text_field_embedder/token_characters/encoder/hidden_size"] = [25, 50, 75]
# c4
choices["model/text_field_embedder/token_characters/encoder/dropout"] = [0.1, 0.3, 0.5]

input_size_field = "model/stacked_encoder/input_size"
# c0 + 2*c3

# c5
choices["model/stacked_encoder/hidden_size"] = [100, 200, 300, 500]
# c6
choices["model/stacked_encoder/dropout"] = [0.1, 0.3, 0.5]
# c7
choices["iterator/batch_size"] = [16, 32, 64]

fields = [
    "model/text_field_embedder/tokens/pretrained_file",
    "model/text_field_embedder/token_characters/embedding/embedding_dim",
    "model/text_field_embedder/token_characters/encoder/input_size",
    "model/text_field_embedder/token_characters/encoder/hidden_size",
    "model/text_field_embedder/token_characters/encoder/dropout",
    "model/stacked_encoder/hidden_size",
    "model/stacked_encoder/dropout",
    "iterator/batch_size"
]

all_commands = []
with open('tagger_crf_template.json', 'r') as f:
    data = json.load(f)
    keys = choices.keys()

    cnt = 0
    sel = 0
    for c0 in choices[fields[0]]:
        data = set_data(data, fields[0], pemb[c0])
        data = set_data(data, word_dim_field, pemb_dim[c0])
        for c1 in choices[fields[1]]:
            data = set_data(data, fields[1], c1)
            for c2 in choices[fields[2]]:
                data = set_data(data, fields[2], c2)
                for c3 in choices[fields[3]]:
                    data = set_data(data, fields[3], c3)
                    data = set_data(data, input_size_field, pemb_dim[c0] + 2 * c3)
                    for c4 in choices[fields[4]]:
                        data = set_data(data, fields[4], c4)
                        for c5 in choices[fields[5]]:
                            data = set_data(data, fields[5], c5)
                            for c6 in choices[fields[6]]:
                                data = set_data(data, fields[6], c6)
                                for c7 in choices[fields[7]]:
                                    data = set_data(data, fields[7], c7)
                                    cnt += 1
                                    file_ext = '_'.join([str(c) for c in [c0, c1, c2, c3, c4, c5, c6, c7]])
                                    # print(file_ext)
                                    res_filepath = '/data/tagger/ner-iobes_conll03_tagcrf_' + file_ext + '/metrics.json'
                                    if os.path.exists(res_filepath):
                                        with open(res_filepath, 'r') as fr:
                                            results = json.load(fr)
                                            print(file_ext,
                                                  results['test_f1-measure-overall'],
                                                  results['validation_f1-measure-overall'])




