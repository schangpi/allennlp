import os
import optparse
import dataset_utils
from dataset_utils import update_tag_scheme, word_mapping, tag_mapping
"""
python dataset/preprocess_datasets_seq2seq.py --domain uni --tasks xpos_upos --tag_schemes none_none
python dataset/preprocess_datasets_seq2seq.py --domain conll03 --tasks xpos_chunk_ner --tag_schemes none_iobes_iobes
python dataset/preprocess_datasets_seq2seq.py --domain conll02 --tasks chunk --tag_schemes iobes
"""

# Read parameters from command line
optparser = optparse.OptionParser()
optparser.add_option("--src_dir", default="./dataset", help="Directory for raw data")
optparser.add_option("--tgt_dir", default="./dataset/seq2seq", help="Directory for preprocessed data")
optparser.add_option("--domain", default="", help="Domain name")
optparser.add_option("--tasks", default="", help="Task names in order")
optparser.add_option("--tag_schemes", default=None, help="Tagging scheme (IOB or IOBES) in order")
# Paths to train, validation, and test data
optparser.add_option("--train", default="", help="Train set location")
optparser.add_option("--dev", default="", help="Dev set location")
optparser.add_option("--test", default="", help="Test set location")
opts = optparser.parse_args()[0]

dataset_dir = opts.src_dir
processed_dataset_dir = opts.tgt_dir
if opts.domain== 'uni':
    opts.train = os.path.join(dataset_dir, "uni/en-ud-train.conllu")
    opts.dev = os.path.join(dataset_dir, "uni/en-ud-dev.conllu")
    opts.test = os.path.join(dataset_dir, "uni/en-ud-test.conllu")
    load_func = dataset_utils.load_sentences_uni
elif opts.domain == 'conll03':
    opts.train = os.path.join(dataset_dir, "conll03/eng.train")
    opts.dev = os.path.join(dataset_dir, "conll03/eng.testa")
    opts.test = os.path.join(dataset_dir, "conll03/eng.testb")
    load_func = dataset_utils.load_sentences_conll03
elif opts.domain == 'conll02':
    opts.train = os.path.join(dataset_dir, "conll02/eng_chunking_train.conll")
    opts.dev = os.path.join(dataset_dir, "conll02/eng_chunking_dev.conll")
    opts.test = os.path.join(dataset_dir, "conll02/eng_chunking_test.conll")
    load_func = dataset_utils.load_sentences_conll02

sentences = {}
sentences['train'] = load_func(opts.train, zeros=False, lower=False)
sentences['dev'] = load_func(opts.dev, zeros=False, lower=False)
sentences['test'] = load_func(opts.test, zeros=False, lower=False)

task_list = opts.tasks.split("_")
tag_scheme_list = opts.tag_schemes.split("_")

assert(len(task_list) == len(tag_scheme_list))

# Write sequence to sequence
for idx, task in enumerate(task_list):
    print(idx, task)
    tag_scheme = tag_scheme_list[idx]
    if tag_scheme != "none":
        assert(tag_scheme in ['iob', 'iobes'])
        # Use selected tagging scheme (IOB / IOBES)
        update_tag_scheme(sentences['train'], tag_scheme, idx+1)
        update_tag_scheme(sentences['dev'], tag_scheme, idx+1)
        update_tag_scheme(sentences['test'], tag_scheme, idx+1)
    for data_subset in ['train', 'dev', 'test']:
        with open(os.path.join(processed_dataset_dir, data_subset,
                               task + "_" + opts.domain + "_" + data_subset + '.txt'), 'w') as f:
            for s in sentences[data_subset]:
                for w in s:
                    f.write(str(w[0]) + " ")
                f.write("\t")
                for w in s:
                    f.write(str(w[idx+1]) + " ")
                f.write("\n")

# Create a dictionary and a mapping for words and tags
_, _, id_to_word_train = word_mapping(sentences['train'], idx=0)
_, _, id_to_word = word_mapping(sentences['train'] + sentences['dev'] + sentences['test'], idx=0)
_, _, id_to_tag = tag_mapping(sentences['train'] + sentences['dev'] + sentences['test'], idx=-1)
with open(os.path.join(processed_dataset_dir, opts.domain + '_vocab.source.train'), 'w') as f:
    for k, v in id_to_word_train.items():
        f.write(v + "\n")
with open(os.path.join(processed_dataset_dir, 'vocab_stats', opts.domain + '_vocab.source'), 'w') as f:
    for k, v in id_to_word.items():
        f.write(v + "\n")
with open(os.path.join(processed_dataset_dir, 'vocab_stats', opts.domain + '_vocab.target'), 'w') as f:
    for k, v in id_to_tag.items():
        f.write(v + "\n")