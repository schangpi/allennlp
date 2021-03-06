import os
import optparse

import dataset_utils
from dataset_utils import update_tag_scheme

"""
python dataset/preprocess_datasets_tagger.py --domain uni --tasks xpos_upos --tag_schemes none_none
python dataset/preprocess_datasets_tagger.py --domain conll03 --tasks xpos_chunk_ner --tag_schemes none_iobes_iobes
python dataset/preprocess_datasets_tagger.py --domain conll02 --tasks chunk --tag_schemes iobes
python dataset/preprocess_datasets_tagger.py --domain conll03 --tasks xpos_chunk_ner --tag_schemes none_none_none
python dataset/preprocess_datasets_tagger.py --domain conll02 --tasks chunk --tag_schemes none
python dataset/preprocess_datasets_tagger.py --domain ccg --tasks ccg --tag_schemes none
python dataset/preprocess_datasets_tagger.py --domain semtraits --tasks semtr --tag_schemes none
python dataset/preprocess_datasets_tagger.py --domain streusle --tasks xpos_upos_mwe_smwe_wmwe_supsense --tag_schemes none_none_none_none_none_none
python dataset/preprocess_datasets_tagger.py --domain broadcast1 --tasks com --tag_schemes none
python dataset/preprocess_datasets_tagger.py --domain broadcast2 --tasks com --tag_schemes none
python dataset/preprocess_datasets_tagger.py --domain broadcast3 --tasks com --tag_schemes none
python dataset/preprocess_datasets_tagger.py --domain semcor --tasks xpos_wsd_sem --tag_schemes none_none_none
python dataset/preprocess_datasets_tagger.py --domain fnt --tasks frame --tag_schemes none
python dataset/preprocess_datasets_tagger.py --domain hyp --tasks hyp --tag_schemes none
"""

# python dataset/preprocess_datasets_tagger.py --domain google --tasks com --tag_schemes none

# Read parameters from command line
optparser = optparse.OptionParser()
optparser.add_option("--src_dir", default="./dataset", help="Directory for raw data")
optparser.add_option("--tgt_dir", default="./dataset/tagger", help="Directory for preprocessed data")
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
elif opts.domain == 'ccg':
    opts.train = os.path.join(dataset_dir, "ccg/eng_ccg_train.conll")
    opts.dev = os.path.join(dataset_dir, "ccg/eng_ccg_dev.conll")
    opts.test = os.path.join(dataset_dir, "ccg/eng_ccg_test.conll")
    load_func = dataset_utils.load_sentences_general
elif opts.domain == 'semcor':
    opts.train = os.path.join(dataset_dir, "semcor/semcor_train.conll")
    opts.dev = os.path.join(dataset_dir, "semcor/semcor_dev.conll")
    opts.test = os.path.join(dataset_dir, "semcor/semcor_test.conll")
    load_func = dataset_utils.load_sentences_general
elif opts.domain == 'semtraits':
    opts.train = os.path.join(dataset_dir, "semtraits/semcor_semantictraits_train.conll.txt")
    opts.dev = os.path.join(dataset_dir, "semtraits/semcor_semantictraits_dev.conll.txt")
    opts.test = os.path.join(dataset_dir, "semtraits/semcor_semantictraits_test.conll.txt")
    load_func = dataset_utils.load_sentences_general
elif opts.domain == 'streusle':
    opts.train = os.path.join(dataset_dir, "streusle/streusle.ud_train.json")
    opts.dev = os.path.join(dataset_dir, "streusle/streusle.ud_dev.json")
    opts.test = os.path.join(dataset_dir, "streusle/streusle.ud_test.json")
    load_func = dataset_utils.load_sentences_streusle
elif 'broadcast' in opts.domain:
    opts.train = os.path.join(dataset_dir, "broadcast/" + opts.domain + "_com_train.conll")
    opts.dev = os.path.join(dataset_dir, "broadcast/" + opts.domain + "_com_dev.conll")
    opts.test = os.path.join(dataset_dir, "broadcast/" + opts.domain + "_com_test.conll")
    load_func = dataset_utils.load_sentences_general_no_O
elif opts.domain == 'fnt':
    opts.train = os.path.join(dataset_dir, "fnt/eng_fnt_train.conll")
    opts.dev = os.path.join(dataset_dir, "fnt/eng_fnt_dev.conll")
    opts.test = os.path.join(dataset_dir, "fnt/eng_fnt_test.conll")
    load_func = dataset_utils.load_sentences_general_noutfok
elif opts.domain == 'hyp':
    opts.train = os.path.join(dataset_dir, "hyp/eng_hyperlinks_train.conll")
    opts.dev = os.path.join(dataset_dir, "hyp/eng_hyperlinks_dev.conll")
    opts.test = os.path.join(dataset_dir, "hyp/eng_hyperlinks_test.conll")
    load_func = dataset_utils.load_sentences_general_noutfok
"""
elif opts.domain == 'google':
    opts.train = os.path.join(dataset_dir, "com/streusle.ud_train.conllulex")
    opts.dev = os.path.join(dataset_dir, "com/streusle.ud_dev.conllulex")
    opts.test = os.path.join(dataset_dir, "com/streusle.ud_test.conllulex")
    load_func = dataset_utils.load_sentences_streusle

"""

sentences = {}
sentences['train'] = load_func(opts.train, zeros=False, lower=False)
sentences['dev'] = load_func(opts.dev, zeros=False, lower=False)
sentences['test'] = load_func(opts.test, zeros=False, lower=False)

task_list = opts.tasks.split("_")
tag_scheme_list = opts.tag_schemes.split("_")

assert(len(task_list) == len(tag_scheme_list))

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
        filename = task + "_" + opts.domain + "_" + data_subset + '.txt'
        if tag_scheme == 'iobes':
            filename = task + '-iobes' + "_" + opts.domain + "_" + data_subset + '.txt'
        with open(os.path.join(processed_dataset_dir, data_subset, filename), 'w') as f:
            for s in sentences[data_subset]:
                for w in s:
                    # print(w)
                    f.write(str(w[0]) + "<Deli>" + str(w[idx+1]) + " ")
                f.write("\n")