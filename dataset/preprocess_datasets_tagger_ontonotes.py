import os
import dataset_utils
from dataset_utils import update_tag_scheme

"""
dataset_dir = "./dataset/ontonotes_chunking"
processed_dataset_dir = "./dataset/ontonotes_chunking_evaluate"
for fn in os.listdir(dataset_dir):
    load_func = dataset_utils.load_sentences_general
    sentences = {}
    testpath = os.path.join(dataset_dir, fn)
    sentences['test'] = load_func(testpath, zeros=False, lower=False)
    task_list = ["chunk"]
    tag_scheme_list = ["iobes"]
    assert(len(task_list) == len(tag_scheme_list))
    for idx, task in enumerate(task_list):
        print(idx, task)
        tag_scheme = tag_scheme_list[idx]
        if tag_scheme != "none":
            assert(tag_scheme in ['iob', 'iobes'])
            # Use selected tagging scheme (IOB / IOBES)
            update_tag_scheme(sentences['test'], tag_scheme, idx+1)
        for data_subset in ['test']:
            filename = task + "_" + '-'.join(fn.split('_')[1:]).replace('.chunks', '')
            # if tag_scheme == 'iobes':
            #     filename = task + '-iobes' + "_" + opts.domain + "_" + data_subset + '.txt'
            os.mkdir(os.path.join(processed_dataset_dir, filename))
            with open(os.path.join(processed_dataset_dir, filename, 'chunk_conll02_test.txt'), 'w') as f:
                for s in sentences[data_subset]:
                    for w in s:
                        # print(w)
                        f.write(str(w[0]) + "<Deli>" + str(w[idx+1]) + " ")
                    f.write("\n")
"""

dataset_dir = "./dataset/ontonotes_xpos"
processed_dataset_dir = "./dataset/ontonotes_xpos_evaluate"
for fn in os.listdir(dataset_dir):
    load_func = dataset_utils.load_sentences_general
    sentences = {}
    testpath = os.path.join(dataset_dir, fn)
    sentences['test'] = load_func(testpath, zeros=False, lower=False)
    task_list = ["xpos"]
    tag_scheme_list = ["none"]
    assert(len(task_list) == len(tag_scheme_list))
    for idx, task in enumerate(task_list):
        print(idx, task)
        tag_scheme = tag_scheme_list[idx]
        if tag_scheme != "none":
            assert(tag_scheme in ['iob', 'iobes'])
            # Use selected tagging scheme (IOB / IOBES)
            update_tag_scheme(sentences['test'], tag_scheme, idx+1)
        for data_subset in ['test']:
            filename = task + "_" + '-'.join(fn.split('_')[1:]).replace('.tags', '')
            # if tag_scheme == 'iobes':
            #     filename = task + '-iobes' + "_" + opts.domain + "_" + data_subset + '.txt'
            os.mkdir(os.path.join(processed_dataset_dir, filename))
            with open(os.path.join(processed_dataset_dir, filename, 'xpos_uni_test.txt'), 'w') as f:
                for s in sentences[data_subset]:
                    for w in s:
                        # print(w)
                        f.write(str(w[0]) + "<Deli>" + str(w[idx+1]) + " ")
                    f.write("\n")