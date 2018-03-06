import json
import os

task_suffix = '_tagger_template'
task_crf_suffix = '_tagger_crf_template'
task_domains = [
    'upos_uni',
    'upos_streusle',
    'xpos_uni',
    'xpos_streusle',
    'xpos_conll03',
    'chunk-iobes_conll02',
    'chunk-iobes_conll03',
    'chunk_conll02',
    'chunk_conll03',
    'com_broadcast1',
    'com_broadcast2',
    'com_broadcast3',
    'ner-iobes_conll03',
    'ner_conll03',
    'supsense_streusle',
    'mwe_streusle',
    'smwe_streusle',
    'sem_semcor',
    'semtr_semtraits']
# 'ccg_ccg'

multi_path = '/data/tagger/multitagger_multi_'
task_emb_path = '/data/tagger/taskembtagger_task_embedding_tagger_'
task_emb_prepend_path = '/data/tagger/taskembtagger_task_prepend_embedding_tagger_'
exts = ['xpos', 'upos', 'uni', 'conll03', 'uposchunk']
exts += ["nerupos", "nerxpos", "chunkner", "nersemtr", "nersem", "nersupsense", "comner",
         "commwe", "chunkmwe", "mwexpos", "mweupos", "mwesupsense", "mwesemtr", "mwesem", "mwener",
         "semxpos", "semupos", "chunksem", "semsemtr", "comsem", "semsupsense",
         "semtrxpos", "semtrupos", "chunksemtr", "comsemtr", "semtrsupsense",
         "supsensexpos", "supsenseupos", "chunksupsense", "comsupsense",
         "comxpos", "comupos", "chunkcom"
         ]

filepaths = []
filepaths += ['/data/tagger/' + td + task_suffix for td in task_domains]
filepaths += ['/data/tagger/' + td + task_crf_suffix for td in task_domains]
filepaths += ([multi_path + ext for ext in exts] +
              [task_emb_path + ext for ext in exts] +
              [task_emb_prepend_path + ext for ext in exts])
# print(filepaths)
for filepath in filepaths:
    res_filepath = os.path.join(filepath, 'metrics.json')
    if os.path.exists(res_filepath):
        with open(res_filepath, 'r') as fr:
            results = json.load(fr)
            print(filepath)
            for k in results:
                print(k, results[k])
            print('')
            # print(results['test_f1-measure-overall'],
            #       results['validation_f1-measure-overall'])
    else:
        print(filepath, ' does not exist')




