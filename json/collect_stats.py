import os

task_suffix = '_tagger_crf_template'
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
    'semtr_semtraits',
    'ccg_ccg'
]

filepaths = ['/data/tagger/' + td + task_suffix + '/vocabulary' for td in task_domains]
for filepath in filepaths:
    print(filepath)
    with open(os.path.join(filepath, 'tokens.txt'), 'r') as fr:
        cnt = 0
        for i, l in enumerate(fr):
            if i == 0 and l != '@@UNKNOWN@@\n':
                print("Warnings:", l)
            if l.strip() != '':
                cnt += 1
            else:
                print(i, l)
        print("vocab size: ", cnt-1)
    with open(os.path.join(filepath, 'labels.txt'), 'r') as fr:
        cnt = 0
        for i, l in enumerate(fr):
            if l.strip() != '':
                cnt += 1
            else:
                print(i, l)
        print("vocab size: ", cnt)



