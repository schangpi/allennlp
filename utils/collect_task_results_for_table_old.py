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
all_tasks = ["upos", "xpos", "chunk", "ner", "mwe", "sem", "semtr", "supsense", "com"]

current_tsks = ["upos", "upos",
                "xpos", "xpos", "xpos",
                "chunk", "chunk",
                "ner",
                "mwe",
                "sem",
                "semtr",
                "supsense",
                "com", "com", "com"]
current_tsk_domains = ["upos_uni", "upos_streusle",
                       "xpos_uni", "xpos_streusle", "xpos_conll03",
                       "chunk_conll02", "chunk_conll03",
                       "ner_conll03",
                       "mwe_streusle",
                       "sem_semcor",
                       "semtr_semtraits",
                       "supsense_streusle",
                       "com_broadcast1",
                       "com_broadcast2",
                       "com_broadcast3"]
for current_tsk, current_tsk_domain in zip(current_tsks, current_tsk_domains):
    # print(current_tsk, current_tsk_domain)
    top_table = '\\begin{table*}[t]\n\\centering\n\\footnotesize{\n\\begin{tabular}{c|c|c|c}\n'
    top_table += 'Trained with & \\multicolumn{3}{|c}{\\task{' + current_tsk
    top_table += '} on ' + current_tsk_domain.split('_')[-1]
    top_table += '} \\\\ \\cline{2-4}\n'
    top_table += '& Multiple Decoders & Task Embeddings (All Steps) & Task Embeddings (Prepend) \\\\ \\hline'
    print(top_table)
    current_tsk_domain_single = current_tsk_domain
    if 'ner' in current_tsk_domain:
        current_tsk_domain_single = current_tsk_domain.replace('ner', 'ner-iobes')
    if 'chunk' in current_tsk_domain:
        current_tsk_domain_single = current_tsk_domain.replace('chunk', 'chunk-iobes')
    other_tasks = []
    exts = []
    for tsk in all_tasks:
        if tsk == current_tsk:
            continue
        exts.append(''.join(sorted([tsk, current_tsk])))
        other_tasks.append(tsk)
    exts.append("all")
    other_tasks.append("all")

    res_filepath = os.path.join('/data/tagger/' + current_tsk_domain_single + task_crf_suffix, 'metrics.json')
    print("Self only & ", end=' ')
    with open(res_filepath, 'r') as fr:
        results = json.load(fr)
        print('\\multicolumn{3}{|c}{',end='')
        print(round(100 * results['test_f1-measure-overall'], 2), end='')
        print('}', end=' ')
    print(' \\\\ \\hline')
    filepaths = []
    filepaths += [multi_path + ext for ext in exts]
    te_filepaths = ['results/test_' + current_tsk_domain + '_task_embedding_tagger_' + ext + '_screenlog' for ext in exts]
    tpe_filepaths = ['results/test_' + current_tsk_domain + '_task_prepend_embedding_tagger_' + ext + '_screenlog'
                     for ext in exts]
    for i, filepath in enumerate(filepaths):
        print('+\\task{' + other_tasks[i] + '}', end=' ')
        print('&', end=' ')
        res_filepath = os.path.join(filepath, 'metrics.json')
        if os.path.exists(res_filepath):
            with open(res_filepath, 'r') as fr:
                results = json.load(fr)
                print(round(100*results['test_' + current_tsk + '-f1-measure-overall'], 2), end=' ')
        print('&', end=' ')
        res_filepath = te_filepaths[i]
        if os.path.exists(res_filepath):
            with open(res_filepath, 'r') as fr:
                for line in fr:
                    pass
                try:
                    print(round(100*float(line.split()[-1]), 2), end=' ')
                except:
                    pass
        print('&', end=' ')
        res_filepath = tpe_filepaths[i]
        if os.path.exists(res_filepath):
            with open(res_filepath, 'r') as fr:
                for line in fr:
                    pass
                try:
                    print(round(100*float(line.split()[-1]), 2), end=' ')
                except:
                    pass
        print(' \\\\ ')
        if i >= len(filepaths)-2:
            print(' \\hline ')
    bottom_table = '\\end{tabular}\n\\caption{\\small F1-Score}\\label{tMulti'
    bottom_table += ''.join(current_tsk_domain.split('_'))
    bottom_table += '}}\n\\end{table*}'
    print(bottom_table)
    print('')




