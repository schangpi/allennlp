import json
import os

single_path = '/data/beer/tagger_clean/'
task_suffix = '_tagger_'
task_crf_suffix = '_tagger_crf_'
multi_path = '/data/beer/tagger_clean/multitagger_multi_'
teonly_path = '/data/beer/tagger_clean/taskembtagger_taskonly_embedding_tagger_'
tpeonly_path = '/data/beer/tagger_clean/taskembtagger_taskonly_prepend_embedding_tagger_'

single_path_old = '/data/tagger/'
multi_path_old = '/data/tagger/multitagger_multi_'
teonly_path_old = '/data/tagger/taskembtagger_taskonly_embedding_tagger_'
tpeonly_path_old = '/data/tagger/taskembtagger_taskonly_prepend_embedding_tagger_'

all_tasks = ["upos", "xpos", "chunk", "ner", "mwe", "sem", "semtr", "supsense", "com", "frame", "hyp"]
current_tsk_domains = ["upos_uni",
                       "xpos_uni",
                       "chunk_conll02",
                       "ner_conll03",
                       "mwe_streusle",
                       "sem_semcor",
                       "semtr_semcor",
                       "supsense_streusle",
                       "com_broadcast1",
                       "frame_fnt",
                       "hyp_hyp"
                       ]

def load_score_json(filedir, att):
    filepath = os.path.join(filedir, 'metrics.json')
    if os.path.exists(filepath):
        with open(filepath, 'r') as fr:
            results = json.load(fr)
            return results[att]
    else:
        return 0.0

for current_tsk_domain in current_tsk_domains:
    current_tsk, all_domain_strs = current_tsk_domain.split('_')
    top_table = '\\begin{table*}[t]\n\\centering\n\\footnotesize{\n\\begin{tabular}{c|c|c|c}\n'
    top_table += 'Trained with & \\multicolumn{3}{|c}{\\task{' + current_tsk
    top_table += '} on ' + all_domain_strs
    top_table += '} \\\\ \\cline{2-4}\n'
    top_table += '& Multiple & Task Embs & Task Embs \\\\'
    top_table += '& Decoders & (All Steps) & (Prepend) \\\\ \\hline'
    print(top_table)
    other_tasks = []
    exts = []
    useolds = []
    problematic = ["upos", "xpos", "chunk", "sem", "semtr", "frame", "hyp"]
    for tsk in all_tasks:
        if tsk == current_tsk:
            continue
        exts.append(''.join(sorted([tsk, current_tsk])))
        if tsk not in problematic and current_tsk not in problematic:
            useolds.append(True)
        else:
            useolds.append(False)
        other_tasks.append(tsk)
    exts.append("all")
    useolds.append(False)
    other_tasks.append("all")

    multi_filepaths = [multi_path + ext for ext in exts]
    multi_filepaths_old = [multi_path_old + ext for ext in exts]
    teonly_filepaths = [teonly_path + ext for ext in exts]
    tpeonly_filepaths = [tpeonly_path + ext for ext in exts]
    teonly_filepaths_old = [teonly_path_old + ext for ext in exts]
    tpeonly_filepaths_old = [tpeonly_path_old + ext for ext in exts]

    """
    single_f1 = load_score_json(single_path + current_tsk_domain + task_suffix + current_tsk,
                                'test_f1-measure-overall')
    single_f1_old = load_score_json(single_path_old + current_tsk_domain + task_suffix + 'template',
                                    'test_f1-measure-overall')
    print("Self only (no CRF) & ", end=' ')
    print('\\multicolumn{3}{|c}{', end='')
    print(round(100 * single_f1, 2), '/', round(100 * single_f1_old, 2), end='')
    print('}', end=' ')
    print(' \\\\ \\hline')
    """

    single_crf_f1 = load_score_json(single_path + current_tsk_domain + task_crf_suffix + current_tsk,
                                    'test_f1-measure-overall')
    single_crf_f1_old = load_score_json(single_path_old + current_tsk_domain + task_crf_suffix + 'template',
                                        'test_f1-measure-overall')
    # print("Self only (CRF) & ", end=' ')
    print("Self only & ", end=' ')
    print('\\multicolumn{3}{|c}{', end='')
    print(round(100 * single_crf_f1, 2), '/', round(100 * single_crf_f1_old, 2), end='')
    print('}', end=' ')
    print(' \\\\ \\hline')

    for i, mfilepath in enumerate(multi_filepaths):
        print('+\\task{' + other_tasks[i] + '}', end=' ')
        print('&', end=' ')

        multi_f1 = load_score_json(mfilepath, 'test_' + current_tsk + '-f1-measure-overall')
        multi_f1_old = load_score_json(multi_filepaths_old[i], 'test_' + current_tsk + '-f1-measure-overall')
        # if useolds[i]:
        #     multi_f1 = multi_f1_old
        print(round(100 * multi_f1, 2), '/', round(100 * multi_f1_old, 2), end=' ')
        print('&', end=' ')

        teonly_f1 = load_score_json(teonly_filepaths[i], 'test_' + current_tsk + '-f1-measure-overall')
        teonly_f1_old = load_score_json(teonly_filepaths_old[i], 'test_' + current_tsk + '-f1-measure-overall')
        # if useolds[i]:
        #     teonly_f1 = teonly_f1_old
        print(round(100 * teonly_f1, 2), '/', round(100 * teonly_f1_old, 2), end=' ')
        print('&', end=' ')

        tpeonly_f1 = load_score_json(tpeonly_filepaths[i], 'test_' + current_tsk + '-f1-measure-overall')
        tpeonly_f1_old = load_score_json(tpeonly_filepaths_old[i], 'test_' + current_tsk + '-f1-measure-overall')
        # if useolds[i]:
        #     tpeonly_f1 = tpeonly_f1_old
        print(round(100 * tpeonly_f1, 2), '/', round(100 * tpeonly_f1_old, 2), end=' ')
        print(' \\\\ ', end=' ')
        if i >= len(multi_filepaths) - 2:
            print(' \\hline ', end=' ')
        print('')
    # print(' \\hline ')
    caption = 'F1 score tested on the task \\task{' + current_tsk + '} in different training scenarios'
    bottom_table = '\\end{tabular}\n\\caption{\\small ' + caption + '}\\label{tMultiTask'
    bottom_table += current_tsk + all_domain_strs
    bottom_table += '}}\n\\end{table*}'
    print(bottom_table)
    print('')
