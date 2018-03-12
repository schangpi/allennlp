import json
import os
import numpy as np

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

seedsufs = ['', '0', '1']
def load_score_json(filedir, att):
    f1s = []
    for seedsuf in seedsufs:
        filepath = os.path.join(filedir + seedsuf, 'metrics.json')
        if os.path.exists(filepath):
            with open(filepath, 'r') as fr:
                results = json.load(fr)
                f1s.append(results[att])
        else:
            f1s.append(0.0)
    return f1s

def f1s_to_txt(f1s, lowbnd=None, upbnd=None, k=1.5):
    npf1s = np.array(f1s)
    npf1s[npf1s == 0] = np.nan
    mean_f1s = 100 * np.nanmean(npf1s)
    std_f1s = 100 * np.nanstd(npf1s)
    toreturn = ''
    if sum(np.isnan(npf1s)) > 0:
        toreturn +=  ' '.join([str(round(100 * f1, 2)) for f1 in f1s]) + ' = '
    if lowbnd is not None and mean_f1s + k*std_f1s <= lowbnd:
        toreturn += '{\\color{red}'
    elif upbnd is not None and mean_f1s - k*std_f1s >= upbnd:
            toreturn += '{\\color{green}'
    toreturn += str(round(mean_f1s, 2))
    toreturn += ' $\\pm$ '
    toreturn += str(round(std_f1s, 2))
    if lowbnd is not None and mean_f1s + k*std_f1s <= lowbnd:
        toreturn += '$\\downarrow$ }'
    elif upbnd is not None and mean_f1s - k*std_f1s >= upbnd:
        toreturn += '$\\uparrow$ }'
    return toreturn

def f1s_to_stats(f1s):
    npf1s = np.array(f1s)
    npf1s[npf1s == 0] = np.nan
    mean_f1s = 100 * np.nanmean(npf1s)
    std_f1s = 100 * np.nanstd(npf1s)
    return mean_f1s, std_f1s

for current_tsk_domain in current_tsk_domains:
    current_tsk, all_domain_strs = current_tsk_domain.split('_')
    top_table = '\\begin{table*}[t]\n\\centering\n\\footnotesize{\n\\begin{tabular}{c|c|c|c}\n'
    top_table += 'Trained with & \\multicolumn{3}{|c}{\\task{' + current_tsk + '}'
    top_table += ' on ' + all_domain_strs
    top_table += '} \\\\ \\cline{2-4}\n'
    top_table += '& \\textbf{Multi-Dec} & \\textbf{TE-Dec} & \\textbf{TE-Enc} \\\\ \\hline'
    print(top_table)
    firstcols = []
    exts = []
    # useolds = []
    problematic = ["upos", "xpos", "chunk", "sem", "semtr", "frame", "hyp"]
    for tsk in all_tasks:
        if tsk == current_tsk:
            continue
        exts.append(''.join(sorted([tsk, current_tsk])))
        # if tsk not in problematic and current_tsk not in problematic:
        #     useolds.append(True)
        # else:
        #     useolds.append(False)
        firstcols.append("+" + tsk)
    for tsk in all_tasks:
        if tsk == current_tsk:
            continue
        exts.append("allminus_" + tsk)
        firstcols.append("all - " + tsk)
    exts.append("all")
    # useolds.append(False)
    firstcols.append("all")

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
    print(f1s_to_txt(single_crf_f1), end='')
    print('}', end=' ')
    print(' \\\\ \\hline')

    mean_single_crf, std_single_crf = f1s_to_stats(single_crf_f1)
    k = 1.5
    lowbnd = mean_single_crf - k*std_single_crf
    upbnd = mean_single_crf + k*std_single_crf

    multi_pairwise_f1s = []
    teonly_pairwise_f1s = []
    tpeonly_pairwise_f1s = []

    for i, mfilepath in enumerate(multi_filepaths):
        print('\\task{' + firstcols[i] + '}', end=' ')
        print('&', end=' ')

        multi_f1 = load_score_json(mfilepath, 'test_' + current_tsk + '-f1-measure-overall')
        multi_f1_old = load_score_json(multi_filepaths_old[i], 'test_' + current_tsk + '-f1-measure-overall')
        # if useolds[i]:
        #     multi_f1 = multi_f1_old
        print(f1s_to_txt(multi_f1, lowbnd, upbnd), end=' ')
        print('&', end=' ')

        teonly_f1 = load_score_json(teonly_filepaths[i], 'test_' + current_tsk + '-f1-measure-overall')
        teonly_f1_old = load_score_json(teonly_filepaths_old[i], 'test_' + current_tsk + '-f1-measure-overall')
        # if useolds[i]:
        #     teonly_f1 = teonly_f1_old
        print(f1s_to_txt(teonly_f1, lowbnd, upbnd), end=' ')
        print('&', end=' ')

        tpeonly_f1 = load_score_json(tpeonly_filepaths[i], 'test_' + current_tsk + '-f1-measure-overall')
        tpeonly_f1_old = load_score_json(tpeonly_filepaths_old[i], 'test_' + current_tsk + '-f1-measure-overall')
        # if useolds[i]:
        #     tpeonly_f1 = tpeonly_f1_old
        print(f1s_to_txt(tpeonly_f1, lowbnd, upbnd), end=' ')
        print(' \\\\ ', end=' ')
        if 'all' not in mfilepath:
            multi_pairwise_f1s.append(f1s_to_stats(multi_f1)[0]/100)
            teonly_pairwise_f1s.append(f1s_to_stats(teonly_f1)[0]/100)
            tpeonly_pairwise_f1s.append(f1s_to_stats(tpeonly_f1)[0]/100)
        if i == len(all_tasks) - 2 or i == 2*len(all_tasks) - 3:
            print(' \\hline ', end=' ')
        if i == len(all_tasks) - 2:
            print('')
            print('Average (pairwise)', end=' ')
            print('&', end=' ')
            pairwise_mean, _ = f1s_to_stats(multi_pairwise_f1s)
            print(str(round(pairwise_mean, 2)), end=' ')
            print('&', end=' ')
            pairwise_mean, _ = f1s_to_stats(teonly_pairwise_f1s)
            print(str(round(pairwise_mean, 2)), end=' ')
            print('&', end=' ')
            pairwise_mean, _ = f1s_to_stats(tpeonly_pairwise_f1s)
            print(str(round(pairwise_mean, 2)), end=' ')
            print(' \\\\ \\hline ', end=' ')
        print('')
    # print(' \\hline ')
    caption = 'F1 score tested on the task \\task{' + current_tsk + '} in different training scenarios'
    bottom_table = '\\end{tabular}\n\\caption{\\small ' + caption + '}\\label{tMultiTask'
    bottom_table += current_tsk + all_domain_strs
    bottom_table += '}}\n\\end{table*}'
    print(bottom_table)
    print('')
