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

def combine_tasks(tsk1, tsk2):
    tsk1_list = tsk1.split('-')
    tsk2_list = tsk2.split('-')
    if len(tsk1_list) == 1 and len(tsk2_list) == 1:
        return ''.join(sorted(tsk1_list + tsk2_list))
    else:
        return '-'.join(sorted(tsk1_list + tsk2_list))

oracles = {'upos': ['', 'chunk', ''], 'xpos': ['', 'chunk', ''], 'chunk': ['sem-upos-xpos', 'ner-upos-xpos', 'upos-xpos'], 'mwe': ['chunk-ner-sem-semtr-supsense-upos-xpos', 'chunk-ner-sem-semtr-supsense-upos-xpos', 'chunk-sem-semtr-supsense-upos-xpos'], 'ner': ['', '', ''], 'sem': ['chunk-upos-xpos', 'chunk-upos-xpos', 'chunk-supsense-upos'], 'semtr': ['chunk-mwe-ner-sem-supsense-upos-xpos', 'chunk-mwe-sem-supsense-upos-xpos', 'chunk-frame-mwe-sem-supsense-upos-xpos'], 'supsense': ['chunk-ner-sem-semtr-upos-xpos', 'sem-semtr-upos-xpos', 'ner-sem-upos-xpos'], 'com': ['', '', ''], 'frame': ['', '', ''], 'hyp': ['chunk-sem-supsense-xpos', 'chunk-upos-xpos', 'xpos']}


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

def load_score_json_get_commands(filedir):
    commands = []
    for seedsuf in seedsufs:
        filepath = os.path.join(filedir + seedsuf, 'metrics.json')
        if not os.path.exists(filepath):
            commands.append(filepath)
    return commands

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
            toreturn += '{\\color{olive}'
    toreturn += str(round(mean_f1s, 2))
    # toreturn += '\\scriptsize{ $\\pm$ ' + str(round(std_f1s, 2)) + '}'
    toreturn += '\\tiny{ $\\pm$ ' + str(round(std_f1s, 2)) + '}'
    if lowbnd is not None and mean_f1s + k*std_f1s <= lowbnd:
        toreturn += ' $\\downarrow$ }'
    elif upbnd is not None and mean_f1s - k*std_f1s >= upbnd:
        toreturn += ' $\\uparrow$ }'
    return toreturn

def f1s_to_smalltxt(f1s, lowbnd=None, upbnd=None, k=1.5):
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
            toreturn += '{\\color{olive}'
    toreturn += '\\scriptsize{' + str(round(mean_f1s, 2)) + '}'
    # toreturn +=  '\\footnotesize{ $\\pm$ ' + str(round(std_f1s, 2)) + '}'
    if lowbnd is not None and mean_f1s + k*std_f1s <= lowbnd:
        toreturn += ' $\\downarrow$ }'
    elif upbnd is not None and mean_f1s - k*std_f1s >= upbnd:
        toreturn += ' $\\uparrow$ }'
    return toreturn

def f1s_to_stats(f1s):
    npf1s = np.array(f1s)
    npf1s[npf1s == 0] = np.nan
    mean_f1s = 100 * np.nanmean(npf1s)
    std_f1s = 100 * np.nanstd(npf1s)
    return mean_f1s, std_f1s

avg_dict = {}

all_commands = []

num_tasks = len(all_tasks)
tskid = {}
for i, task in enumerate(all_tasks):
    tskid[task] = i

# pairwise information
multi_meanpairwise_table = np.zeros((num_tasks, num_tasks))
teonly_meanpairwise_table = np.zeros((num_tasks, num_tasks))
tpeonly_meanpairwise_table = np.zeros((num_tasks, num_tasks))
multi_stdpairwise_table = np.zeros((num_tasks, num_tasks))
teonly_stdpairwise_table = np.zeros((num_tasks, num_tasks))
tpeonly_stdpairwise_table = np.zeros((num_tasks, num_tasks))
multi_pairwise_table = [['' for _ in range(num_tasks)] for _ in range(num_tasks)]
teonly_pairwise_table = [['' for _ in range(num_tasks)] for _ in range(num_tasks)]
tpeonly_pairwise_table = [['' for _ in range(num_tasks)] for _ in range(num_tasks)]
multi_pwavg_f1s = []
teonly_pwavg_f1s = []
tpeonly_pwavg_f1s = []

# all minus information
multi_meanallminus_table = np.zeros((num_tasks, num_tasks))
teonly_meanallminus_table = np.zeros((num_tasks, num_tasks))
tpeonly_meanallminus_table = np.zeros((num_tasks, num_tasks))
multi_stdallminus_table = np.zeros((num_tasks, num_tasks))
teonly_stdallminus_table = np.zeros((num_tasks, num_tasks))
tpeonly_stdallminus_table = np.zeros((num_tasks, num_tasks))
multi_allminus_table = [['' for _ in range(num_tasks)] for _ in range(num_tasks)]
teonly_allminus_table = [['' for _ in range(num_tasks)] for _ in range(num_tasks)]
tpeonly_allminus_table = [['' for _ in range(num_tasks)] for _ in range(num_tasks)]
multi_allminusavg_f1s = []
teonly_allminusavg_f1s = []
tpeonly_allminusavg_f1s = []

# all information
all_items = {"all": 1, "oracle": 2}
num_all_items = len(all_items)
multi_meanall_table = np.zeros((num_tasks, num_all_items))
teonly_meanall_table = np.zeros((num_tasks, num_all_items))
tpeonly_meanall_table = np.zeros((num_tasks, num_all_items))
multi_stdall_table = np.zeros((num_tasks, num_all_items))
teonly_stdall_table = np.zeros((num_tasks, num_all_items))
tpeonly_stdall_table = np.zeros((num_tasks, num_all_items))
multi_all_f1s = []
teonly_all_f1s = []
tpeonly_all_f1s = []

for current_tsk_domain in current_tsk_domains:
    current_tsk, all_domain_strs = current_tsk_domain.split('_')
    current_tsk_id = tskid[current_tsk]
    top_table = '\\begin{table*}[t]\n\\centering\n\\scriptsize{\n\\begin{tabular}{c|c|c|c|c}\n'
    # top_table = '\\begin{table*}[t]\n\\centering\n\\footnotesize{\n\\begin{tabular}{c|c|c|c|c}\n'
    top_table += '\multicolumn{2}{c}{Trained with} & \\multicolumn{3}{|c}{Tested on \\task{' + current_tsk + '}'
    # top_table += ' on ' + all_domain_strs
    top_table += '} \\\\ \\cline{3-5}\n'
    top_table += ' \multicolumn{2}{c|}{} & \\textbf{Multi-Dec} & \\textbf{TE-Dec} & \\textbf{TE-Enc} \\\\ \\hline'
    print(top_table)
    firstcols = []
    exts = []
    for tsk in all_tasks:
        if tsk == current_tsk:
            continue
        exts.append(''.join(sorted([tsk, current_tsk])))
        firstcols.append("+" + tsk)
    for tsk in all_tasks:
        if tsk == current_tsk:
            continue
        exts.append("allminus_" + tsk)
        firstcols.append("all - " + tsk)
    exts.append("all")
    firstcols.append("all")

    multi_filepaths = [multi_path + ext for ext in exts]
    teonly_filepaths = [teonly_path + ext for ext in exts]
    tpeonly_filepaths = [tpeonly_path + ext for ext in exts]

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

    multi_meanpairwise_table[current_tsk_id, current_tsk_id] = f1s_to_stats(single_crf_f1)[0]
    teonly_meanpairwise_table[current_tsk_id, current_tsk_id] = f1s_to_stats(single_crf_f1)[0]
    tpeonly_meanpairwise_table[current_tsk_id, current_tsk_id] = f1s_to_stats(single_crf_f1)[0]
    multi_stdpairwise_table[current_tsk_id, current_tsk_id] = f1s_to_stats(single_crf_f1)[1]
    teonly_stdpairwise_table[current_tsk_id, current_tsk_id] = f1s_to_stats(single_crf_f1)[1]
    tpeonly_stdpairwise_table[current_tsk_id, current_tsk_id] = f1s_to_stats(single_crf_f1)[1]
    multi_pairwise_table[current_tsk_id][current_tsk_id] = f1s_to_smalltxt(single_crf_f1)
    teonly_pairwise_table[current_tsk_id][current_tsk_id] = f1s_to_smalltxt(single_crf_f1)
    tpeonly_pairwise_table[current_tsk_id][current_tsk_id] = f1s_to_smalltxt(single_crf_f1)

    # print("Self only (CRF) & ", end=' ')
    print("\multicolumn{2}{c}{ \\task{" + current_tsk + "} only } & ", end=' ')
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
    multi_allminus_f1s = []
    teonly_allminus_f1s = []
    tpeonly_allminus_f1s = []
    for i, mfilepath in enumerate(multi_filepaths):
        if i == 0:
            print('\\parbox[t]{1mm}{\\multirow{' + str(num_tasks) + '}{*}{\\rotatebox[origin = c]{90}{Pairwise}}}',
                  end = '')
        elif i == num_tasks-1:
            print('\\parbox[t]{1mm}{\\multirow{' + str(num_tasks-1) +
                  '}{*}{\\rotatebox[origin = c]{90}{All but one}}}', end='')
        if i < len(multi_filepaths) - 1:
            print('&', end=' ')
            print('\\task{' + firstcols[i] + '}', end=' ')
        else:
            print('\\multicolumn{2}{c|}{\\task{' + firstcols[i] + '}}', end=' ')
        print('&', end=' ')

        multi_f1 = load_score_json(mfilepath, 'test_' + current_tsk + '-f1-measure-overall')
        print(f1s_to_txt(multi_f1, lowbnd, upbnd), end=' ')
        print('&', end=' ')

        teonly_f1 = load_score_json(teonly_filepaths[i], 'test_' + current_tsk + '-f1-measure-overall')
        print(f1s_to_txt(teonly_f1, lowbnd, upbnd), end=' ')
        print('&', end=' ')

        tpeonly_f1 = load_score_json(tpeonly_filepaths[i], 'test_' + current_tsk + '-f1-measure-overall')
        print(f1s_to_txt(tpeonly_f1, lowbnd, upbnd), end=' ')
        print(' \\\\ ', end=' ')
        if 'all' not in mfilepath:
            multi_pairwise_f1s.append(f1s_to_stats(multi_f1)[0]/100)
            teonly_pairwise_f1s.append(f1s_to_stats(teonly_f1)[0]/100)
            tpeonly_pairwise_f1s.append(f1s_to_stats(tpeonly_f1)[0]/100)

            another_task_id = tskid[firstcols[i][1:]]
            multi_meanpairwise_table[current_tsk_id, another_task_id] = f1s_to_stats(multi_f1)[0]
            teonly_meanpairwise_table[current_tsk_id, another_task_id] = f1s_to_stats(teonly_f1)[0]
            tpeonly_meanpairwise_table[current_tsk_id, another_task_id] = f1s_to_stats(tpeonly_f1)[0]
            multi_stdpairwise_table[current_tsk_id, another_task_id] = f1s_to_stats(multi_f1)[1]
            teonly_stdpairwise_table[current_tsk_id, another_task_id] = f1s_to_stats(teonly_f1)[1]
            tpeonly_stdpairwise_table[current_tsk_id, another_task_id] = f1s_to_stats(tpeonly_f1)[1]
            multi_pairwise_table[current_tsk_id][another_task_id] = f1s_to_smalltxt(multi_f1, lowbnd, upbnd)
            teonly_pairwise_table[current_tsk_id][another_task_id] = f1s_to_smalltxt(teonly_f1, lowbnd, upbnd)
            tpeonly_pairwise_table[current_tsk_id][another_task_id] = f1s_to_smalltxt(tpeonly_f1, lowbnd, upbnd)
        elif 'allminus' in mfilepath:
            multi_allminus_f1s.append(f1s_to_stats(multi_f1)[0]/100)
            teonly_allminus_f1s.append(f1s_to_stats(teonly_f1)[0]/100)
            tpeonly_allminus_f1s.append(f1s_to_stats(tpeonly_f1)[0]/100)
        elif 'all' in mfilepath:
            multi_all_f1s.append(f1s_to_stats(multi_f1)[0])
            teonly_all_f1s.append(f1s_to_stats(teonly_f1)[0])
            tpeonly_all_f1s.append(f1s_to_stats(tpeonly_f1)[0])
            multi_allminusavg_f1s.append(f1s_to_stats(multi_allminus_f1s)[0])
            teonly_allminusavg_f1s.append(f1s_to_stats(teonly_allminus_f1s)[0])
            tpeonly_allminusavg_f1s.append(f1s_to_stats(tpeonly_allminus_f1s)[0])
        if i == 2*num_tasks - 3 or i == len(multi_filepaths) - 1:
            print(' \\hline ', end=' ')
        elif i == num_tasks - 2 :
            print(' \\cline{2-5} ', end=' ')
        if i == num_tasks - 2:
            print('')
            print('& Average', end=' ')
            print('&', end=' ')
            pairwise_mean, _ = f1s_to_stats(multi_pairwise_f1s)
            multi_pwavg_f1s.append(pairwise_mean)
            print(str(round(pairwise_mean, 2)), end=' ')
            print('&', end=' ')
            pairwise_mean, _ = f1s_to_stats(teonly_pairwise_f1s)
            teonly_pwavg_f1s.append(pairwise_mean)
            print(str(round(pairwise_mean, 2)), end=' ')
            print('&', end=' ')
            pairwise_mean, _ = f1s_to_stats(tpeonly_pairwise_f1s)
            tpeonly_pwavg_f1s.append(pairwise_mean)
            print(str(round(pairwise_mean, 2)), end=' ')
            print(' \\\\ \\hline ', end=' ')
        print('')


    print("\multicolumn{2}{c|}{ Oracle } & ", end=' ')
    current_oracles = oracles[current_tsk]
    if current_oracles[0] != '':
        f1_oracle = load_score_json(multi_path + combine_tasks(current_tsk, current_oracles[0]),
                                    'test_' + current_tsk + '-f1-measure-overall')
        all_commands += load_score_json_get_commands(multi_path + combine_tasks(current_tsk, current_oracles[0]))
        print(f1s_to_txt(f1_oracle), end=' ')
    else:
        print(' None ', end=' ')
    print('&', end=' ')
    if current_oracles[1] != '':
        f1_oracle = load_score_json(teonly_path + combine_tasks(current_tsk, current_oracles[1]),
                                    'test_' + current_tsk + '-f1-measure-overall')
        all_commands += load_score_json_get_commands(teonly_path + combine_tasks(current_tsk, current_oracles[1]))
        print(f1s_to_txt(f1_oracle), end=' ')
    else:
        print(' None ', end=' ')
    print('&', end=' ')
    if current_oracles[2] != '':
        f1_oracle = load_score_json(tpeonly_path + combine_tasks(current_tsk, current_oracles[2]),
                                    'test_' + current_tsk + '-f1-measure-overall')
        all_commands += load_score_json_get_commands(tpeonly_path + combine_tasks(current_tsk, current_oracles[2]))
        print(f1s_to_txt(f1_oracle), end=' ')
    else:
        print(' None ', end=' ')
    print(' \\\\ \\hline')
    # print(' \\hline ')
    caption = 'F1 score tested on the task \\task{' + current_tsk + '} in different training scenarios'
    bottom_table = '\\end{tabular}\n\\caption{\\small ' + caption + '}\\label{tMultiTask'
    bottom_table += current_tsk + all_domain_strs
    bottom_table += '}}\n\\end{table*}'
    print(bottom_table)
    print('')

for command in sorted(list(set(all_commands))):
    print('%', command)

def get_compare_mtl_str(table, methods, metrics):
    compare_mtl_str = '\\begin{table*}[t]\n\\centering\n\\scriptsize{\n\\begin{tabular}{c|c|c|c|c|c|c|c|c|c|c|c|c||c}\n'
    compare_mtl_str += ' Trained with & MTL method'
    for j in range(num_tasks):
        compare_mtl_str += ' & ' + '\\tinytask{' + all_tasks[j] + '}'
    compare_mtl_str += ' & ' + '{\\scriptsize{Average}}'
    compare_mtl_str += '\\\\ \\hline\n'
    for i in range(len(table)):
        compare_mtl_str += metrics[i]
        compare_mtl_str += '& ' + methods[i]
        for j in range(num_tasks):
            compare_mtl_str += '& ' + '{\\tiny{' + str(round(table[i][j], 2)) + '}}'
        compare_mtl_str += '& ' + '{\\tiny{' + str(round(np.mean(np.array(table[i])), 2)) + '}}'
        # compare_mtl_str += ' $\pm$ ' + str(round(np.std(np.array(table[i])), 2)) + '}}'
        if i % 3 != 2:
            compare_mtl_str += '\\\\ \\cline{2-14}\n'
        else:
            compare_mtl_str += '\\\\ \\hline\n'
    compare_mtl_str += '\\end{tabular}\n'
    compare_mtl_str += '\\caption{\small Comparison between MTL approaches}\\label{tCompareMtl}}\n'
    compare_mtl_str += '\\end{table*}'
    return compare_mtl_str

def get_toptask_commands(table):
    command = []
    all_good_dicts = {}
    for i in range(num_tasks):
        tsk_numbers = []
        for j in range(num_tasks):
            if i == j:
                tsk_numbers.append(0.0)
            else:
                tsk_numbers.append(table[i,j])
        indices = np.argsort(-np.array(tsk_numbers))
        command.append('-'.join(sorted([all_tasks[i]] + [all_tasks[j] for j in indices[:2]])))
        all_good_dicts[all_tasks[i]] = ['-'.join(sorted([all_tasks[j] for j in indices[:2]]))]
    return list(set(command)), all_good_dicts

def get_rel_table(table):
    rel_table = np.zeros((num_tasks, num_tasks))
    for i in range(num_tasks):
        for j in range(num_tasks):
            rel_table[i,j] = 100*(table[i,j] - table[i,i]) / table[i,i]
    return rel_table

def get_pairwise_str(table, all_table, allminus_table, oracle_table, method, cap, isstr=False):
    pairwise_str = '\\begin{table*}[t]\n\\centering\n\\scriptsize{\n\\begin{tabular}{c|c|c|c|c|c|c|c|c|c|c|c}\n'
    for j in range(num_tasks):
        pairwise_str += ' & ' + '\\task{' + all_tasks[j] + '}'
    pairwise_str += '\\\\ \\hline\n'
    for i in range(num_tasks):
        pairwise_str += '\\task{' + all_tasks[i] +'} '
        for j in range(num_tasks):
            if isstr:
                content = table[j][i]
                if i == j:
                    pairwise_str += '& ' + '{\\color{blue}' + content + '}'
                else:
                    if 'color' in content:
                        pairwise_str += '& ' + content
                    elif 'scriptsize' in content:
                        pairwise_str += '& ' + content.replace('scriptsize', 'tiny')
            else:
                content = table[j,i]
                pairwise_str += '& '
                if i == j:
                    pairwise_str += '{\\color{blue}' + str(round(content, 2)) + '}'
                else:
                    if content < -0.5:
                        pairwise_str += '{\\color{red}' + str(round(content, 2)) + '}'
                    elif content > 0.5:
                        pairwise_str += '{\\color{olive}' + str(round(content, 2)) + '}'
                    else:
                        pairwise_str += '{\\tiny{' + str(round(content, 2)) + '}}'
        pairwise_str += '\\\\ \\hline\n'
    pairwise_str += '\\task{all} '
    for j in range(num_tasks):
        if isstr:
            content = all_table[j]
            if 'color' in content:
                pairwise_str += '& ' + content
            elif 'scriptsize' in content:
                pairwise_str += '& ' + content.replace('scriptsize', 'tiny')

    pairwise_str += '\\end{tabular}\n'
    pairwise_str += '\\caption{\small ' + cap + '}\\label{t' + method + '}}\n'
    pairwise_str += '\\end{table*}'
    return pairwise_str

caption_ending = ' For each number, we train on corresponding ``row" and ``column" tasks jointly and test on the column task.'

print(get_pairwise_str(get_rel_table(multi_meanpairwise_table), 'RelMultiDec', 'Relative in \\% F1 scores for \\textbf{Multi-Dec}.' + caption_ending))
print(get_pairwise_str(get_rel_table(teonly_meanpairwise_table), 'RelTEDec', 'Relative in \\% F1 scores for \\textbf{TE-Dec}.' + caption_ending))
print(get_pairwise_str(get_rel_table(tpeonly_meanpairwise_table), 'RelTEEnc', 'Relative in \\% F1 scores for \\textbf{TE-Enc}.' + caption_ending))

print(get_pairwise_str(multi_pairwise_table, 'PwMultiDec', 'Pairwise F1 scores for \\textbf{Multi-Dec}.' + caption_ending, isstr=True))
print(get_pairwise_str(teonly_pairwise_table, 'PwTEDec', 'Pairwise F1 scores for \\textbf{TE-Dec}.' + caption_ending, isstr=True))
print(get_pairwise_str(tpeonly_pairwise_table, 'PwTEEnc', 'Pairwise F1 scores for \\textbf{TE-Enc}.' + caption_ending, isstr=True))

table = [multi_pwavg_f1s, teonly_pwavg_f1s, tpeonly_pwavg_f1s,
         multi_all_f1s, teonly_all_f1s, tpeonly_all_f1s,
         multi_allminusavg_f1s, teonly_allminusavg_f1s, tpeonly_allminusavg_f1s]
methods = ['\\textbf{Multi-Dec}',
           '\\textbf{TE-Dec}',
           '\\textbf{TE-Enc}',
           '\\textbf{Multi-Dec}',
           '\\textbf{TE-Dec}',
           '\\textbf{TE-Enc}',
           '\\textbf{Multi-Dec}',
           '\\textbf{TE-Dec}',
           '\\textbf{TE-Enc}']
metrics = ['Mean Average',
           'Pairwise',
           '',
           'Mean',
           'All',
           '',
           'Mean Average',
           'All-but-one',
           '']
print(get_compare_mtl_str(table, methods, metrics))

"""
command, good_dicts = get_toptask_commands(multi_meanpairwise_table)
for tt in command:
    print('for ext in ', end='')
    print('"' + tt + '"', '\"' + tt + '0"', '"' + tt + '1"')
    print('do')
    print('./multi_multi.sh "$ext" "$tmpdir" "$gpu"')
    print('done')
    print('')
all_good_dicts = good_dicts

command, good_dicts = get_toptask_commands(teonly_meanpairwise_table)
for tt in command:
    print('for ext in ', end='')
    print('"' + tt + '"', '\"' + tt + '0"', '"' + tt + '1"')
    print('do')
    print('./multi_teonly.sh "$ext" "$tmpdir" "$gpu"')
    print('done')
    print('')
for k, v in good_dicts.items():
    all_good_dicts[k] = all_good_dicts[k] + v

command, good_dicts = get_toptask_commands(tpeonly_meanpairwise_table)
for tt in command:
    print('for ext in ', end='')
    print('"' + tt + '"', '\"' + tt + '0"', '"' + tt + '1"')
    print('do')
    print('./multi_tpeonly.sh "$ext" "$tmpdir" "$gpu"')
    print('done')
    print('')
for k, v in good_dicts.items():
    all_good_dicts[k] = list(set(all_good_dicts[k] + v))

print(all_good_dicts)
"""