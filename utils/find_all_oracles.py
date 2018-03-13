import json
import os
import numpy as np

single_path = '/data/beer/tagger_clean/'
task_suffix = '_tagger_'
task_crf_suffix = '_tagger_crf_'
multi_path = '/data/beer/tagger_clean/multitagger_multi_'
teonly_path = '/data/beer/tagger_clean/taskembtagger_taskonly_embedding_tagger_'
tpeonly_path = '/data/beer/tagger_clean/taskembtagger_taskonly_prepend_embedding_tagger_'

all_tasks = ["upos", "xpos", "chunk", "ner", "mwe", "sem", "semtr", "supsense", "com", "frame", "hyp"]
all_domains = ["uni", "conll03", "conll02", "streusle", "semcor", "broadcast1", "fnt", "hyp"]
tskds = {"upos": ["uni"],
         "xpos": ["uni"],
         "chunk": ["conll02"],
         "mwe": ["streusle"],
         "ner": ["conll03"],
         "sem": ["semcor"],
         "semtr": ["semcor"],
         "supsense": ["streusle"],
         "com": ["broadcast1"],
         "frame": ["fnt"],
         "hyp": ["hyp"]}

seedsufs = ['', '0', '1']

dataset_tasks = "dataset_reader/tasks"
dataset_domains = "dataset_reader/domains"
model_tasks = "model/tasks"
model_domains = "model/domains"
iter_tasks = "train_iterator/tasks"
random_seed_key = "random_seed"
numpy_seed_key = "numpy_seed"
torch_seed_key = "pytorch_seed"
# random_seed: 13370
# numpy_seed: 1337
# torch_seed = 133
seeds = [(11170, 1117, 111), (12270, 1227, 122)]

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

def check_f1s(f1s, lowbnd=None, upbnd=None, k=1.5):
    npf1s = np.array(f1s)
    npf1s[npf1s == 0] = np.nan
    mean_f1s = 100 * np.nanmean(npf1s)
    std_f1s = 100 * np.nanstd(npf1s)
    if lowbnd is not None and mean_f1s + k*std_f1s <= lowbnd:
        return -1
    elif upbnd is not None and mean_f1s - k*std_f1s >= upbnd:
        return 1
    return 0

def f1s_to_stats(f1s):
    npf1s = np.array(f1s)
    npf1s[npf1s == 0] = np.nan
    mean_f1s = 100 * np.nanmean(npf1s)
    std_f1s = 100 * np.nanstd(npf1s)
    return mean_f1s, std_f1s

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

def set_taskdomains(toread, towrite, tasks, domains, random_seed, numpy_seed, torch_seed):
    with open(toread, 'r') as f:
        data = json.load(f)
        data = set_data(data, random_seed_key, random_seed)
        data = set_data(data, numpy_seed_key, numpy_seed)
        data = set_data(data, torch_seed_key, torch_seed)
        data = set_data(data, dataset_tasks, tasks)
        data = set_data(data, model_tasks, tasks)
        data = set_data(data, iter_tasks, tasks)
        data = set_data(data, dataset_domains, domains)
        data = set_data(data, model_domains, domains)
        with open(towrite, 'w') as fw:
            json.dump(data, fw, indent=4)

def find_oracle(current_tsk, k):
    exts = []
    others = []
    for tsk in all_tasks:
        if tsk == current_tsk:
            continue
        exts.append(''.join(sorted([tsk, current_tsk])))
        others.append(tsk)

    multi_filepaths = [multi_path + ext for ext in exts]
    teonly_filepaths = [teonly_path + ext for ext in exts]
    tpeonly_filepaths = [tpeonly_path + ext for ext in exts]

    single_crf_f1 = load_score_json(single_path + current_tsk + '_' +tskds[current_tsk][0] + task_crf_suffix + current_tsk,
                                    'test_f1-measure-overall')
    mean_single_crf, std_single_crf = f1s_to_stats(single_crf_f1)
    lowbnd = mean_single_crf - k*std_single_crf
    upbnd = mean_single_crf + k*std_single_crf

    all_goods = []
    all_goods_json = []
    goods = []
    for i, mfilepath in enumerate(multi_filepaths):
        multi_f1 = load_score_json(mfilepath, 'test_' + current_tsk + '-f1-measure-overall')
        if check_f1s(multi_f1, lowbnd, upbnd, k) == 1:
            goods.append(others[i])
    print('Multi:', current_tsk, goods)
    if len(goods) > 1:
        all_goods.append('-'.join(sorted(goods)))
        all_goods_json.append('-'.join(sorted(goods)))
    elif len(goods) == 1:
        all_goods_json.append(''.join(sorted(goods)))
    else:
        all_goods_json.append('')

    goods = []
    for i, teonlyfilepath in enumerate(teonly_filepaths):
        teonly_f1 = load_score_json(teonlyfilepath, 'test_' + current_tsk + '-f1-measure-overall')
        if check_f1s(teonly_f1, lowbnd, upbnd, k) == 1:
            goods.append(others[i])
    print('TE:', current_tsk, goods)
    if len(goods) > 1:
        all_goods.append('-'.join(sorted(goods)))
        all_goods_json.append('-'.join(sorted(goods)))
    elif len(goods) == 1:
        all_goods_json.append(''.join(sorted(goods)))
    else:
        all_goods_json.append('')

    goods = []
    for i, tpeonlyfilepath in enumerate(tpeonly_filepaths):
        tpeonly_f1 = load_score_json(tpeonlyfilepath, 'test_' + current_tsk + '-f1-measure-overall')
        if check_f1s(tpeonly_f1, lowbnd, upbnd, k) == 1:
            goods.append(others[i])
    print('TPE:', current_tsk, goods)
    if len(goods) > 1:
        all_goods.append('-'.join(sorted(goods)))
        all_goods_json.append('-'.join(sorted(goods)))
    elif len(goods) == 1:
        all_goods_json.append(''.join(sorted(goods)))
    else:
        all_goods_json.append('')

    return list(set(all_goods)), all_goods_json

multi_json_template = 'multi_all.json'
te_json_template = 'task_embedding_tagger_all.json'
tpe_json_template = 'task_prepend_embedding_tagger_all.json'
teonly_json_template = 'taskonly_embedding_tagger_all.json'
tpeonly_json_template = 'taskonly_prepend_embedding_tagger_all.json'
all_json_templates = [multi_json_template,
                      te_json_template,
                      tpe_json_template,
                      teonly_json_template,
                      tpeonly_json_template]

json_multi_path = 'multi/multi_'
json_te_path = 'task_embedding/task_embedding_tagger_'
json_tpe_path = 'task_prepend_embedding/task_prepend_embedding_tagger_'
json_teonly_path = 'taskonly_embedding/taskonly_embedding_tagger_'
json_tpeonly_path = 'taskonly_prepend_embedding/taskonly_prepend_embedding_tagger_'
all_json_paths = [json_multi_path,
                  json_te_path,
                  json_tpe_path,
                  json_teonly_path,
                  json_tpeonly_path]

combined_all_good_dicts = {}
for k in [1.5, 2, 2.5]:
    all_goods_dict = {}
    all_goods_dict_json = {}
    for t1, ds1 in tskds.items():
        all_goods, all_goods_json = find_oracle(t1, k)
        all_goods_dict[t1] = all_goods
        all_goods_dict_json[t1] = all_goods_json
    print(k, all_goods_dict)
    print(k, all_goods_dict_json)
    for ky, val in all_goods_dict.items():
        current_val = combined_all_good_dicts.get(ky, [])
        combined_all_good_dicts[ky] = current_val + val

for ky, val in combined_all_good_dicts.items():
    toprint = "\"" + ky + "\"" + ":"
    print(toprint, list(set(val)), ",")