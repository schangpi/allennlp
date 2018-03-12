import json
import os
import numpy as np

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

# 1   {'upos': [], 'xpos': [], 'chunk': ['uposxpos', 'neruposxpos', 'semsemtruposxpos'], 'mwe': ['chunknersemsemtrsupsenseuposxpos', 'chunksemsemtrsupsenseuposxpos'], 'ner': [], 'sem': ['chunknersupsenseuposxpos', 'chunkuposxpos'], 'semtr': ['chunkframemwesemsupsenseuposxpos', 'chunkmwenersemsupsenseuposxpos', 'chunkmwesemsupsenseuposxpos'], 'supsense': ['nersemsemtruposxpos', 'chunknersemuposxpos', 'chunknersemsemtruposxpos'], 'com': [], 'frame': [], 'hyp': ['chunksemsupsenseuposxpos', 'chunkneruposxpos', 'neruposxpos']}
# 1.5 {'upos': [], 'xpos': [], 'chunk': ['uposxpos', 'neruposxpos', 'semuposxpos'], 'mwe': ['chunknersemsemtrsupsenseuposxpos', 'chunksemsemtrsupsenseuposxpos'], 'ner': [], 'sem': ['chunksupsenseupos', 'chunkuposxpos'], 'semtr': ['chunkframemwesemsupsenseuposxpos', 'chunkmwenersemsupsenseuposxpos', 'chunkmwesemsupsenseuposxpos'], 'supsense': ['nersemuposxpos', 'semsemtruposxpos', 'chunknersemsemtruposxpos'], 'com': [], 'frame': [], 'hyp': ['chunksemsupsensexpos', 'chunkuposxpos']}
# 2   {'upos': [], 'xpos': [], 'chunk': ['uposxpos', 'neruposxpos', 'semuposxpos'], 'mwe': ['chunknersemsemtrsupsenseuposxpos', 'chunksemsemtrsupsenseuposxpos'], 'ner': [], 'sem': ['chunksupsenseupos', 'chunkuposxpos'], 'semtr': ['chunksemsupsenseuposxpos', 'chunkframemwesemsupsenseuposxpos', 'chunkmwenersemsupsenseuposxpos'], 'supsense': ['nersemupos', 'uposxpos', 'semsemtr'], 'com': [], 'frame': [], 'hyp': ['supsensexpos', 'chunkuposxpos']}
# 2.5 {'upos': [], 'xpos': [], 'chunk': ['uposxpos', 'neruposxpos', 'semuposxpos'], 'mwe': ['chunksemsupsenseuposxpos', 'chunknersemsemtrsupsenseuposxpos'], 'ner': [], 'sem': ['chunksupsenseupos', 'chunkuposxpos'], 'semtr': ['chunksemsupsenseuposxpos', 'chunkframemwesemsupsenseuposxpos', 'chunknersemuposxpos'], 'supsense': ['uposxpos', 'nersem'], 'com': [], 'frame': [], 'hyp': []}

all_good_dicts = {"upos": [] ,
"xpos": [] ,
"chunk": ['upos-xpos', 'sem-upos-xpos', 'ner-upos-xpos', 'sem-semtr-upos-xpos'] ,
"mwe": ['chunk-ner-sem-semtr-supsense-upos-xpos', 'chunk-sem-semtr-supsense-upos-xpos', 'chunk-sem-supsense-upos-xpos'] ,
"ner": [] ,
"sem": ['chunk-ner-supsense-upos-xpos', 'chunk-upos-xpos', 'chunk-supsense-upos'] ,
"semtr": ['chunk-ner-sem-upos-xpos', 'chunk-mwe-sem-supsense-upos-xpos', 'chunk-frame-mwe-sem-supsense-upos-xpos', 'chunk-mwe-ner-sem-supsense-upos-xpos', 'chunk-sem-supsense-upos-xpos'] ,
"supsense": ['ner-sem-semtr-upos-xpos', 'sem-semtr', 'sem-semtr-upos-xpos', 'ner-sem-upos-xpos', 'chunk-ner-sem-upos-xpos', 'upos-xpos', 'chunk-ner-sem-semtr-upos-xpos', 'ner-sem-upos', 'ner-sem'] ,
"com": [] ,
"frame": [] ,
"hyp": ['chunk-upos-xpos', 'supsense-xpos', 'chunk-ner-upos-xpos', 'ner-upos-xpos', 'chunk-sem-supsense-xpos', 'chunk-sem-supsense-upos-xpos']
}

def get_domains_for_tasks(tsks):
    dmns = []
    for tsk in tsks:
        dmns.append(tskds[tsk][0])
    return sorted(list(set(dmns)))

for iseed, seed in enumerate(seeds):
    random_seed = seed[0]
    numpy_seed = seed[1]
    torch_seed = seed[2]
    for t1, helpful_list in all_good_dicts.items():
        if len(helpful_list) == 0:
            continue
        for helpful_item in helpful_list:
            tasks = sorted([t1] + helpful_item.split('-'))
            domains = get_domains_for_tasks(tasks)
            for templ, pth in zip(all_json_templates, all_json_paths):
                set_taskdomains(templ, pth + '-'.join(tasks) + str(iseed) + '.json',
                                tasks, domains, random_seed, numpy_seed, torch_seed)

all_commands = []
for t1, helpful_list in all_good_dicts.items():
    if len(helpful_list) == 0:
        continue
    for helpful_item in helpful_list:
        tasks = sorted([t1] + helpful_item.split('-'))
        domains = get_domains_for_tasks(tasks)
        for templ, pth in zip(all_json_templates, all_json_paths):
            set_taskdomains(templ, pth + '-'.join(tasks) + '.json',
                            tasks, domains, 13370, 1337, 133)
            all_commands.append('-'.join(tasks))
            # all_commands.append('-'.join(tasks) + '0')
            # all_commands.append('-'.join(tasks) + '1')

print(list(set(all_commands)))