import json

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
all_domains = ["uni", "conll03", "conll02", "streusle", "semcor", "broadcast1", "fnt", "hyp"]
dataset_tasks = "dataset_reader/tasks"
dataset_domains = "dataset_reader/domains"
model_tasks = "model/tasks"
model_domains = "model/domains"
iter_tasks = "train_iterator/tasks"
random_seed_key = "random_seed"
numpy_seed_key = "numpy_seed"
torch_seed_key = "torch_seed"
# random_seed: 13370
# numpy_seed: 1337
# torch_seed = 133
seeds = [(11170, 1117, 111), (12270, 1227, 122)]

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

multi_template = 'multi_all.json'
te_template = 'task_embedding_tagger_all.json'
tpe_template = 'task_prepend_embedding_tagger_all.json'
teonly_template = 'taskonly_embedding_tagger_all.json'
tpeonly_template = 'taskonly_prepend_embedding_tagger_all.json'
all_templates = [multi_template,
                 te_template,
                 tpe_template,
                 teonly_template,
                 tpeonly_template]

multi_path = 'multi/multi_'
te_path = 'task_embedding/task_embedding_tagger_'
tpe_path = 'task_prepend_embedding/task_prepend_embedding_tagger_'
teonly_path = 'taskonly_embedding/taskonly_embedding_tagger_'
tpeonly_path = 'taskonly_prepend_embedding/taskonly_prepend_embedding_tagger_'
all_paths = [multi_path,
             te_path,
             tpe_path,
             teonly_path,
             tpeonly_path]

for iseed, seed in enumerate(seeds):
    random_seed = seed[0]
    numpy_seed = seed[1]
    torch_seed = seed[2]
    tasks = sorted(tskds.keys())
    domains = sorted(all_domains)
    for templ, pth in zip(all_templates, all_paths):
        set_taskdomains(templ, pth + 'all' + str(iseed) + '.json',
                        tasks, domains, random_seed, numpy_seed, torch_seed)

    for t1, ds1 in tskds.items():
        tasks = sorted(tskds.keys())
        tasks.remove(t1)
        domains = sorted(list(all_domains))
        if ds1[0] != 'semcor' and ds1[0] != 'uni':
            domains.remove(ds1[0])
        for templ, pth in zip(all_templates, all_paths):
            set_taskdomains(templ, pth + 'allminus_' + t1 + str(iseed) + '.json',
                            tasks, domains, random_seed, numpy_seed, torch_seed)

    for t1, ds1 in tskds.items():
        for t2, ds2 in tskds.items():
            if t1 != t2:
                tasks = sorted(list(set([t1, t2])))
                domains = sorted(list(set(ds1 + ds2)))
                for templ, pth in zip(all_templates, all_paths):
                    set_taskdomains(templ, pth + ''.join(tasks) + str(iseed) + '.json',
                                    tasks, domains, random_seed, numpy_seed, torch_seed)
