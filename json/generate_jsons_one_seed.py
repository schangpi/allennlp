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
domains = ["uni", "conll03", "conll02", "streusle", "semcor", "broadcast1", "fnt", "hyp"]
dataset_tasks = "dataset_reader/tasks"
dataset_domains = "dataset_reader/domains"
model_tasks = "model/tasks"
model_domains = "model/domains"
random_seed_key = "random_seed"
numpy_seed_key = "numpy_seed"
torch_seed_key = "torch_seed"
# random_seed: 13370
# numpy_seed: 1337
# torch_seed = 133
seeds = [(11170, 1117, 111), (12270, 1227, 122)]

def set_singles(toread, towrite, tasks, random_seed, numpy_seed, torch_seed):
    with open(toread, 'r') as f:
        data = json.load(f)
        data = set_data(data, random_seed_key, random_seed)
        data = set_data(data, numpy_seed_key, numpy_seed)
        data = set_data(data, torch_seed_key, torch_seed)
        data = set_data(data, model_tasks, tasks)
        with open(towrite, 'w') as fw:
            json.dump(data, fw, indent=4)

def set_multiples(toread, towrite, tasks, domains, random_seed, numpy_seed, torch_seed):
    with open(toread, 'r') as f:
        data = json.load(f)
        data = set_data(data, random_seed_key, random_seed)
        data = set_data(data, numpy_seed_key, numpy_seed)
        data = set_data(data, torch_seed_key, torch_seed)
        data = set_data(data, dataset_tasks, tasks)
        data = set_data(data, model_tasks, tasks)
        data = set_data(data, dataset_domains, domains)
        data = set_data(data, model_domains, domains)
        with open(towrite, 'w') as fw:
            json.dump(data, fw, indent=4)

for iseed, seed in enumerate(seeds):
    random_seed = seed[0]
    numpy_seed = seed[1]
    torch_seed = seed[2]
    for t1, ds1 in tskds.items():
        tasks = [t1]
        domains = sorted(list(set(ds1)))
        set_singles('tagger_template.json', 'single/tagger_' + ''.join(tasks) + str(iseed) + '.json',
                    tasks, random_seed, numpy_seed, torch_seed)
        set_singles('tagger_crf_template.json', 'single/tagger_crf_' + ''.join(tasks) + str(iseed) + '.json',
                    tasks, random_seed, numpy_seed, torch_seed)

        modelprefix = 'multi/multi_'
        set_multiples(modelprefix + 'upos.json', modelprefix + ''.join(tasks) + str(iseed) + '.json',
                      tasks, domains, random_seed, numpy_seed, torch_seed)
        modelprefix = 'task_embedding/task_embedding_tagger_'
        set_multiples(modelprefix + 'upos.json', modelprefix + ''.join(tasks) + str(iseed) + '.json',
                      tasks, domains, random_seed, numpy_seed, torch_seed)
        model_prefix = 'task_prepend_embedding/task_prepend_embedding_tagger_'
        set_multiples(modelprefix + 'upos.json', modelprefix + ''.join(tasks) + str(iseed) + '.json',
                      tasks, domains, random_seed, numpy_seed, torch_seed)
        model_prefix = 'taskonly_embedding/taskonly_embedding_tagger_'
        set_multiples(modelprefix + 'upos.json', modelprefix + ''.join(tasks) + str(iseed) + '.json',
                      tasks, domains, random_seed, numpy_seed, torch_seed)
        model_prefix = 'taskonly_prepend_embedding/taskonly_prepend_embedding_tagger_'
        set_multiples(modelprefix + 'upos.json', modelprefix + ''.join(tasks) + str(iseed) + '.json',
                      tasks, domains, random_seed, numpy_seed, torch_seed)
