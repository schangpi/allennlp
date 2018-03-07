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

# "upos": ["uni", "streusle"],
# "xpos": ["uni", "streusle", "conll03"],
# "chunk": ["conll02", "conll03"],

# tskds = {"mwe": ["streusle"],
#          "ner": ["conll03"],
#          "sem": ["semcor"],
#          "semtr": ["semtraits"],
#          "supsense": ["streusle"],
#          "com": ["broadcast1"]}
tskds = {"sem": ["semcor"],
         "frame": ["fnt"],
         "hyp": ["hyp"]}

domains = ["uni", "conll03", "conll02", "streusle", "semcor", "semtraits", "broadcast1", "broadcast2", "broadcast3",
           "fnt", "hyp"]
dataset_tasks = "dataset_reader/tasks"
dataset_domains = "dataset_reader/domains"
model_tasks = "model/tasks"
model_domains = "model/domains"

with open('tagger_template.json', 'r') as f:
    data = json.load(f)
    for t1, ds1 in tskds.items():
        tasks = [t1]
        domains = sorted(list(set(ds1)))
        # data = set_data(data, dataset_tasks, tasks)
        data = set_data(data, model_tasks, tasks)
        # data = set_data(data, dataset_domains, domains)
        # data = set_data(data, model_domains, domains)
        with open('single/tagger_' + ''.join(tasks) + '.json', 'w') as fw:
            json.dump(data, fw, indent=4)

with open('tagger_crf_template.json', 'r') as f:
    data = json.load(f)
    for t1, ds1 in tskds.items():
        tasks = [t1]
        domains = sorted(list(set(ds1)))
        # data = set_data(data, dataset_tasks, tasks)
        data = set_data(data, model_tasks, tasks)
        # data = set_data(data, dataset_domains, domains)
        # data = set_data(data, model_domains, domains)
        with open('single/tagger_crf_' + ''.join(tasks) + '.json', 'w') as fw:
            json.dump(data, fw, indent=4)

with open('multi/multi_upos.json', 'r') as f:
    data = json.load(f)
    for t1, ds1 in tskds.items():
        tasks = [t1]
        domains = sorted(list(set(ds1)))
        data = set_data(data, dataset_tasks, tasks)
        data = set_data(data, model_tasks, tasks)
        data = set_data(data, dataset_domains, domains)
        data = set_data(data, model_domains, domains)
        with open('multi/multi_' + ''.join(tasks) + '.json', 'w') as fw:
            json.dump(data, fw, indent=4)

with open('task_embedding/task_embedding_tagger_upos.json', 'r') as f:
    data = json.load(f)
    for t1, ds1 in tskds.items():
        tasks = [t1]
        domains = sorted(list(set(ds1)))
        data = set_data(data, dataset_tasks, tasks)
        data = set_data(data, model_tasks, tasks)
        data = set_data(data, dataset_domains, domains)
        data = set_data(data, model_domains, domains)
        with open('task_embedding/task_embedding_tagger_' + ''.join(tasks) +'.json', 'w') as fw:
            json.dump(data, fw, indent=4)

with open('task_prepend_embedding/task_prepend_embedding_tagger_upos.json', 'r') as f:
    data = json.load(f)
    for t1, ds1 in tskds.items():
        tasks = [t1]
        domains = sorted(list(set(ds1)))
        data = set_data(data, dataset_tasks, tasks)
        data = set_data(data, model_tasks, tasks)
        data = set_data(data, dataset_domains, domains)
        data = set_data(data, model_domains, domains)
        with open('task_prepend_embedding/task_prepend_embedding_tagger_' + ''.join(tasks) + '.json', 'w') as fw:
            json.dump(data, fw, indent=4)

with open('taskonly_embedding/taskonly_embedding_tagger_upos.json', 'r') as f:
    data = json.load(f)
    for t1, ds1 in tskds.items():
        tasks = [t1]
        domains = sorted(list(set(ds1)))
        data = set_data(data, dataset_tasks, tasks)
        data = set_data(data, model_tasks, tasks)
        data = set_data(data, dataset_domains, domains)
        data = set_data(data, model_domains, domains)
        with open('taskonly_embedding/taskonly_embedding_tagger_' + ''.join(tasks) +'.json', 'w') as fw:
            json.dump(data, fw, indent=4)

with open('taskonly_prepend_embedding/taskonly_prepend_embedding_tagger_upos.json', 'r') as f:
    data = json.load(f)
    for t1, ds1 in tskds.items():
        tasks = [t1]
        domains = sorted(list(set(ds1)))
        data = set_data(data, dataset_tasks, tasks)
        data = set_data(data, model_tasks, tasks)
        data = set_data(data, dataset_domains, domains)
        data = set_data(data, model_domains, domains)
        with open('taskonly_prepend_embedding/taskonly_prepend_embedding_tagger_' + ''.join(tasks) + '.json',
                  'w') as fw:
            json.dump(data, fw, indent=4)