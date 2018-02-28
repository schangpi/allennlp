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

tskds = {"upos": ["uni", "streusle"],
         "xpos": ["uni", "streusle", "conll03"],
         "chunk": ["conll02", "conll03"],
         "mwe": ["streusle"],
         "ner": ["conll03"],
         "sem": ["semcor"],
         "semtr": ["semtraits"],
         "supsense": ["streusle"],
         "com": ["broadcast1"]}
domains = ["uni", "conll03", "conll02", "streusle", "semcor", "semtraits", "broadcast1", "broadcast2", "broadcast3"]
dataset_tasks = "dataset_reader/tasks"
dataset_domains = "dataset_reader/domains"
model_tasks = "model/tasks"
model_domains = "model/domains"
iter_tasks = "train_iterator/tasks"

with open('multi_all.json', 'r') as f:
    data = json.load(f)
    with open('multi/multi_all.json', 'w') as fw:
        json.dump(data, fw, indent=4)
    for t1, ds1 in tskds.items():
        for t2, ds2 in tskds.items():
            if t1 != t2:
                tasks = sorted(list(set([t1, t2])))
                domains = sorted(list(set(ds1 + ds2)))
                data = set_data(data, dataset_tasks, tasks)
                data = set_data(data, model_tasks, tasks)
                data = set_data(data, iter_tasks, tasks)
                data = set_data(data, dataset_domains, domains)
                data = set_data(data, model_domains, domains)
                with open('multi/multi_' + ''.join(tasks) + '.json', 'w') as fw:
                    json.dump(data, fw, indent=4)

with open('task_embedding_tagger_all.json', 'r') as f:
    data = json.load(f)
    with open('task_embedding/task_embedding_tagger_all.json', 'w') as fw:
        json.dump(data, fw, indent=4)
    for t1, ds1 in tskds.items():
        for t2, ds2 in tskds.items():
            if t1 != t2:
                tasks = sorted(list(set([t1, t2])))
                domains = sorted(list(set(ds1 + ds2)))
                data = set_data(data, dataset_tasks, tasks)
                data = set_data(data, model_tasks, tasks)
                data = set_data(data, iter_tasks, tasks)
                data = set_data(data, dataset_domains, domains)
                data = set_data(data, model_domains, domains)
                with open('task_embedding/task_embedding_tagger_' + ''.join(tasks) +'.json', 'w') as fw:
                    json.dump(data, fw, indent=4)

with open('task_prepend_embedding_tagger_all.json', 'r') as f:
    data = json.load(f)
    with open('task_prepend_embedding/task_prepend_embedding_tagger_all.json', 'w') as fw:
        json.dump(data, fw, indent=4)
    for t1, ds1 in tskds.items():
        for t2, ds2 in tskds.items():
            if t1 != t2:
                tasks = sorted(list(set([t1, t2])))
                domains = sorted(list(set(ds1 + ds2)))
                data = set_data(data, dataset_tasks, tasks)
                data = set_data(data, model_tasks, tasks)
                data = set_data(data, iter_tasks, tasks)
                data = set_data(data, dataset_domains, domains)
                data = set_data(data, model_domains, domains)
                with open('task_prepend_embedding/task_prepend_embedding_tagger_' + ''.join(tasks) + '.json',
                          'w') as fw:
                    json.dump(data, fw, indent=4)

with open('taskonly_embedding_tagger_all.json', 'r') as f:
    data = json.load(f)
    with open('taskonly_embedding/taskonly_embedding_tagger_all.json', 'w') as fw:
        json.dump(data, fw, indent=4)
    for t1, ds1 in tskds.items():
        for t2, ds2 in tskds.items():
            if t1 != t2:
                tasks = sorted(list(set([t1, t2])))
                domains = sorted(list(set(ds1 + ds2)))
                data = set_data(data, dataset_tasks, tasks)
                data = set_data(data, model_tasks, tasks)
                data = set_data(data, iter_tasks, tasks)
                data = set_data(data, dataset_domains, domains)
                data = set_data(data, model_domains, domains)
                with open('taskonly_embedding/taskonly_embedding_tagger_' + ''.join(tasks) +'.json', 'w') as fw:
                    json.dump(data, fw, indent=4)

with open('taskonly_prepend_embedding_tagger_all.json', 'r') as f:
    data = json.load(f)
    with open('taskonly_prepend_embedding/taskonly_prepend_embedding_tagger_all.json', 'w') as fw:
        json.dump(data, fw, indent=4)
    for t1, ds1 in tskds.items():
        for t2, ds2 in tskds.items():
            if t1 != t2:
                tasks = sorted(list(set([t1, t2])))
                domains = sorted(list(set(ds1 + ds2)))
                data = set_data(data, dataset_tasks, tasks)
                data = set_data(data, model_tasks, tasks)
                data = set_data(data, iter_tasks, tasks)
                data = set_data(data, dataset_domains, domains)
                data = set_data(data, model_domains, domains)
                with open('taskonly_prepend_embedding/taskonly_prepend_embedding_tagger_' + ''.join(tasks) + '.json',
                          'w') as fw:
                    json.dump(data, fw, indent=4)