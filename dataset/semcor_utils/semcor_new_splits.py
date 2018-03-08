def load_sentences(path):
    sentences = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            wordtags = line.strip().split()
            words = [w.split('<Deli>')[0] for w in wordtags]
            sentences.append(words)
    # print(sentences)
    return sentences

def load_sentences_and_tags(path, verbose=False):
    sentences = []
    sentence_tags = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            words = []
            tags = []
            wordtags = line.strip().split()
            for w in wordtags:
                if verbose and '_' in w:
                    print('found _')
                wordtag = w.split('<Deli>')
                multiword = wordtag[0].split('_')
                words += multiword
                tags += [wordtag[1]] * len(multiword)
            sentences.append(words)
            sentence_tags.append(tags)
    # print(sentences)
    return sentences, sentence_tags

dataset_path = '../multi_tagger/'
train_semcor_sentences, train_semcor_tags = load_sentences_and_tags('../semcor/sem_semcor_train_old.txt')
dev_semcor_sentences, dev_semcor_tags = load_sentences_and_tags('../semcor/sem_semcor_dev_old.txt')
test_semcor_sentences, test_semcor_tags = load_sentences_and_tags('../semcor/sem_semcor_test_old.txt')
semcor_dict = {}
tbd = []
for sent, tags in zip(train_semcor_sentences, train_semcor_tags):
    key = '_'.join(sent)
    if key in semcor_dict:
        if semcor_dict[key] !=  '_'.join(tags):
            # print('Warning', key, semcor_dict[key], '_'.join(tags))
            tbd.append(key)
    else:
        semcor_dict[key] = '_'.join(tags)
for sent, tags in zip(dev_semcor_sentences, dev_semcor_tags):
    key = '_'.join(sent)
    if key in semcor_dict:
        if semcor_dict[key] !=  '_'.join(tags):
            # print('Warning', key, semcor_dict[key], '_'.join(tags))
            tbd.append(key)
    else:
        semcor_dict[key] = '_'.join(tags)
for sent, tags in zip(test_semcor_sentences, test_semcor_tags):
    key = '_'.join(sent)
    if key in semcor_dict:
        if semcor_dict[key] !=  '_'.join(tags):
            # print('Warning', key, semcor_dict[key], '_'.join(tags))
            tbd.append(key)
    else:
        semcor_dict[key] = '_'.join(tags)

print(len(tbd), tbd)
for d in list(set(tbd)):
    del semcor_dict[d]

train_semtr_sentences = load_sentences(dataset_path + 'train/semtr_semtraits_train.txt')
dev_semtr_sentences = load_sentences(dataset_path + 'dev/semtr_semtraits_dev.txt')
test_semtr_sentences = load_sentences(dataset_path + 'test/semtr_semtraits_test.txt')

train_notfound = []
with open('sem_semcor_train.txt', 'w') as f:
    for sent in train_semtr_sentences:
        key = '_'.join(sent)
        if key in semcor_dict:
            for w, t in zip(sent, semcor_dict[key].split('_')):
                f.write(w + '<Deli>' + t + ' ')
            f.write('\n')
        else:
            train_notfound.append(sent)

dev_notfound = []
with open('sem_semcor_dev.txt', 'w') as f:
    for sent in dev_semtr_sentences:
        key = '_'.join(sent)
        if key in semcor_dict:
            for w, t in zip(sent, semcor_dict[key].split('_')):
                f.write(w + '<Deli>' + t + ' ')
            f.write('\n')
        else:
            dev_notfound.append(sent)

test_notfound = []
with open('sem_semcor_test.txt', 'w') as f:
    for sent in test_semtr_sentences:
        key = '_'.join(sent)
        if key in semcor_dict:
            for w, t in zip(sent, semcor_dict[key].split('_')):
                f.write(w + '<Deli>' + t + ' ')
            f.write('\n')
        else:
            test_notfound.append(sent)

print(len(train_notfound), ' not found.')
print(len(dev_notfound), ' not found.')
print(len(test_notfound), ' not found.')

train_semtr_sentences, train_semtr_tags = load_sentences_and_tags(dataset_path + 'train/semtr_semtraits_train.txt',
                                                                  True)
dev_semtr_sentences, dev_semtr_tags = load_sentences_and_tags(dataset_path + 'dev/semtr_semtraits_dev.txt',
                                                              True)
test_semtr_sentences, test_semtr_tags = load_sentences_and_tags(dataset_path + 'test/semtr_semtraits_test.txt',
                                                                True)
semtr_dict = {}
tbd = []
for sent, tags in zip(train_semtr_sentences, train_semtr_tags):
    key = '_'.join(sent)
    if key in semtr_dict:
        if semtr_dict[key] !=  '_'.join(tags):
            tbd.append(key)
    else:
        semtr_dict[key] = '_'.join(tags)
for sent, tags in zip(dev_semtr_sentences, dev_semtr_tags):
    key = '_'.join(sent)
    if key in semtr_dict:
        if semtr_dict[key] !=  '_'.join(tags):
            tbd.append(key)
    else:
        semtr_dict[key] = '_'.join(tags)
for sent, tags in zip(test_semtr_sentences, test_semtr_tags):
    key = '_'.join(sent)
    if key in semtr_dict:
        if semtr_dict[key] !=  '_'.join(tags):
            tbd.append(key)
    else:
        semtr_dict[key] = '_'.join(tags)

print(len(tbd), tbd)
for d in list(set(tbd)):
    del semtr_dict[d]

train_notfound = []
with open('semtr_semcor_train.txt', 'w') as f:
    for sent in train_semtr_sentences:
        key = '_'.join(sent)
        if key in semtr_dict:
            for w, t in zip(sent, semtr_dict[key].split('_')):
                f.write(w + '<Deli>' + t + ' ')
            f.write('\n')
        else:
            train_notfound.append(sent)

dev_notfound = []
with open('semtr_semcor_dev.txt', 'w') as f:
    for sent in dev_semtr_sentences:
        key = '_'.join(sent)
        if key in semtr_dict:
            for w, t in zip(sent, semtr_dict[key].split('_')):
                f.write(w + '<Deli>' + t + ' ')
            f.write('\n')
        else:
            dev_notfound.append(sent)

test_notfound = []
with open('semtr_semcor_test.txt', 'w') as f:
    for sent in test_semtr_sentences:
        key = '_'.join(sent)
        if key in semtr_dict:
            for w, t in zip(sent, semtr_dict[key].split('_')):
                f.write(w + '<Deli>' + t + ' ')
            f.write('\n')
        else:
            test_notfound.append(sent)

print(len(train_notfound), ' not found.')
print(len(dev_notfound), ' not found.')
print(len(test_notfound), ' not found.')