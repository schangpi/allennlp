from collections import Counter
import scipy.stats

def load_sentences(path):
    sentences = []
    all_tags = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            wordtags = line.strip().split()
            words = [w.split('<Deli>')[0].lower() for w in wordtags]
            tags = [w.split('<Deli>')[1].lower() for w in wordtags]
            sentences.append(words)
            all_tags.append(tags)
    return sentences, all_tags

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

def get_token_type_ratio(sentences):
    wordfreq = Counter()
    for sent in sentences:
        for word in sent:
            wordfreq[word] += 1
    num_tokens = sum(wordfreq.values())
    num_types = len(wordfreq.keys())
    return float(num_tokens) / num_types

def get_tag_stats(all_tags):
    tagfreq = Counter()
    for tags in all_tags:
        for tag in tags:
            tagfreq[tag] += 1
    return len(tagfreq.keys()), scipy.stats.entropy(list(tagfreq.values()))

def get_info_list(st):
    return [st["train_sentences"], st["train_toktyp"], st["train_nlab"], st["train_labent"],
            st["all_sentences"], st["all_toktyp"], st["all_nlab"], st["all_labent"]]

data_stats = {}
for tsk, domain in tskds.items():
    st = {}
    print(tsk)
    train_sentences, train_labels = load_sentences('multi_tagger_clean/train/' + tsk + '_' + domain[0] + '_train.txt')
    dev_sentences, dev_labels = load_sentences('multi_tagger_clean/dev/' + tsk + '_' + domain[0] + '_dev.txt')
    test_sentences, test_labels = load_sentences('multi_tagger_clean/test/' + tsk + '_' + domain[0] + '_test.txt')
    st["train_sentences"] = len(train_sentences)
    st["dev_sentences"] = len(dev_sentences)
    st["test_sentences"] = len(test_sentences)
    st["all_sentences"] = len(train_sentences + dev_sentences + test_sentences)
    st["train_toktyp"] = get_token_type_ratio(train_sentences)
    st["dev_toktyp"] = get_token_type_ratio(dev_sentences)
    st["test_toktyp"] = get_token_type_ratio(test_sentences)
    st["all_toktyp"] = get_token_type_ratio(train_sentences + dev_sentences + test_sentences)
    st["train_nlab"], st["train_labent"] = get_tag_stats(train_labels)
    st["dev_nlab"], st["dev_labent"] = get_tag_stats(dev_labels)
    st["test_nlab"], st["test_labent"] = get_tag_stats(test_labels)
    st["all_nlab"], st["all_labent"] = get_tag_stats(train_labels + dev_labels + test_labels)
    data_stats[tsk] = st

dstxt = {"uni": "Universal Dependencies v1.4",
         "conll02": "CoNLL-2000" ,
         "conll03": "CoNLL-2003" ,
         "streusle": "Streusle 4.0",
         "semcor": "SemCor",
         "broadcast1": "Broadcast News 1",
         "fnt": "FrameNet 1.5",
         "hyp": "Hyper-Text Corpus"}

task_groups = [["upos", "xpos"], ["chunk"], ["ner"], ["mwe", "supsense"], ["sem", "semtr"], ["com"], ["frame"], ["hyp"]]
dg = 1
latstr = ''
for tg in task_groups:
    num_rows = len(tg)
    if num_rows > 1:
        st = get_info_list(data_stats[tg[0]])
        latstr += '\\multirow{' + str(num_rows) + '}{*}{' + dstxt[tskds[tg[0]][0]] + '} &' + \
                  '\\multirow{' + str(num_rows) + '}{*}{' + str(st[0]) + '/' + str(st[4]) + '} & ' + \
                  '\\multirow{' + str(num_rows) + '}{*}{' + str(round(st[1], dg)) + '/' + str(round(st[5], dg)) + '} & ' + \
                  '\\task{' + tg[0] + '} & ' + \
                  str(st[6]) + ' & ' + \
                  str(round(st[3], dg)) + \
                  '\\\\\n'
                  # str(round(st[3], dg)) + '/' + str(round(st[7], dg)) + \
        for j in range(1, num_rows):
            current_tsk = tg[j]
            st = get_info_list(data_stats[current_tsk])
            latstr += '& ' + \
                      '& ' + \
                      '& ' + \
                      '\\task{' + current_tsk + '} & ' + \
                      str(st[6]) + ' & ' + \
                      str(round(st[3], dg)) + \
                      '\\\\'
            if j == num_rows - 1:
                latstr += ' \\hline'
            latstr += ' \n'
    else:
        st = get_info_list(data_stats[tg[0]])
        latstr += dstxt[tskds[tg[0]][0]] + ' & ' + \
                  str(st[0]) + '/' + str(st[4]) + ' & ' + \
                  str(round(st[1], dg)) + '/' + str(round(st[5], dg)) + ' & ' + \
                  '\\task{' + tg[0] + '} & ' + str(st[6])
        if tg[0] == "ner" or tg[0] == "chunk":
            latstr += ' (IOBES)'
        latstr += ' & ' + \
                  str(round(st[3], dg)) + \
                  '\\\\ \\hline \n'
                  # str(round(st[3], dg)) + '/' + str(round(st[7], dg)) + \
print(latstr)