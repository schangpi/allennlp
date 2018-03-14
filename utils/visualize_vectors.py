import sys
import codecs
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

def load_embeddings(file_name, topk=None):
    with codecs.open(file_name, 'r', 'utf-8') as f_in:
        vocabulary, wv = zip(*[line.strip().split(' ', 1) for line in
                               f_in])
    wv = np.loadtxt(wv)
    if topk is not None:
        wv = wv[:topk]
        vocabulary = vocabulary[:topk]
    return wv, vocabulary

def get_topk(Y, topk, vocab):
    cnt = 0
    new_Y = []
    new_vocab = []
    num_tasks = 0
    for i, v in enumerate(vocab):
        if '<<' in v and '>>' in v:
            new_Y.append(Y[i])
            new_vocab.append(v)
            num_tasks += 1
            if num_tasks == 11:
                break
        else:
            if cnt < topk:
                new_Y.append(Y[i])
                new_vocab.append(v)
                cnt += 1
    return np.array(new_Y), new_vocab

# pref_te = 'C:\\Users\\martbeerina\\Desktop\\task_embeddings\\taskembtagger_taskonly_embedding_tagger_'
# pref_tpe = 'C:\\Users\\martbeerina\\Desktop\\task_embeddings\\taskembtagger_taskonly_prepend_embedding_tagger_'

pref_te = 'task_embeddings/taskembtagger_taskonly_embedding_tagger_'
pref_tpe = 'task_embeddings/taskembtagger_taskonly_prepend_embedding_tagger_'

embeddings_files_te = [pref_te + 'all_task_word_vectors.txt',
                       pref_te + 'all0_task_word_vectors.txt',
                       pref_te + 'all1_task_word_vectors.txt']
embeddings_files_tpe = [pref_tpe + 'all_task_word_vectors.txt',
                        pref_tpe + 'all0_task_word_vectors.txt',
                        pref_tpe + 'all1_task_word_vectors.txt']
embeddings_words_files_tpe = [pref_tpe + 'all_topwords_vectors.txt',
                              pref_tpe + 'all0_topwords_vectors.txt',
                              pref_tpe + 'all1_topwords_vectors.txt']
embeddings_files = embeddings_files_te + embeddings_files_tpe

embeddings_titles_te = ['TE-Dec Round 1',
                        'TE-Dec Round 2',
                        'TE-Dec Round 3']
embeddings_titles_tpe = ['TE-Enc Round 1',
                         'TE-Enc Round 2',
                         'TE-Enc Round 3']
embeddings_words_titles = ['TE-Enc Round 1',
                           'TE-Enc Round 2',
                           'TE-Enc Round 3']
Ys = {}
vocabularys = {}
for i, embeddings_file in enumerate(embeddings_files):
    wv, vocabulary = load_embeddings(embeddings_file)
    wv = normalize(wv, axis=1, norm='l2')
    tsne_model = TSNE(perplexity=10,
                      learning_rate=5.0,
                      early_exaggeration=1.0,
                      n_components=2,
                      init='pca',
                      n_iter=5000,
                      n_iter_without_progress=100,
                      random_state=211,
                      method='exact',
                      verbose=1)
    print(tsne_model.n_iter)
    Ys[embeddings_file] = tsne_model.fit_transform(wv)
    vocabularys[embeddings_file] = vocabulary
    np.set_printoptions(suppress=True)

for i, embeddings_file in enumerate(embeddings_files_te):
    fig = plt.figure(1 + i)
    ax = fig.add_subplot(111)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_title(embeddings_titles_te[i], fontsize=18)
    Y = Ys[embeddings_file]
    plt.scatter(Y[:, 0], Y[:, 1])
    for label, x, y in zip(vocabularys[embeddings_file], Y[:, 0], Y[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', fontsize=16)
    fig.savefig('taskemb_tedec' + str(i) + '.eps')
    fig.savefig('taskemb_tedec' + str(i) + '.png')

for i, embeddings_file in enumerate(embeddings_files_tpe):
    fig = plt.figure(4 + i)
    ax = fig.add_subplot(111)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_title(embeddings_titles_tpe[i], fontsize=18)
    Y = Ys[embeddings_file]
    plt.scatter(Y[:, 0], Y[:, 1])
    for label, x, y in zip(vocabularys[embeddings_file], Y[:, 0], Y[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', fontsize=16)
    fig.savefig('taskemb_teenc' + str(i) + '.eps')
    fig.savefig('taskemb_teenc' + str(i) + '.png')

for i, embeddings_file in enumerate(embeddings_words_files_tpe):
    wv, vocabulary = load_embeddings(embeddings_file, 5000)
    wv = normalize(wv, axis=1, norm='l2')
    tsne_model = TSNE(perplexity=25.0,
                      learning_rate=10.0,
                      n_components=2,
                      init='pca',
                      n_iter=5000,
                      n_iter_without_progress=300,
                      random_state=211,
                      verbose=1)
    print(tsne_model.n_iter)
    Ys[embeddings_file] = tsne_model.fit_transform(wv)
    vocabularys[embeddings_file] = vocabulary
    np.set_printoptions(suppress=True)

topk = 100
for i, embeddings_file in enumerate(embeddings_words_files_tpe):
    fig = plt.figure(7 + i)
    ax = fig.add_subplot(111)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_title(embeddings_words_titles[i], fontsize=18)
    Y, vocab = get_topk(Ys[embeddings_file], topk, vocabularys[embeddings_file])
    plt.scatter(Y[:, 0], Y[:, 1])
    for label, x, y in zip(vocab, Y[:, 0], Y[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', fontsize=16)
    fig.savefig('taskemb_teencw' + str(i) + '.eps')
    fig.savefig('taskemb_teencw' + str(i) + '.png')

# plt.show()