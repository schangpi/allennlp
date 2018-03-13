import sys
import codecs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

def load_embeddings(file_name):
    with codecs.open(file_name, 'r', 'utf-8') as f_in:
        vocabulary, wv = zip(*[line.strip().split(' ', 1) for line in
                               f_in])
    wv = np.loadtxt(wv)
    return wv, vocabulary

pref = 'C:\\Users\\martbeerina\\Desktop\\task_embeddings\\taskembtagger_taskonly_embedding_tagger_'
pref2 = 'C:\\Users\\martbeerina\\Desktop\\task_embeddings\\taskembtagger_taskonly_prepend_embedding_tagger_'
embeddings_files = [pref + 'all_task_word_vectors.txt',
                    pref + 'all0_task_word_vectors.txt',
                    pref + 'all1_task_word_vectors.txt',
                    pref2 + 'all_task_word_vectors.txt',
                    pref2 + 'all0_task_word_vectors.txt',
                    pref2 + 'all1_task_word_vectors.txt']
embeddings_titles = ['TE-Dec Round 1',
                     'TE-Dec Round 2',
                     'TE-Dec Round 3',
                     'TE-Enc Round 1',
                     'TE-Enc Round 2',
                     'TE-Enc Round 3']
Ys = {}
vocabulary = None
for embeddings_file in embeddings_files:
    wv, vocabulary = load_embeddings(embeddings_file)
    wv = normalize(wv, axis=1, norm='l2')
    tsne_model = TSNE(perplexity=10,
                      learning_rate=10.0,
                      early_exaggeration=1.0,
                      n_components=2,
                      init='pca',
                      n_iter=50000,
                      n_iter_without_progress=1000,
                      random_state=211,
                      method='exact')
    print(tsne_model.n_iter)
    Ys[embeddings_file] = tsne_model.fit_transform(wv)
    np.set_printoptions(suppress=True)

fig = plt.figure()
for i, embeddings_file in enumerate(embeddings_files):
    ax = fig.add_subplot(231 + i)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_title(embeddings_titles[i], fontsize=13)
    Y = Ys[embeddings_file]
    plt.scatter(Y[:, 0], Y[:, 1])
    # plt.axis('off')
    for label, x, y in zip(vocabulary, Y[:, 0], Y[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', fontsize=11)
plt.show()