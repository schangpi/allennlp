# Check whether there are any sentences in dev and test of semtraits that are in semcor

def load_sentences(path):
    sentences = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            wordtags = line.strip().split()
            words = [w.split('<Deli>')[0].lower() for w in wordtags]
            sentences.add('_'.join(words))
    # print(sentences)
    return sentences

train_str_sentences = load_sentences('multi_tagger_clean/train/semtr_semtraits_train.txt')
dev_str_sentences = load_sentences('multi_tagger_clean/dev/semtr_semtraits_dev.txt')
test_str_sentences = load_sentences('multi_tagger_clean/test/semtr_semtraits_test.txt')

train_uni_sentences = load_sentences('multi_tagger_clean/train/sem_semcor_train.txt')
dev_uni_sentences = load_sentences('multi_tagger_clean/dev/sem_semcor_dev.txt')
test_uni_sentences = load_sentences('multi_tagger_clean/test/sem_semcor_test.txt')

train_intersect = list(train_str_sentences & train_uni_sentences)
dev_intersect = list(dev_str_sentences & dev_uni_sentences)
test_intersect = list(test_str_sentences & test_uni_sentences)
print(len(train_intersect), len(list(train_str_sentences)), len(list(train_uni_sentences)))
print(len(dev_intersect), len(list(dev_str_sentences)), len(list(dev_uni_sentences)))
print(len(test_intersect), len(list(test_str_sentences)), len(list(test_uni_sentences)))
for sent in list(train_str_sentences - train_uni_sentences):
    print(sent, sent in dev_uni_sentences, sent in test_uni_sentences)
for sent in list(dev_str_sentences - dev_uni_sentences):
    print(sent, sent in train_uni_sentences, sent in test_uni_sentences)
eval_intersect = list((dev_str_sentences | test_str_sentences) & train_uni_sentences)
print(len(eval_intersect))
# for sent in eval_intersect:
#     print(sent, sent in dev_uni_sentences, sent in test_uni_sentences)

opp_eval_intersect = list((dev_uni_sentences | test_uni_sentences) & train_str_sentences)
print(len(opp_eval_intersect))
for sent in opp_eval_intersect:
    print(sent, sent in dev_str_sentences, sent in test_str_sentences)