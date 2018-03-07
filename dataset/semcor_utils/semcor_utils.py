# import nltk
# nltk.download('semcor')
from nltk.corpus import semcor
from nltk import Tree

supsense_sentences = []
sense_to_supsense = {}
c = 0
with open('english_semcor_all.conll', 'r') as f:
    sent = []
    for line in f:
        line = line.strip()
        if line != '':
            sent.append(line.split('\t'))
        else:
            if len(sent) > 0:
                supsense_sentences.append(sent)
                sent = []
        c += 1

print(supsense_sentences)
print(c)

# print(semcor.words())
# print(semcor.chunks())
i = 0
semcor_sents = semcor.tagged_sents(tag='both')
# print(semcor_sents)
with open('semcor_all.conll', 'w') as f:
    for sent in semcor_sents:
        if i == len(supsense_sentences):
            break
        ref_sent = supsense_sentences[i]
        for j, ch in enumerate(sent):
            # print(ch)
            # f.write(str(ch) + '\n')
            rt = ch.label()
            sense = 'O'
            if not isinstance(ch[0], str):
                sense = rt
                if isinstance(ch[0][0], str):
                    pos = ch[0].label()
                    word = '_'.join([child for child in ch[0]])
                    # if len(ch[0]) > 1:
                    #     print([child for child in ch[0]])
                    #     print(word)
                else:
                    pos = ch[0][0].label()
                    word = '_'.join(ch[0][0])
            else:
                if len(ch) == 1:
                    word = ch[0]
                else:
                    word = '_'.join([child for child in ch])
                if rt is not None:
                    pos = rt
                else:
                    pos = word
            # print(word, pos, sense)
            if word != ref_sent[j][0]:
                print(ch)
                print(word, ref_sent[j][0])
                # assert(word == ref_sent[j][0])
            supsense = ref_sent[j][1]
            # k = word + '_' + sense
            # if k in sense_to_supsense:
            #     if sense_to_supsense[k] != supsense :
            #         print(ch, word, ref_sent[j][0])
            #         print(k, sense_to_supsense[k], supsense)
            #         assert(sense_to_supsense[sense] == supsense)
            # else:
            #     sense_to_supsense[k] = supsense
            f.write(word + '\t' + pos + '\t' + sense + '\t' + supsense + '\n')
        # print('\n')
        f.write('\n')
        i += 1
