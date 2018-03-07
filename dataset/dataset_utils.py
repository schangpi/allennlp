import re
import io
import json

def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True


def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def iobes_iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags

def update_tag_scheme(sentences, tag_scheme, idx=-1):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.
    """
    for i, s in enumerate(sentences):
        tags = [w[idx] for w in s]
        # Check that tags are given in the IOB format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                word[idx] = new_tag
        elif tag_scheme == 'iobes':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[idx] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')

def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico

def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item

def word_mapping(sentences, idx=0):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    words = [[x[idx] for x in s] for s in sentences]
    dico = create_dico(words)
    word_to_id, id_to_word = create_mapping(dico)
    print("Found %i unique words (%i in total)" % (len(dico), sum(len(x) for x in words)))
    return dico, word_to_id, id_to_word

def tag_mapping(sentences, idx=-1):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    dico, tag_to_id, id_to_tag = word_mapping(sentences, idx)
    print("Found %i unique tags" % len(dico))
    return dico, tag_to_id, id_to_tag

def f_process(x, zeros, lower):
    x = x.lower() if lower else x
    x = zero_digits(x) if zeros else x
    return x

def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', 'DG', s)

def load_sentences_uni(path, zeros, lower):
    sentences = []
    sentence = []
    for line in io.open(path, 'r', encoding='utf8'):
        if line.startswith('#') or (line.find('-') != -1 and line.find('-') < line.find('\t')):
            continue
        l = line.split()
        if len(l) > 0:
            # form, fpos, pos
            word = [f_process(l[1], zeros, lower), l[4], l[3]]
            sentence.append(word)
        else:
            sentences.append(sentence)
            sentence = []
    return sentences

def load_sentences_conll03(path, zeros, lower):
    # Load sentences. A line must contain at least a word and its tag.
    # Sentences are separated by empty lines.
    sentences = []
    sentence = []
    for line in io.open(path, 'r', encoding='utf8'):
        line = line.rstrip()
        if not line:
            if len(sentence) > 0:
                if f_process('DOCSTART', False, lower) not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            l = line.split()
            word = [f_process(l[0], zeros, lower)] + l[1:]
            assert len(word) >= 2
            sentence.append(word)
    if len(sentence) > 0:
        if f_process('DOCSTART', False, lower) not in sentence[0][0]:
            sentences.append(sentence)
    return sentences

def load_sentences_conll02(path, zeros, lower):
    # Load sentences. A line must contain at least a word and its tag.
    # Sentences are separated by empty lines.
    sentences = []
    sentence = []
    for line in io.open(path, 'r', encoding='utf8'):
        line = line.rstrip()
        if not line:
            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []
        else:
            l = line.split()
            assert len(l) >= 2
            word = [f_process(l[0], zeros, lower)] + l[1:]
            sentence.append(word)
    if len(sentence) > 0:
        sentences.append(sentence)
    return sentences

def load_sentences_general(path, zeros, lower):
    # Load sentences. A line must contain at least a word and its tag.
    # Sentences are separated by empty lines.
    sentences = []
    sentence = []
    for line in io.open(path, 'r', encoding='utf8'):
        line = line.rstrip()
        if not line:
            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []
        else:
            l = line.split()
            assert len(l) >= 2
            word = [f_process(l[0], zeros, lower)] + l[1:]
            sentence.append(word)
    if len(sentence) > 0:
        sentences.append(sentence)
    return sentences

def load_sentences_general_noutfok(path, zeros, lower):
    # Load sentences. A line must contain at least a word and its tag.
    # Sentences are separated by empty lines.
    sentences = []
    sentence = []
    for line in open(path, 'r', encoding='utf8', errors="ignore"):
        line = line.rstrip()
        if not line:
            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []
        else:
            l = line.split()
            assert len(l) >= 2
            word = [f_process(l[0], zeros, lower)] + l[1:]
            sentence.append(word)
    if len(sentence) > 0:
        sentences.append(sentence)
    return sentences

def load_sentences_general_no_O(path, zeros, lower):
    # Load sentences. A line must contain at least a word and its tag.
    # Sentences are separated by empty lines.
    sentences = []
    sentence = []
    for line in io.open(path, 'r', encoding='utf8'):
        line = line.rstrip()
        if not line:
            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []
        else:
            l = line.split()
            assert len(l) >= 2
            word = [f_process(l[0], zeros, lower)] + [t if t != 'O' else 'KEEP' for t in l[1:]]
            sentence.append(word)
    if len(sentence) > 0:
        sentences.append(sentence)
    return sentences

def load_sentences_streusle(path, zeros, lower):
    sentences = []
    with open(path, 'r') as f:
        data = json.load(f)
        for sent in data:
            sentence = []
            tokens = sent['toks']
            for tok in tokens:
                splitted_laxtag = tok['lextag'].split('-')
                mwe = splitted_laxtag[0]
                # if mwe == 'o':
                #     mwe = 'O'
                # if mwe != 'O':
                #     mwe = '@@' + mwe
                mwe = mwe.upper()
                smwe = 'O'
                wmwe = 'O'
                if tok['smwe'] is not None:
                    assert (mwe != 'O')
                    smwe = mwe[0]
                    mwe = mwe[0]
                if tok['wmwe'] is not None:
                    wmwe = mwe[0]
                    mwe = mwe[0]
                    if mwe == 'O':
                        print('Warning: weak MWE with O')
                supersense = 'O'
                if len(splitted_laxtag) == 3:
                    supersense = splitted_laxtag[2]
                word = [f_process(tok['word'], zeros, lower),
                        tok['xpos'],
                        tok['upos'],
                        mwe,
                        smwe,
                        wmwe,
                        supersense]
                sentence.append(word)
            if len(sentence) > 0:
                sentences.append(sentence)
    return sentences