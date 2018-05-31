import numpy as np
import collections
from itertools import chain

"""
    Define plays dictionary, R: Rock, S: Scissors, P: Paper
"""
p = {'R': 1, 'S': 2, 'P': 3}


def read_play_sequences(file_name):
    with open(file_name) as f:
        plays_seq = f.readlines()
    plays_seq = [x.strip() for x in plays_seq]
    plays_seq = [plays_seq[i].split() for i in range(len(plays_seq))]
    plays_seq = np.array(plays_seq)
    plays_seq = np.reshape(plays_seq, [-1, ])
    return plays_seq


def read_data(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    content = [content[i].split() for i in range(len(content))]
    content = np.array(content)
    content = list(chain(*content))
    # content = np.reshape(content, [-1, ])
    return content


def build_encode_decode_dictionary(text):
    if text is not None:
        # Index words by frequency of appearance in the text
        count = collections.Counter(text).most_common()
        encode = dict()
        for word, _ in count:
            # key is the word, value is the position of the word in dictionary
            encode[word] = len(encode)
        decode = dict(zip(encode.values(), encode.keys()))
        return encode, decode

def find_index(training_data,original_text):
    for i in range(len(training_data)-2):
        if original_text[0] == training_data[i]:
            if original_text[1] == training_data[i+1]:
                if original_text[2] == training_data[i+2]:
                    return i+3
    return -1
