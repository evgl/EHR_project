# import pandas as pd
# import nltk.data
# import re
import json
import bcolz
import pickle
import numpy as np

# from tqdm import tqdm
# from nltk.tokenize import word_tokenize
# from nltk.stem import PorterStemmer
# from nltk.corpus import stopwords

# # words that do not have meaning (can be modified later)
# USELESS_WORDS = ["a", "the", "he", "she", ",", ".", "?", "!", ":", ";", "+", "*", "**"\
#                  "your", "you"]

# # create set of words to ignore in text
# stop_words = set(stopwords.words('english'))

# stemmer = PorterStemmer()
# tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# alive_df = pd.read_csv('alive_df.csv')
# dead_df = pd.read_csv('dead_df.csv')

# all_notes_df = pd.concat([alive_df, dead_df])

# # Get all rows from clinical notes dataframe
# rows = all_notes_df['TEXT'].values.tolist()

# # Read the file back into a Python list object
# with open('patient_node_set_list.txt', 'r') as f:
#     patient_node_set = json.loads(f.read())


# orig_word_dict = dict((k, set()) for k in patient_node_set) 
# for row in tqdm(rows):
#     # Pre-processing
#     # Get rid of punctuations
#     _line = re.sub(r'\[\*\*(.*?)\*\*\]|[_,\d\*:~=\.\-\+\\/]+', ' ', row)
#     # Tokenize note text
#     tokens = word_tokenize(_line)
#     words = [w for w in tokens if not w in stop_words]

#     for word in words:
#         stem_w = stemmer.stem(word.lower())
#         if stem_w in orig_word_dict:
#             orig_word_dict[stem_w].add(word.lower())

# for key, value in orig_word_dict.items():
#     orig_word_dict[key] = list(value)

# # Write json
# with open('orig_word_dict.json', 'w') as fp:
#     json.dump(orig_word_dict, fp)

# Read json
with open('orig_word_dict.json', 'r') as fp:
    orig_word_dict = json.load(fp)

with open('patient_node_set_list.txt', 'r') as f:
    patient_node_set = json.loads(f.read())

# Glove
glove_path = 'glove.6B'
# words = []
# idx = 0
# word2idx = {}
# vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/6B.100.dat', mode='w')

# with open(f'{glove_path}/glove.6B.100d.txt', 'rb') as f:
#     for l in f:
#         line = l.decode().split()
#         word = line[0]
#         words.append(word)
#         word2idx[word] = idx
#         idx += 1
#         vect = np.array(line[1:]).astype(np.float)
#         vectors.append(vect)
    
# vectors = bcolz.carray(vectors[1:].reshape((400000, 100)), rootdir=f'{glove_path}/6B.100.dat', mode='w')
# vectors.flush()
# pickle.dump(words, open(f'{glove_path}/6B.100_words.pkl', 'wb'))
# pickle.dump(word2idx, open(f'{glove_path}/6B.100_idx.pkl', 'wb'))

vectors = bcolz.open(f'{glove_path}/6B.100.dat')[:]
words = pickle.load(open(f'{glove_path}/6B.100_words.pkl', 'rb'))
word2idx = pickle.load(open(f'{glove_path}/6B.100_idx.pkl', 'rb'))

glove = {w: vectors[word2idx[w]] for w in words}


matrix_len = len(list(patient_node_set))
emb_dim = 100
glove_emb = {}
words_not_found = 0

for i, word in enumerate(list(patient_node_set)):
    try: 
        glove_emb[word] = glove[word]
    except KeyError:

        if any(w in glove for w in list(orig_word_dict[word])):
            for w in list(orig_word_dict[word]):
                if w in glove:
                    glove_emb[word] = glove[w]
                    break
        else:
            words_not_found += 1
            print(f"{word} : {list(orig_word_dict[word])}")
            glove_emb[word] = np.random.normal(scale=0.6, size=(emb_dim, ))

# Visualize 