"""Get rows from dataframe. Each row is a note. 
Each note is a text, that can be several sentences. 
Format is string.

Problem 1:
The code is messy because the patient nodes contains tokens that are already stemmed.
However Bert takes the original text and takes care of pre-processings. The only 
limitation is that the model can take only 512 tokens. 

Solution:
Cut the text into parts if it >> 512. Or take only part where the words are contained.

Problem 2:
To find stemmed tokens from notes in a patient_node_set. Then find original words (not stemmed) in a note and
do embeddings on them. The challenge is that words can repeat several times in a text for the same token but 
in Bert model they have different embeddings in different sentences. 

Solution:
Search along the tokens one by one and cut the tokens list when found. The next token
will be searched in the cut list. This way we can find tokens even if they repeat

Problem 3:
Our goal is to create a dictionary of single node (word) : single vector. However the Bert model can
have several different vectors for one word.

Solution:
Concantenate last 4 vectors into one, or take average of the vectors
"""
from functions import find_cooc_per_patient
import json
import re
import nltk.data
import pandas as pd
import time
from biobert_embedding.embedding import BiobertEmbedding
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

alive_df = pd.read_csv('alive_df.csv')
dead_df = pd.read_csv('dead_df.csv')

# Combine all notes
all_notes_df = pd.concat([alive_df, dead_df])

# # Read json
# with open('word_dict.json', 'r') as fp:
#     word_dict = json.load(fp)

# patient_node_0, patient_cooc_0, patient_note_num_0 = find_cooc_per_patient(dead_df, word_dict, 0.15)
# patient_node_1, patient_cooc_1, patient_note_num_1 = find_cooc_per_patient(alive_df, word_dict, 0.15)

# # Leave only unique values
# patient_node_set = set()

# for _, v in patient_node_0.items():
#     for item in v:
#         patient_node_set.add(item)
    
# for _, v in patient_node_1.items():
#     for item in v:
#         patient_node_set.add(item)

# with open('patient_node_set_list.txt', 'w') as f:
#     f.write(json.dumps(list(patient_node_set)))

#Now read the file back into a Python list object
with open('patient_node_set_list.txt', 'r') as f:
    patient_node_set = json.loads(f.read())

# def _bbert_candidate_node_check(bbert_node):
#     """Function to execute the same changes that were done to the patient node,
#     in order to compare them
    
#     We return list because biobert tokenize ex. cp/dizzines as one token, but our case
#     after cleaning the token split into two ['cp', 'dizzi']

#     Arguments:
#         bbert_node (str): biobert token from sentence

#     Returns:
#         list: list of transformed token"""
#     candidate_token = re.sub(r'\[\*\*(.*?)\*\*\]|[_,\d\*:~=\.\-\+\\/]+', ' ', bbert_node)
#     candidate_token =  word_tokenize(candidate_token)
#     candidate_token = [stemmer.stem(word.lower()) for word in candidate_token]
#     return candidate_token


# # Dictionary for embeddings
# biobert_emb = {}

# # Get all rows from clinical notes dataframe
# rows = all_notes_df['TEXT'].values.tolist()

# # Class Initialization (You can set default 'model_path=None' as your finetuned BERT model path while Initialization)
# biobert = BiobertEmbedding()

# count_errors = 0
# count_all = 0
# start = time.time()
# # Iterate through each row
# for row in tqdm(rows):
#     lines_list = tokenizer.tokenize(row)
#     for line in lines_list:
    #     # Pre-processing
    
    #     # Get rid of punctuations
    #     _line = re.sub(r'\[\*\*(.*?)\*\*\]|[_,\d\*:~=\.\-\+\\/]+', ' ', line)
    # #     print(_line)
    #     # Tokenize note text
    #     tokens = word_tokenize(_line)
    #     # tokens = word_tokenize(line)
    # #     print(tokens)
        
    
#         # Stemm each word in a list of tokens
#         words = [stemmer.stem(word.lower()) for word in tokens]
#         # Look for words that need to be added to a dictionary
#         words_for_embeddings = []
#         idx_for_embeddings = []
#         original_word_for_embeddings = []
#         # Iterate each word
#         for i, word in enumerate(words):
#             # If word in a note is in the set of patient nodes
#             if word in list(patient_node_set):
#                 idx_for_embeddings.append(i)
#                 words_for_embeddings.append(word)
#                 original_word_for_embeddings.append(tokens[i])

#         # If list of indexes for word that need to be embedded is not empty
#         if original_word_for_embeddings:
            

#             # print('=' * 60)
#             # print(original_word_for_embeddings)
#             # print(words_for_embeddings)
#             # print(idx_for_embeddings)
#             # print('=' * 60)

#             # Do the biobert embeddings
#             """WARNING! Bert model can handle only 512 tokens. 
#             So the tokens number should be checked before doing the embeddings."""
#             # print(line)
#             try:
#                 word_embeddings = biobert.word_vector(line)
#                 count_all += 1
#             except Exception as e:
#                 count_errors += 1
#                 print(len(tokens))
#                 print("Oops!", e.__class__, "occurred.")
                

# print(f"\nNumber of all: {count_all}\nNumber of errors: {count_errors}")
# end = time.time()
# print(f"Time take to finish: {end - start}")




#             # print('=' * 60)
#             # print ('Shape of Word Embeddings: %d x %d' % (len(word_embeddings), len(word_embeddings[0])))
#             # print('=' * 60)
        
#             """ In order to find right word among the bert tokens...
#             The words are appearing in order"""
#             _emb_idx = 0
#             # Index start from 1
#             # Iterate through stemmed words
#             for word in words_for_embeddings:

#                 if word in biobert_emb:
#                     for i in range(_emb_idx, len(biobert.tokens)):
#                         if biobert.tokens[i] == biobert_emb[word]["original_word"]:
#                             # print(biobert.tokens)
#                             # print(f"\ni = {i}, _emb_idx: {_emb_idx}, biobert.tokens[i]: {biobert.tokens[i]}")
                            
#                             biobert_emb[word]["vector"].append(word_embeddings[i])
#                             _emb_idx = i
#                             break
#                 else:
#                     # Find a right token in Bert model tokens
#                     for i in range(_emb_idx, len(biobert.tokens)):
                      
#                         candidate_token = biobert.tokens[i]
#                         _candidate_token = _bbert_candidate_node_check(candidate_token)

#                         # Found node among bert
#                         if word in _candidate_token:
# #                             print(f"Found a new word: {word}")
# #                             print(f"\ni = {i}, _emb_idx: {_emb_idx}")
#                             # Add a node to a 
#                             biobert_emb[word] = {"original_word" : biobert.tokens[i], "vector" : [word_embeddings[i]]}
#                             # Update the start index for BioBert tokens
#                             _emb_idx = i
#                             break
#                             # print('=' * 60)

# Number of all: 95983
# Number of errors: 7304
# Time take to finish: 11269.214816093445