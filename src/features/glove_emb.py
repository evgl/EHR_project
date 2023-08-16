""" Program to train GloVe embediings from our dataset
"""

import pandas as pd
import json
import string
import re
from functions import find_cooc_per_patient
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tqdm import tqdm
from glove import Corpus, Glove
import matplotlib.pyplot as plt
import numpy as np

# Step 1
alive_df = pd.read_csv('alive_df.csv')
dead_df = pd.read_csv('dead_df.csv')
print(f"Number of patients in label_0: {dead_df['SUBJECT_ID_x'].nunique()}")
print(f"Number of patients in label_1: {alive_df['SUBJECT_ID_x'].nunique()}")

# Read json
with open('word_dict.json', 'r') as fp:
    word_dict = json.load(fp)

patient_node_0, patient_cooc_0, patient_note_num_0 = find_cooc_per_patient(dead_df, word_dict, 0.15)
patient_node_1, patient_cooc_1, patient_note_num_1 = find_cooc_per_patient(alive_df, word_dict, 0.15)

# Leave only unique values
patient_node_set = set()

for k, v in patient_node_0.items():
    for item in v:
        patient_node_set.add(item)
    
for k, v in patient_node_1.items():
    for item in v:
        patient_node_set.add(item)

patient_square_node_data = pd.DataFrame({'node':list(patient_node_set)})

patient_square_node_id_data = patient_square_node_data.set_index("node")
patient_square_node_id_data['subject'] = "positive"
patient_subjects = patient_square_node_id_data["subject"]

# ------------------------------
all_notes_df = pd.concat([alive_df, dead_df])
text_lines = list()
lines = all_notes_df['TEXT'].values.tolist()

# words that do not have meaning (can be modified later)
USELESS_WORDS = ["a", "the", "he", "she", ",", ".", "?", "!", ":", ";", "+", "*", "**"\
                 "your", "you"]

# count up the frequency of every word in every disease file
stemmer = PorterStemmer()
# create set of words to ignore in text
stop_words = set(stopwords.words('english'))

for word in USELESS_WORDS:
    stop_words.add(word)

for line in tqdm(lines):
    
    line = re.sub(r'\[\*\*(.*?)\*\*\]|[_,\d\*:~=\.\-\+\\/]+', ' ', line)
    tokens = word_tokenize(line)
#     tokens = [w.lower() for w in tokens]
    words = [stemmer.stem(word.lower()) for word in tokens]
    # filter out stop words
    words = [w for w in words if not w in stop_words]
    
    text_lines.append(words)

print(len(text_lines))

# ----------------------
# GloVe
# creating a corpus object
corpus = Corpus() 
#training the corpus to generate the co occurence matrix which is used in GloVe
corpus.fit(text_lines, window=5)

# Creating a Glove object which will use the matrix created in the above lines to create embeddings
# We can set the learning rate as it uses Gradient Descent and number of components

# Train glove embeddings
glove = Glove(no_components=128, learning_rate=0.05)
 
glove.fit(corpus.matrix, epochs=1, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)
glove.save('glove.model')
# glove = Glove.load('glove.model')

# Create np.array only for set of patient nodes
emb_list = []

for node in tqdm(list(patient_node_set)):
    if node not in glove.dictionary:
        print(node)
    else:
        emb_list.append(glove.word_vectors[glove.dictionary[node]])

# Visualise node embeddings generated by GloVe
# Retrieve node embeddings and corresponding subjects
patient_node_ids = list(patient_node_set)  # list of node IDs

# the gensim ordering may not match the StellarGraph one, so rearrange
patient_node_targets = patient_subjects.loc[patient_node_ids].astype("category")

patient_node_embeddings = np.asarray(emb_list)
print(f"patient_node_embeddings.shape: {patient_node_embeddings.shape}")

# from sklearn.manifold import TSNE
# Apply t-SNE transformation on node embeddings
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42)
patient_node_embeddings_2d = tsne.fit_transform(patient_node_embeddings)

# draw the points
import matplotlib.pyplot as plt
alpha = 0.7

plt.figure(figsize=(10, 8))
plt.scatter(
    patient_node_embeddings_2d[:, 0],
    patient_node_embeddings_2d[:, 1],
    c=patient_node_targets.cat.codes,
    cmap="jet",
    alpha=0.7,
)
plt.savefig('glove_emb.png')
plt.show()