import pandas as pd
import numpy as np
import json
import tensorflow as tf
import matplotlib.pyplot as plt
from functions import extract_data, count_notes_per_patient, logger, count_words_per_patient, find_frequent_word, find_cooc_per_patient
from functions import cooc_log_odd_score, train_embeddings
from functions import create_graphs_lists, train_model
from nltk.stem import PorterStemmer
from sklearn import model_selection

# Variables

# Input vars --->
disease_name = 'PNEUMONIA'
database_path = '../MIMIC-III'

patient_id_to_num_notes = {}

number_of_patients = {}
note_appearance_counter = {}
## Step 4
n_fold = float(3)
threshold = float(0.01)
frequent_word_lists = {}

min_sup = 0.15
# Input vars ---<

# Save dataframe ------------------->
# alive_df.to_csv('alive_df.csv', index = False, header=True)
# dead_df.to_csv('dead_df.csv', index = False, header=True)
# Save dataframe -------------------<

# Step 1
# alive_df, dead_df = extract_data(disease_name, database_path)
alive_df = pd.read_csv('alive_df.csv')
dead_df = pd.read_csv('dead_df.csv')

logger.info(f"Number of patients in label_0: {dead_df['SUBJECT_ID_x'].nunique()}")
logger.info(f"Number of patients in label_1: {alive_df['SUBJECT_ID_x'].nunique()}")

# # Step 2
# logger.info("Count notes per patient...")
# patient_id_to_num_notes['label_0'] = count_notes_per_patient(dead_df)
# patient_id_to_num_notes['label_1'] = count_notes_per_patient(alive_df)

# # Step 3
# logger.info("Count words per patient...")
# number_of_patients['label_0'], note_appearance_counter['label_0'] = count_words_per_patient(dead_df, patient_id_to_num_notes['label_0'])
# number_of_patients['label_1'], note_appearance_counter['label_1'] = count_words_per_patient(alive_df, patient_id_to_num_notes['label_1'])

# # Step 4
# logger.info("Find frequent words...")
# word_dict = find_frequent_word(note_appearance_counter, number_of_patients, n_fold, threshold)


# # Write json
# with open('word_dict.json', 'w') as fp:
#     json.dump(word_dict, fp)

# # TODO: Step 2-4 can be merged into one

# Read json
with open('word_dict.json', 'r') as fp:
    word_dict = json.load(fp)


# Step 5
logger.info("Count co-occurrences per patient...")
patient_node_0, patient_cooc_0, patient_note_num_0 = find_cooc_per_patient(dead_df, word_dict, 0.15)
patient_node_1, patient_cooc_1, patient_note_num_1 = find_cooc_per_patient(alive_df, word_dict, 0.15)

# Step 6
logger.info("Get and normalize weights in co-occurrences...")
normalized_cooc_odd_scores = cooc_log_odd_score(patient_cooc_0, patient_cooc_1)

# Step 7
logger.info("Train embeddings...")
sequence2vec = train_embeddings(patient_node_0, patient_node_1, normalized_cooc_odd_scores)

# Step 8
# Create graphs, graph labels, train and test data
logger.info("Create graphs, graph labels, train and test data...")
graphs, graph_labels, train_index, test_index = create_graphs_lists(patient_cooc_0, patient_cooc_1, normalized_cooc_odd_scores, sequence2vec)
# -------------------

# Step 9
# Train model
logger.info("Train model...")

test_accs = train_model(graphs, graph_labels, train_index, test_index)

logger.info(f"Accuracy over all folds mean: {np.mean(test_accs)*100:.3}% and std: {np.std(test_accs)*100:.2}%")

# plt.figure(figsize=(8, 6))
# plt.hist(test_accs)
# plt.xlabel("Accuracy")
# plt.ylabel("Count")
# plt.savefig('result_chart.png')
#plt.show()

# TODO: Divide patients into two datasets before creaing a word dictionary
# Do the embedding comparison with other embedding algorithms like
# word2vec, elmo, bert, fasttext and etc.

# Weighted random walk model
# 78.7% and std: 2.1%
# 79.4% and std: 2.4%

# Not weighted random walk model
# 78.5% and std: 5.2%
# 78.9% and std: 3.4%