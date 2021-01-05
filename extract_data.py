import os
import pandas as pd
import numpy as np
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import functions
from functions import extract_data, count_notes_per_patient, logger, count_words_per_patient, find_frequent_word, find_cooc_per_patient
from functions import cooc_log_odd_score, sequence2vec, other_emb
from functions import create_graphs_lists, train_model
from nltk.stem import PorterStemmer
from sklearn import model_selection

# Variables

# Input vars --->
disease_name = 'CONGESTIVE HEART FAILURE'
database_path = '../MIMIC-III'
inputs_path = os.path.join('data/inputs/', disease_name)

patient_id_to_num_notes = {}

number_of_patients = {}
note_appearance_counter = {}
## Step 4
n_fold = float(3)
threshold = float(0.01)
frequent_word_lists = {}

min_sup = 0.15
# Input vars ---<

# Step 1
# alive_df, dead_df = extract_data(disease_name, database_path)

# Save dataframe ------------------->
# alive_df.to_csv(os.path.join(inputs_path, 'alive_df.csv'), index = False, header=True)
# dead_df.to_csv(os.path.join(inputs_path, 'dead_df.csv'), index = False, header=True)
# Save dataframe -------------------<

alive_df = pd.read_csv(os.path.join(inputs_path,'alive_df.csv'))
dead_df = pd.read_csv(os.path.join(inputs_path,'dead_df.csv'))

logger.info(f"Number of patients in label_0: {dead_df['SUBJECT_ID_x'].nunique()}")
logger.info(f"Number of patients in label_1: {alive_df['SUBJECT_ID_x'].nunique()}")

# Step 2
# logger.info("Count notes per patient...")
# patient_id_to_num_notes['label_0'] = count_notes_per_patient(dead_df)
# patient_id_to_num_notes['label_1'] = count_notes_per_patient(alive_df)

# Step 3
# logger.info("Count words per patient...")
# number_of_patients['label_0'], note_appearance_counter['label_0'] = count_words_per_patient(dead_df, patient_id_to_num_notes['label_0'])
# number_of_patients['label_1'], note_appearance_counter['label_1'] = count_words_per_patient(alive_df, patient_id_to_num_notes['label_1'])

# Step 4
# logger.info("Find frequent words...")
# word_dict = find_frequent_word(note_appearance_counter, number_of_patients, n_fold, threshold)


# Write json
# with open(os.path.join(inputs_path,'word_dict.json'), 'w') as fp:
#     json.dump(word_dict, fp)

# # TODO: Step 2-4 can be merged into one

# # Read json
with open(os.path.join(inputs_path,'word_dict.json'), 'r') as fp:
    word_dict = json.load(fp)


# # Step 5
# logger.info("Count co-occurrences per patient...")
# patient_node_0, patient_cooc_0, patient_note_num_0 = find_cooc_per_patient(dead_df, word_dict, 0.15)
# patient_node_1, patient_cooc_1, patient_note_num_1 = find_cooc_per_patient(alive_df, word_dict, 0.15)

# Write json
# with open(os.path.join(inputs_path,'patient_node_0.json'), 'w') as fp:
#     json.dump(patient_node_0, fp)
# with open(os.path.join(inputs_path,'patient_node_1.json'), 'w') as fp:
#     json.dump(patient_node_1, fp)

# Because key is a tuple we store it as a string
# with open(os.path.join(inputs_path,'patient_cooc_0.txt'),'w+') as f:
#     f.write(str(patient_cooc_0))
# with open(os.path.join(inputs_path,'patient_cooc_1.txt'),'w+') as f:
#     f.write(str(patient_cooc_1))

# Read json
with open(os.path.join(inputs_path,'patient_node_0.json'), 'r') as fp:
    patient_node_0 = json.load(fp)
with open(os.path.join(inputs_path,'patient_node_1.json'), 'r') as fp:
    patient_node_1 = json.load(fp)
# Read txt with tuples
dic = ''
with open(os.path.join(inputs_path,'patient_cooc_0.txt'),'r') as f:
         for i in f.readlines():
            dic=i #string
patient_cooc_0 = eval(dic) # this is orignal dict with instace dict

dic = ''
with open(os.path.join(inputs_path,'patient_cooc_1.txt'),'r') as f:
         for i in f.readlines():
            dic=i #string
patient_cooc_1 = eval(dic) # this is orignal dict with instace dict

# Step 6
logger.info("Get and normalize weights in co-occurrences...")
normalized_cooc_odd_scores = cooc_log_odd_score(patient_cooc_0, patient_cooc_1, )

# Step 7
# logger.info("Train embeddings...")
# word2vec_emb, fasttext_emb, glove_emb = other_emb(alive_df, dead_df, patient_node_0, patient_node_1)
# sequence2vec = sequence2vec(patient_node_0, patient_node_1, normalized_cooc_odd_scores)
# sequence2vec_notWeighted = functions.sequence2vec(patient_node_0, patient_node_1, normalized_cooc_odd_scores, weighted=False)

# Save dic with val of nparrays into npy 
# np.save(os.path.join(inputs_path,'sequence2vec.npy'),sequence2vec)
# np.save(os.path.join(inputs_path,'word2vec_emb.npy'), word2vec_emb)
# np.save(os.path.join(inputs_path,'fasttext_emb.npy'), fasttext_emb)
# np.save(os.path.join(inputs_path,'glove_emb.npy'), glove_emb)
# np.save(os.path.join(inputs_path,'sequence2vec_notWeighted.npy'), sequence2vec_notWeighted)

# # # Load npy file
# sequence2vec = np.load(os.path.join(inputs_path,'sequence2vec.npy'), allow_pickle=True)
# word2vec_emb = np.load(os.path.join(inputs_path,'word2vec_emb.npy'), allow_pickle=True)
# fasttext_emb = np.load(os.path.join(inputs_path,'fasttext_emb.npy'), allow_pickle=True)
# glove_emb = np.load(os.path.join(inputs_path,'glove_emb.npy'), allow_pickle=True)
# sequence2vec_notWeighted = np.load(os.path.join(inputs_path,'sequence2vec_notWeighted.npy'), allow_pickle=True)

# print(f"word2vec_emb of cmo:\n {word2vec_emb['cmo']}")
# print(f"fasttext_emb of cmo:\n {fasttext_emb['cmo']}")
# print(f"glove_emb of cmo:\n {glove_emb['cmo']}")
# print(f"sequence2vec of cmo:\n {sequence2vec['cmo']}")
# print(f"sequence2vec_notWeighted of cmo:\n {sequence2vec_notWeighted['cmo']}")


# # Step 8
# # Create graphs, graph labels, train and test data
# logger.info("Create graphs, graph labels, train and test data...")
# graphs, graph_labels, train_index, test_index = create_graphs_lists(patient_cooc_0, patient_cooc_1, normalized_cooc_odd_scores, sequence2vec)
# # -------------------

# # Step 9
# # Train model
# logger.info("Train model...")
# test_accs, test_f1_score, test_precision, test_recall, test_auc = train_model(graphs, graph_labels, train_index, test_index, "seq2vec", disease_name)
# logger.info(f"Accuracy over all folds mean: {np.mean(test_accs)*100:.3}% and std: {np.std(test_accs)*100:.2}%")
# logger.info(f"F1_socre over all folds mean: {np.mean(test_f1_score)*100:.3}% and std: {np.std(test_f1_score)*100:.2}%")
# logger.info(f"Precision over all folds mean: {np.mean(test_precision)*100:.3}% and std: {np.std(test_precision)*100:.2}%")
# logger.info(f"Recall over all folds mean: {np.mean(test_recall)*100:.3}% and std: {np.std(test_recall)*100:.2}%")
# logger.info(f"AUC over all folds mean: {np.mean(test_auc)*100:.3}% and std: {np.std(test_auc)*100:.2}%")

"""PNEUMONIA
2020-12-10 14:28:52,504 - Accuracy over all folds mean: 79.0% and std: 2.2%
2020-12-10 14:28:52,505 - F1_socre over all folds mean: 82.5% and std: 1.5%
2020-12-10 14:28:52,505 - Precision over all folds mean: 70.8% and std: 2.4%
2020-12-10 14:28:52,505 - Recall over all folds mean: 99.4% and std: 1.8%
2020-12-10 14:28:52,505 - AUC over all folds mean: 94.7% and std: 1.7%
"""
"""SEPSIS
2021-01-04 17:47:27,307 - Accuracy over all folds mean: 82.6% and std: 2.6%
2021-01-04 17:47:27,308 - F1_socre over all folds mean: 84.0% and std: 4.2%
2021-01-04 17:47:27,308 - Precision over all folds mean: 73.7% and std: 5.1%
2021-01-04 17:47:27,308 - Recall over all folds mean: 98.4% and std: 4.5%
2021-01-04 17:47:27,308 - AUC over all folds mean: 94.1% and std: 1.2%
"""

"""Train other embeddings"""
# logger.info("Train other embeddings sequence2vec_notWeighted...")
# logger.info("Create graphs, graph labels, train and test data...")
# graphs, graph_labels, train_index, test_index = create_graphs_lists(patient_cooc_0, patient_cooc_1, normalized_cooc_odd_scores, sequence2vec_notWeighted)
# logger.info("Train model...")
# test_accs, test_f1_score, test_precision, test_recall, test_auc = train_model(graphs, graph_labels, train_index, test_index, "sequence2vec_notWeighted", disease_name)
# logger.info(f"sequence2vec_notWeighted Accuracy over all folds mean: {np.mean(test_accs)*100:.3}% and std: {np.std(test_accs)*100:.2}%")
# logger.info(f"sequence2vec_notWeighted F1_socre over all folds mean: {np.mean(test_f1_score)*100:.3}% and std: {np.std(test_f1_score)*100:.2}%")
# logger.info(f"sequence2vec_notWeighted Precision over all folds mean: {np.mean(test_precision)*100:.3}% and std: {np.std(test_precision)*100:.2}%")
# logger.info(f"sequence2vec_notWeighted Recall over all folds mean: {np.mean(test_recall)*100:.3}% and std: {np.std(test_recall)*100:.2}%")
# logger.info(f"sequence2vec_notWeighted AUC over all folds mean: {np.mean(test_auc)*100:.3}% and std: {np.std(test_auc)*100:.2}%")

"""PNEUMONIA
2020-12-23 15:58:54,575 - sequence2vec_notWeighted Accuracy over all folds mean: 78.4% and std: 4.1%
2020-12-23 15:58:54,575 - sequence2vec_notWeighted F1_socre over all folds mean: 82.0% and std: 2.6%
2020-12-23 15:58:54,575 - sequence2vec_notWeighted Precision over all folds mean: 71.3% and std: 4.4%
2020-12-23 15:58:54,576 - sequence2vec_notWeighted Recall over all folds mean: 97.7% and std: 4.0%
2020-12-23 15:58:54,576 - sequence2vec_notWeighted AUC over all folds mean: 93.0% and std: 1.5%
"""
"""SEPSIS
2021-01-04 17:58:29,885 - sequence2vec_notWeighted Accuracy over all folds mean: 80.9% and std: 4.8%
2021-01-04 17:58:29,886 - sequence2vec_notWeighted F1_socre over all folds mean: 78.5% and std: 5.6%
2021-01-04 17:58:29,886 - sequence2vec_notWeighted Precision over all folds mean: 70.6% and std: 7.4%
2021-01-04 17:58:29,886 - sequence2vec_notWeighted Recall over all folds mean: 90.9% and std: 8.9%
2021-01-04 17:58:29,887 - sequence2vec_notWeighted AUC over all folds mean: 93.3% and std: 1.2%
"""
# logger.info("Train other embeddings word2vec_emb...")
# logger.info("Create graphs, graph labels, train and test data...")
# graphs, graph_labels, train_index, test_index = create_graphs_lists(patient_cooc_0, patient_cooc_1, normalized_cooc_odd_scores, word2vec_emb)
# logger.info("Train model...")
# test_accs, test_f1_score, test_precision, test_recall, test_auc = train_model(graphs, graph_labels, train_index, test_index, "word2vec", disease_name)
# logger.info(f"word2vec_emb Accuracy over all folds mean: {np.mean(test_accs)*100:.3}% and std: {np.std(test_accs)*100:.2}%")
# logger.info(f"word2vec_emb F1_socre over all folds mean: {np.mean(test_f1_score)*100:.3}% and std: {np.std(test_f1_score)*100:.2}%")
# logger.info(f"word2vec_emb Precision over all folds mean: {np.mean(test_precision)*100:.3}% and std: {np.std(test_precision)*100:.2}%")
# logger.info(f"word2vec_emb Recall over all folds mean: {np.mean(test_recall)*100:.3}% and std: {np.std(test_recall)*100:.2}%")
# logger.info(f"word2vec_emb AUC over all folds mean: {np.mean(test_auc)*100:.3}% and std: {np.std(test_auc)*100:.2}%")

"""PNEUMONIA
2020-12-23 16:11:33,875 - word2vec_emb Accuracy over all folds mean: 57.7% and std: 8.6%
2020-12-23 16:11:33,875 - word2vec_emb F1_socre over all folds mean: 70.2% and std: 4.1%
2020-12-23 16:11:33,875 - word2vec_emb Precision over all folds mean: 55.1% and std: 6.0%
2020-12-23 16:11:33,875 - word2vec_emb Recall over all folds mean: 98.2% and std: 3.2%
2020-12-23 16:11:33,876 - word2vec_emb AUC over all folds mean: 89.0% and std: 2.7%
"""
"""SEPSIS
2021-01-04 18:10:23,649 - word2vec_emb Accuracy over all folds mean: 77.2% and std: 4.7%
2021-01-04 18:10:23,649 - word2vec_emb F1_socre over all folds mean: 74.1% and std: 6.3%
2021-01-04 18:10:23,649 - word2vec_emb Precision over all folds mean: 65.6% and std: 6.5%
2021-01-04 18:10:23,650 - word2vec_emb Recall over all folds mean: 87.7% and std: 1e+01%
2021-01-04 18:10:23,650 - word2vec_emb AUC over all folds mean: 90.3% and std: 2.2%
"""
# logger.info("Train other embeddings fasttext_emb...")
# logger.info("Create graphs, graph labels, train and test data...")
# graphs, graph_labels, train_index, test_index = create_graphs_lists(patient_cooc_0, patient_cooc_1, normalized_cooc_odd_scores, fasttext_emb)
# logger.info("Train model...")
# test_accs, test_f1_score, test_precision, test_recall, test_auc = train_model(graphs, graph_labels, train_index, test_index, "fasttext", disease_name)
# logger.info(f"fasttext_emb Accuracy over all folds mean: {np.mean(test_accs)*100:.3}% and std: {np.std(test_accs)*100:.2}%")
# logger.info(f"fasttext_emb F1_socre over all folds mean: {np.mean(test_f1_score)*100:.3}% and std: {np.std(test_f1_score)*100:.2}%")
# logger.info(f"fasttext_emb Precision over all folds mean: {np.mean(test_precision)*100:.3}% and std: {np.std(test_precision)*100:.2}%")
# logger.info(f"fasttext_emb Recall over all folds mean: {np.mean(test_recall)*100:.3}% and std: {np.std(test_recall)*100:.2}%")
# logger.info(f"fasttext_emb AUC over all folds mean: {np.mean(test_auc)*100:.3}% and std: {np.std(test_auc)*100:.2}%")

"""PNEUMONIA
2020-12-23 16:24:29,499 - fasttext_emb Accuracy over all folds mean: 58.3% and std: 8.9%
2020-12-23 16:24:29,499 - fasttext_emb F1_socre over all folds mean: 70.5% and std: 4.4%
2020-12-23 16:24:29,499 - fasttext_emb Precision over all folds mean: 55.5% and std: 6.1%
2020-12-23 16:24:29,500 - fasttext_emb Recall over all folds mean: 98.3% and std: 2.6%
2020-12-23 16:24:29,500 - fasttext_emb AUC over all folds mean: 89.0% and std: 2.7%
"""
"""SEPSIS
2021-01-04 18:21:48,876 - fasttext_emb Accuracy over all folds mean: 70.2% and std: 7.6%
2021-01-04 18:21:48,876 - fasttext_emb F1_socre over all folds mean: 69.2% and std: 5.5%
2021-01-04 18:21:48,877 - fasttext_emb Precision over all folds mean: 58.2% and std: 7.1%
2021-01-04 18:21:48,877 - fasttext_emb Recall over all folds mean: 89.1% and std: 9.4%
2021-01-04 18:21:48,877 - fasttext_emb AUC over all folds mean: 90.3% and std: 1.9%
"""
# logger.info("Train other embeddings glove_emb...")
# logger.info("Create graphs, graph labels, train and test data...")
# graphs, graph_labels, train_index, test_index = create_graphs_lists(patient_cooc_0, patient_cooc_1, normalized_cooc_odd_scores, glove_emb)
# logger.info("Train model...")
# test_accs, test_f1_score, test_precision, test_recall, test_auc = train_model(graphs, graph_labels, train_index, test_index, "glove", disease_name)
# logger.info(f"glove_emb Accuracy over all folds mean: {np.mean(test_accs)*100:.3}% and std: {np.std(test_accs)*100:.2}%")
# logger.info(f"glove_emb F1_socre over all folds mean: {np.mean(test_f1_score)*100:.3}% and std: {np.std(test_f1_score)*100:.2}%")
# logger.info(f"glove_emb Precision over all folds mean: {np.mean(test_precision)*100:.3}% and std: {np.std(test_precision)*100:.2}%")
# logger.info(f"glove_emb Recall over all folds mean: {np.mean(test_recall)*100:.3}% and std: {np.std(test_recall)*100:.2}%")
# logger.info(f"glove_emb AUC over all folds mean: {np.mean(test_auc)*100:.3}% and std: {np.std(test_auc)*100:.2}%")

"""PNEUMONIA
2020-12-23 16:39:19,289 - glove_emb Accuracy over all folds mean: 70.2% and std: 8.0%
2020-12-23 16:39:19,289 - glove_emb F1_socre over all folds mean: 75.8% and std: 4.1%
2020-12-23 16:39:19,289 - glove_emb Precision over all folds mean: 64.5% and std: 6.1%
2020-12-23 16:39:19,289 - glove_emb Recall over all folds mean: 93.1% and std: 3.6%
2020-12-23 16:39:19,289 - glove_emb AUC over all folds mean: 86.7% and std: 1.4%
"""
"""SEPSIS
2021-01-04 18:32:57,542 - glove_emb Accuracy over all folds mean: 76.9% and std: 6.1%
2021-01-04 18:32:57,549 - glove_emb F1_socre over all folds mean: 69.2% and std: 4.8%
2021-01-04 18:32:57,550 - glove_emb Precision over all folds mean: 61.9% and std: 6.0%
2021-01-04 18:32:57,550 - glove_emb Recall over all folds mean: 80.7% and std: 7.6%
2021-01-04 18:32:57,550 - glove_emb AUC over all folds mean: 84.7% and std: 3.1%
"""
# # plt.figure(figsize=(8, 6))
# # plt.hist(test_accs)
# # plt.xlabel("Accuracy")
# # plt.ylabel("Count")
# # plt.savefig('result_chart.png')
# #plt.show()

# # # TODO: Divide patients into two datasets before creaing a word dictionary
# # # Do the embedding comparison with other embedding algorithms like
# # # word2vec, elmo, bert, fasttext and etc.




# """Seq2Vec"""
# # # Weighted random walk model
# # # 78.7% and std: 2.1%
# # # 79.4% and std: 2.4%

# # # Not weighted random walk model
# # # 78.5% and std: 5.2%
# # # 78.9% and std: 3.4%

# """Word2Vec"""
# # 59.3% and std: 1e+01%
# # 59.2% and std: 1e+01%

# """FastText"""
# # 56.8% and std: 8.3%
# # 56.6% and std: 7.9%

# """Glove"""
# # 71.1% and std: 7.2%
# # 70.2% and std: 8.5%

# # Different node embeddings
# # https://github.com/shenweichen/GraphEmbedding