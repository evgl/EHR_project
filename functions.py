import pandas as pd
import numpy as np
import json
import os
import sys
import re
import logging
import math
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
from tqdm import tqdm
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder

import stellargraph as sg
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import GCNSupervisedGraphClassification
from stellargraph import StellarGraph

from stellargraph import datasets

from sklearn import model_selection
from IPython.display import display, HTML

from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt


logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(asctime)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

# define constants
MIN_SEQ_LEN = 4
USE_1_N_SEQ = 2

# words that do not have meaning (can be modified later)
STOP_WORDS = ["a", "the", "he", "she", ",", ".", "?", "!", ":", ";", "+", "*", "**"\
                 "your", "you"]

# count up the frequency of every word in every disease file
stemmer = PorterStemmer()
# create set of words to ignore in text
stop_words = set(stopwords.words('english'))

for word in STOP_WORDS:
    stop_words.add(word)

def extract_data(disease_name, database_path):

    with open('variables.json') as f:
        variable = json.loads(f.read())

    logger.info("Loading data. Please wait...")
    admissions_df = pd.read_csv(os.path.join(database_path, "ADMISSIONS.csv"))
    noteevents_df = pd.read_csv(os.path.join(database_path, "NOTEEVENTS.csv")) 

    logger.info("Preprocessing...")
    # Left join of two dataframes
    note_admiss_df_left = noteevents_df.merge(admissions_df, on='HADM_ID', how='left', indicator=True)

    # Extract notes only for certain disease 
    disease_df = note_admiss_df_left.loc[note_admiss_df_left["DIAGNOSIS"] == disease_name, variable['disease_df']]
    # Remove notes with discharge summary
    no_disch_df = disease_df.loc[disease_df["CATEGORY"] != 'Discharge summary', variable['disease_df']]

    # Divide notes into two lables 
    alive_no_disch = no_disch_df[no_disch_df.DEATHTIME.isnull()]
    dead_no_disch = no_disch_df[no_disch_df.DEATHTIME.notnull()]

    # Sort notes 
    alive_no_disch = alive_no_disch.sort_values(by=variable['order_df'])
    dead_no_disch = dead_no_disch.sort_values(by=variable['order_df'])

    # Remove not needed fields
    alive_no_disch = alive_no_disch[variable['d_df']]
    dead_no_disch = dead_no_disch[variable['d_df']]

    return alive_no_disch, dead_no_disch


def count_notes_per_patient(disease_df):
    patient_id_to_num_notes = {}
    patient_id = -1
    note_counter = 0
            
    for index, row in tqdm(disease_df.iterrows(), total=disease_df.shape[0]):
        patient_id_check = int(row['SUBJECT_ID_x'])
                
        if not patient_id == patient_id_check:
            patient_id_to_num_notes[patient_id] = note_counter
            note_counter = 1
        else:
            note_counter += 1
                    
        patient_id = patient_id_check
                
    patient_id_to_num_notes[patient_id] = note_counter
    del patient_id_to_num_notes[-1]
    return patient_id_to_num_notes

def count_words_per_patient(disease_df, patient_id_to_num_notes):
    note_appearance_counter = {}
    number_of_patients = 0 # number of patients
    note_counter = 0
# -----------
    patient_id = -1
    word_set = set()
    note_event_counter = 0

    # Iterate through each note
    for index, row in tqdm(disease_df.iterrows(), total=disease_df.shape[0]):     
        patient_id_check = int(row['SUBJECT_ID_x'])
    
        # if patient id has changed, end sequence and start new sequence
        if not patient_id == patient_id_check:
            number_of_patients += 1
            note_event_counter = 0
        
            for word in word_set:
                if word in note_appearance_counter:
                    note_appearance_counter[word] += 1
                else:
                    note_appearance_counter[word] = 1
            # reset word_set
            word_set = set()
        
        # update patient id
        patient_id = patient_id_check

            
        if patient_id_to_num_notes[patient_id_check] <= MIN_SEQ_LEN:
            continue
            
        if note_event_counter < patient_id_to_num_notes[patient_id] // USE_1_N_SEQ:
            note_event_counter += 1
            continue
                
        note_counter += 1
        note = re.sub(r'\[\*\*(.*?)\*\*\]|[_,\d\*:~=\.\-\+\\/]+', ' ', row['TEXT'])
        tokenized_note = word_tokenize(note)
        
        
        for word in tokenized_note:
            stemmed_word = stemmer.stem(word.lower())
            if not stemmed_word in stop_words:
                word_set.add(stemmed_word)

    return number_of_patients, note_appearance_counter

def find_frequent_word(note_appearance_counter, number_of_patients, n_fold, threshold):

    frequent_word_lists = {}
    factor = {}
    # calculate normalizing factor for each disease
    note_sum = 0

    # Count from two labels
    for disease in number_of_patients:
        note_sum += float(number_of_patients[disease])
    
    for disease in number_of_patients:
        factor[disease] = number_of_patients[disease] / note_sum

    # determine frequent word for each disease file
    for disease in note_appearance_counter:
        frequent_word_lists[disease] = []

        logger.info(disease + " has " + str(len(note_appearance_counter[disease])) + " unique words!")

        for word in note_appearance_counter[disease]:
        
            freq_check = True
            for check_disease in note_appearance_counter:
            
                if not disease == check_disease:
                    if word in note_appearance_counter[check_disease]:
                        if not (note_appearance_counter[disease][word] / note_appearance_counter[check_disease][word] / factor[disease] * factor[check_disease] > n_fold \
                            and note_appearance_counter[disease][word] > (number_of_patients[disease] * threshold)):

                            freq_check = False
                            break

                    else:
                        if not (note_appearance_counter[disease][word] > n_fold and note_appearance_counter[disease][word] > (number_of_patients[disease] * threshold)):
                            freq_check = False
                            break
            if freq_check:
                frequent_word_lists[disease].append((word))

    logger.info(f"Frequent word label_0: {len(frequent_word_lists['label_0'])}")
    logger.info(f"Frequent word label_1: {len(frequent_word_lists['label_1'])}")

    FREQUENT_WORD_LIST = frequent_word_lists['label_0'] + frequent_word_lists['label_1']

    word_dict = {}
    word_id = 1
    stemmer = PorterStemmer()


    for word in FREQUENT_WORD_LIST:
        if not word == "WORD":
            word_dict[stemmer.stem(word.strip())] = word_id
            word_id += 1
    
    return word_dict

def find_cooc_per_patient(disease_df, word_dict, min_support, label):
    
    # For dataframe
    patient_id_lst = []
    patient_cooc_lst = []
    patient_cooc_minsup_lst = []
    patient_label_lst = []
    patient_note_cnt = []
    patient_feature_lst = []
    patient_feature_idx_lst = []
    patinet_log_odd_ratio_lst = []
    
    # --------------    
    patient_id = -1
    note_cnt = 0
    patient_note_list = []
    
    # read line in from file (each line is one note)
    for index, row in tqdm(disease_df.iterrows(), total=disease_df.shape[0]):
        
        # only regard certain type of notes
        patient_id_check = int(row['SUBJECT_ID_x'])
        note = re.sub(r'\[\*\*(.*?)\*\*\]|[_,\d\*:~=\.\-\+\\/]+', ' ', row['TEXT'])
        patient_word_set = set()
    
#         print(f"patient_id_check: {patient_id_check}, patient_id: {patient_id}")
        # if patient id has changed, end sequence and start new sequence
        if not patient_id == patient_id_check and not patient_id == -1:
            te = TransactionEncoder()
            te_ary = te.fit(patient_note_list).transform(patient_note_list)
            df = pd.DataFrame(te_ary, columns=te.columns_)
            df_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)
            
            cooc_tmp = []
            cooc_minsup_tmp = []
            cooc_feature_tmp = []
            cooc_feature_idx_tmp = []
            cooc_log_odds_ratio_tmp = []
            
            for index, row in df_itemsets.iterrows():
                if len(row['itemsets']) == 1:
                    word = list(row['itemsets'])[0]
                    min_sup = row['support']
                    cooc_feature_tmp.append(min_sup)
                    cooc_feature_idx_tmp.append(word)
                    cooc_log_odds_ratio_tmp.append(math.log(min_sup/((1-min_sup) + 0.0001)))
                    
                if len(row['itemsets']) == 2:
                    cooc_ = list(row['itemsets'])
                    cooc_tmp.append(cooc_)
                    cooc_minsup_tmp.append(row['support'])      
          
            # Update glob lists
            patient_id_lst.append(patient_id)
            patient_cooc_lst.append(cooc_tmp)
            patient_cooc_minsup_lst.append(cooc_minsup_tmp)
            patient_label_lst.append(label)
            patient_note_cnt.append(note_cnt)
            
            patient_feature_lst.append(cooc_feature_tmp)
            patient_feature_idx_lst.append(cooc_feature_idx_tmp)
            patinet_log_odd_ratio_lst.append(cooc_log_odds_ratio_tmp)
            
            # Reset local lists
            patient_note_list = []
            note_cnt = 0
                    
        # update patient id
        patient_id = patient_id_check
        tokenized_note = word_tokenize(note)
        note_cnt += 1

        # loop through each word in note to count word belonging to each disease
        for word in tokenized_note:
            stemmed_word = stemmer.stem(word.lower())       
            if stemmed_word in word_dict:
                    patient_word_set.add(stemmed_word)

        templst = []
        for word in patient_word_set:
            templst.append(word)

        if templst:
            patient_note_list.append(templst)
    
    # Last patient info
    te = TransactionEncoder()
    te_ary = te.fit(patient_note_list).transform(patient_note_list)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    df_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)
            
    cooc_tmp = []
    cooc_minsup_tmp = []
    #-----
    cooc_feature_tmp = []
    cooc_feature_idx_tmp = []
    cooc_log_odds_ratio_tmp = []
            
    for index, row in df_itemsets.iterrows():
        if len(row['itemsets']) == 1:
            word = list(row['itemsets'])[0]
            min_sup = row['support']
            cooc_feature_tmp.append(min_sup)
            cooc_feature_idx_tmp.append(word)
            cooc_log_odds_ratio_tmp.append(math.log(min_sup/((1-min_sup) + 0.0001)))

        if len(row['itemsets']) == 2:
            cooc_ = list(row['itemsets'])
            cooc_tmp.append(cooc_)
            cooc_minsup_tmp.append(row['support'])
                    
    # Update glob lists
    patient_id_lst.append(patient_id)
    patient_cooc_lst.append(cooc_tmp)
    patient_cooc_minsup_lst.append(cooc_minsup_tmp)
    patient_label_lst.append(label)
    patient_note_cnt.append(note_cnt)
    
    patient_feature_lst.append(cooc_feature_tmp)
    patient_feature_idx_lst.append(cooc_feature_idx_tmp)
    patinet_log_odd_ratio_lst.append(cooc_log_odds_ratio_tmp)
    
    return pd.DataFrame({"patient_id":patient_id_lst, "patient_cooc":patient_cooc_lst, "cooc_minsup":patient_cooc_minsup_lst, "cooc_feature":patient_feature_lst, "log_odd_ratio":patinet_log_odd_ratio_lst, "cooc_feature_idx":patient_feature_idx_lst ,"label":patient_label_lst, "note_cnt":patient_note_cnt})

# Train model
# Load data
def create_graph_list(pd_df):
    graphs = []
    labels = []

    for index, row in pd_df.iterrows():
        if row['patient_cooc']:
            source = []
            target = []
            edge_feature = []
            
            feature_node = []
            feature_idx = []
            
            for cooc in row['patient_cooc']:
                source.append(cooc[0])
                target.append(cooc[1])
                
            for feature in row['cooc_minsup']:
                edge_feature.append(feature)
            
            # Node feature and index
            for node_feature in row['log_odd_ratio']:
                feature_node.append(node_feature)
                
            for node_idx in row['cooc_feature_idx']:
                feature_idx.append(node_idx)
                
            square_node_data = pd.DataFrame({"feature_src":feature_node}, index=feature_idx)
            square_edges = pd.DataFrame({"source": source, "target": target})
          
            square = StellarGraph(square_node_data, square_edges)
            graphs.append(square)
            labels.append(row['label'])   
            
    return graphs, labels

def create_graph_classification_model(generator):
    gc_model = GCNSupervisedGraphClassification(
        layer_sizes=[64, 64],
        activations=["relu", "relu"],
        generator=generator,
        dropout=0.5,
    )
    x_inp, x_out = gc_model.in_out_tensors()
    predictions = Dense(units=32, activation="relu")(x_out)
    predictions = Dense(units=16, activation="relu")(predictions)
    predictions = Dense(units=1, activation="sigmoid")(predictions)

    # Let's create the Keras model and prepare it for training
    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(optimizer=Adam(0.005), loss=binary_crossentropy, metrics=["acc"])

    return model

# Train the model
def train_fold(model, train_gen, test_gen, epochs):

    es = EarlyStopping(
    monitor="val_loss", min_delta=0, patience=25, restore_best_weights=True)

    history = model.fit(
        train_gen, epochs=epochs, validation_data=test_gen, verbose=0, callbacks=[es],
    )
    # calculate performance on the test data and return along with history
    test_metrics = model.evaluate(test_gen, verbose=0)
    test_acc = test_metrics[model.metrics_names.index("acc")]

    return history, test_acc

def get_generators(generator, train_index, test_index, graph_labels, batch_size):
    train_gen = generator.flow(
        train_index, targets=graph_labels.iloc[train_index].values, batch_size=batch_size
    )
    test_gen = generator.flow(
        test_index, targets=graph_labels.iloc[test_index].values, batch_size=batch_size
    )

    return train_gen, test_gen