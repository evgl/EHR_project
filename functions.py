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
from collections import defaultdict
from tqdm import tqdm
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder

import stellargraph as sg
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import GCNSupervisedGraphClassification
from stellargraph import StellarGraph

from sklearn import model_selection
from IPython.display import display, HTML

from gensim.models import Word2Vec
from stellargraph.data import BiasedRandomWalk

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

def find_cooc_per_patient(disease_df, word_dict, min_support):
    """ Function counts co-occurences per each patient

    Arguments:
        disease_df: Desease dataframe without discharge notes
        word_dict: Dictionary of all words
        min_support: minimum support

    Returns:
    Type dictionary
        patient_node : nodes and support for each patient
        patient_cooc : co-occurrences and support for each patient
        patient_note_num : number of notes for each patient
    """
    # For dataframe
    patient_cooc_dict = {}
    patient_node_dict = {}
    patient_note_cnt = {}
    
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
    
        # if patient id has changed, end sequence and start new sequence
        if not patient_id == patient_id_check and not patient_id == -1:
            te = TransactionEncoder()
            te_ary = te.fit(patient_note_list).transform(patient_note_list)
            df = pd.DataFrame(te_ary, columns=te.columns_)
            df_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)
            
            cooc_tmp = []
            cooc_minsup_tmp = []
            node_minsup_tmp = []
            cooc_node_idx_tmp = []
            
            for index, row in df_itemsets.iterrows():
                if len(row['itemsets']) == 1:
                    word = list(row['itemsets'])[0]
                    min_sup = row['support']

                    node_minsup_tmp.append(len(patient_note_list) * row['support'])
                    cooc_node_idx_tmp.append(word)
                    
                if len(row['itemsets']) == 2:
                    cooc_ = sorted(list(row['itemsets']))
                    cooc_tmp.append(cooc_)
                    cooc_minsup_tmp.append(len(patient_note_list) * row['support'])
            
            cooc_dict = {}

            for num, i in enumerate(cooc_tmp):
                if tuple(i) not in cooc_dict:
                    cooc_dict[tuple(i)] = cooc_minsup_tmp[num]

            # dictionary = dict(zip(keys, values))
            node_dict = {}
            for num, i in enumerate(cooc_node_idx_tmp):
                if i not in node_dict:
                    node_dict[i] = node_minsup_tmp[num]
                    
            # Update glob lists
            if patient_id not in patient_cooc_dict:
                patient_cooc_dict[patient_id] = cooc_dict
                patient_node_dict[patient_id] = node_dict
                patient_note_cnt[patient_id] = note_cnt
            else:
                print(f"patient_id: {patient_id} is already in the dictionary!")
            
            
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
    node_minsup_tmp = []
    cooc_node_idx_tmp = []
            
    for index, row in df_itemsets.iterrows():
        if len(row['itemsets']) == 1:
            word = list(row['itemsets'])[0]
            node_minsup_tmp.append(len(patient_note_list) * row['support'])
            cooc_node_idx_tmp.append(word)

        if len(row['itemsets']) == 2:
            cooc_ = sorted(list(row['itemsets']))
            cooc_tmp.append(cooc_)
            cooc_minsup_tmp.append(len(patient_note_list) * row['support'])
    
                    
    cooc_dict = {}

    for num, i in enumerate(cooc_tmp):
        if tuple(i) not in cooc_dict:
            cooc_dict[tuple(i)] = cooc_minsup_tmp[num]

    # dictionary = dict(zip(keys, values))
    node_dict = {}
    for num, i in enumerate(cooc_node_idx_tmp):
        if i not in node_dict:
            node_dict[i] = node_minsup_tmp[num]

    # Update glob lists
    if patient_id not in patient_cooc_dict:
        patient_cooc_dict[patient_id] = cooc_dict
        patient_node_dict[patient_id] = node_dict
        patient_note_cnt[patient_id] = note_cnt
    else:
        print(f"patient_id: {patient_id} is already in the dictionary!")

    return patient_node_dict, patient_cooc_dict, patient_note_cnt


def cooc_log_odd_score(patient_cooc_0, patient_cooc_1):
    """ Function to count co-occurence log-odds-scores 
    and normalize them
    """
    # Create a set of all unique co-occurrences from both group
    patient_cooc_set = set()
    patient_cooc_0_dict = {}
    patient_cooc_1_dict = {}

    for k, v in patient_cooc_0.items():
        for item in v:
            patient_cooc_set.add(item)
            if item not in patient_cooc_0_dict:
                patient_cooc_0_dict[item] = v[item]
            else:
                patient_cooc_0_dict[item] = patient_cooc_0_dict[item] + v[item]
            
    
    for k, v in patient_cooc_1.items():
        for item in v:
            patient_cooc_set.add(item)
            if item not in patient_cooc_1_dict:
                patient_cooc_1_dict[item] = v[item]
            else:
                patient_cooc_1_dict[item] = patient_cooc_1_dict[item] + v[item]

    # Get the set of coocurrencies from two groups
    # Count log_odd_score

    patient_cooc_odd_scores = {}

    for set_item in patient_cooc_set:
        if set_item in patient_cooc_0_dict and set_item in patient_cooc_1_dict: 
            d_prob = patient_cooc_0_dict[set_item]/(patient_cooc_0_dict[set_item] + patient_cooc_1_dict[set_item])
            a_prob = patient_cooc_1_dict[set_item]/(patient_cooc_0_dict[set_item] + patient_cooc_1_dict[set_item])
            log_odd_score = math.log((a_prob + 0.001)/(d_prob+0.001))
            patient_cooc_odd_scores[set_item] = log_odd_score
        elif set_item in patient_cooc_0_dict:
            log_odd_score = math.log((0.001)/(1.001))
            patient_cooc_odd_scores[set_item] = log_odd_score
        elif set_item in patient_cooc_1_dict:
            log_odd_score = math.log((1.001)/(0.001))
            patient_cooc_odd_scores[set_item] = log_odd_score

    # Normalization
    def norm_arr(array):
        arr = np.array(list(array))
        start = 0
        end = 1
        width = end - start
        res = (arr - arr.min())/(arr.max() - arr.min()) * width + start
        return res.tolist()

    cooc_keys, cooc_values = zip(*patient_cooc_odd_scores.items())
    normalized_cooc_odd_scores = dict(zip(cooc_keys, norm_arr(cooc_values)))

    return normalized_cooc_odd_scores


def train_embeddings(patient_node_0, patient_node_1, new_patient_cooc_odd_scores):

    # Get unique patient nodes
    patient_node_set = set()
    
    for _, v in patient_node_0.items():
        for item in v:
            patient_node_set.add(item)
    
    for _, v in patient_node_1.items():
        for item in v:
            patient_node_set.add(item)

    patient_square_node_data = pd.DataFrame({'node':list(patient_node_set)})
    patient_square_node_id_data = patient_square_node_data.set_index("node")

    # Create bi-directional df with edge weight
    bidirect_source = []
    bidirect_target = []
    edge_weight = []

    for item, val in new_patient_cooc_odd_scores.items():
        bidirect_source.extend([item[0], item[1]])
        bidirect_target.extend([item[1], item[0]])
        edge_weight.extend([val, val])
    
    logger.info(f"source: {len(bidirect_source)}, target: {len(bidirect_target)}")

    weighted_patient_bidirect_square_edge_data = pd.DataFrame(
        {
            "source": bidirect_source,
            "target": bidirect_target,
            "weight": edge_weight,
        })
    
    # Create a graph with bi-directional edges
    G_weighted_patient_bidirect = StellarGraph(
        {"corner": patient_square_node_id_data}, 
        {"line": weighted_patient_bidirect_square_edge_data}
    )
    
    # Perform random walk
    rw_weighted_patient_bidirect = BiasedRandomWalk(G_weighted_patient_bidirect)

    # Weighted
    weighted_walks_patient_bidirect = rw_weighted_patient_bidirect.run(
        nodes=G_weighted_patient_bidirect.nodes(),  # root nodes
        length=100,  # maximum length of a random walk
        n=10,  # number of random walks per root node
        p=0.5,  # Defines (unormalised) probability, 1/p, of returning to source node
        q=2.0,  # Defines (unormalised) probability, 1/q, for moving away from source node
        weighted=True,  # for weighted random walks
        seed=42,  # random seed fixed for reproducibility
    )
    logger.info("Number of random walks: {}".format(len(weighted_walks_patient_bidirect)))

    # Embeddings
    weighted_model_patient = Word2Vec(
        weighted_walks_patient_bidirect, size=128, window=5, min_count=0, sg=1, workers=4, iter=1
    )

    patient_weighted_node_emb_dict = {}
    for index, _ in patient_square_node_id_data.iterrows():
        if index not in patient_weighted_node_emb_dict:
            patient_weighted_node_emb_dict[index] = weighted_model_patient.wv[index]
        else:
            print(f"index: {index} is already in a dictionary!")

    return patient_weighted_node_emb_dict

# Train model
# Load data
def create_graphs_lists(patient_cooc_0, patient_cooc_1, normalized_cooc_odd_scores, sequence2vec):

    def _create_graph_list(patient_cooc_dict, cooc_odd_scores, node_emb_dict, label):
    
        graphs = []
        labels = []
    
        for _,row in patient_cooc_dict.items():
     
            if row:
                source = []
                target = []
                edge_weight = []
                node_idx = []
                for cooc in row:
                    source.extend([cooc[0], cooc[1]])
                    target.extend([cooc[1], cooc[0]])
                    edge_weight.extend([cooc_odd_scores[cooc], cooc_odd_scores[cooc]])
        
                node_idx = list(set(source + target))
    
                # Create a dataframe of only nodes
                square_node_data = pd.DataFrame(
                    index=node_idx)
        
                # Create a dictionary for each column for a vector
                node_features = defaultdict(list)
                for node in node_idx:
                    for i, vec in enumerate(node_emb_dict[node]):
                        node_features['w_' + str(i)].append(vec)
        
                # Add columns to a dataframe
                for k, v in node_features.items():
              
                    square_node_data[k] = v

                square_edges = pd.DataFrame({ 
                    "source": source, 
                    "target": target, 
                    "weight":edge_weight
                })
        
                square = StellarGraph({"corner": square_node_data}, {"line": square_edges})
                graphs.append(square)
                labels.append(label)
            
        return graphs, labels

    # patient_weighted_node_emb_dict
    graph_0, label_0 = _create_graph_list(patient_cooc_0, normalized_cooc_odd_scores, sequence2vec, -1)
    graph_1, label_1 = _create_graph_list(patient_cooc_1, normalized_cooc_odd_scores, sequence2vec, 1)

    graphs = graph_0 + graph_1
    labels = label_0 + label_1

    # TODO: Later this part should be changed
    # prepare test and train datasets
    test_cnt = int(len(graphs)*0.1)/2
    print(test_cnt)

    pos_start = len(graph_0)
    print(pos_start)

    test_arr = []
    train_arr = []
    for i, v in enumerate(graphs):
        # Take first items for neg set
        if i < test_cnt:
            test_arr.append(i)    
        elif i > pos_start and i <= pos_start + test_cnt:
            test_arr.append(i)    
        else:
            train_arr.append(i)

    # Shuffle test_arr elements randomly
    import random
    seed = 42
    # Put elements of test arr into c and shuffle
    c = list(test_arr)
    random.Random(seed).shuffle(c)
    test_arr =  c

    # Put elements of train arr in c and shuffle
    c = list(train_arr)
    random.Random(seed).shuffle(c)
    train_arr =  c

    # Shuffled train and test arrays
    train_index = np.array(train_arr)
    test_index = np.array(test_arr)

    # Convert labels from list to pd.Series
    graph_labels = pd.Series(labels)
    # Convert labels to pandas dataframe
    graph_labels = pd.get_dummies(graph_labels, drop_first=True)

    return graphs, graph_labels, train_index, test_index

def train_model(graphs, graph_labels, train_index, test_index):

    # Initialize generator
    generator = PaddedGraphGenerator(graphs=graphs)
    epochs = 200  # maximum number of training epochs
    
    es = EarlyStopping(
        monitor="val_loss", min_delta=0, patience=25, restore_best_weights=True
    )

    def _create_graph_classification_model(generator):
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

    

    # To train in folds
    def _train_fold(model, train_gen, test_gen, es, epochs):
        model.fit(
            train_gen, epochs=epochs, validation_data=test_gen, verbose=0, callbacks=[es],
        )
        # calculate performance on the test data and return along with history
        test_metrics = model.evaluate(test_gen, verbose=0)
        test_acc = test_metrics[model.metrics_names.index("acc")]

        return test_acc

    # To train in folds
    def _get_generators(train_index, test_index, graph_labels, batch_size):
        train_gen = generator.flow(
            train_index, targets=graph_labels.iloc[train_index].values, weighted=True, batch_size=batch_size, shuffle=False, seed=42
        )
        test_gen = generator.flow(
            test_index, targets=graph_labels.iloc[test_index].values, weighted=True, batch_size=batch_size, shuffle=False, seed=42
        )

        return train_gen, test_gen


    # logger.info(f"Num GPUs Available: {len(tf.config.experimental.list_physical_devices('GPU'))}")
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     # Restrict TensorFlow to only use the first GPU
    #     try:
    #         tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
    #         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #         logger.info(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    #     except RuntimeError as e:
    #         # Visible devices must be set before GPUs have been initialized
    #         logger.info(e)


    # To train 50 folds
    test_accs = []

    for i in range(50):
        print(f"Training and evaluating on fold {i+1}...")
    
        train_gen, test_gen = _get_generators(
            train_index, test_index, graph_labels, batch_size=30
        )

        model = _create_graph_classification_model(generator)
        acc  = _train_fold(model, train_gen, test_gen, es, epochs)
        test_accs.append(acc)

    return test_accs
