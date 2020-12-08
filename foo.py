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
from pathlib import Path

import stellargraph as sg
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import GCNSupervisedGraphClassification
from stellargraph import StellarGraph

from sklearn import model_selection
from sklearn.metrics import roc_auc_score
from IPython.display import display, HTML

from gensim.models import Word2Vec, FastText
from stellargraph.data import BiasedRandomWalk

from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import metrics
from tensorflow.keras import models
import matplotlib.pyplot as plt


from tensorflow.keras import backend as K

from glove import Corpus, Glove

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
                    # Case 1: Use in defaul embeddings training 
                    # for i, vec in enumerate(node_emb_dict[node]):
                    # Case 2: Use when load npy file
                    for i, vec in enumerate(node_emb_dict[()][node]):
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
    for i, _ in enumerate(graphs):
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


def train_model(graphs, graph_labels, train_index, test_index, model_name, disease_name):
    # Save model path --------------->
    save_model_path = os.path.join("data/models/", disease_name)
    # Create a path to save the model
    create_model_path = os.path.join(save_model_path, model_name)
    Path(create_model_path).mkdir(parents=True, exist_ok=True)
    # Save model path ---------------<

    # Initialize generator
    generator = PaddedGraphGenerator(graphs=graphs)
    epochs = 200  # maximum number of training epochs
    
    es = EarlyStopping(
        monitor="val_loss", min_delta=0, patience=25, restore_best_weights=True
    )

    # precision_recall ------->
    def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def f1_m(y_true, y_pred):
        precision = precision_m(y_true, y_pred)
        recall = recall_m(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))

    # precision_recall -------<

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
        # model.compile(optimizer=Adam(0.005), loss=binary_crossentropy, metrics=["acc"])
        model.compile(optimizer=Adam(0.005), loss=binary_crossentropy, metrics=["acc", f1_m, precision_m, recall_m, metrics.AUC(name='auc')])

        return model

    

    # To train in folds
    def _train_fold(model, train_gen, test_gen, es, epochs, fold):
        # model.fit(
        #     train_gen, epochs=epochs, validation_data=test_gen, verbose=0, callbacks=[es],
        # )

        # # save model / If want to say each model
        save_model_name = model_name  + "_" + str(fold) + ".h5"
        fold_model_path = os.path.join(save_model_path, model_name, save_model_name)
        # model.save(fold_model_path)

        print(f"fold_model_path: {fold_model_path}")
        model = model.load_weights(fold_model_path)

        # calculate performance on the test data and return along with history
        test_metrics = model.evaluate(test_gen, verbose=0)
        loss, accuracy, f1_score, precision, recall, auc = test_metrics
        # print(test_metrics)
        # [0.3395552337169647, 0.8181818127632141, 0.8459383845329285, 0.7362573146820068, 1.0]
        test_acc = test_metrics[model.metrics_names.index("acc")]
        # print(test_acc)

        # return test_acc
        return accuracy, f1_score, precision, recall, auc

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
    test_f1_score = []
    test_precision = []
    test_recall = []
    test_auc = []

    for i in range(10):
        fold = i + 1
        print(f"Training and evaluating on fold {fold}...")
    
        train_gen, test_gen = _get_generators(
            train_index, test_index, graph_labels, batch_size=30
        )

        model = _create_graph_classification_model(generator)
        accuracy, f1_score, precision, recall, auc  = _train_fold(model, train_gen, test_gen, es, epochs, fold)

        test_accs.append(accuracy)
        test_f1_score.append(f1_score)
        test_precision.append(precision)
        test_recall.append(recall)
        test_auc.append(auc)

        # Save model accuracy of each fold
        model_accuracies_file = model_name + ".txt"
        with open(os.path.join(save_model_path, model_accuracies_file), 'w') as f:
            f.write(json.dumps(test_accs))
            
    return test_accs, test_f1_score, test_precision, test_recall, test_auc

# Initialize generator
generator = PaddedGraphGenerator(graphs=graphs)

model = _create_graph_classification_model(generator)

fold_model_path = "data/models/PNEUMONIA/seq2vec/seq2vec_1.h5"
model = model.load_weights(fold_model_path)


"""
# Count accuraccies for the models
path = "./data/models/PNEUMONIA/"

_file = os.path.join(path, "seq2vec.txt")
with open(_file, 'r') as f:
    test_accs = json.loads(f.read())
print(f"seq2vec Accuracy over all folds mean: {np.mean(test_accs)*100:.3}% and std: {np.std(test_accs)*100:.2}%")

_file = os.path.join(path, "sequence2vec_notWeighted.txt")
with open(_file, 'r') as f:
    test_accs = json.loads(f.read())
print(f"sequence2vec_notWeighted Accuracy over all folds mean: {np.mean(test_accs)*100:.3}% and std: {np.std(test_accs)*100:.2}%")

_file = os.path.join(path, "word2vec.txt")
with open(_file, 'r') as f:
    test_accs = json.loads(f.read())
print(f"word2vec Accuracy over all folds mean: {np.mean(test_accs)*100:.3}% and std: {np.std(test_accs)*100:.2}%")

_file = os.path.join(path, "fasttext.txt")
with open(_file, 'r') as f:
    test_accs = json.loads(f.read())
print(f"fasttext_emb Accuracy over all folds mean: {np.mean(test_accs)*100:.3}% and std: {np.std(test_accs)*100:.2}%")

_file = os.path.join(path, "glove.txt")
with open(_file, 'r') as f:
    test_accs = json.loads(f.read())
print(f"glove Accuracy over all folds mean: {np.mean(test_accs)*100:.3}% and std: {np.std(test_accs)*100:.2}%")
"""
