import pandas as pd
import json
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
from tqdm import tqdm
import sys
import re

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

    print("Loading data. Please wait...")
    admissions_df = pd.read_csv(os.path.join(database_path, "ADMISSIONS.csv"))
    noteevents_df = pd.read_csv(os.path.join(database_path, "NOTEEVENTS.csv")) 

    print("Preprocessing...")
    note_admiss_df_left = noteevents_df.merge(admissions_df, on='HADM_ID', how='left', indicator=True)

    disease_df = note_admiss_df_left.loc[note_admiss_df_left["DIAGNOSIS"] == disease_name, variable['disease_df']]
    no_disch_df = disease_df.loc[disease_df["CATEGORY"] != 'Discharge summary', variable['disease_df']]

    alive_no_disch = no_disch_df[no_disch_df.DEATHTIME.isnull()]
    dead_no_disch = no_disch_df[no_disch_df.DEATHTIME.notnull()]

    alive_no_disch = alive_no_disch.sort_values(by=variable['order_df'])
    dead_no_disch = dead_no_disch.sort_values(by=variable['order_df'])

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