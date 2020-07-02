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
import logging

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
    
    return frequent_word_lists