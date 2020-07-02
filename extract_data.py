import pandas as pd
from functions import extract_data, count_notes_per_patient, logger, count_words_per_patient, find_frequent_word
import json
from nltk.stem import PorterStemmer
"""
# Variables

# Input vars --->
disease_name = 'PNEUMONIA'
database_path = '../MIMIC-III'

patient_id_to_num_notes = {}

number_of_patients = {}
note_appearance_counter = {}

n_fold = float(3)
threshold = float(0.02)
frequent_word_lists = {}
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

# Step 2
logger.info("Count notes per patient...")
patient_id_to_num_notes['label_0'] = count_notes_per_patient(dead_df)
patient_id_to_num_notes['label_1'] = count_notes_per_patient(alive_df)

# Step 3
logger.info("Count words per patient...")
number_of_patients['label_0'], note_appearance_counter['label_0'] = count_words_per_patient(dead_df, patient_id_to_num_notes['label_0'])
number_of_patients['label_1'], note_appearance_counter['label_1'] = count_words_per_patient(alive_df, patient_id_to_num_notes['label_1'])

# Step 4
logger.info("Find frequent words...")
frequent_word_lists = find_frequent_word(note_appearance_counter, number_of_patients, n_fold, threshold)
logger.info(len(frequent_word_lists['label_0']))
logger.info(len(frequent_word_lists['label_1']))

with open('frequent_word_lists.json', 'w') as fp:
    json.dump(frequent_word_lists, fp)
"""
# TODO: Step 2-4 can be merged into one

with open('frequent_word_lists.json') as f:
    frequent_word_lists = json.load(f)

# Later can be done inside the find_frequent_word function
FREQUENT_WORD_LIST = frequent_word_lists['label_0'] + frequent_word_lists['label_1']

"""function description:
generates frequent word set for the disease
"""
word_dict = {}
word_id = 1
stemmer = PorterStemmer()


for word in FREQUENT_WORD_LIST:
    if not word == "WORD":
        word_dict[stemmer.stem(word.strip())] = word_id
        word_id += 1
             
print("\nword dictionary created!\n")
print(word_dict)