import pandas as pd
import numpy as np
from functions import extract_data, count_notes_per_patient, logger, count_words_per_patient, find_frequent_word, find_cooc_per_patient
from functions import create_graph_list, PaddedGraphGenerator, train_fold, create_graph_classification_model, get_generators
import json
from nltk.stem import PorterStemmer
from sklearn import model_selection

# Variables

# Input vars --->
disease_name = 'PNEUMONIA'
database_path = '../MIMIC-III'

patient_id_to_num_notes = {}

number_of_patients = {}
note_appearance_counter = {}

n_fold = float(3)
threshold = float(0.01)
frequent_word_lists = {}

min_sup = 0.7
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
word_dict = find_frequent_word(note_appearance_counter, number_of_patients, n_fold, threshold)


# Write json
# with open('frequent_word_lists.json', 'w') as fp:
#     json.dump(frequent_word_lists, fp)

# TODO: Step 2-4 can be merged into one

# Read json
# with open('frequent_word_lists.json') as f:
#     frequent_word_lists = json.load(f)

# Step 5
lable_0_cooc_df = find_cooc_per_patient(dead_df, word_dict, min_sup, -1)
lable_1_cooc_df = find_cooc_per_patient(alive_df, word_dict, min_sup, 1)

# Step 6
# Train model
graphs = []
labels = []
features = []

graph_0, label_0 = create_graph_list(lable_0_cooc_df)
graph_1, label_1 = create_graph_list(lable_1_cooc_df)

logger.info(f"graphs_0: {len(graph_0)}, labels_0: {len(label_0)}")
logger.info(f"graph_1: {len(graph_1)}, labels_1: {len(label_1)}")

graphs.extend(graph_0)
labels.extend(label_0)

graphs.extend(graph_1)
labels.extend(label_1)

logger.info(f"graphs: {len(graphs)}, labels: {len(labels)}")
logger.info(f"graph info:\n{graphs[0].info()}")


summary = pd.DataFrame(
    [(g.number_of_nodes(), g.number_of_edges()) for g in graphs],
    columns=["nodes", "edges"],
)
logger.info(f"Summary:\n{summary.describe().round(1)}")

graph_labels = pd.Series(labels)
graph_labels = pd.get_dummies(graph_labels, drop_first=True)
generator = PaddedGraphGenerator(graphs=graphs)

# Train the model
epochs = 200  # maximum number of training epochs
folds = 10  # the number of folds for k-fold cross validation
n_repeats = 5  # the number of repeats for repeated k-fold cross validation

test_accs = []

stratified_folds = model_selection.RepeatedStratifiedKFold(
    n_splits=folds, n_repeats=n_repeats
).split(graph_labels, graph_labels)

for i, (train_index, test_index) in enumerate(stratified_folds):
    logger.info(f"Training and evaluating on fold {i+1} out of {folds * n_repeats}...")
    train_gen, test_gen = get_generators(
        generator, train_index, test_index, graph_labels, batch_size=30
    )

    model = create_graph_classification_model(generator)
    history, acc = train_fold(model, train_gen, test_gen, epochs)
    test_accs.append(acc)

logger.info(f"Accuracy over all folds mean: {np.mean(test_accs)*100:.3}% and std: {np.std(test_accs)*100:.2}%")
