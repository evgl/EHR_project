import json
import os
import numpy as np

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