import pandas as pd
from biobert_embedding.embedding import BiobertEmbedding
from tqdm import tqdm


alive_df = pd.read_csv('alive_df.csv')
dead_df = pd.read_csv('dead_df.csv')

# Combine all notes
all_notes_df = pd.concat([alive_df, dead_df])

text_lines = list()
lines = all_notes_df['TEXT'].values.tolist()

text = ""
for line in tqdm(lines):
    text += ' ' + line

# Class Initialization (You can set default 'model_path=None' as your finetuned BERT model path while Initialization)
biobert = BiobertEmbedding()

word_embeddings = biobert.word_vector(text)
sentence_embedding = biobert.sentence_vector(text)

# print("Text Tokens: ", biobert.tokens)
# Text Tokens:  ['breast', 'cancers', 'with', 'her2', 'amplification', 'have', 'a', 'higher', 'risk', 'of', 'cns', 'metastasis', 'and', 'poorer', 'prognosis', '.']

print ('Shape of Word Embeddings: %d x %d' % (len(word_embeddings), len(word_embeddings[0])))
# Shape of Word Embeddings: 16 x 768

print("Shape of Sentence Embedding = ",len(sentence_embedding))
# Shape of Sentence Embedding =  768