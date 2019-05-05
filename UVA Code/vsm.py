

import pandas as pd
import os
import nltk
import sqlite3
import numpy as np

db_name = '____' # Fill this in!

os.chdir('../../Data')
os.getcwd()

OHCO = ['year', 'month', 'num_day', 'weekday', 'section', 'docID', 'sentence_id', 'token_id']

with sqlite3.connect(db_name) as db:
    K = pd.read_sql('SELECT * FROM token', db, index_col=OHCO)
    V = pd.read_sql('SELECT * FROM vocab', db, index_col='term_id')


# Creating DTM without stopwords
WORDS = (K.punc == 0) & (K.num == 0) & K.term_id.isin(V[V.stop==0].index)

# Extract BOW from Tokens
BOW = K[WORDS].groupby(OHCO[:1]+['term_id'])['term_id'].count()

# Convert BOW to DTM
DTM = BOW.unstack().fillna(0)


# TF
alpha = .000001 # We introduce an arbitrary smoothing value
alpha_sum = alpha * V.shape[0]
TF = DTM.apply(lambda x: (x + alpha) / (x.sum() + alpha_sum), axis=1)

# TFIDF
N_docs = DTM.shape[0]
V['df'] = DTM[DTM > 0].count()
# vocab['idf'] = np.log10(N_docs / vocab.df) # if we want idf separately, it is the piece below
TFIDF = TF * np.log2(N_docs / V[V.stop==0]['df'])


# Add stats to the vocabulary table
V['tfidf_sum'] = TFIDF.sum()
V['tfidf_mean'] = TFIDF.mean()
V['tfidf_max'] = TFIDF.max()

# Create Document Table (Each Article)
D = DTM.sum(1).astype('int').to_frame().rename(columns={0:'term_count'})
D['tf'] = D.term_count / D.term_count.sum()

# Document Pairs
chap_ids = D.index.tolist()
pairs = [(i,j) for i in chap_ids for j in chap_ids if j > i]
P = pd.DataFrame(pairs).reset_index(drop=True).set_index([0,1])
P.index.names = ['doc_x','doc_y']


# Get top words and trim matrix
def get_top_terms(vocab, no_stops=True, sort_col='n', k=1000):
    if no_stops:
        V = vocab[vocab.stop == 0]
    else:
        V = vocab
    return V.sort_values(sort_col, ascending=False).head(k)

top_n = 1000
TOPV = get_top_terms(V, sort_col='n', k = top_n)

# reduce tfidf small
tfidf_small = TFIDF[TOPV.index].stack().to_frame().rename(columns={0:'w'})


from scipy.spatial import distance
P['cosine'] = P.apply(distance.cosine, 1)

with sqlite3.connect(db_name) as db:
    V.to_sql('vocab', db, if_exists='replace', index=True)
    K.to_sql('token', db, if_exists='replace', index=True)
    D.to_sql('doc', db, if_exists='replace', index=True)
    P.to_sql('docpair', db, if_exists='replace', index=True)
    tfidf_small.to_sql('tfidf_small', db, if_exists='replace', index=True)
    TFIDF.to_sql('tfidf_small', db, if_exists='replace', index=True)
    # BOW.to_frame().rename(columns={'term_id':'n'}).to_sql('bow', db, if_exists='replace', index=True)
