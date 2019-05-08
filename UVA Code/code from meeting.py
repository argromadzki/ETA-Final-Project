

import pandas as pd
import sqlite3
import numpy as np
import os


os.chdir('../../Data/Reduced')
os.getcwd()

# converting files to csv's since the overhead destroyed us

# with sqlite3.connect('WSJ-processed-full.db') as db:
#      K = pd.read_sql('SELECT * FROM token', db)
#      V = pd.read_sql('SELECT * FROM vocab', db)
# K.to_csv('vocab_data_01.csv')
# V.to_csv('vocab_data_01.csv')

K = pd.read_csv('token_data_01.csv')
V = pd.read_csv('vocab_data_01.csv')

WORDS = (K.punc == 0) & (K.num == 0) & K.term_id.isin(V[V.stop==0].index)

K = K[['year','month','num_day','section','docID','sentence_id','token_id','token_str', 'term_str', 'term_id', 'punc', 'num']]

K = K.rename(columns={'num_day':'day', 'docID':'doc_key', 'sentence_id':'sentence_num','token_id':'token_num'})

K.head()


# fix the months to save memory
import calendar
d = dict((value,key) for key,value in enumerate(calendar.month_name))
K.month =K.month.map(d)
K.head()

# # Extract DOC table
D = K.iloc[:,:5].drop_duplicates().reset_index(drop=True)
D.index.name = 'doc_id'
D.head()


# ## Remove duplicate doc_key
D.doc_key.value_counts().head()
D[D.doc_key=='j000000020011011dxab00025']
D = D.drop(24291)
D = D.reset_index(drop=True)
D.index.name = 'doc_id'
D.head()


# # Extract TOKEN table (originally called this, but we are just modifying existing one to match framework)

K = K.iloc[:,4:].reset_index(drop=True)
K.index.name = 'token_id'
K.head()

K['doc_id'] = K.doc_key.map(D.reset_index().set_index('doc_key').doc_id)
K.head()


# ## Remove doc_key from DOC and TOKEN

D = D.drop('doc_key', axis=1)
K = K.drop('doc_key', axis=1)

K.head()

'''
# # Extract VOCAB table (again, we had done this before)
# K = K[K.punc==0]
V = K[K.punc==0].term_str.value_counts().to_frame().reset_index().rename(columns={'index':'term_str', 'term_str':'n'})
V.index.name = 'term_id'
V.head()
K['term_id'] = K.term_str.map(V.reset_index().set_index('term_str').term_id).fillna(-1)
K.term_id = K.term_id.astype('int')
'''


# # Create DTM


# K = K[K.punc==0] # shouldn't have done this line
K.head()
BOW = K[WORDS].groupby(['doc_id','term_id']).term_id.count().to_frame().rename(columns={'term_id':'n'})
BOW.head()
# DTM = BOW.unstack()
# DTM.head()


A = BOW.reset_index().groupby(['doc_id']).n.sum()
B = BOW.reset_index().groupby(['doc_id','term_id']).n.sum()

TF = (B / A).to_frame()
TF.head()

## DF
DF = BOW.reset_index().groupby('term_id').doc_id.count().to_frame().rename(columns={'doc_id':'n'})
DF.head()
N = D.shape[0]

IDF = np.log10(N/DF)

TFIDF = TF * IDF

TFIDF.head()

TFIDF = TFIDF.rename(columns={'n':'f'})

TFIDF = TFIDF.join(V['term_str'], on='term_id', how='left')

TFIDF.sort_values('f', ascending=True)

TFIDF.f.describe()

x = TFIDF.f.quantile(.75)

TFIDF[TFIDF.f > x]


# ========== added after meeting ========== 

V['tfidf_sum'] = TFIDF.sum()
# V['tfidf_mean'] = TFIDF.mean() # computationally expensive -- don't need all of them
# V['tfidf_max'] = TFIDF.max()

# create pairs (only run this if we want comparisons)
doc_ids = D.index.tolist()
pairs = [(i,j) for i in doc_ids for j in doc_ids if j > i]
P = pd.DataFrame(pairs).reset_index(drop=True).set_index([0,1])
P.index.names = ['doc_x','doc_y']

# reduce vocabulary based on TFIDF 

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

V.to_csv('vocab_01.csv')
K.to_csv('token_01.csv')
D.to_csv('doc_01.csv')
P.to_csv('docpair_01.csv')
tfidf_small.to_csv('tfidf_small_01.csv')
TFIDF.to_csv('tfidf_small_01.csv')

