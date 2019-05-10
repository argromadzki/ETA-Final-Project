

import pandas as pd
import sqlite3
import numpy as np
import os


os.chdir('../../Data/')
os.getcwd()


K = pd.read_csv('redo_tokens.csv')
V = pd.read_csv('redo_vocab.csv')

# sum(K.docID.isna()) # not missing to start with

WORDS = (K.punc == 0) & (K.num == 0) & K.term_id.isin(V[V.stop==0].index)
del(V)
K = K[['year','month','num_day','section','docID','sentence_id','token_id','token_str', 'term_str', 'term_id', 'punc', 'num']]

K = K.rename(columns={'num_day':'day', 'docID':'doc_key', 'sentence_id':'sentence_num','token_id':'token_num'})

# sum(K.doc_key.isna()) # still not missing

# fix the months to save memory
import calendar
d = dict((value,key) for key,value in enumerate(calendar.month_name))
K.month =K.month.map(d)
K.head()

K.sort_values(by=['year','month'], inplace= True)
# sum(K.doc_key.isna())  # still good..


# # Extract DOC table
# K.iloc[:,:5].duplicated()
D = K.iloc[:,:5].drop_duplicates().reset_index(drop=True) 
# !!! think i got it -- since the ID's were messed up for rows that wrapped (and often followed similar patterns)
D.index.name = 'doc_id'
D.head()


# ## Remove duplicate doc_key
D.doc_key.value_counts().head() # check to see if there are any in the full thing

# D_original = D
# D = D_original
D = D.drop(list(D[D.doc_key=='pm'].index))
D = D.drop(list(D[D.doc_key=='billion'].index))
D = D.drop(list(D[D.doc_key=='million'].index))
D = D.drop(list(D[D.doc_key=='am'].index))
D = D.reset_index(drop=True)
D.index.name = 'doc_id'
D.head()


# # Extract TOKEN table (originally called this, but we are just modifying existing one to match framework)

K = K.iloc[:,4:].reset_index(drop=True)
K.index.name = 'token_id'
K.head()

K['doc_id'] = K.doc_key.map(D.reset_index().set_index('doc_key').doc_id)
K.head()
# sum(K.doc_key.isna()) # still nothing missing!
sum(K.doc_id.isna()) # confirmed -- cannot get unique documents based on non-unique names 

K= K[ -K.doc_id.isna()]


# ## Remove doc_key from DOC and TOKEN

D = D.drop('doc_key', axis=1)
K = K.drop('doc_key', axis=1)

K.head()


# # Create DTM


# K = K[K.punc==0] # shouldn't have done this line
K.head()
BOW = K[WORDS].groupby(['doc_id','term_id']).term_id.count().to_frame().rename(columns={'term_id':'n'})
BOW.head()
# DTM = BOW.unstack()
# DTM.head()



K.doc_id = K.doc_id.astype('int')

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

x = TFIDF.f.quantile(.50)
TFIDF[TFIDF.f > x]


# ========== added after meeting ========== 
# TFIDF.loc[0].sum() # debugging what our TFIDF looks like -- it is different from those in Textman files
# TFIDF.head().max()

# term_id_headers = V.term_id
# doc_id_rows = list(D.index)

# reshaped_TFIDF = pd.DataFrame(index = doc_id_rows, columns=term_id_headers) # this would give 180k x 30k dataframe -- memory error

# V['tfidf_sum'] = TFIDF.sum()
# V['tfidf_mean'] = TFIDF.mean() # computationally expensive -- don't need all of them
# V['tfidf_max'] = TFIDF.max()

# create pairs (only run this if we want comparisons) -- too memory intensive, again
# doc_ids = D.index.tolist()
# pairs = [(i,j) for i in doc_ids for j in doc_ids if j > i]
# P = pd.DataFrame(pairs).reset_index(drop=True).set_index([0,1])
# P.index.names = ['doc_x','doc_y']

# reduce vocabulary based on TFIDF 

def get_top_terms(vocab, no_stops=True, sort_col='n', k=1000):
    if no_stops:
        V = vocab[vocab.stop == 0]
    else:
        V = vocab
    return V.sort_values(sort_col, ascending=False).head(k)


# TFIDF.T
top_n = 10000
TOPV = get_top_terms(V, sort_col='n', k = top_n)

# pd.set_option('display.max_rows', 1000)
# possible_stopwords = get_top_terms(V, sort_col='n', k = 50)['term_str']

# copyTOPV = TOPV
# bucket1 = list(copyTOPV['term_str'].iloc[0:49])
# copyTOPV = copyTOPV[50:]
# bucket2 = list(copyTOPV['term_str'].iloc[0:49])
# copyTOPV = copyTOPV[50:]
# bucket3 = list(copyTOPV['term_str'].iloc[0:49])
# copyTOPV = copyTOPV[50:]
# bucket4 = list(copyTOPV['term_str'].iloc[0:49])
# copyTOPV = copyTOPV[50:]
# bucket5 = list(copyTOPV['term_str'].iloc[0:49])
# copyTOPV = copyTOPV[50:]
# bucket6 = list(copyTOPV['term_str'].iloc[0:49])
# copyTOPV = copyTOPV[50:]

# bucket1
# bucket2
# bucket3
# bucket4
# bucket5
# bucket6

# reduce tfidf small
tfidf_small = TFIDF.loc[TOPV.index].stack().to_frame().rename(columns={0:'w'})


# from scipy.spatial import distance
# P['cosine'] = P.apply(distance.cosine, 1)

K.doc_id = K.doc_id.astype('int32')

K[K.doc_id.isna()]

K.to_csv('redo_token_mod.csv')
D.to_csv('redo_doc.csv')
tfidf_small.to_csv('redo_tfidf_small.csv')
TFIDF.to_csv('redo_tfidf.csv')

