
#%%
import pandas as pd
import numpy as np
import sqlite3
from gensim.models import word2vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

# os.getcwd()
os.chdir('../../Data/')
os.getcwd()


#%% # # Process
tokens = pd.read_csv('redo_token_mod.csv')
tokens = tokens[~tokens.term_str.isna()]
#%% # ## Import tokens and convert to a corpus for Gensim

documents = pd.read_csv('redo_doc.csv') # , index_col='doc_id'
combo = pd.merge(tokens, documents, on='doc_id')
del(documents)
del(tokens)

#%%
OHCO = ['year','month','day','section','sentence_num']

corpus = combo.groupby(OHCO).term_str.apply(lambda  x:  x.tolist())    .reset_index()['term_str'].tolist()


#%%

# for article in corpus: # removed by the second line in tokens processes
#     for word in article:
#         if word =='nan':
#             del(word)

corpus[:5]
#%% # ## Generate word embeddings with Gensim's library

#%%
model = word2vec.Word2Vec(corpus, size=246, window=5, min_count=200, workers=4)
del(corpus)

#%% # ## Visualize with tSNE
#%% # ### Generate coordinates to plot

#%%
coords = pd.DataFrame(index=range(len(model.wv.vocab)))
coords['label'] = [w for w in model.wv.vocab]
coords['vector'] = coords['label'].apply(lambda x: model.wv.get_vector(x))

#%% # ### Use ScikitLearn's TSNE library

#%%
tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
tsne_values = tsne_model.fit_transform(coords['vector'].tolist())


#%%
coords['x'] = tsne_values[:,0]
coords['y'] = tsne_values[:,1]


#%%
coords.head()
#%%

coords_full = coords
coords_full.to_csv("tsne_coordinates.csv")
coords = coords.sample(500)

#%%
plt.figure(figsize=(800, 800)) 
for i in range(len(coords)):
    plt.scatter(coords.x[i],coords.y[i])
    plt.annotate(coords['label'][i],
                 xy=(coords.x[i], coords.y[i]),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')
plt.savefig('wsj wordcloud.png')
plt.show()