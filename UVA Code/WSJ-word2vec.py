
#%%
import pandas as pd
import numpy as np
import sqlite3
from gensim.models import word2vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import sys

#%% # # Plot options
num_tsne_points = 2500
limit_to_nouns = True
section_of_interest = None
process_files = True
token_output = 'redo_token_mod.csv'
token_output_pos = 'token_output_pos.csv'
token_output_noun = 'redo_token_mod_noun.csv'
token_output_section = 'redo_token_mod_{}.csv'.format(section_of_interest)
token_output_noun_section = 'redo_token_mod_noun_{}.csv'.format(section_of_interest)
tsne_coordinates = "tsne_coordinates.csv"
tsne_coordinates_nouns_all_sections = "tsne_coordinates_nouns_all_sections.csv"

coords_to_graph = tsne_coordinates_nouns_all_sections
#%% # get datasets
os.chdir('../../Data/')
os.getcwd()
V = pd.read_csv('redo_vocab.csv')
if process_files is not True:
    coords = pd.read_csv(coords_to_graph, index_col = 'Unnamed: 0')
    coords = coords.sample(num_tsne_points)
    coords.reset_index(inplace = True, drop=True)

    plt.figure(figsize=(100, 100)) 
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
    response = input('Continue to processing (Y/N): ').strip().lower()
    if response != 'y':
        sys.exit("Confirmed -- exiting now")
#%%
tokens = pd.read_csv(token_output)


#%% # # Process

tokens = tokens[~tokens.term_str.isna()]

if limit_to_nouns == True:
    from nltk.tag import pos_tag
    tokens['pos'] = pd.DataFrame(pos_tag(list(tokens.token_str)))[1]
    # tokens.to_csv('token_output_noun')
    WORDS = (tokens.punc == 0) & (tokens.num == 0) & tokens.term_id.isin(V[V.stop==0].index) & (tokens.pos == 'NN')
    tokens = tokens.loc[WORDS, :]
    tokens.to_csv(token_output_noun)

else:
    WORDS = (tokens.punc == 0) & (tokens.num == 0) & tokens.term_id.isin(V[V.stop==0].index)
    tokens = tokens.loc[WORDS, :]



#%% # ## Import tokens and convert to a corpus for Gensim

documents = pd.read_csv('redo_doc.csv') # , index_col='doc_id'
combo = pd.merge(tokens, documents, on='doc_id')


if section_of_interest is not None:
    combo = combo[combo.section == section_of_interest]

del(documents)
del(tokens)

#%%
OHCO = ['year','month','day','section','sentence_num']

corpus = combo.groupby(OHCO).term_str.apply(lambda  x:  x.tolist()).reset_index()['term_str'].tolist()


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
coords_full.to_csv(coords_to_graph)
coords = coords.sample(num_tsne_points)
coords.reset_index(inplace = True, drop=True)

#%%
plt.figure(figsize=(100, 100)) 
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

#%%
