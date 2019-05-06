
slug = 'FILL THIS IN'
db_file = 'FILL THIS IN.db'
# vocab_weight = 'tfidf_sum'
vocab_weight = 'tf_sum'
vocab_weight_quantile = .94
vocab_min_n = 3
kde_kernel = 'gaussian'
kde_bandwidth = 2000
kde_samples = 1000


import pandas as pd
import numpy as np
import scipy as sp
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.neighbors import KernelDensity as KDE


from IPython.display import display, HTML
# % matplotlib inline
# % pwd

# basic functions
def get_term_id(vocab, term_str):
    return vocab[vocab.term_str == term_str].index[0]

def get_term_str(vocab, term_id):
    return vocab.loc[term_id].term_str

# imports
with sqlite3.connect(db_file) as db:
    V = pd.read_sql("SELECT * FROM vocab WHERE stop = 0", db, index_col='term_id')
    K = pd.read_sql("SELECT term_id, term_str FROM token WHERE term_id >= 0", db)

# token index
K.rename_axis('offset', inplace=True)

# filter vocab
V1 = V[(V[vocab_weight] > V[vocab_weight].quantile(vocab_weight_quantile)) & (V.n >= vocab_min_n)]
V1.sort_values(vocab_weight, ascending=False).head(10).T


# list of top terms
TOP_TERMS = V.sort_values(vocab_weight, ascending=False).term_str.head(30).tolist()

# filter tokens by vocab
K = K[K.term_id.isin(V1.index)]

# create offset arrays for each term
B = K.reset_index().groupby(['term_str']).offset.apply(lambda x: x.tolist()).to_frame()
B['x'] = B.apply(lambda x: np.array(x.offset)[:, np.newaxis], 1)

# KDE for each term
scale_max = K.index.max() # THIS IS CRUCIAL
x_axis = np.linspace(0, scale_max, kde_samples)[:, np.newaxis]
B['kde'] = B.apply(lambda row: KDE(kernel=kde_kernel, bandwidth=kde_bandwidth).fit(row.x), 1)
B['scores'] = B.apply(lambda row: row.kde.score_samples(x_axis), axis=1)
# B['scaled'] = B.apply(lambda row: np.exp(row.scores) * (scale_max / kde_samples), axis=1)

# visualize KDE
PLOTS = B.apply(lambda row: pd.Series(np.exp(row.scores) * (scale_max / kde_samples)), axis=1)
FIG = dict(figsize=(15, 5))
# ======== words ===========
PLOTS.loc['EXAMPLE1'].plot(**FIG)
PLOTS.loc['EXAMPLE2'].plot(**FIG)
PLOTS.loc['EXAMPLE3'].plot(**FIG)


# SCORE PAIRS
pairs = pd.DataFrame([(x,y) for x in B.index for y in B.index if y > x] , columns=['x','y'])

# compute overlap -- will likely give us memory errors
def overlap(row):
    kde1 = PLOTS.loc[row.x]
    kde2 = PLOTS.loc[row.y]
    overlap = np.minimum(kde1, kde2)
    return np.trapz(overlap)
pairs['overlap'] = pairs.apply(overlap, axis=1)

def paircorr(row):
    return PLOTS.T[[row.x,row.y]].corr().values[0][1]
pairs['corr'] = pairs.apply(paircorr, axis=1)

# skim top words
pairs.overlap.plot.hist()
pairs[pairs.overlap > .6 ].sort_values('overlap', ascending=False)

pairs2 = pairs.copy().rename(columns={'x':'y', 'y':'x'})
pairs3 = pd.concat([pairs, pairs2], sort=True)


# see related words to top term
DETAIL = '<table>'
for i, term in enumerate(TOP_TERMS):
    friends = pairs3[pairs3.x == term].sort_values('overlap', ascending=False).head(10)
    DETAIL += "<tr><td colspan=1><b>{}. {}</b></td></tr>".format(i+1, term)
    for row in friends.reset_index(drop=True)[['y', 'overlap']].values:
        bar = round(row[1] * 100) * '|'
        DETAIL += "<tr><td>{}</td><td style='text-align:left;'>{} ({})</td></tr>".format(row[0], bar, row[1])
DETAIL += "</table>"

display(HTML(DETAIL))



# exploring term correlations
CORR = pd.crosstab(pairs3.x, pairs3.y, pairs3.overlap, aggfunc='sum').fillna(1)
def corr_plot_terms(terms, dtm, title='Foo'):
    plt.figure(figsize = (20,20))
    print(title)
    corr = dtm[terms].corr()
    sns.heatmap(corr, vmax=.3, annot=True, center=0, 
              cmap='RdYlGn',
              square=True, linewidths=.5, 
              cbar_kws={"shrink": .5})
    plt.show()

corr_plot_terms(TOP_TERMS, PLOTS.T, title='TEST')


# export graph
import networkx as nx
G = nx.Graph()
edges = pairs[['x','y','overlap']].sort_values('overlap', ascending=False).head(1000).apply(lambda x: (x.x, x.y, x.overlap), axis=1).values
G.add_weighted_edges_from(edges)
nx.write_gexf(G, "{}.gexf".format(slug))

with sqlite3.connect(db_file) as db:
    pairs.to_sql('term_pair', db, if_exists='replace', index=True)
    PLOTS.T.to_sql('term_kde', db, if_exists='replace', index=True)
#     vocab.to_sql('vocab', db, if_exists='replace', index=True)

