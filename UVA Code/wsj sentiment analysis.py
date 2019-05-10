#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os


os.chdir('../../Data/')
os.getcwd()


#%% 
# # Synopsis
# 
# Applying Syuzhet methods to *Persuasion* using NRC, Bing, Syuzhet, and VADER.
#%% 
# # Configuration

#%%

# Developed in other notebook
lex_db = 'C:\\Users\\Nick\\Documents\\UVA\\DSI\\Spring 2019\\ETA\\Github Repo ETA\\DS5559\\labs\\2019-04-11_Lab11\\lexicons.db'

# Developed in previous lab
OHCO = ['year','month','day','section', 'doc_id', 'sentence_num']
YEAR = OHCO[:1]
MONTH = OHCO[:2]
DATE = OHCO[:3]
SECT = OHCO[:4]
ARTICLE = OHCO[:5]
SENTS = OHCO[:6]

# emo = 'bing_sentiment'
emo = 'syu_sentiment'

# For KDE
kde_kernel = 'gaussian'
# kde_bandwidth = 100
kde_samples = 1000

#%% 
# # Libraries

#%%
import pandas as pd
import numpy as np
import scipy as sp
import sys
import scipy.fftpack as fftpack
from sklearn.neighbors import KernelDensity as KDE
from sklearn.preprocessing import scale

import sqlite3

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.display import display, HTML

#%% 
# # Pragmas

#%%

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()
plt.style.use('fivethirtyeight')

#%% 
# # Process
#%% 
# ## Get Lexicons

#%%
with sqlite3.connect(lex_db) as db:
    combo = pd.read_sql('SELECT * FROM combo', db, index_col='term_str')


#%%
combo.head()

#%% 
# ### Get lexicon columns

#%%
# emo_cols = [col for col in combo.columns if 'nrc_' in col]
emo_cols = combo.columns


#%%
emo_cols

#%% 
# ## Get Text

#%%

tokens = pd.read_csv("redo_token_mod.csv")
tokens = tokens[~tokens.term_str.isna()]
vocab = pd.read_csv("redo_vocab.csv", index_col='term_id')
docs = pd.read_csv("redo_doc.csv")
# just renaming so I can delete items
full_tokens = pd.merge(tokens, docs, on='doc_id')
del(tokens)
tokens = full_tokens
tokens.set_index(OHCO)


#%% 
# ### Merge sentiment lexicon with vocab

#%%
tokens = tokens.join(combo, on='term_str', how='left')


#%%
tokens[emo].sample(10)


#%%
tokens[emo_cols] = tokens[emo_cols].fillna(0)


#%%
tokens.head(10)

#%% 
# ## Sentiment by OHCO

#%%
FIG = dict(figsize=(20, 5), legend=True, fontsize=14, rot=45)

#%% 
# ### By Article

#%%
by_article = tokens.groupby(ARTICLE)[emo].sum().plot(**FIG)

#%%

by_internationalmonth = tokens[tokens.section=="International"].groupby(MONTH)[emo_cols].sum()

max_x = by_internationalmonth.shape[0]
xticks = list(range(0, max_x, 100))
xticks.append(max_x - 1)


by_internationalmonth['nrc_anger'].plot(**FIG, xticks=xticks)
by_internationalmonth['nrc_sadness'].plot(**FIG, xticks=xticks)
by_internationalmonth['nrc_fear'].plot(**FIG, xticks=xticks)

#%%

by_internationalmonth[emo_cols[:5]].plot(**FIG, xticks=xticks)
by_month[emo_cols[:5]].plot(**FIG, xticks=xticks)

#%%

by_internationalmonth = tokens[tokens.section=="International"].groupby(MONTH)[emo_cols].sum()

max_x = by_internationalmonth.shape[0]
xticks = list(range(0, max_x, 100))
xticks.append(max_x - 1)


by_internationalmonth['nrc_anger'].plot(**FIG, xticks=xticks)
by_internationalmonth['nrc_sadness'].plot(**FIG, xticks=xticks)
by_internationalmonth['nrc_fear'].plot(**FIG, xticks=xticks)


#%%
by_date = tokens.groupby(DATE)[emo_cols].sum()


#%%
max_x = by_date.shape[0]
xticks = list(range(0, max_x, 100))
xticks.append(max_x - 1)


#%%
by_date[emo].plot(**FIG, xticks=xticks)

#%% 
# ### By Sentence

#%%
by_month = tokens.groupby(MONTH)[emo_cols].sum()


#%%
max_x = by_month.shape[0]
xticks = list(range(0, max_x, 250))
xticks.append(max_x - 1)


#%%
by_month[emo].plot(**FIG, xticks=xticks)


#%% 
# ### Add text to do spot checks

#%%
#tokens['html'] =  tokens.apply(lambda x: 
 #                              "<span class='sent{}'>{}</span>".format(int(np.sign(x[emo])), x.token_str), 1)


#%%
#tokens['html'].head()


#%%
by_month['month_str'] = tokens.groupby(MONTH).term_str.apply(lambda x: x.str.cat(sep=' '))
#by_month['html_str'] = tokens.groupby(MONTH).html.apply(lambda x: x.str.cat(sep=' '))


#%%
by_month[['month_str']].head()

#%% 
# ### Histogram

#%%
by_month[emo].plot.hist(**FIG, bins=50)

#%% 
# ### Look at a sample of sentences

#%%
rows = []
for idx in by_month.sample(10).index:
    
    valence = round(by_month.loc[idx, emo], 4)     
    t = 0
    if valence > t: color = '#ccffcc'
    elif valence < t: color = '#ffcccc'
    else: color = '#f2f2f2'
    z=0
    rows.append("""<tr style="background-color:{0};padding:.5rem 1rem;font-size:110%;">
    <td>{1}</td><td>{3}</td><td width="400" style="text-align:left;">{2}</td>
    </tr>""".format(color, valence, by_month.loc[idx, 'html_str'], idx))
    
display(HTML('<style>#sample1 td{font-size:120%;vertical-align:top;} .sent-1{color:red;font-weight:bold;} .sent1{color:green;font-weight:bold;}</style>'))
display(HTML('<table id="sample1"><tr><th>Sentiment</th><th>ID</th><th width="600">Sentence</th></tr>'+''.join(rows)+'</table>'))

#%% 
# ### Plot with Xticks

#%%
xticks = [0]
xticks += by_month[emo].reset_index().sort_values(emo).head(7).index.values.tolist()
xticks += by_month[emo].reset_index().sort_values(emo).tail(7).index.values.tolist()
xticks += [by_month.shape[0] - 1]


#%%
by_month[emo].plot(**FIG, xticks=xticks, title=emo)


#%%
by_month.iloc[xticks][emo].sort_index().to_frame()

#%% 
# ### Inspect Positives

#%%
by_month.sort_values(emo, ascending=False)[[emo,'month_str']].head(20)

#%% 
# ### Inspect Negatives

#%%
by_month.sort_values(emo, ascending=True)[[emo,'month_str']].head(20)

#%% 
# ## Apply Transforms#%% 
# ### Adapt Jockers' revised DCT function
# 
# Discrete Cosine Transform. A DCT is a Fourier-related transform similar to the discrete Fourier transform (DFT), but using only real numbers. 

#%%
def get_dct_transform(raw_values, low_pass_size = 5, x_reverse_len = 100):
    if low_pass_size > len(raw_values):
        raise ValueError("low_pass_size must be less than or equal to the length of raw_values input vector")
    values_dct = fftpack.dct(raw_values, type = 2)
    keepers = values_dct[:low_pass_size]
    padded_keepers = list(keepers) + list(np.zeros(x_reverse_len - low_pass_size))
    dct_out = fftpack.idct(padded_keepers)
    return(dct_out)


#%%
FFTCFG1 = dict(
    low_pass_size = 5,
    x_reverse_len = 100
)
X1 = get_dct_transform(by_month[emo].values, **FFTCFG1)


#%%
FIG['legend'] = True
FIG['figsize'] = (12,5)
pd.Series(X1).plot(**FIG)

#%% 
#%% 
# ### Using Rolling
# **Window types**: 
# boxcar
# triang
# blackman
# hamming
# bartlett
# parzen
# bohman
# blackmanharris
# nuttall
# barthann
# kaiser (needs beta)
# gaussian (needs std)
# general_gaussian (needs power, width)
# slepian (needs width).

#%%
# Config for sentences
CFG1 = dict(
    win_type='triang',
    window=3000,
    center=True
)


#%%
FIG['figsize'] = (20,5)
by_month[emo].fillna(0).rolling(**CFG1).sum().plot(**FIG)
#does not plot


#%%
# Config for tokens
CFG2 = dict(
    win_type='triang',
    window= int(tokens.shape[0]/ 9),
    center=True
)
FIG['legend'] = True
CFG2['window']


#%%
#tokens[emo].fillna(0).rolling(**CFG2).sum().plot(**FIG)
#hangs 


#%%
tokens[['nrc_positive','nrc_negative']].rolling(**CFG2).sum().plot(**FIG)


#%%
tokens[['bing_positive','bing_negative']].fillna(0).rolling(**CFG2).sum().plot(**FIG)

#%% 
# ## Multinomial Emotions

#%%
EOM = tokens[emo_cols].reset_index(drop=True)


#%%
EOM.head()


#%%
EOM.sum().sort_values().plot.barh(figsize=(7,7), fontsize=18)

#%% 
# ### Shape data for Sklearn's KDE function

#%%
emo_offsets = []
for col in emo_cols:
    x = EOM[EOM[col] > 0].index.tolist()
    y = [[i] for i in x]
    emo_offsets.append((col, y))
B = pd.DataFrame(emo_offsets, columns=['emo', 'x'])
B = B.set_index('emo')


#%%
B

#%% 
# ### Get KDE for each emotion

#%%
scale_max = EOM.shape[0]
kde_bandwidth = 1000
# kde_bandwidth = 250
x_axis = np.linspace(0, scale_max, kde_samples)[:, np.newaxis]
#hangs below
B['kde'] = B.apply(lambda row: KDE(kernel=kde_kernel, bandwidth=kde_bandwidth).fit(row.x), 1)
B['scores'] = B.apply(lambda row: row.kde.score_samples(x_axis), axis=1)

#%% 
# ### Visualize KDE plots

#%%
PLOTS = B.apply(lambda row: pd.Series(np.exp(row.scores) * (scale_max / kde_samples)), axis=1).T


#%%
PLOTS.plot(**FIG)


#%%
PLOTS[['nrc_joy','nrc_sadness']].plot(**FIG)


#%%
PLOTS[['nrc_joy','nrc_positive']].plot(**FIG)


#%%
PLOTS[['nrc_joy','nrc_anticipation']].plot(**FIG)


#%%
PLOTS[['nrc_joy','nrc_sadness']].plot(**FIG)


#%%
PLOTS[['nrc_anticipation','nrc_surprise']].plot(**FIG)


#%%
PLOTS[['nrc_positive','nrc_negative']].plot(**FIG)


#%%
PLOTS[['bing_positive','bing_negative']].plot(**FIG)

#%% 
# ### Compare KDE to Rolling graphs of Sentiment

#%%
PLOTS[emo].plot(**FIG)


#%%
tokens[emo].fillna(0).rolling(**CFG2).mean().plot(**FIG)


#%%
fig, axs = plt.subplots(len(emo_cols),2)
for i, emo in enumerate(emo_cols):
    FIGa = dict(figsize=(20,50), ax=axs[i,0], title=emo + ' (KDE)')
    FIGb = dict(figsize=(20,50), ax=axs[i,1], title=emo + ' (rolling)')
    PLOTS[emo].plot(**FIGa)
    tokens[emo].rolling(**CFG2).mean().plot(**FIGb)

#%% 
# ## Peak Joy
#%% 
# ### Find max joy from PLOTS

#%%
PLOTS.head()


#%%
PEAK_JOY = int((PLOTS['nrc_joy'].idxmax() / 1000) * tokens.shape[0])


#%%
PEAK_JOY_CHAP_NUM = tokens.iloc[PEAK_JOY].name[0]


#%%
PEAK_JOY_CHAP_NUM + 1

#%% 
# ### Display Chapter

#%%
PEAK_JOY_CHAP = tokens.loc[PEAK_JOY_CHAP_NUM].groupby(['para_num']).token_str.apply(lambda x: ' '.join(x)).tolist()


#%%
display(HTML('<br/><br/>'.join(PEAK_JOY_CHAP)))
