# add stopwords 
import os
import pandas as pd
import nltk
import sqlite3
import regex

os.getcwd()
extra_stopwords_file = open('extra_stopwords.txt', 'r').readlines()

extra_stopwords = list(pd.DataFrame(extra_stopwords_file).iloc[:,0].apply((lambda x: x.strip('\n'))))

stopwords = set(nltk.corpus.stopwords.words('english')  + extra_stopwords) # add extra stopwords piece within the set if defined

sw = pd.DataFrame({'x':1}, index=stopwords)

os.chdir('../../Data/Reduced')
os.getcwd()


with sqlite3.connect('WSJ-processed-full.db') as db:
    V = pd.read_sql('SELECT * FROM vocab', db, index_col='term_id')


V['stop'] = V.term_str.map(sw.x).fillna(0).astype('int')

with sqlite3.connect('WSJ-processed-full.db') as db:
    V.to_sql('vocab', db, if_exists='replace', index=True)


