# add stopwords 
import os
import pandas as pd
import nltk

os.getcwd()
extra_stopwords_file = open('extra_stopwords.txt', 'r').readlines()

extra_stopwords = pd.DataFrame(extra_stopwords_file).iloc[:,0].apply((lambda x: x.strip('\n')))

stopwords = set(nltk.corpus.stopwords.words('english')  + extra_stopwords) # add extra stopwords piece within the set if defined

sw = pd.DataFrame({'x':1}, index=stopwords)
# vocab['stop'] = vocab.term_str.map(sw.x).fillna(0).astype('int')