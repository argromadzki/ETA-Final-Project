
import pandas as pd
import os
import nltk
import sqlite3

os.chdir('../../Data')
os.getcwd()


def create_df(year):
    '''
    Can be used on its own to set up a basic dataframe from a single year of the WSJ data.

    Input: Year (type = int)
    '''
    df = pd.read_csv('{}_clean.csv'.format(year), header=-1)
    df.columns =  ['doc_number', 'year', 'month', 'num_day', 'weekday', 'section', 'text']
    df = get_ID(df)
    df = strip_docID(df)
    # df['year'] = year
    return df

# test = create_df(2000)
# sum(test.docID.isnull())


def get_ID(df):
    '''
    Creates columns of factiva ID's used to index articles by extracting it from the end of each text

    Input: Dataframe (single year)
    '''
    df['docID'] = df.text.apply(lambda x: x.split()[-1:][0])
    return df

def strip_docID(df):
    '''
    After extracting factiva ID's, strip them off of the end of each text

    Input: Dataframe (single year) that has already been processed by `get_ID`
    '''
    df['text'] = df.text.apply(lambda x: ' '.join(x.split()[:-2]))
    return df

# df2000 = create_df(2000)
# df1999 = create_df(1999)
# df1999

# test_99_00 = [df1999,df2000]

# combo_df = df1999.append(df2000) # .append(df2001).append(df2002)
# combo_df

def combine_years(list_of_dfYears):
    '''
    This is a helper function to `make_multiple_dfs`, used to combine single-year dataframes.
    
    Alternatively, Can be called on its own to manually combine multiple processed single-year dataframes

    Input: list of the individual year dataframes.

    Note: unlike `make_multiple_dfs`, the input used here is the dataframes themselves

    '''
    # FirstYear= True
    for i in range(0,len(list_of_dfYears)):
        if i == 0:
            new_df = list_of_dfYears[0]
        else:
            new_df = new_df.append(list_of_dfYears[i])
    return new_df

# combo_df =combine_years(test_99_00)
# combo_df

def make_multiple_dfs(list_of_years):
    '''
    This function can be used to execute the full preprocessing pipeline through till the point of actual text processing
    
    Input: list of years (type = int)

    Dependencies: `create_df`, `get_docID`, `strip_docID`, and `combine_years`


    '''
    list_of_dfs = []
    for year in list_of_years:
        print('\nProcessing CSV for year {}...'.format(year))
        current_df = create_df(year)
        list_of_dfs.append(current_df)
    big_df = combine_years(list_of_dfs)
    return big_df


if __name__ == "__main__":

    list_years = [1999,2000,2001,2002]
    df = make_multiple_dfs(list_years).reset_index(drop=True)
    pd.set_option('display.max_columns', 8)

    df['fulldate'] = df["month"] + "-" + df["num_day"].map(str) + "-" + df["year"].map(str)

    df['fulldate'] = pd.to_datetime(df['fulldate'])

    df.sort_values(by=['fulldate', 'weekday', 'section'], inplace= True)

    # df.iloc[1].text

    df['section'].value_counts()

####################################################
OHCO = ['year', 'month', 'num_day', 'weekday', 'section', 'docID', 'sentence_id', 'token_id']
ARTICLE = OHCO[:6]
SENTS = OHCO[:7]

dfi = df.set_index(ARTICLE)
# df_texts = dfi['text']
# df_texts

# df_texts.index[0] # gets the indices of the first thing
# list(df_texts.index.get_level_values('section').unique())

sents = dfi.text\
    .apply(lambda x: pd.Series(nltk.sent_tokenize(x)))\
    .stack()\
    .to_frame()\
    .rename(columns={0:'sent_str'})
sents.index.names = SENTS


tokens = sents.sent_str\
    .apply(lambda x: pd.Series(nltk.pos_tag(nltk.word_tokenize(x))))\
    .stack()\
    .to_frame()\
    .rename(columns={0:'pos_tuple'})
print('success')
tokens.index.names = OHCO
tokens['pos'] = tokens.pos_tuple.apply(lambda x: x[1])
tokens['token_str'] = tokens.pos_tuple.apply(lambda x: x[0])
tokens = tokens.drop('pos_tuple', 1)
del(sents)

tokens['punc'] = tokens.token_str.str.match(r'^[\W_]*$').astype('int')
tokens['num'] = tokens.token_str.str.match(r'^.*\d.*$').astype('int')

WORDS = (tokens.punc == 0) & (tokens.num == 0)
tokens.loc[WORDS, 'term_str'] = tokens.token_str.str.lower()\
    .str.replace(r'["_*.]', '')
vocab = tokens[tokens.punc == 0].term_str.value_counts().to_frame()\
    .reset_index()\
    .rename(columns={'index':'term_str', 'term_str':'n'})
vocab = vocab.sort_values('term_str').reset_index(drop=True)
vocab.index.name = 'term_id'

vocab['p'] = vocab.n / vocab.n.sum()

stemmer = nltk.stem.porter.PorterStemmer()
vocab['port_stem'] = vocab.term_str.apply(lambda x: stemmer.stem(x))

stopwords = set(nltk.corpus.stopwords.words('english') + extra_stopwords)

sw = pd.DataFrame({'x':1}, index=stopwords)
vocab['stop'] = vocab.term_str.map(sw.x).fillna(0).astype('int')
del(sw)

tokens['term_id'] = tokens['term_str'].map(vocab.reset_index()\
    .set_index('term_str').term_id).fillna(-1).astype('int')

with sqlite3.connect('WSJ.db') as db:
    tokens.to_sql('token', db, if_exists='replace', index=True)
    vocab.to_sql('vocab', db, if_exists='replace', index=True)