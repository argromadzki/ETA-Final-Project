
import pandas as pd
import os
import nltk
import sqlite3
import numpy as np


os.chdir('../../Data/Reduced')
os.getcwd()


def create_df(year):
    '''
    Can be used on its own to set up a basic dataframe from a single year of the WSJ data.

    Input: Year (type = int)
    '''
    df = pd.read_csv('{}_manual_clean.csv'.format(year), header=-1)
    df.columns =  ['docID', 'year', 'month', 'num_day', 'weekday', 'title', 'text','section'] # note this is different with new file
    df['text'] = df['title']+' '+df['text']
    del df['title']

    # df.drop_duplicates(['doc_number'], keep='last') # shouldn't need this now that character limit fixed
    # df = get_ID(df)                       # removed considering we have factiva ID read in now as its own column
    df = strip_docID(df)
    # df['year'] = year
    return df


def strip_docID(df):
    '''
    After extracting factiva ID's, strip them off of the end of each text

    Input: Dataframe (single year) that has already been processed by `get_ID`
    '''
    df['text'] = df.text.apply(lambda x: ' '.join(str(x).split()[:-2]))
    return df


def combine_years(list_of_dfYears):
    '''
    This is a helper function to `make_multiple_dfs`, used to combine single-year dataframes.
    
    Alternatively, Can be called on its own to manually combine multiple processed single-year dataframes

    Input: list of the individual year dataframes.

    Note: unlike `make_multiple_dfs`, the input used here is the dataframes themselves

    '''

    for i in range(0,len(list_of_dfYears)):
        if i == 0:
            new_df = list_of_dfYears[0]
        else:
            new_df = new_df.append(list_of_dfYears[i])
    return new_df


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

# implement however many years
list_years = [2001] # reduced this for computing time
df = make_multiple_dfs(list_years).reset_index(drop=True)
pd.set_option('display.max_columns', 8)

pd.set_option('display.max_rows', 100)

df = df[df.year != None]
# df[df['year'].isna()]
df['year'] = df.year.astype(np.int64)

df['fulldate'] = df["month"] + "-" + df["num_day"].map(str) + "-" + df["year"].map(str)
df['fulldate'] = pd.to_datetime(df['fulldate'])

# df[df.fulldate =='6,165-A1-nan'] # one article had this
# df[df.fulldate =='B7C-nan-nan']
# df[df.fulldate =='B7G-nan-nan']
# df[df.fulldate =='B2-nan-nan']
# df[df.fulldate =='0-7,045- other passengers recall.  "']
# df.loc['August-1-2001.0' in df.fulldate, df.year] = 'August-1-2001' # call npint64 earlier to solve this



df.sort_values(by=['fulldate', 'weekday', 'section'], inplace= True)

OHCO = ['year', 'month', 'num_day', 'weekday', 'section', 'docID', 'sentence_id', 'token_id']
ARTICLE = OHCO[:6]
SENTS = OHCO[:7]

dfi = df.set_index(ARTICLE)

print('Processing sentences...')
sents = dfi.text.apply(lambda x: pd.Series(nltk.sent_tokenize(x))).stack().to_frame().rename(columns={0:'sent_str'})
sents.index.names = SENTS



def to_years(list_of_years):
    '''
    Used to break up the dataframe into more feasible sizes before word level tokenization

    Input: list of years
    '''
    
    dictionary = {}
    for one_year in list_of_years:
        dictionary[one_year] = sents.loc[one_year]
    return dictionary


def to_months(list_of_years):
    '''
    Used to break up the dataframe into more feasible sizes before word level tokenization after `to_years`

    Input: list of years
    '''

    month_dictionary = {}
    for one_year in list_of_years:
        for one_month in ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']:
            month_dictionary[str(one_year) +" " + one_month] = sents.loc[one_year].loc[one_month]

    return month_dictionary

test_dict_month = to_months([2001]) # again reduced this here
list_keys = test_dict_month.keys()


def monthly_tokenizer(key_yearmonth):
    tokens = test_dict_month[key_yearmonth].sent_str\
        .apply(lambda x: pd.Series(nltk.word_tokenize(x)))\
        .stack()\
        .to_frame()\
        .rename(columns={0:'token_str'})
    tokens.index.names = OHCO[2:]
    return tokens


def full_monthly_tokenizer(list_keys): 
    for key_yearmonth in list_keys:
        print('Processing {}'.format(key_yearmonth)) # note this isn't popping up
        tokens = test_dict_month[key_yearmonth].sent_str\
            .apply(lambda x: pd.Series(nltk.word_tokenize(x)))\
            .stack()\
            .to_frame()\
            .rename(columns={0:'token_str'})
        tokens.index.names = OHCO[2:]
        with sqlite3.connect('WSJ-Monthly.db') as db:
            tokens.to_sql(str(key_yearmonth), db, if_exists='replace', index=True)
        print("Finished {}".format(key_yearmonth))
        del(tokens)



year01 = list(list_keys)[0:12]
full_monthly_tokenizer(year01)

def reorient_things(year_month):
    with sqlite3.connect('WSJ-Monthly.db') as db:
        month_df = pd.read_sql('SELECT * FROM [{}]'.format(year_month), db, index_col=OHCO[2:])
    str_year, str_month = year_month.split()
    month_df['month'] = str_month
    month_df['year'] = str_year
    month_df.reset_index(level=OHCO[2:], inplace= True)
    month_df.set_index(OHCO, inplace= True)
    month_df.index.names = OHCO
    return month_df
        
list_keys = ['2001 January', '2001 February', '2001 March', '2001 April', '2001 May', '2001 June', '2001 July', '2001 August', '2001 September', '2001 October', '2001 November', '2001 December']


def reassemble(list_keys):
    for key_yearmonth in list_keys:
        tokens = reorient_things(key_yearmonth)
        with sqlite3.connect('WSJ-reindexed.db') as db:
                tokens.to_sql(str(key_yearmonth), db, if_exists='replace', index=True)
        print("Finished with {}".format(key_yearmonth))
reassemble(list_keys)


def reassemble2(list_keys):
     for key_yearmonth in list_keys:
         tokens = reorient_things(key_yearmonth)
         with sqlite3.connect('WSJ-benchmark.db') as db:
                 tokens.to_sql('token', db, if_exists='append', index=True)
         print("Finished with {}".format(key_yearmonth))


with sqlite3.connect('WSJ-benchmark.db') as db:
     tokens = pd.read_sql('SELECT * FROM token', db, index_col = OHCO)

###############################################
with sqlite3.connect('WSJ-reindexed.db') as db:
    df25 = pd.read_sql('SELECT * FROM [2001 January]', db)

with sqlite3.connect('WSJ-reindexed.db') as db:
    df26 = pd.read_sql('SELECT * FROM [2001 February]', db)

with sqlite3.connect('WSJ-reindexed.db') as db:
    df27 = pd.read_sql('SELECT * FROM [2001 March]', db)

with sqlite3.connect('WSJ-reindexed.db') as db:
    df28 = pd.read_sql('SELECT * FROM [2001 April]', db)

with sqlite3.connect('WSJ-reindexed.db') as db:
    df29 = pd.read_sql('SELECT * FROM [2001 May]', db)

with sqlite3.connect('WSJ-reindexed.db') as db:
    df30 = pd.read_sql('SELECT * FROM [2001 June]', db)

with sqlite3.connect('WSJ-reindexed.db') as db:
    df31 = pd.read_sql('SELECT * FROM [2001 July]', db)

with sqlite3.connect('WSJ-reindexed.db') as db:
    df32 = pd.read_sql('SELECT * FROM [2001 August]', db)

with sqlite3.connect('WSJ-reindexed.db') as db:
    df33 = pd.read_sql('SELECT * FROM [2001 September]', db)

with sqlite3.connect('WSJ-reindexed.db') as db:
    df34 = pd.read_sql('SELECT * FROM [2001 October]', db)

with sqlite3.connect('WSJ-reindexed.db') as db:
    df35 = pd.read_sql('SELECT * FROM [2001 November]', db)

with sqlite3.connect('WSJ-reindexed.db') as db:
    df36 = pd.read_sql('SELECT * FROM [2001 December]', db)
######################################################

dflist = [df25, df26, df27, df28, df29, df30, df31, df32, df33, df34, df35, df36]
total = df25.append(dflist[1:]) 



total.set_index(OHCO, inplace= True)


with sqlite3.connect('WSJ-full.db') as db: 
    total.to_sql("full", db, if_exists='replace', index=True)

tokens = total


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

stopwords = set(nltk.corpus.stopwords.words('english') ) 

sw = pd.DataFrame({'x':1}, index=stopwords)
vocab['stop'] = vocab.term_str.map(sw.x).fillna(0).astype('int')
del(sw)

tokens['term_id'] = tokens['term_str'].map(vocab.reset_index()\
    .set_index('term_str').term_id).fillna(-1).astype('int')


with sqlite3.connect('WSJ-processed-full.db') as db:
    vocab.to_sql('vocab', db, if_exists='replace', index=True)

def chunk_writer(df, increment=5000):
    n = len(df)
    start = 0
    repeat = True
    while repeat == True:
        end = start + increment
        if end > n:
            current = df.loc[start:n]
            with sqlite3.connect('WSJ-processed-full.db') as db:
                current.to_sql('token', db, if_exists='append', index=True, chunksize=1000)
            repeat = False

        else:
            current = df.loc[start:end]
            with sqlite3.connect('WSJ-processed-full.db') as db:
                current.to_sql('token', db, if_exists='append', index=True, chunksize=1000)
        
        start = end + 1



chunk_writer(tokens)




mystring = 'j000000020010711dx1p001ce'
mystring2 = 'j000000020010711dx1p001ce'
mystring3= 'j000000020010711dx1p001ce'
mystring4 = 'j000000020010711dx1p001ce'
mystring5 = 'fudge'


mystring[:2]

mylist = [mystring, mystring2, mystring3, mystring4, mystring5]
filtered = []
for item in mylist:
    if item[:2] == 'j0':
        filtered.append(item)




