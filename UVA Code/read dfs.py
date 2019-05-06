
import pandas as pd
import os
import nltk
import sqlite3


os.chdir('../../Data')
os.getcwd()

OHCO = ['year', 'month', 'num_day', 'weekday', 'section', 'docID', 'sentence_id', 'token_id']
ARTICLE = OHCO[:6]
SENTS = OHCO[:7]


#with sqlite3.connect('WSJ-Monthly.db') as db:
#    test = pd.read_sql('SELECT * FROM [1999 April]', db, index_col=OHCO[2:])


#test.head()

#test['month'] = 'April'

#test

# test2 = test.set_index('month')
# test3 = test.reset_index(level=OHCO[2:])

# test4 = test3.set_index(OHCO[1:])

# test.index.names = OHCO[1:]

# test['num_day'] = test.index.loc[num_day]

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
        
# april = reorient_things('1999 April')



# could use the function from before to get this, but just copied output for time's sake
list_keys = ['1999 January', '1999 February', '1999 March', '1999 April', '1999 May', '1999 June', '1999 July', '1999 August', '1999 September', '1999 October', '1999 November', '1999 December', '2000 January', '2000 February', '2000 March', '2000 April', '2000 May', '2000 June', '2000 July', '2000 August', '2000 September', '2000 October', '2000 November', '2000 December', '2001 January', '2001 February', '2001 March', '2001 April', '2001 May', '2001 June', '2001 July', '2001 August', '2001 September', '2001 October', '2001 November', '2001 December', '2002 January', '2002 February', '2002 March', '2002 April', '2002 May', '2002 June', '2002 July', '2002 August', '2002 September', '2002 October', '2002 November', '2002 December']


def reassemble(list_keys):
    for key_yearmonth in list_keys:
        tokens = reorient_things(key_yearmonth)
        with sqlite3.connect('WSJ-reindexed.db') as db:
                tokens.to_sql(str(key_yearmonth), db, if_exists='replace', index=True)
        print("Finished with {}".format(key_yearmonth))
reassemble(list_keys)




def reassemble2(list_keys):
    '''
    @ nick, I was thinking we might be able to actually do this in one step if we were to use this
    '''
    for key_yearmonth in list_keys:
        tokens = reorient_things(key_yearmonth)
        with sqlite3.connect('WSJ-reindexed2.db') as db:
                tokens.to_sql('reindexed2', db, if_exists='append', index=True)
        print("Finished with {}".format(key_yearmonth))
# reassemble(list_keys)

# after this, stack in order and we should have full df!
################################################
with sqlite3.connect('WSJ-reindexed.db') as db:
    df1 = pd.read_sql('SELECT * FROM [1999 January]', db)

with sqlite3.connect('WSJ-reindexed.db') as db:
    df2 = pd.read_sql('SELECT * FROM [1999 February]', db)

with sqlite3.connect('WSJ-reindexed.db') as db:
    df3 = pd.read_sql('SELECT * FROM [1999 March]', db)

with sqlite3.connect('WSJ-reindexed.db') as db:
    df4 = pd.read_sql('SELECT * FROM [1999 April]', db)

with sqlite3.connect('WSJ-reindexed.db') as db:
    df5 = pd.read_sql('SELECT * FROM [1999 May]', db)

with sqlite3.connect('WSJ-reindexed.db') as db:
    df6 = pd.read_sql('SELECT * FROM [1999 June]', db)

with sqlite3.connect('WSJ-reindexed.db') as db:
    df7 = pd.read_sql('SELECT * FROM [1999 July]', db)

with sqlite3.connect('WSJ-reindexed.db') as db:
    df8 = pd.read_sql('SELECT * FROM [1999 August]', db)

with sqlite3.connect('WSJ-reindexed.db') as db:
    df9 = pd.read_sql('SELECT * FROM [1999 September]', db)

with sqlite3.connect('WSJ-reindexed.db') as db:
    df10 = pd.read_sql('SELECT * FROM [1999 October]', db)

with sqlite3.connect('WSJ-reindexed.db') as db:
    df11 = pd.read_sql('SELECT * FROM [1999 November]', db)

with sqlite3.connect('WSJ-reindexed.db') as db:
    df12 = pd.read_sql('SELECT * FROM [1999 December]', db)
############################################
with sqlite3.connect('WSJ-reindexed.db') as db:
    df13 = pd.read_sql('SELECT * FROM [2000 January]', db)

with sqlite3.connect('WSJ-reindexed.db') as db:
    df14 = pd.read_sql('SELECT * FROM [2000 February]', db)

with sqlite3.connect('WSJ-reindexed.db') as db:
    df15 = pd.read_sql('SELECT * FROM [2000 March]', db)

with sqlite3.connect('WSJ-reindexed.db') as db:
    df16 = pd.read_sql('SELECT * FROM [2000 April]', db)

with sqlite3.connect('WSJ-reindexed.db') as db:
    df17 = pd.read_sql('SELECT * FROM [2000 May]', db)

with sqlite3.connect('WSJ-reindexed.db') as db:
    df18 = pd.read_sql('SELECT * FROM [2000 June]', db)

with sqlite3.connect('WSJ-reindexed.db') as db:
    df19 = pd.read_sql('SELECT * FROM [2000 July]', db)

with sqlite3.connect('WSJ-reindexed.db') as db:
    df20 = pd.read_sql('SELECT * FROM [2000 August]', db)

with sqlite3.connect('WSJ-reindexed.db') as db:
    df21 = pd.read_sql('SELECT * FROM [2000 September]', db)

with sqlite3.connect('WSJ-reindexed.db') as db:
    df22 = pd.read_sql('SELECT * FROM [2000 October]', db)

with sqlite3.connect('WSJ-reindexed.db') as db:
    df23 = pd.read_sql('SELECT * FROM [2000 November]', db)

with sqlite3.connect('WSJ-reindexed.db') as db:
    df24 = pd.read_sql('SELECT * FROM [2000 December]', db)
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
with sqlite3.connect('WSJ-reindexed.db') as db:
    df37 = pd.read_sql('SELECT * FROM [2002 January]', db)

with sqlite3.connect('WSJ-reindexed.db') as db:
    df38 = pd.read_sql('SELECT * FROM [2002 February]', db)

with sqlite3.connect('WSJ-reindexed.db') as db:
    df39 = pd.read_sql('SELECT * FROM [2002 March]', db)

with sqlite3.connect('WSJ-reindexed.db') as db:
    df40 = pd.read_sql('SELECT * FROM [2002 April]', db)

with sqlite3.connect('WSJ-reindexed.db') as db:
    df41 = pd.read_sql('SELECT * FROM [2002 May]', db)

with sqlite3.connect('WSJ-reindexed.db') as db:
    df42 = pd.read_sql('SELECT * FROM [2002 June]', db)

with sqlite3.connect('WSJ-reindexed.db') as db:
    df43 = pd.read_sql('SELECT * FROM [2002 July]', db)

with sqlite3.connect('WSJ-reindexed.db') as db:
    df44 = pd.read_sql('SELECT * FROM [2002 August]', db)

with sqlite3.connect('WSJ-reindexed.db') as db:
    df45 = pd.read_sql('SELECT * FROM [2002 September]', db)

with sqlite3.connect('WSJ-reindexed.db') as db:
    df46 = pd.read_sql('SELECT * FROM [2002 October]', db)

with sqlite3.connect('WSJ-reindexed.db') as db:
    df47 = pd.read_sql('SELECT * FROM [2002 November]', db)

with sqlite3.connect('WSJ-reindexed.db') as db:
    df48 = pd.read_sql('SELECT * FROM [2002 December]', db)
    
#######################################################

# df1b = df1
dflist = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16, df17, df18, df19, df20, df21, df22, df23, df24, df25, df26, df27, df28, df29, df30, df31, df32, df33, df34, df35, df36, df37, df38, df39, df40, df41, df42, df43, df44, df45, df46, df47, df48]
total = df1.append(dflist[1:]) 

# run from here, double checking if total tail works --> should be Dec2002

total.tail()
total_backup = total # to save us a headache if we mess up total somehow
# total.reset_index(level=OHCO, inplace= True)  # ended up not needing this line -- there was some random index number for each row within a month





total.set_index(OHCO, inplace= True) # columns to multiindex





with sqlite3.connect('WSJ-full.db') as db: 
    total.to_sql("full", db, if_exists='replace', index=True)

# with sqlite3.connect('WSJ-thru-01.db') as db:
#         total.to_sql("thru-01", db, if_exists='replace', index=True)

# After creating the full indexed and tokenized df

with sqlite3.connect('WSJ-processed-full.db') as db:
    tokens = pd.read_sql('SELECT * FROM token', db, index_col = OHCO)

#with sqlite3.connect('WSJ-thru-01.db') as db:
#    tokens = pd.read_sql('SELECT * FROM thru-01', db)

tokens['punc'] = tokens.token_str.str.match(r'^[\W_]*$').astype('int')
tokens['num'] = tokens.token_str.str.match(r'^.*\d.*$').astype('int') # does this regex capture something like "1dot6"

WORDS = (tokens.punc == 0) & (tokens.num == 0) # BREAKS AFTER THIS!!!!!!!!!!!
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

stopwords = set(nltk.corpus.stopwords.words('english') ) # + extra_stopwords # add extra stopwords piece within the set if defined

sw = pd.DataFrame({'x':1}, index=stopwords)
vocab['stop'] = vocab.term_str.map(sw.x).fillna(0).astype('int')
del(sw)

tokens['term_id'] = tokens['term_str'].map(vocab.reset_index()\
    .set_index('term_str').term_id).fillna(-1).astype('int')

# with sqlite3.connect('WSJ-processed.db') as db:
#     tokens.to_sql('token', db, if_exists='replace', index=True)

# with sqlite3.connect('WSJ-processed.db') as db:
#     vocab.to_sql('vocab', db, if_exists='replace', index=True)


# with sqlite3.connect('WSJ-processed.db') as db:
#     tokens.to_sql('token', db, if_exists='replace', index=True)


tokens_test = tokens

tokens_test.year[1]

tokens_test99 = tokens_test.head()[tokens_test.head().year=='1999',:]
len(tokens_test)


OHCO2 = ['year'] # 'month', 'num_day', 'weekday', 'section', 'docID', 'sentence_id', 'token_id', 'term_id'] 
tokens.set_index(OHCO2, inplace= True)

# OHCO3 = ['year', 'month']


# tokens.tail().year
# tokens[year==1999]
# tokens3 = tokens.loc[5001:10000]
#tokens00 = tokens.loc[2000]
#tokens01 = tokens.loc[2001]
#tokens02 = tokens.loc[2002]






# with sqlite3.connect('WSJ-processed.db') as db:
#     tokens3.to_sql('token', db, if_exists='append', index=True, chunksize=1000)

# with sqlite3.connect('WSJ-processed.db') as db:
#     tokens00.to_sql('token', db, if_exists='append', index=True)

# with sqlite3.connect('WSJ-processed.db') as db:
#     tokens01.to_sql('token', db, if_exists='append', index=True)

# with sqlite3.connect('WSJ-processed.db') as db:
#     tokens02.to_sql('token', db, if_exists='append', index=True)

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


# create initial  overalldf using jan 99
# read in wtih sql for each mo after
# fix index at current yearmonth
# append current year month to overall df
