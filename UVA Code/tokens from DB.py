
import pandas as pd
import os
import nltk
import sqlite3

os.chdir('../../Data')
os.getcwd()

OHCO = ['year', 'month', 'num_day', 'weekday', 'section', 'docID', 'sentence_id', 'token_id']
ARTICLE = OHCO[:6]
SENTS = OHCO[:7]


with sqlite3.connect('WSJ.db') as db:
    sents = pd.read_sql('SELECT * FROM benchmark', db, index_col=OHCO[:-1])

sents.head()

def to_years(list_of_years):
    '''
    Used to break up the dataframe into more feasible sizes before word level tokenization

    Input: list of years
    '''
    
    dictionary = {}
    for one_year in list_of_years:
        dictionary[one_year] = sents.loc[one_year]
    return dictionary

# test_dict = to_years([1999,2000,2001,2002])

# test_dict


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

test_dict_month = to_months([1999,2000,2001,2002])
list_keys = test_dict_month.keys()


def monthly_tokenizer(key_yearmonth):
    tokens = test_dict_month[key_yearmonth].sent_str\
        .apply(lambda x: pd.Series(nltk.word_tokenize(x)))\
        .stack()\
        .to_frame()\
        .rename(columns={0:'token_str'})
    tokens.index.names = OHCO[2:]
    return tokens


def full_monthly_tokenizer(list_keys): #### haven't called this, but it should run everything
    for key_yearmonth in list_keys:
        tokens = test_dict_month[key_yearmonth].sent_str\
            .apply(lambda x: pd.Series(nltk.word_tokenize(x)))\
            .stack()\
            .to_frame()\
            .rename(columns={0:'token_str'})
        tokens.index.names = OHCO[2:]
        with sqlite3.connect('WSJ-Monthly.db') as db:
            tokens.to_sql(str(key_yearmonth), db, if_exists='replace', index=True)


##################################################
Jan99 = monthly_tokenizer('1999 January')
with sqlite3.connect('WSJ-Monthly.db') as db:
    Jan99.to_sql('Jan99', db, if_exists='replace', index=True)

Feb99 = monthly_tokenizer('1999 February')
with sqlite3.connect('WSJ-Monthly.db') as db:
    Feb99.to_sql('Feb99', db, if_exists='replace', index=True)

Mar99 = monthly_tokenizer('1999 March')
with sqlite3.connect('WSJ-Monthly.db') as db:
    Mar99.to_sql('Mar99', db, if_exists='replace', index=True)

Apr99 = monthly_tokenizer('1999 April')
with sqlite3.connect('WSJ-Monthly.db') as db:
    Apr99.to_sql('Apr99', db, if_exists='replace', index=True)

May99 = monthly_tokenizer('1999 May')
with sqlite3.connect('WSJ-Monthly.db') as db:
    May99.to_sql('May99', db, if_exists='replace', index=True)

June99 = monthly_tokenizer('1999 June')
with sqlite3.connect('WSJ-Monthly.db') as db:
    June99.to_sql('June99', db, if_exists='replace', index=True)

July99 = monthly_tokenizer('1999 July')
with sqlite3.connect('WSJ-Monthly.db') as db:
    July99.to_sql('July99', db, if_exists='replace', index=True)

Aug99 = monthly_tokenizer('1999 August')
with sqlite3.connect('WSJ-Monthly.db') as db:
    Aug99.to_sql('Aug99', db, if_exists='replace', index=True)

Sept99 = monthly_tokenizer('1999 September')
with sqlite3.connect('WSJ-Monthly.db') as db:
    Sept99.to_sql('Sept99', db, if_exists='replace', index=True)

Oct99 = monthly_tokenizer('1999 October')
with sqlite3.connect('WSJ-Monthly.db') as db:
    Oct99.to_sql('Oct99', db, if_exists='replace', index=True)

Nov99 = monthly_tokenizer('1999 November')
with sqlite3.connect('WSJ-Monthly.db') as db:
    Nov99.to_sql('Nov99', db, if_exists='replace', index=True)

Dec99 = monthly_tokenizer('1999 December')
with sqlite3.connect('WSJ-Monthly.db') as db:
    Dec99.to_sql('Dec99', db, if_exists='replace', index=True)


##################################################

Jan00 = monthly_tokenizer('2000 January')
with sqlite3.connect('WSJ-Monthly.db') as db:
    Jan00.to_sql('Jan00', db, if_exists='replace', index=True)

Feb00 = monthly_tokenizer('2000 February')
with sqlite3.connect('WSJ-Monthly.db') as db:
    Feb00.to_sql('Feb00', db, if_exists='replace', index=True)

Mar00 = monthly_tokenizer('2000 March')
with sqlite3.connect('WSJ-Monthly.db') as db:
    Mar00.to_sql('Mar00', db, if_exists='replace', index=True)
    
Apr00 = monthly_tokenizer('2000 April')
with sqlite3.connect('WSJ-Monthly.db') as db:
    Apr00.to_sql('Apr00', db, if_exists='replace', index=True)
    
May00 = monthly_tokenizer('2000 May')
with sqlite3.connect('WSJ-Monthly.db') as db:
    May00.to_sql('May00', db, if_exists='replace', index=True)
    
June00 = monthly_tokenizer('2000 June')
with sqlite3.connect('WSJ-Monthly.db') as db:
    June00.to_sql('June00', db, if_exists='replace', index=True)
    
July00 = monthly_tokenizer('2000 July')
with sqlite3.connect('WSJ-Monthly.db') as db:
    July00.to_sql('July00', db, if_exists='replace', index=True)
    
Aug00 = monthly_tokenizer('2000 August')
with sqlite3.connect('WSJ-Monthly.db') as db:
    Aug00.to_sql('Aug00', db, if_exists='replace', index=True)
    
Sept00 = monthly_tokenizer('2000 September')
with sqlite3.connect('WSJ-Monthly.db') as db:
    Sept00.to_sql('Sept00', db, if_exists='replace', index=True)
    
Oct00 = monthly_tokenizer('2000 October')
with sqlite3.connect('WSJ-Monthly.db') as db:
    Oct00.to_sql('Oct00', db, if_exists='replace', index=True)
    
Nov00 = monthly_tokenizer('2000 November')
with sqlite3.connect('WSJ-Monthly.db') as db:
    Nov00.to_sql('Nov00', db, if_exists='replace', index=True)
    
Dec00 = monthly_tokenizer('2000 December')
with sqlite3.connect('WSJ-Monthly.db') as db:
    Dec00.to_sql('Dec00', db, if_exists='replace', index=True)
    
##################################################


Jan01 = monthly_tokenizer('2001 January')
with sqlite3.connect('WSJ-Monthly.db') as db:
    Jan01.to_sql('Jan01', db, if_exists='replace', index=True)

Feb01 = monthly_tokenizer('2001 February')
with sqlite3.connect('WSJ-Monthly.db') as db:
    Feb01.to_sql('Feb01', db, if_exists='replace', index=True)

Mar01 = monthly_tokenizer('2001 March')
with sqlite3.connect('WSJ-Monthly.db') as db:
    Mar01.to_sql('Mar01', db, if_exists='replace', index=True)
    
Apr01 = monthly_tokenizer('2001 April')
with sqlite3.connect('WSJ-Monthly.db') as db:
    Apr01.to_sql('Apr01', db, if_exists='replace', index=True)
    
May01 = monthly_tokenizer('2001 May')
with sqlite3.connect('WSJ-Monthly.db') as db:
    May01.to_sql('May01', db, if_exists='replace', index=True)
    
June01 = monthly_tokenizer('2001 June')
with sqlite3.connect('WSJ-Monthly.db') as db:
    June01.to_sql('June01', db, if_exists='replace', index=True)
    
July01 = monthly_tokenizer('2001 July')
with sqlite3.connect('WSJ-Monthly.db') as db:
    July01.to_sql('July01', db, if_exists='replace', index=True)
    
Aug01 = monthly_tokenizer('2001 August')
with sqlite3.connect('WSJ-Monthly.db') as db:
    Aug01.to_sql('Aug01', db, if_exists='replace', index=True)
    
Sept01 = monthly_tokenizer('2001 September')
with sqlite3.connect('WSJ-Monthly.db') as db:
    Sept01.to_sql('Sept01', db, if_exists='replace', index=True)
    
Oct01 = monthly_tokenizer('2001 October')
with sqlite3.connect('WSJ-Monthly.db') as db:
    Oct01.to_sql('Oct01', db, if_exists='replace', index=True)
    
Nov01 = monthly_tokenizer('2001 November')
with sqlite3.connect('WSJ-Monthly.db') as db:
    Nov01.to_sql('Nov01', db, if_exists='replace', index=True)
    
Dec01 = monthly_tokenizer('2001 December')
with sqlite3.connect('WSJ-Monthly.db') as db:
    Dec01.to_sql('Dec01', db, if_exists='replace', index=True)
    
##################################################


Jan02 = monthly_tokenizer('2002 January')
with sqlite3.connect('WSJ-Monthly.db') as db:
    Jan02.to_sql('Jan02', db, if_exists='replace', index=True)

Feb02 = monthly_tokenizer('2002 February')
with sqlite3.connect('WSJ-Monthly.db') as db:
    Feb02.to_sql('Feb02', db, if_exists='replace', index=True)

Mar02 = monthly_tokenizer('2002 March')
with sqlite3.connect('WSJ-Monthly.db') as db:
    Mar02.to_sql('Mar02', db, if_exists='replace', index=True)
    
Apr02 = monthly_tokenizer('2002 April')
with sqlite3.connect('WSJ-Monthly.db') as db:
    Apr02.to_sql('Apr02', db, if_exists='replace', index=True)
    
May02 = monthly_tokenizer('2002 May')
with sqlite3.connect('WSJ-Monthly.db') as db:
    May02.to_sql('May02', db, if_exists='replace', index=True)
    
June02 = monthly_tokenizer('2002 June')
with sqlite3.connect('WSJ-Monthly.db') as db:
    June02.to_sql('June02', db, if_exists='replace', index=True)
    
July02 = monthly_tokenizer('2002 July')
with sqlite3.connect('WSJ-Monthly.db') as db:
    July02.to_sql('July02', db, if_exists='replace', index=True)
    
Aug02 = monthly_tokenizer('2002 August')
with sqlite3.connect('WSJ-Monthly.db') as db:
    Aug02.to_sql('Aug02', db, if_exists='replace', index=True)
    
Sept02 = monthly_tokenizer('2002 September')
with sqlite3.connect('WSJ-Monthly.db') as db:
    Sept02.to_sql('Sept02', db, if_exists='replace', index=True)
    
Oct02 = monthly_tokenizer('2002 October')
with sqlite3.connect('WSJ-Monthly.db') as db:
    Oct02.to_sql('Oct02', db, if_exists='replace', index=True)
    
Nov02 = monthly_tokenizer('2002 November')
with sqlite3.connect('WSJ-Monthly.db') as db:
    Nov02.to_sql('Nov02', db, if_exists='replace', index=True)
    
Dec02 = monthly_tokenizer('2002 December')
with sqlite3.connect('WSJ-Monthly.db') as db:
    Dec02.to_sql('Dec02', db, if_exists='replace', index=True)
    





# test_dict_month['1999 January']

# tokens = test_dict_month['1999 January'].sent_str\
#     .apply(lambda x: pd.Series(nltk.word_tokenize(x)))\
#     .stack()\
#     .to_frame()\
#     .rename(columns={0:'token_str'})
# tokens.index.names = OHCO[2:]



# just_99 = sents.loc[1999]
# just_00 = sents.loc[2000]
# just_01 = sents.loc[2001]
# just_02 = sents.loc[2002]


# len(just_99) + len(just_00) + len(just_01) + len(just_02) # testing total rows

# tokens = just_99.sent_str\
#     .apply(lambda x: pd.Series(nltk.word_tokenize(x)))\
#     .stack()\
#     .to_frame()\
#     .rename(columns={0:'token_str'})
# tokens.index.names = OHCO