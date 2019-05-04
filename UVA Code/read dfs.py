
import pandas as pd
import os
import nltk
import sqlite3


os.chdir('../../Data')
os.getcwd()

OHCO = ['year', 'month', 'num_day', 'weekday', 'section', 'docID', 'sentence_id', 'token_id']
ARTICLE = OHCO[:6]
SENTS = OHCO[:7]


with sqlite3.connect('WSJ-Monthly.db') as db:
    test = pd.read_sql('SELECT * FROM [1999 April]', db, index_col=OHCO[2:])


test.head()

test['month'] = 'April'

test

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
        
april = reorient_things('1999 April')



# could use the function from before to get this, but just copied output for time's sake
list_keys = ['1999 January', '1999 February', '1999 March', '1999 April', '1999 May', '1999 June', '1999 July', '1999 August', '1999 September', '1999 October', '1999 November', '1999 December', '2000 January', '2000 February', '2000 March', '2000 April', '2000 May', '2000 June', '2000 July', '2000 August', '2000 September', '2000 October', '2000 November', '2000 December', '2001 January', '2001 February', '2001 March', '2001 April', '2001 May', '2001 June', '2001 July', '2001 August', '2001 September', '2001 October', '2001 November', '2001 December', '2002 January', '2002 February', '2002 March', '2002 April', '2002 May', '2002 June', '2002 July', '2002 August', '2002 September', '2002 October', '2002 November', '2002 December']


def reassemble(list_keys):
    for key_yearmonth in list_keys:
        tokens = reorient_things(key_yearmonth)
        with sqlite3.connect('WSJ-reindexed.db') as db:
                tokens.to_sql(str(key_yearmonth), db, if_exists='replace', index=True)
        print("Finished with {}".format(key_yearmonth))
reassemble(list_keys)

# after this, stack in order and we should have full df!

