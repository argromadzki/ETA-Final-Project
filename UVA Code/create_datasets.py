
import pandas as pd
import os

os.chdir('../../Data')
os.getcwd()


def create_df(year):
    df = pd.read_csv('{}_clean.csv'.format(year), header=-1)
    df.columns =  ['doc_number', 'year', 'month', 'num_day', 'weekday', 'section', 'text']
    df = get_ID(df)
    df = strip_docID(df)
    # df['year'] = year
    return df

# test = create_df(2000)
# sum(test.docID.isnull())

# create columns of factiva ID's
def get_ID(df):
    df['docID'] = df.text.apply(lambda x: x.split()[-1:][0])
    return df

def strip_docID(df):
    df['text'] = df.text.apply(lambda x: ' '.join(x.split()[:-2]))
    return df

# df2000 = create_df(2000)
# df1999 = create_df(1999)
# df1999

# test_99_00 = [df1999,df2000]

# combo_df = df1999.append(df2000) # .append(df2001).append(df2002)
# combo_df

def combine_years(list_of_dfYears):
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
    list_of_dfs = []
    for year in list_of_years:
        print('\nProcessing CSV for year {}...'.format(year))
        current_df = create_df(year)
        list_of_dfs.append(current_df)
    big_df = combine_years(list_of_dfs)
    return big_df


list_years = [1999,2000,2002]
df = make_multiple_dfs(list_years).reset_index(drop=True)
pd.set_option('display.max_columns', 8)

df['fulldate'] = df["month"] + "-" + df["num_day"].map(str) + "-" + df["year"].map(str)

df['fulldate'] = pd.to_datetime(df['fulldate'])

df.sort_values(by=['fulldate', 'weekday', 'section'], inplace= True)

df.iloc[1].text

####################################################
OHCO = ['year', 'month', 'num_day', 'weekday', 'section', 'docID']
dfi = df.set_index(OHCO)
df_texts = dfi['text']
df_texts

# df_texts.index[0] # gets the indices of the first thing
# df_texts.index.get_level_values('section')
