
import pandas as pd
import os

os.chdir('../../Data')
os.getcwd()


def create_df(year):
    df = pd.read_csv('{}_clean.csv'.format(year), header=0)
    df.columns =  ['doc_number', 'section', 'text'] # assuming we only have df with these features
                                                    # (update to add year to reflect CSVs)
    df = get_ID(df)
    df = strip_docID(df)
    df['Year'] = year
    return df

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

combo_df_v2 = make_multiple_dfs([1999,2000])