
import pandas as pd
import os
import nltk
import sqlite3
import regex

os.chdir('../../Data/')
os.getcwd()


tokens = pd.read_csv('WSJ-processed-full.csv')


# teststring = "asdf93_-2    test."
# teststring2 = "asdf93_-2    test."
# teststring3 = "asdf93_-2    test."

# testdf =pd.DataFrame([teststring,teststring2,teststring3])
# testdf.columns = ['token_str']
# testdf['term_str'] = None
# testdf.loc[:,'term_str'] = testdf.token_str.str.lower()\
#     .str.replace(r'["'"'"'_*.\\\/\d\s -]', '')



WORDS = (tokens.punc == 0) & (tokens.num == 0) 
tokens.loc[WORDS, 'term_str'] = tokens.token_str.str.lower()\
    .str.replace(r'["'"'"'_*.\\\/\d\s -]', '') # alex says "sorry raf for this ugly expression"

vocab = tokens[tokens.punc == 0].term_str.value_counts().to_frame()\
    .reset_index()\
    .rename(columns={'index':'term_str', 'term_str':'n'})
vocab = vocab.sort_values('term_str').reset_index(drop=True)
vocab.index.name = 'term_id'

vocab['p'] = vocab.n / vocab.n.sum()

stemmer = nltk.stem.porter.PorterStemmer()
vocab['port_stem'] = vocab.term_str.apply(lambda x: stemmer.stem(x))


extra_stopwords = ["'s", "said", "mr", "'t", "new", "says", "company", "million", "year", "us", "would", "one", "also", "billion", "inc", "market", "last", "companies", "could", "stock", "two", "years", "first", "people", "time", "business", "many", "york", "cents", "president", "share", "corp", "may", "group", "street", "sales", "even", "wall", "n", "journal", "week", "shares", "much", "say", "government", "still", "like", "investors", "since", "world", "quarter", "staff", "chief", "price", "make", "months", "bush", "three", "get", "trading", "bank", "state", "financial", "industry", "deal", "according", "days", "american", "expected", "analysts", "firm", "money", "big", "back", "executive", "funds", "revenue", "economy", "work", "earnings", "officials", "next", "federal", "co", "made", "prices", "ms", "growth", "well", "rate", "month", "stocks", "yesterday", "including", "take", "fell", "way", "cut", "parttechnology", "fund", "past", "reporter", "international", "among", "day", "rose", "another", "exchange", "economic", "house", "see", "recent", "earlier", "unit", "tax", "trade", "use", "public", "washington", "end", "service", "'re", "high", "rates", "investment", "services", "markets", "ago", "news", "though", "net", "home", "help", "major", "several", "na", "going", "already", "long", "chairman", "index", "pay", "securities", "sept", "system", "nasdaq", "customers", "income", "less", "capital", "former", "general", "buy", "far", "value", "report", "online", "number", "today", "used", "loss", "early", "debt", "court", "average", "board", "city", "based", "interest", "might", "office", "call", "costs", "five", "top", "without", "case", "research", "four", "go", "likely", "around", "move", "employees", "products", "points", "small", "sellpm", "information", "least", "want", "set", "late", "profit", "come", "results", "think", "department", "second", "current", "global", "put", "close", "credit", "called", "director", "six", "little", "issue", "data", "total", "point", "period", "site", "software", "assets", "weeks", "management", "better", "offer", "friday", "need", "yet", "spending", "center", "reported", "found", "war", "recently", "maker", "however", "compared", "program", "workers", "spokesman", "corporate", "operations", "right", "largest", "sold", "old", "john", "large", "firms", "expects", "law", "making", "bill", "times", "due", "cost", "show", "foreign", "higher", "offering", "lower", "systems", "march", "health", "local", "best", "increase", "executives", "support", "bonds", "demand", "change", "half", "political", "later", "white", "nearly", "europe", "ca", "analyst", "party"]
stopwords = set(nltk.corpus.stopwords.words('english') + extra_stopwords) # add extra stopwords piece within the set if defined

sw = pd.DataFrame({'x':1}, index=stopwords)
vocab['stop'] = vocab.term_str.map(sw.x).fillna(0).astype('int')
del(sw)

tokens['term_id'] = tokens['term_str'].map(vocab.reset_index()\
    .set_index('term_str').term_id).fillna(-1).astype('int')


vocab.to_csv('redo_vocab.csv')
tokens.to_csv('redo_tokens.csv')
