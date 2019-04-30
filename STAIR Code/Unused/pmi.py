# encoding: utf-8

# pmi.py
# by Maurits van der Veen
# last modified 2017-10-03

# Functions to find top co-occurring words using pointwise mutual information (PMI).
# Uses:
# - expand list of top words for a corpus to use in filtering etc.
# - corpus-specific expansion of valence dictionary

# TODO: write a function to add particular words/phrases that are not in the global dictionary
# This can happen when the occurrence rate falls below mincount/roundsize


# ******************************* (s)top word info ***************************

# For Muslim corpus, found 240487 distinct words, with top frequencies as follows:

# with is in 0.898493 of corpus
# who is in 0.809798 of corpus
# an is in 0.849439 of corpus
# as is in 0.899129 of corpus
# at is in 0.836452 of corpus
# is is in 0.943224 of corpus
# it is in 0.897528 of corpus
# in is in 0.990918 of corpus
# but is in 0.847706 of corpus
# xpossess is in 0.908628 of corpus
# this is in 0.771538 of corpus
# on is in 0.925169 of corpus
# he is in 0.771231 of corpus
# been is in 0.756927 of corpus
# and is in 0.988263 of corpus
# to is in 0.993506 of corpus
# that is in 0.939583 of corpus
# . is in 1.000000 of corpus
# has is in 0.835684 of corpus
# have is in 0.871421 of corpus
# from is in 0.855889 of corpus
# of is in 0.994077 of corpus
# not is in 0.869184 of corpus
# muslim is in 0.773951 of corpus
# a is in 0.990128 of corpus
# for is in 0.946756 of corpus
# be is in 0.861418 of corpus
# they is in 0.780861 of corpus
# the is in 0.998684 of corpus
# was is in 0.872869 of corpus
# by is in 0.907531 of corpus
# are is in 0.842836 of corpus


# ****************************** frequency distributions *********************

def freqDist(aText, skipperiods=True):
    """Generate a term frequency dictionary."""
    # textwords = aText.split()
    # if skipperiods:
    #     textwords = [x for x in textwords if x != '.']
    # return freqDistList(textwords)
    return freqDistList(aText.replace('.', '').split() if skipperiods else aText.split())


def freqDist_ngram(aText, ngram, skipperiods=True):
    """Generate a term frequency dictionary for multi-word phrases."""
    textwords = aText.replace('.', '').split() if skipperiods else aText.split()
    ngramX = ngram-1
    ngrams = []
    for index in range(ngramX, len(textwords)):
        ngrams.append(' '.join(textwords[index-ngramX:index+1]))
    return freqDistList(ngrams)


def freqDistList(aList):
    """Generate a term frequency dictionary."""
    fD = {}
    for aWord in aList:
        if aWord in fD:
            fD[aWord] += 1
        else:
            fD[aWord] = 1
    return fD, len(aList)


def addFreqDist(freqD, newFD):
    """Merge new frequency dict into corpus data on frequency."""
    for key, val in newFD.iteritems():
        if key in freqD:
            freqD[key] += val
        else:
            freqD[key] = val
    return freqD


def addFreqDistX(freqD, presenceD, newFD):
    """Merge new frequency dict into corpus data on frequency & presence."""
    for key, val in newFD.iteritems():
        if key in freqD:
            freqD[key] += val
        else:
            freqD[key] = val
        if key in presenceD:
            presenceD[key] += 1
        else:
            presenceD[key] = 1
    return freqD, presenceD


def mergeFreqDist(fd1, fd2):
    """Merge two frequency distributions.

    Note: to save memory, overwrite fd1 in-place,
    so be aware of that in the calling function!
    """
    if fd1 == dict():
        return fd2
    elif fd2 == dict():
        return fd1
    else:
        for key, val in fd2.iteritems():
            if key in fd1:
                fd1[key] += val
            else:
                fd1[key] = val
    return fd1


def mergeFreqDist_copy(fd1, fd2):
    """Merge two frequency distributions.

    Note: Create new copy to avoid confusion.
    """
    newFD = dict()
    for key, val in fd1.iteritems():
        newFD[key] = val
    for key, val in fd2.iteritems():
        if key in newFD:
            newFD[key] += val
        else:
            newFD[key] = val
    return newFD


def mergeFreqDists_copy(fdlist):
    """Merge multiple frequency distributions.

    Note: Create new copy to avoid confusion.
    """
    newFD = dict()
    for fd in fdlist:
        for key, val in fd.iteritems():
            if key in newFD:
                newFD[key] += val
            else:
                newFD[key] = val
    return newFD


def corpusFreqDist(texts):
    """Calculate frequency distribution as well as nr articles in which
    a word occurs, for all words in the corpus."""
    fD, pD = {}, {}
    for aText in texts:
        fD, pD = addFreqDistX(fD, pD, freqDist(aText)[0])
    return fD, pD


def saveFD(filename, corpusFD, corpusPD, nrdocs, nrwords):
    """Save frequency distribution information for a corpus."""
    import pickle

    with open(filename, 'wb') as picklefile:
        pickle.dump((corpusFD, corpusPD, nrdocs, nrwords), picklefile)
    return filename


def combineFDs(filelist, outputfile):
    """Combine multiple frequency distribution files."""
    import pickle

    combinedFD = dict()
    combinedPD = dict()
    comb_nrdocs, comb_nrwords = 0, 0
    for fdfile in filelist:
        with open(fdfile, 'rb') as fdpickle:
            corpusFD, corpusPD, nrdocs, nrwords = pickle.load(fdpickle)
            combinedFD = mergeFreqDist_copy(combinedFD, corpusFD)
            combinedPD = mergeFreqDist_copy(combinedPD, corpusPD)
            comb_nrdocs += nrdocs
            comb_nrwords += nrwords

    with open(outputfile, 'wb') as picklefile:
        pickle.dump((combinedFD, combinedPD, comb_nrdocs, comb_nrwords), picklefile)
    return outputfile


def convertFD2csv(picklefile, outputfile, whichFD='count'):
    """Convert a pickle-file frequency dictionary to a csv version."""
    import csv
    import pickle

    with open(picklefile, 'rb') as fdpickle:
        corpusFD, corpusPD, nrdocs, nrwords = pickle.load(fdpickle)
    targetFD = corpusFD if whichFD == 'count' else corpusPD
    with open(outputfile, 'wb') as outf:
        outwriter = csv.writer(outf)
        outwriter.writerow(('nrdocs', nrdocs))
        outwriter.writerow(('nrwords', nrwords))
        for key, value in targetFD.iteritems():
            outwriter.writerow((key, value))


def readFD_csv(csvfile, whichFD='count'):
    """Load a csv-format dictionary file."""
    import csv

    freqdict = {}
    with open(csvfile, 'rU') as inf:
        inreader = csv.reader(inf)

        # Get overall information from first 2 lines
        docrow = next(inreader)
        if docrow[0] != 'nrdocs':
            print "Error, expected to find number of docs in first row."
            return
        nrdocs = int(docrow[1])

        wordrow = next(inreader)
        if wordrow[0] != 'nrwords':
            print "Error, expected to find number of words in second row."
            return
        nrwords = int(wordrow[1])

        # Run through rest of csv file
        for key, value in inreader:  # note: assumes each row has 2 column values only!!
            freqdict[key] = int(value)

    # Figure out which dictionary is wanted
    if whichFD == 'count':
        return freqdict, dict(), nrdocs, nrwords
    else:  # presence dictionary
        return dict(), freqdict, nrdocs, nrwords


def calc_corpusFD(corpusfile, picklefile, textcol=1, skipperiods=True, capsmatter=False):
    """Calculate frequency & presence data for the overall corpus."""
    import csv
    import sys
    csv.field_size_limit(1000000000)

    # initialize frequency & presence dictionaries
    corpusFD, corpusPres = {}, {}
    totalwords = 0

    # Run through the file
    with open(corpusfile, 'rU') as corpusdata:
        for count, row in enumerate(csv.reader(corpusdata)):
            text = row[textcol] if capsmatter else row[textcol].lower()
            fd, nrwords = freqDist(text, skipperiods)
            totalwords += nrwords
            corpusFD, corpusPres = addFreqDistX(corpusFD, corpusPres, fd)
            if count % 100000 == 0:
                print "Processing document", count
        nrdocs = count + 1

    # Finish up
    print "Processed %d texts containing a total of %d words" % (nrdocs, totalwords)
    saveFD(picklefile, corpusFD, corpusPres, nrdocs, totalwords)
    return nrdocs, totalwords


def pmi2DTmatrix(textsfiles, pmifiles,
                 pmithres=0.05, mindf=1, maxdf=1,
                 corpusheader=False, textcol=1, pmicol=1,
                 capsmatter=False, singlewords=True, combine_parts=True,
                 translate2US=False, dialectfile='',
                 replacefile='', translationfunction=None):
    """Load texts & convert to document-term matrix after filtering on pmi."""
    import csv
    from operator import itemgetter
    import translate
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    csv.field_size_limit(1000000000)

    # Would it work well to use a regular expression for the filtering?
    # Would want to do an re search on word boundaries for any words other than those we have.
    # Worth testing execution speed; although it's already pretty fast

    # from the sklearn documentation:
    # While the TruncatedSVD transformer works with any(sparse)
    # feature matrix, using it on tf–idf matrices is recommended over
    # raw frequency counts in an LSA / document processing setting.
    # Moreover, sublinear scaling and inverse document frequency
    # should be turned on (sublinear_tf=True, use_idf=True)
    # to bring the feature values closer to a Gaussian distribution,
    # compensating for LSA’s erroneous assumptions about textual data.

    # Load all texts -- might become problematic for large corpora!
    texts = []
    for textsfile in textsfiles:
        with open(textsfile, 'rU') as textsf:
            textdata = csv.reader(textsf)
            if corpusheader:
                dummy = next(textdata)

            if combine_parts: # combine multi-part texts by id
                curid = -1
                curtexts = []
                for row in textdata:
                    rowtext = row[textcol] if capsmatter else row[textcol].lower()
                    if row[0] != curid:
                        if len(curtexts) > 0:
                            texts.append(' '.join(curtexts))
                        curid = row[0]
                        curtexts = [rowtext,]
                    else:
                        curtexts.append(rowtext)
                # flush last text
                if len(curtexts) > 0:
                    texts.append(' '.join(curtexts))

            else:  # simply take every row as an item
                for row in textdata:
                    texts.append(row[textcol] if capsmatter else row[textcol].lower())
    print "Working with %d texts" % (len(texts))

    # Load all pmi words exceeding pmithres; convert to set
    pmilist = []
    pmiwordcol = 0  # This should always be the case, so safe to hardcode

    for pmifile in pmifiles:
        with open(pmifile, 'rU') as pmif:
            for pmidata in csv.reader(pmif):
                if float(pmidata[pmicol]) >= pmithres:
                    termwords = pmidata[pmiwordcol].split()
                    if not singlewords or len(termwords) == 1:
                        for pmiword in termwords:  # handle ngrams
                            try:
                                isnum = float(pmiword)
                            except:  # can't convert to float -> has characters
                                pmilist.append(pmiword if capsmatter else pmiword.lower())
    pmiset = set(pmilist)  # eliminate duplicates

    # Translate & merge words if necessary, for comparison across UK & US texts
    # E.g. If we have 'naturalisation' in a UK source we should leave 'naturalization'
    # in as well, and if we have both, we want to merge them
    if translate2US:
        pmiwords = ' '.join(pmiset)
        translatedwords = translate.translate_1text_only(pmiwords, dialectfile,
                                                         replacefile, translationfunction)
        # See which words have changed.
        # Note: this assumes all mappings are 1-word to 1-word (no phrases)!!
        # If that is not true, need to set up loop differently (possible, but less elegant)
        mappings = {}
        for origword, translword in zip(pmiwords.split(), translatedwords.split()):
            if origword != translword:
                mappings[origword] = translword
                print "Mapping %s to %s" % (origword, translword)
        translatedset = set(translatedwords.split())

    filterednr = len(translatedset) if translate2US else len(pmiset)
    print "Filtering down to %d terms" % (filterednr,)

    # Filter all texts by pmi words
    filteredtexts = [' '.join(word for word in text.split() if word in pmiset) \
                     for text in texts]
    # Map translated words as appropriate
    if translate2US:
        filteredtexts = [' '.join([mappings.get(word, word) for word in text.split()]) \
                         for text in filteredtexts]

    # Vectorize texts
    # No need to filter stopwords, since we're already down to pmi words
    vectorizer = CountVectorizer(ngram_range=(1, 1),
                                 min_df=mindf, max_df=maxdf,
                                 stop_words=None)
    try:
        X = vectorizer.fit_transform(filteredtexts)
    # Remove inclusion restrictions if corpus otherwise too small
    except ValueError:
        vectorizer = CountVectorizer(ngram_range=(1, 1),
                                     min_df=1, max_df=1.0,
                                     stop_words=None)
        X = vectorizer.fit_transform(texts)
    # Get the words matching the index for the entries in the vectorized matrix
    vocab = [v[0] for v in sorted(vectorizer.vocabulary_.items(), key=itemgetter(1))]

    # Now generate 2 different document-term matrices:
    # (both use the sub-linear adjustment to word counts)
    # 1. tfidf
    vectorTransformer1 = TfidfTransformer(use_idf=True, sublinear_tf=True)
    DT_tfidf = vectorTransformer1.fit_transform(X)
    # 2. adjusted count
    vectorTransformer2 = TfidfTransformer(use_idf=False, sublinear_tf=True)
    DT_adjcount = vectorTransformer2.fit_transform(X)

    return texts, vocab, DT_tfidf, DT_adjcount


def write_DTplusmeta(outputfile, DTreduced, corpusmetas, metaheader=True):
    """Take the document-topic matrix and combine it with metadata."""
    import csv
    csv.field_size_limit(1000000000)

    # Load all metadata -- might become problematic for large corpora!
    metadata = []
    for metafile in corpusmetas:
        with open(metafile, 'rU') as metaf:
            meta_in = csv.reader(metaf)
            if metaheader:
                header_info = next(meta_in)
            metadata += [row for row in meta_in]

    # Double-check number of docs for which we have info
    if len(metadata) != len(DTreduced):
        print "Error: doc-topic matrix has %d entries; have metadata for %d" % \
              (len(DTreduced), len(metadata))

    # Write out results
    header_data = header_info if metaheader else \
                        ['metadata' + str(counter) for counter in range(len(metadata[0]))]
    header_data += ['topic' + str(counter) for counter in range(len(DTreduced[0]))]
    with open(outputfile, 'wb') as outf:
        outwriter = csv.writer(outf)
        outwriter.writerow(header_data)
        outwriter.writerows([x + list(y) for x, y in zip(metadata, DTreduced)])
    return


def get_topics(DTmatrix, topicmodel='NMF', nrtopics=5,
               hyperparams=()):
    """Perform Latent Semantic Analysis or LDA on a document-term matrix."""

    from sklearn.decomposition import NMF, TruncatedSVD
    from sklearn.decomposition import LatentDirichletAllocation

    random_init = 42

    if topicmodel == 'SVD':
        tm = TruncatedSVD(nrtopics)
    elif topicmodel == 'NMF':
        tm = NMF(nrtopics, random_state=random_init, alpha=.1, l1_ratio=.5, init='nndsvd')
    else:  # LDA
        if len(hyperparams) == 3:
            max_iter = hyperparams[0]
            dt_prior = hyperparams[1]
            tw_prior = hyperparams[2]
        else:
            max_iter=100
            dt_prior = 0.1
            tw_prior = 0.001
        tm = LatentDirichletAllocation(n_components=nrtopics, max_iter=max_iter,
                                       doc_topic_prior=dt_prior, topic_word_prior=tw_prior,
                                       learning_method='online', learning_offset=20.,
                                       random_state=random_init)
    themodel = tm.fit(DTmatrix)
    DTreduced = themodel.transform(DTmatrix)
    # topicwords = themodel.components_
    return themodel, DTreduced


def display_topics(texts, tmW, tmH, vocab, nrwords=10, nrtexts=3):
    """Display top words for each topic, along with text snippets reflecting topics."""
    import numpy as np

    # Loop over the topics
    for topic_nr, topic in enumerate(tmH):

        # Get top n words, using argsort on np array of words
        # Earlier code, less elegant (though maybe marginally faster):
        #    topwords = [vocab[i] for i in topic.argsort()[:-nrwords - 1:-1]]
        topwords = [vocab[i] for i in topic.argsort()[::-1][:nrwords]]
        print "\nTopic %d: %s" % (topic_nr, ", ".join(topwords))

        # Get indices for texts that have the highest proportion of this topic
        top_doc_indices = np.argsort(tmW[:,topic_nr])[::-1]

        # Now get the original, unfiltered texts for these indices
        # Note: in case of duplicates, skip
        snippetsfound = []
        for doc_index in top_doc_indices:
            doc_text_sents = texts[doc_index].split('.')
            snippet = '.'.join(doc_text_sents[:2])  # title & opening sentence
            if len(doc_text_sents) > 2:
                snippet += ' ' + bestsnippet('.'.join(doc_text_sents[2:]), topwords)
            if snippet not in snippetsfound and len(snippet.split()) > 30:
                snippetsfound.append(snippet)
                if len(snippetsfound) == nrtexts:
                    break  # exit for loop
        for snippet in snippetsfound:
            print '\n', snippet


def bestsnippet(text, keywords, length=60):
    """Return snippet of specified length with largest keyword count in text."""

    # Note: right now assumes everything lower-case.
    # Improve to handle capitalization correctly.

    keys = set(keywords)
    textlen = len(text)
    if textlen <= length:
        return text

    textwords = text.split()
    # Mark locations of keywords in text
    textvals = [1 if word in keys else 0 for word in textwords]
    # Sum # keywords in all snippets of length
    textvals_sum = [(sum(textvals[x-length:x]), x) for x in range(length, textlen+1)]
    # Get the largest # (don't worry if tie)
    bestind = sorted(textvals_sum)[-1][1]

    # Return appropriate snippet
    return ' '.join(textwords[bestind-length:bestind])


def vectorize2FD(texts, ngrams_min=1, ngrams_max=1,
                 min_df=1, max_df=1,
                 stopwords=None):
    """Helper to calc_corpusFD_ngrams - vectorize text & turn into freq dict."""

    import numpy as np
    from operator import itemgetter
    from sklearn.feature_extraction.text import CountVectorizer

    # Make document frequency adjustments for very small document sets
    if len(texts) == 0:
        print "Problem: no texts in corpus (returning empty dictionary."
        return dict()
    if len(texts) <= min_df:
        min_df = 1
        max_df = 1.0

    vectorizer = CountVectorizer(ngram_range=(ngrams_min, ngrams_max),
                                 min_df=min_df, max_df=max_df,
                                 stop_words=stopwords)  # binary=True to make it binary
    try:
        X = vectorizer.fit_transform(texts)
    except ValueError:  # happens when corpus too small -> eliminate inclusion restrictions
        min_df = 1
        max_df = 1.0
        vectorizer = CountVectorizer(ngram_range=(ngrams_min, ngrams_max),
                                     min_df=min_df, max_df=max_df,
                                     stop_words=stopwords)  # binary=True to make it binary
        X = vectorizer.fit_transform(texts)

    freq = np.ravel(X.sum(axis=0))  # sum each column to get total counts for each word
    # repeat above steps with 'binary=True' if calcpresence is True

    # Vocabulary is a dict mapping words to index in the vectorized data
    # To make sure ordering matches after converting from dict to items(), sort on index
    vocab = [v[0] for v in sorted(vectorizer.vocabulary_.items(), key=itemgetter(1))]
    # Turn into frequency distribution dictionary
    return dict(zip(vocab, freq))

    # vectorizer = CountVectorizer(ngram_range=(ngrams_min, ngrams_max),
    #                              min_df=minprescount, max_df=max_df,
    #                              stop_words=stopwords)  # binary=True to make it binary
    # X = vectorizer.fit_transform(roundtexts)
    # freq = np.ravel(X.sum(axis=0))  # sum each column to get total counts for each word
    # # repeat above steps with 'binary=True' if calcpresence is True
    #
    # # Get vocabulary keys, sorted by value
    # vocab = [v[0] for v in sorted(vectorizer.vocabulary_.items(), key=itemgetter(1))]
    # # Turn into frequency distribution dictionary
    # freqdist = dict(zip(vocab, freq))


def calc_corpusFD_ngrams(corpusfile, picklefile,
                         selectionfile='', selectioncol=-1, selectedfile='',
                         ngrams_min=1, ngrams_max=1, textcol=1,
                         roundsize=50000, minprescount=1,
                         calcpresence=False, stripterms=(), wild='*',
                         skipperiods=True, skipstopwords=False,
                         capsmatter=False,
                         corpusheader=False, selectionheader=True):
    """Calculate frequency & presence data for a corpus."""
    import csv
    # import numpy as np
    import os
    import re
    # from operator import itemgetter
    # from sklearn.feature_extraction.text import CountVectorizer

    max_df = 0.995 if skipperiods else 1  # periods should occur in just about every text
    # From list at the top of this module, it is likely this will also filter out:
    # in, and, to, of, the, a (all around 0.99) as well as: is, that, for (all around 0.95)
    # these are also all in the English stopwords list below, so the net effect should be
    # periods only
    stopwords = 'english' if skipstopwords else None

    corpusFD = {}
    roundtexts = []
    totalterms = 0
    included = 0

    re_strips = [re.sub('\\' + wild, '[\w-]*', stripterm) for stripterm in stripterms]
    # keys (with wildcard) must be bounded by word boundaries
    re_strips = ['\\b' + re_strip + '\\b' for re_strip in re_strips]

    # includeids = []
    includerows = []
    if selectionfile != '' and selectioncol != -1:
        # print "Reading in selection criterion..."
        with open(selectionfile, 'rU') as featuref:
            featuredata = csv.reader(featuref)
            if selectionheader:
                skipheader = next(featuredata)
            for counter, row in enumerate(featuredata):
                if len(row) <= selectioncol:
                    print "Error: line %d has no data in column %d (the selection column)" % (counter, selectioncol)
                includerows.append(int(row[selectioncol]) > 0)
                # includeids.append(row[0])

        nrwithfeatureinfo = len(includerows)
        print "Number of articles for which we have feature information: %d" % (nrwithfeatureinfo,)
    else:
        nrwithfeatureinfo = -1

    if selectedfile == '':
        writeoutput = False
        outputfile = 'empty.csv'
    else:
        writeoutput = True
        outputfile = selectedfile

    # print "Writing out selected articles..."
    with open(corpusfile, 'rU') as corpusf, open(outputfile, 'wb') as outf:
        corpusdata = csv.reader(corpusf)
        outwriter = csv.writer(outf)
        if corpusheader:
            skipheader = next(corpusdata)
        for counter, row in enumerate(corpusdata):
            if len(row) <= textcol:
                print "Error: line %d has no data in column %d (the text column)" % (counter, textcol)
            # if row[0] != includeids[counter]:
            #     print "id mismatch at row %d: %s - %s" % (counter, row[0], includeids[counter])
            if nrwithfeatureinfo > -1 and counter >= nrwithfeatureinfo:
                print "Error: exceeded number of texts for which we have feature info, at text nr %d" % (counter)
                print row[textcol]

            if selectioncol == -1 or includerows[counter]:
                if writeoutput:
                    outwriter.writerow(row)
                text = row[textcol] if capsmatter else row[textcol].lower()
                # Optionally filter out words from context
                for stripspec in re_strips:
                    text = re.sub(stripspec, '', text)
                # Now add to the contexts to track
                totalterms += len(text.split())
                roundtexts.append(text)
                included += 1

            if included % roundsize == 0 and included > 0 and len(roundtexts) > 0:
                freqdist = vectorize2FD(roundtexts, ngrams_min=ngrams_min, ngrams_max=ngrams_max,
                                        min_df=minprescount, max_df=max_df, stopwords=stopwords)
                corpusFD = mergeFreqDist_copy(corpusFD, freqdist)

                print "Processed %d texts; included %d" % (counter + 1, included)
                roundtexts = []
        try:
            nrtexts = counter + 1
        except:
            nrtexts = 0

    # Flush remaining texts
    if len(roundtexts) > 0:
        freqdist = vectorize2FD(roundtexts, ngrams_min=ngrams_min, ngrams_max=ngrams_max,
                                            min_df=minprescount, max_df=max_df, stopwords=stopwords)
        corpusFD = mergeFreqDist_copy(corpusFD, freqdist)

    print "Processed %d texts; included %d, containing %d distinct terms (and %d total)" % \
          (nrtexts, included, len(corpusFD), totalterms)
    presdist = {}  # should have an if statement here for calcpresence
    saveFD(picklefile, corpusFD, presdist, nrtexts, totalterms)

    # Finish up
    if selectedfile == '':
        try:
            os.remove(outputfile)
        except:
            pass

    return nrtexts, totalterms, corpusFD


def calc_contextFD_ngrams(corpusfile, contextfile, picklefile,
                          searchspec, window,
                          ngrams_min=1, ngrams_max=1, textcol=1,
                          roundsize=50000, minprescount=1,
                          calcpresence=False,
                          skipperiods=True, capsmatter=False):
    """Calculate frequency & presence data for a context specification."""
    import csv
    import numpy as np
    from operator import itemgetter
    from sklearn.feature_extraction.text import CountVectorizer

    max_df = 0.95 if skipperiods else 1  # periods should occur in just about every text
    # From list at the top of this module, it is likely this will also filter out:
    # in, and, to, of, the, a (all around 0.99) as well as: is, that, for (all around 0.95)
    # these are also all in the English stopwords list below, so the net effect should be
    # periods only

    contextFD = {}
    roundtexts = []
    totalterms = 0
    included = 0

    with open(corpusfile, 'rU') as corpusdata:
        for counter, row in enumerate(csv.reader(corpusdata)):

            text = row[textcol] if capsmatter else row[textcol].lower()
            totalterms += len(text.split())
            roundtexts.append(text)
            included += 1

            if (included) % roundsize == 0:
                # vectorize
                vectorizer = CountVectorizer(ngram_range=(ngrams_min, ngrams_max),
                                             min_df=minprescount, max_df=max_df,
                                             stop_words='english')  # binary=True to make it binary
                X = vectorizer.fit_transform(roundtexts)
                X = vectorizer.fit_transform(roundtexts)
                freq = np.ravel(X.sum(axis=0))  # sum each column to get total counts for each word
                # repeat above steps with 'binary=True' if calcpresence is True

                # Get vocabulary keys, sorted by value
                vocab = [v[0] for v in sorted(vectorizer.vocabulary_.items(), key=itemgetter(1))]
                # Turn into frequency distribution dictionary
                freqdist = dict(zip(vocab, freq))
                corpusFD = mergeFreqDist_copy(corpusFD, freqdist)

                print "Processed %d texts" % (counter + 1,)
                roundtexts = []

    nrtexts = counter
    presdist = {}  # should have an if statement here for calcpresence

    # Finish up
    print "Processed %d texts containing a total of %d terms" % (nrtexts, totalterms)
    saveFD(picklefile, corpusFD, presdist, nrtexts, totalterms)
    return nrtexts, totalterms, corpusFD


def calc_corpusFD_ngramsX(corpusfile, picklefile, ngrams=(1,),
                          textcol=1, mintotcount=1, minprescount=1,
                          skipperiods=True, capsmatter=False):
    """Calculate frequency & presence data for the overall corpus."""
    import csv
    import sys
    csv.field_size_limit(1000000000)

    # initialize frequency & presence dictionaries
    corpusFD, corpusPres = {}, {}
    totalterms = 0
    ngramterms = [0,] * len(ngrams)

    # Run through the file
    with open(corpusfile, 'rU') as corpusdata:
        for count, row in enumerate(csv.reader(corpusdata)):
            text = row[textcol] if capsmatter else row[textcol].lower()
            for ngramcount, ngram in enumerate(ngrams):
                fd, nrterms = freqDist_ngram(text, ngram, skipperiods)
                ngramterms[ngramcount] += nrterms
                totalterms += nrterms
                corpusFD, corpusPres = addFreqDistX(corpusFD, corpusPres, fd)
            if count % 100000 == 0:
                print "Processing document", count
        nrdocs = count + 1

    # Purge rare terms
    tobedeleted = []
    if mintotcount > 1:
        for key in corpusFD:
            if corpusFD[key] <  mintotcount:
                tobedeleted.append(key)
        print "Purging %d terms occurring fewer than %d times in the corpus" % (len(tobedeleted), mintotcount)
    elif minprescount > 1:
        for key in corpusPres:
            if corpusPres[key] < minprescount:
                tobedeleted.append(key)
        print "Purging %d terms present in fewer than %d texts" % (len(tobedeleted), minprescount)
    for key in tobedeleted:
        del corpusFD[key]
        del corpusPres[key]

    # Finish up
    print "Processed %d texts containing a total of %d terms" % (nrdocs, totalterms)
    saveFD(picklefile, corpusFD, corpusPres, nrdocs, totalterms)
    return nrdocs, totalterms, corpusFD


def merge_corpusFDs(fdfiles, outfile):
    """Combine multiple stored corpus frequency distributions for combined corpora."""
    import pickle

    corpusFD, corpusPres = {}, {}
    totaldocs, totalwords = 0, 0

    for fdfile in fdfiles:
        with open(fdfile, 'rb') as fdpickle:
            frequency, presence, docs, words = pickle.load(fdpickle)
            corpusFD = mergeFreqDist_copy(corpusFD, frequency)
            corpusPres = mergeFreqDist_copy(corpusPres, presence)
            totaldocs += docs
            totalwords += words

    print "Combined corpus represents %d texts containing a total of %d words" % \
          (totaldocs, totalwords)
    saveFD(outfile, corpusFD, corpusPres, totaldocs, totalwords)
    return totaldocs, totalwords


def displaycontexts(contextfile, phrase,
                    joinstring=' ... ', maxtodisplay=50,
                    textcol=1, header=False,
                    capsmatter=False):
    """Given a context file and a phrase whose context we wish to see, display the results."""
    import csv

    print "Displaying up to %d contexts for '%s'" % (maxtodisplay, phrase)
    nrfound = 0
    with open(contextfile, 'rU') as contextf:
        contextreader = csv.reader(contextf)
        if header:
            dummy = next(contextreader)
        for row in contextreader:
            text = row[textcol] if capsmatter else row[textcol].lower()
            contexts = text.split(joinstring)
            for context in contexts:
                if phrase in context:
                    nrfound += 1
                    print "\n%d. - %s" % (nrfound, context)
                    if nrfound == maxtodisplay:
                        return
    return


def wordcloud_prep(FDfile, pmifile='', pmithres=0.05,
                   singlesonly=False, aswords=True, maxwords=100000):
    """Prepare data for word cloud generators.

    FD file is a frequency distribution file.
    pmifile, if supplied, should contain a list of words and associated (normalized) pmi values.

    Output is either a word soup the word cloud generator can use,
    or a simple tabular format (word, count) for each word.
    """
    import csv
    import pickle

    with open(FDfile, 'rb') as fdpickle:
        frequencydict, dummy1, dummy2, dummy3 = pickle.load(fdpickle)

    wordlist = []
    if pmifile != '':
        with open(pmifile, 'rU') as pmif:
            pmireader = csv.reader(pmif)
            for pmidata in pmireader:
                if len(pmidata) >= 3:
                    pmival = float(pmidata[1])
                    if pmival < pmithres:
                        break  # end for loop (assumes pmi values sorted in descending order!)
                    wordlist.append(pmidata[0])
    wordset = set(wordlist)

    # Filter frequencydict if called for
    if len(wordset) > 0:
        freqwords = frequencydict.keys()
        for word in freqwords:
            if word not in wordset or (singlesonly and len(word.split()) > 1):
                del frequencydict[word]

    # Scale dictionary counts to avoid overflow
    maxwords = float(maxwords)
    wordcount = sum([value for key, value in frequencydict.iteritems()])
    divisor = wordcount/maxwords

    # Convert to word soup if called for
    allwords = []
    if aswords:
        for word, freq in frequencydict.iteritems():
            allwords += [word for x in range(int(freq/divisor))]
        return ' '.join(allwords)
    else:  # just return in tabular format
        return frequencydict.items()


# ****************************** co-occurrence calculations ******************

# Note: searchWindow returns a window around each found location.
# This means that words will be in there multiple times if they fall
# inside more than 1 window. That's not ideal. To avoid overlaps,
# don't return words, but rather return indices which can be de-duplicated.
# This is done by the wordsinwindow.py module

def searchWindow(aText, theWord, n):
    """Searches for a word, and returns n words on either side of text.

    Returns a list of lists, one for each occurrence of the word in the text."""
    from itertools import izip as zip, count
    wordList = aText.split()
    nrWords = len(wordList)
    indices = [i for i, j in zip(count(), wordList) if j == theWord]
    return [wordList[max(i - n, 0) : i] + \
                        wordList[i + 1 : min(i + n + 1, nrWords)] \
            for i in indices]


def keywordsFDx(texts, keywordsets, window=(10, 10)):
    """Calculate frequency distributions for the contexts around
    one or more (sets of) keywords.

    window = size of window on either side of a keyword to consider.
    """
    from wordsinwindow import wordsinwindow

    keysets = len(keywordsets)
    keyFDs = []
    keysfound = [0,] * keysets
    keycontextsize = [0,] * keysets

    # It should be more efficient to do the keywordsets loop inside the text loop,
    # but things go awry if we try that; so do it the other way around

    # Update keyword-specific freqency distributions
    for setnr, keywordset in enumerate(keywordsets):
        keyFD = {}
        # run down texts
        for count, aText in enumerate(texts):
            keycontext, dummy, nrkeys = \
                wordsinwindow(aText, keywordset, window,
                              includekeys=True, returnval='words')
            textFD, wordcount = freqDistList(keycontext)
            keyFD = addFreqDist(keyFD, textFD)
            keycontextsize[setnr] += wordcount
            keysfound[setnr] += nrkeys
            if count % 10000 == 0:
                print "Calculating FDs for keyword set %d, document %d" % \
                      (setnr, count)
        keyFDs.append(keyFD)

    # for count, aText in enumerate(texts):
    #     # TODO: parallelize across chunks of text
    #     # update keyword-specific freqency distributions
    #     for setnr, keywordset in enumerate(keywordsets):
    #         # TODO: parallelize across keyword sets
    #         keycontext, dummy, nrkeys = \
    #             wordsinwindow(aText, keywordset, window,
    #                           includekeys=True, returnval='words')
    #         keyFD, wordcount = freqDistList(keycontext)
    #         keyFDs[setnr], keyPres[setnr] = \
    #             addFreqDist(keyFDs[setnr], keyPres[setnr], keyFD)
    #         keycontextsize[setnr] += wordcount
    #         keysfound[setnr] += nrkeys
    #     if count % 50000 == 0:
    #         print "Calculating frequency distributions for document %d" % count

    return keyFDs, keysfound, keycontextsize


def keywordsFD(texts, keywordsets, window=(10, 10), includekeys=False,
               parallelmethod='sequential', nrjobs=4):
    """Calculate frequency distributions for the contexts around
    one or more (sets of) keywords.

    window = size of window on either side of a keyword to consider.
    """
    from wordsinwindow import wordsinwindow
    import multiprocessing as mp
    import multiprocessing.dummy as threading
    from functools import partial

    keysets = len(keywordsets)
    keyFDs, keysfound, keycontextsize = [], [], []

    # Set up parallelization
    print "Calculating word frequencies in keyword contexts, parallelization: %s, nr. jobs: %d" % \
          (parallelmethod, nrjobs)
    nrtexts = len(texts)
    chunksize = 1 + (nrtexts / nrjobs)
    textchunks = [texts[x * chunksize : (x + 1) * chunksize] for x in range(nrjobs)]
    partial_processtexts = partial(processtexts,
                                   keywordsets=keywordsets, window=window,
                                   includekeys=includekeys)

    # Three different parallelization options.
    if parallelmethod == 'sequential':
        # 1. Process in sequence (helpful for debugging)
        results = map(partial_processtexts, enumerate(textchunks))
    else: # parallelization
        if parallelmethod == 'threads':  # May not work due to GIL
            # 2. Parallelization using threading
            procPool = threading.Pool(nrjobs)
        else: # parallelmethod == 'multiprocessing'
            #  3. Parallelization using multiple processes (note BLAS issue)
            procPool = mp.Pool(processes=nrjobs)
        results = procPool.map(partial_processtexts, enumerate(textchunks))
        procPool.close()
        procPool.join()

    # Merge results together

    # results will be a list of return items that we'd like to merge
    # each return is keyFDs, keysfound, keycontextsize
    for setnr in range(keysets):
        keyFDs.append(mergeFreqDists_copy([result[0][setnr] for result in results]))
        keysfound.append(sum([result[1][setnr] for result in results]))
        keycontextsize.append((sum([result[2][setnr] for result in results])))
    # for result in results:
    #     for setnr in range(keysets):
    #         keyFDs[setnr] = mergeFreqDist_copy(keyFDs[setnr], result[0][setnr])
    #         keysfound[setnr] += result[1][setnr]
    #         keycontextsize[setnr] += result[2][setnr]
    return keyFDs, keysfound, keycontextsize


def keywordFD(texts, keywordset, window=(10, 10), includekeys=False,
              parallelmethod='sequential', nrjobs=4):
    """Calculate frequency distributions for the contexts around a set of keywords.

    Like keywordsFD, but for a single keywordset only.

    window = size of window on either side of a keyword to consider.
    """
    from wordsinwindow import wordsinwindow
    import multiprocessing as mp
    import multiprocessing.dummy as threading
    from functools import partial

    # Set up parallelization
    print "Processing parallelization: %s, nr. jobs: %d" % \
          (parallelmethod, nrjobs)
    nrtexts = len(texts)
    chunksize = 1 + (nrtexts / nrjobs)
    textchunks = [texts[x * chunksize : (x + 1) * chunksize] for x in range(nrjobs)]
    partial_findkeycontexts = partial(findkeycontexts, keywordset=keywordset,
                                      window=window, includekeys=includekeys)

    # Three different parallelization options.
    if parallelmethod == 'sequential':
        # 1. Process in sequence (helpful for debugging)
        results = map(partial_findkeycontexts, enumerate(textchunks))
    else: # parallelization
        if parallelmethod == 'threads':  # May not work due to GIL
            # 2. Parallelization using threading
            procPool = threading.Pool(nrjobs)
        else: # parallelmethod == 'multiprocessing'
            #  3. Parallelization using multiple processes (note BLAS issue)
            procPool = mp.Pool(processes=nrjobs)
        results = procPool.map(partial_findkeycontexts, enumerate(textchunks))
        procPool.close()
        procPool.join()

    # Merge results together

    # results will be a list of return items that we'd like to merge
    # each return is keyFD, keysfound, keycontextsize
    resultsX = zip(*results)
    keyFD = mergeFreqDists_copy(resultsX[0])
    keysfound = sum(resultsX[1])
    keycontextsize = sum(resultsX[2])

    # Instead of zip* could do 3 list comprehensions:
    # keyFDs.append(mergeFreqDists_copy([result[0] for result in results]))
    # keysfound = sum([result[1] for result in results])
    # keycontextsize = sum([result[2] for result in results])

    # Could also do 1 result at a time, looping
    # for result in results:
    #     for setnr in range(keysets):
    #         keyFDs[setnr] = mergeFreqDist_copy(keyFDs[setnr], result[0][setnr])
    #         keysfound[setnr] += result[1][setnr]
    #         keycontextsize[setnr] += result[2][setnr]
    return keyFD, keysfound, keycontextsize


def keywordFD_filter(texts, filterindices, keywordset, window=(10, 10),
                     includekeys=False, parallelmethod='sequential', nrjobs=4):
    """Calculate frequency distributions for the contexts around a set of keywords.

    Filter by the indices specified

    window = size of window on either side of a keyword to consider.
    """
    from wordsinwindow import wordsinwindow
    import multiprocessing as mp
    import multiprocessing.dummy as threading
    from functools import partial

    # Set up parallelization
    print "Processing parallelization: %s, nr. jobs: %d" % \
          (parallelmethod, nrjobs)
    nrtexts = len(texts)
    chunksize = 1 + (nrtexts / nrjobs)
    textchunks = [texts[x * chunksize : (x + 1) * chunksize] for x in range(nrjobs)]
    filterchunks = [filterindices[x * chunksize : (x + 1) * chunksize] for x in range(nrjobs)]
    partial_findkeycontexts = partial(findkeycontexts_filter, keywordset=keywordset,
                                      window=window, includekeys=includekeys)

    # Three different parallelization options.
    if parallelmethod == 'sequential':
        # 1. Process in sequence (helpful for debugging)
        results = map(partial_findkeycontexts, enumerate(zip(textchunks, filterchunks)))
    else: # parallelization
        if parallelmethod == 'threads':  # May not work due to GIL
            # 2. Parallelization using threading
            procPool = threading.Pool(nrjobs)
        else: # parallelmethod == 'multiprocessing'
            #  3. Parallelization using multiple processes (note BLAS issue)
            procPool = mp.Pool(processes=nrjobs)
        results = procPool.map(partial_findkeycontexts, enumerate(zip(textchunks, filterchunks)))
        procPool.close()
        procPool.join()

    # Merge results together

    # results will be a list of return items that we'd like to merge
    # each return is keyFD, keysfound, keycontextsize
    resultsX = zip(*results)
    keyFD = mergeFreqDists_copy(resultsX[0])
    keysfound = sum(resultsX[1])
    keycontextsize = sum(resultsX[2])

    # Instead of zip* could do 3 list comprehensions:
    # keyFDs.append(mergeFreqDists_copy([result[0] for result in results]))
    # keysfound = sum([result[1] for result in results])
    # keycontextsize = sum([result[2] for result in results])

    # Could also do 1 result at a time, looping
    # for result in results:
    #     for setnr in range(keysets):
    #         keyFDs[setnr] = mergeFreqDist_copy(keyFDs[setnr], result[0][setnr])
    #         keysfound[setnr] += result[1][setnr]
    #         keycontextsize[setnr] += result[2][setnr]
    return keyFD, keysfound, keycontextsize


def keyword_indices(texts, keywordset, window=(10, 10), includekeys=False,
                    parallelmethod='sequential', nrjobs=4):
    """Collect indices for the contexts around a set of keywords.

    Like keywordFD, but for index lists rather than frequency distributions

    window = size of window on either side of a keyword to get.
    """
    from wordsinwindow import wordsinwindow
    import multiprocessing as mp
    import multiprocessing.dummy as threading
    from functools import partial

    # Set up parallelization
    print "Processing parallelization: %s, nr. jobs: %d" % \
          (parallelmethod, nrjobs)
    nrtexts = len(texts)
    chunksize = 1 + (nrtexts / nrjobs)
    textchunks = [texts[x * chunksize : (x + 1) * chunksize] for x in range(nrjobs)]
    partial_findkeycontexts = partial(findkeycontexts_indices, keywordset=keywordset,
                                      window=window, includekeys=includekeys)

    # Three different parallelization options.
    if parallelmethod == 'sequential':
        # 1. Process in sequence (helpful for debugging)
        results = map(partial_findkeycontexts, enumerate(textchunks))
    else: # parallelization
        if parallelmethod == 'threads':  # May not work due to GIL
            # 2. Parallelization using threading
            procPool = threading.Pool(nrjobs)
        else: # parallelmethod == 'multiprocessing'
            #  3. Parallelization using multiple processes (note BLAS issue)
            procPool = mp.Pool(processes=nrjobs)
        results = procPool.map(partial_findkeycontexts, enumerate(textchunks))
        procPool.close()
        procPool.join()

    # Merge results together, in order of chunk number
    # each return is chunknr, nrkeys_list, indices_list
    nrkeys_list = []
    indices_list = []
    results.sort()  # sort by chunk number
    for result in results:
        nrkeys_list += result[1]
        indices_list += result[2]
    return nrkeys_list, indices_list


def findkeycontexts(textchunk, keywordset, window=(10, 10), includekeys=False):
    """Helper function to keywordFD"""
    from wordsinwindow import wordsinwindow

    chunknr, texts = textchunk
    keyFD = {}
    keysfound = 0
    keycontextsize = 0

    # update keyword-specific freqency distributions
    # run down texts
    for count, aText in enumerate(texts):
        keycontext, dummy, nrkeys = \
            wordsinwindow(aText.lower(), keywordset[1], window,
                          includekeys=includekeys, returnval='words')
        textFD, wordcount = freqDistList(keycontext)
        keyFD = addFreqDist(keyFD, textFD)
        keycontextsize += wordcount
        keysfound += nrkeys
        if count % 50000 == 0:
            print "Calculating FDs for keyword set '%s', job %d, document %d" % \
                  (keywordset[0], chunknr, count)
    return keyFD, keysfound, keycontextsize


def findkeycontexts_filter(textchunk, keywordset, window=(10, 10), includekeys=False):
    """Helper function to keywordFD"""
    from wordsinwindow import wordsinwindow_indexfilter

    chunknr, (texts, filters) = textchunk
    keyFD = {}
    keysfound = 0
    keycontextsize = 0

    # update keyword-specific freqency distributions
    # run down texts
    for count, (aText, indices) in enumerate(zip(texts, filters)):
        keycontext, dummy, nrkeys = \
            wordsinwindow_indexfilter(aText.lower(), keywordset[1], indices, window,
                                      includekeys=includekeys, returnval='words')
        textFD, wordcount = freqDistList(keycontext)
        keyFD = addFreqDist(keyFD, textFD)
        keycontextsize += wordcount
        keysfound += nrkeys
        if count % 50000 == 0:
            print "Calculating filtered FDs for keyword set '%s', job %d, document %d" % \
                  (keywordset[0], chunknr, count)
    return keyFD, keysfound, keycontextsize


def findkeycontexts_indices(textchunk, keywordset, window=(10, 10),
                            includekeys=False):
    """Helper function to keywordFD"""
    from wordsinwindow import wordsinwindow

    chunknr, texts = textchunk
    indiceslist = []
    keysfound = []

    # run down texts
    for count, aText in enumerate(texts):
        nrkeys, indices = \
            wordsinwindow(aText.lower(), keywordset[1], window,
                          includekeys=includekeys, returnval='indices')
        indiceslist.append(indices)
        keysfound.append(nrkeys)
        if count % 50000 == 0:
            print "Calculating indices for keyword set '%s', job %d, document %d" % \
                  (keywordset[0], chunknr, count)
    return chunknr, keysfound, indiceslist


def processtexts(textchunk, keywordsets, window=(10, 10), includekeys=False):
    """Helper function to keywordsFD"""
    from wordsinwindow import wordsinwindow

    chunknr, texts = textchunk
    keysets = len(keywordsets)
    keyFDs = []
    keysfound = [0, ] * keysets
    keycontextsize = [0, ] * keysets

    # update keyword-specific freqency distributions
    for setnr, keywordset in enumerate(keywordsets):
        keyFD = {}
        # run down texts
        for count, aText in enumerate(texts):
            keycontext, dummy, nrkeys = \
                wordsinwindow(aText.lower(), keywordset[1], window,
                              includekeys=includekeys, returnval='words')
            textFD, wordcount = freqDistList(keycontext)
            keyFD = addFreqDist(keyFD, textFD)
            keycontextsize[setnr] += wordcount
            keysfound[setnr] += nrkeys
            if count % 10000 == 0:
                print "Calculating FDs for job %d, keyword set %d, document %d" % \
                      (chunknr, setnr, count)
        keyFDs.append(keyFD)
    return keyFDs, keysfound, keycontextsize


def processtextsX(textchunk, keywordsets, window=(10, 10)):
    """Helper function to keywordsFD"""
    from wordsinwindow import wordsinwindow

    chunknr, texts = textchunk
    keysets = len(keywordsets)
    keyFDs, keyPres = [{}, ] * keysets, [{}, ] * keysets
    keysfound = [0, ] * keysets
    keycontextsize = [0, ] * keysets

    for count, aText in enumerate(texts):
        # update keyword-specific freqency distributions
        for setnr, keywordset in enumerate(keywordsets):
            keycontext, dummy, nrkeys = \
                wordsinwindow(aText.lower(), keywordset, window,
                              includekeys=True, returnval='words')
            keyFD, wordcount = freqDistList(keycontext)
            keyFDs[setnr], keyPres[setnr] = \
                addFreqDist(keyFDs[setnr], keyPres[setnr], keyFD)
            keycontextsize[setnr] += wordcount
            keysfound[setnr] += nrkeys
        if count % 1000 == 0:
            print "Calculating frequency distributions for chunk %d, document %d" % \
                  (chunknr, count)
    return keyFDs, keyPres, keysfound, keycontextsize


def keywordsFDplus(texts, keywordsets, window=(10, 10)):
    """Calculate frequency distributions for the overall corpus and
     for the contexts around one or more sets of keywords.

    window = size of window on either side of a keyword to consider.

    Keep track of both an overall total count and a per-text presence flag.
    For now, we don't really use the latter.
    """
    from wordsinwindow import wordsinwindow

    keysets = len(keywordsets)
    # initialize corpus- and word-specific occurrence dictionaries.
    corpusFD, corpusPres = {}, {}
    keyFDs, keyPres = [{},] * keysets, [{},] * keysets
    keysfound = [0,] * keysets

    # run down texts
    for count, aText in enumerate(texts):
        # update overall corpus frequency distribution
        corpusFD, corpusPres = addFreqDist(corpusFD, corpusPres,
                                             freqDist(aText)[0])
        # update keyword-specific freqency distributions
        for setnr, keywordset in enumerate(keywordsets):
            keycontext, dummy, nrkeys = \
                wordsinwindow(aText, keywordset, window,
                              includekeys=True, returnval='words')
            keyFDs[setnr], keyPres[setnr] = \
                addFreqDist(keyFDs[setnr], keyPres[setnr], freqDistList(keycontext)[0])
            keysfound[setnr] += nrkeys
        if count % 10000 == 0:
            print "Calculating frequency distributions for document %d" % count
    return corpusFD, keyFDs, keysfound


def calc_salience(worddata, overallwords, contextwords, pmitype,
                  parallelmethod='sequential', nrjobs=4):
    """Calculate pmi, chi-squared, and Fisher's exact, given counts & frequencies."""

    import multiprocessing as mp
    import multiprocessing.dummy as threading
    from functools import partial

    # Set up parallelization
    print "Calculating pmi & significance values: %s, nr. jobs: %d" % \
          (parallelmethod, nrjobs)
    nrwords = len(worddata)
    chunksize = 1 + (nrwords / nrjobs)
    wordchunks = [worddata[x * chunksize : (x + 1) * chunksize] for x in range(nrjobs)]
    partial_calcstats = partial(calcstats, overallwords=overallwords,
                                contextwords=contextwords, pmitype=pmitype)

    # Three different parallelization options.
    # Note: these don't always work well, and end up hanging; threading should work though
    if parallelmethod == 'sequential':
        # 1. Process in sequence (helpful for debugging)
        results = map(partial_calcstats, enumerate(wordchunks))
    else: # parallelization
        if parallelmethod == 'threads':  # May not work due to GIL
            # 2. Parallelization using threading
            procPool = threading.Pool(nrjobs)
        else: # parallelmethod == 'multiprocessing'
            #  3. Parallelization using multiple processes (note BLAS issue)
            procPool = mp.Pool(processes=nrjobs)
        results = procPool.map(partial_calcstats, enumerate(wordchunks))
        procPool.close()
        procPool.join()

    # Merge results together: simply flatten result lists
    return [word for sublist in results for word in sublist]


def calcstats(wordchunk, overallwords=0, contextwords=0, pmitype=0):
    """Helper function to calc_salience"""

    chunknr, words = wordchunk
    probcontext = contextwords/overallwords
    pmidata = []

    for counter, (word, overall, overallfreq, count, freq) in enumerate(words):
        pmidata.append((word, calcpmi(freq, overallfreq, probcontext, pmitype),
                              calc_pval((count, contextwords), (overall, overallwords))))
        if (counter + 1) % 10000 == 0:
            print "Completed %d words in chunk %d" % (counter + 1, chunknr)
    return pmidata


def fd2pmi (corpusFDfile, keyFDfile, outputfile, pmitype=0, mincount=10,
            parallelmethod='sequential', nrjobs=1, includemissing=True):
    """Calculate PMI & chi2 for a given context-specific frequency distribution."""
    import corpus
    import csv
    import os
    import pickle
    from datetime import datetime

    # Get overall corpus frequency distribution info
    print "Loading corpus frequency distribution info"
    if corpusFDfile[-3:] == 'pkl':
        with open(corpusFDfile, 'rb') as fdpickle:
            Cfrequency, Cpresence, Cdocs, Cwords = pickle.load(fdpickle)
    else:  # assume frequency dictionary is in csv format
        Cfrequency, Cpresence, Cdocs, Cwords = readFD_csv(corpusFDfile)
    # make Cwords float for correct division
    Cwords = float(Cwords)
    # Clear out potentially large-volume variable not needed
    Cpresence = []

    # Get context-specific frequency distribution info
    print "Loading context-specific frequency distribution info"
    if keyFDfile[-3:] == 'pkl':
        with open(keyFDfile, 'rb') as fdpickle:
            Kfrequency, Kpresence, Kdocs, Kwords = pickle.load(fdpickle)
    else:  # assume frequency dictionary is in csv format
        Kfrequency, Kpresence, Kdocs, Kwords = readFD_csv(keyFDfile)
    # make Kwords float for correct division
    Kwords = float(Kwords)
    # Clear out large-volume variable not needed
    Kpresence = []

    # For each word, store overall count & ratio, keycontext count & ratio
    # No need to keep it a dictionary, so make list for speedier processing
    worddata = []
    for word, count in Kfrequency.iteritems():
        if count > mincount:
            overallcount = Cfrequency.get(word, 0)
            if overallcount >= count:  # safety check, in case min_occur wasn't met
                # Could do pmi calculation here too, but if we do that separately,
                # we can parallelize it
                worddata.append((word, overallcount, overallcount/Cwords,
                                       count, count/Kwords))
            elif includemissing:  # min_occur wasn't met for corpus chunks
                worddata.append((word, count, count/Cwords, count, count/Kwords))

    print "Nr terms: %d, total words: %d, context words: %d" % (len(worddata), Cwords, Kwords)

    # Now do pmi calculation, parallelized
    pmidata = calc_salience(worddata, Cwords, Kwords, pmitype,
                            parallelmethod=parallelmethod, nrjobs=nrjobs)
    # Sort by descending normalized pmi
    pmidata.sort(key=lambda l: l[1], reverse=True)

    # Write out & return results
    with open(outputfile, 'wb') as outf:
        outwriter = csv.writer(outf)
        outwriter.writerows(pmidata)
    # Return the sorted pmi list
    return pmidata


def displaypmi(pmidata, infostrings):
    """Display data of the type returned by fd2pmi, intersectpmi, and diffpmi."""
    if len(pmidata) > 0:
        print "\nTop %d phrases for %s-specific %s" % (len(pmidata), infostrings[0], infostrings[1])
        maxtermlen = max([len(x[0]) for x in pmidata])

        if infostrings[2] == 'first' or infostrings[2] == '':
            displaystring = "Term: %" + str(maxtermlen) + "s - %s norm.pmi: %4.2f, p-val: %4.2f"
            for keyinfo in pmidata:
                print displaystring % (keyinfo[0], infostrings[2], keyinfo[1], keyinfo[2])
            print
        else:
            # Display mean or lowest pmi value; no p-value
            nrcols = len(pmidata[0])
            pmicol = nrcols - 2 if infostrings[2] == 'mean' else nrcols - 1
            displaystring = "Term: %" + str(maxtermlen) + "s - %s norm.pmi: %4.2f"
            for keyinfo in pmidata:
                print displaystring % (keyinfo[0], infostrings[2], keyinfo[pmicol])
            print

    return


def keywordsPMI_subset(corpusfile, corpusFDfile, indexfile,
                       keywordsets, window=(10,10), pmitype=1,
                       mincount=10, includekeys=False,
                       parallelmethod='sequential', nrjobs=1):
    """Calculate PMI & chi2 for one or more sets of keywords,

    intersected with a specified set of indices.
    Used to calculate intersections of contexts & their significance.
    """
    import ast
    import corpus
    import csv
    import os
    import pickle
    from datetime import datetime

    # Load texts all at once -> assumes enough memory
    print "Loading texts"
    texts = corpus.readidtextfile_texts(corpusfile)

    # Load filter indices all at once
    print "Loading filtering data"
    filterindexstrings = corpus.readidtextfile_texts(indexfile)
    filterindices = [ast.literal_eval(x) for x in filterindexstrings]

    # Get overall corpus frequency distribution info
    print "Loading corpus frequency distribution info"
    if corpusFDfile[-3:] == 'pkl':
        with open(corpusFDfile, 'rb') as fdpickle:
            Cfrequency, Cpresence, Cdocs, Cwords = pickle.load(fdpickle)
    else:  # assume frequency dictionary is in csv format
        Cfrequency, Cpresence, Cdocs, Cwords = readFD_csv(corpusFDfile)
    # make Cwords float for correct division
    Cwords = float(Cwords)
    # Clear out large-volume variable not needed
    Cpresence = []

    # Loop over the keyword sets. Little to be gained from doing these in parallel,
    # so do each keyword set separately.
    for keyset in keywordsets:
        print "Starting keyword set '%s' at %s" % (keyset[0], datetime.now())

        # Filestem for output; put keyset name & window size in filename
        outstem = '.'.join(corpusfile.split('.')[:-1]) + \
                  '_pmi_' + keyset[0] +  '_L' + str(window[0]) + 'R' + str(window[1])

        # Get frequency distribution info for keyword set
        keyFD, keysfound, keycontextsize = \
                keywordFD_filter(texts, filterindices, keyset,
                                 window=window, includekeys=includekeys,
                                 parallelmethod=parallelmethod, nrjobs=nrjobs)

        # Divide by word counts to get frequency ratios, or 'probabilities'
        keycontextsize = float(keycontextsize)
        # For each word, store overall count & ratio, keycontext count & ratio
        # No need to keep it a dictionary, so make list for speedier processing
        worddata = []
        print "Calculating frequencies"
        for word, count in keyFD.iteritems():
            if count > mincount:
                overallcount = Cfrequency.get(word, 0)
                if overallcount > 0:  # safety check; should not be necessary
                    # Could do pmi calculation here too, but if we do that separately,
                    # we can parallelize it
                    worddata.append((word, overallcount, overallcount/Cwords,
                                           count, count/keycontextsize))
        # Save data so far, so we don't have to re-do it in case of a computer problem
        # outtemp = outstem + '_prep.pkl'
        # with open(outtemp, 'wb') as picklefile:
        #     pickle.dump((worddata, keycontextsize), picklefile)
        # Done with this keyword set's frequency dictionary, so clear it out, to save memory
        keyFD = {}

        # Now do pmi calculation, parallelized
        pmidata = calc_salience(worddata, Cwords, keycontextsize, pmitype,
                                parallelmethod=parallelmethod, nrjobs=nrjobs)
        # Sort by descending normalized pmi
        pmidata.sort(key=lambda l: l[1], reverse=True)
        with open(outstem + '.csv', 'wb') as outf:
            outwriter = csv.writer(outf)
            outwriter.writerows(pmidata)
        # empty out large storage items
        worddata = []
        # delete intermediate picklefile
        # if os.path.isfile(outstem + '_prep.pkl'):
        #     os.remove(outstem + '_prep.pkl')

    # Return the final keyword set's sorted list
    return pmidata


def keywords_indices(corpusfile, keywordsets, window=(10,10),
                     includekeys=False,
                     parallelmethod='sequential', nrjobs=1):
    """Find & save indices for one or more sets of keywords."""
    import corpus
    import csv
    import os
    import pickle
    from datetime import datetime

    # Load texts all at once -> assumes enough memory
    print "Loading texts"
    ids, texts = corpus.readidtextfile(corpusfile)

    # Loop over the keyword sets. Little to be gained from doing these in parallel,
    # so do each keyword set separately.
    for keyset in keywordsets:
        print "Starting keyword set '%s' at %s" % (keyset[0], datetime.now())

        nrkeys_list, indices_list = \
                keyword_indices(texts, keyset, window=window,
                                includekeys=includekeys,
                                parallelmethod=parallelmethod, nrjobs=nrjobs)

        # Filestem for output; put keyset name & window size in filename
        outstem = '.'.join(corpusfile.split('.')[:-1]) + \
                  '_indices_' + keyset[0] + '_L' + str(window[0]) + 'R' + str(window[1])
        with open(outstem + '.csv', 'wb') as outf:
            outwriter = csv.writer(outf)
            outwriter.writerows(zip(ids, indices_list, nrkeys_list))

    print "Done at", datetime.now()
    return


def intersectpmi(pmifiles, outputfile, pmithres=0.1, pthres=0.001, sortby='first'):
    """Report intersection of those words that meet a pmi threshold in each pmi file."""
    import csv

    intersection = {}
    for counter, pmifile in enumerate(pmifiles):
        with open(pmifile, 'rU') as pmif:
            for pmidata in csv.reader(pmif):
                if len(pmidata) < 3:  # blank line sometimes happens
                    continue
                pmival = float(pmidata[1])
                pval = float(pmidata[2])
                if pmival < pmithres:
                    break
                if pval < pthres:
                    if counter == 0:
                        intersection[pmidata[0]] = [pmival, pval]
                    elif pmidata[0] in intersection and \
                            len(intersection[pmidata[0]]) == counter * 2:
                        intersection[pmidata[0]] = intersection[pmidata[0]] + [pmival, pval]
        # Remove any words that didn't make the intersection
        todelete = []
        if counter > 0:
            for word, data in intersection.iteritems():
                if len(data) < (counter + 1) * 2:
                    todelete.append(word)
            for word in todelete:
                del intersection[word]

    nextcol = 1 + len(pmifiles) * 2
    # Sort by first, mean, or lowest pmi value
    sortcol = 1 if sortby == 'first' else (nextcol if sortby == 'mean' else nextcol + 1)
    intersectlist = sorted([[word,] + val + compvals(val) for word, val in intersection.iteritems()],
                           key=lambda l: l[sortcol], reverse=True)
    # write out and return results
    if len(intersectlist) > 0:
        with open(outputfile, 'wb') as outf:
            outwriter = csv.writer(outf)
            outwriter.writerows(intersectlist)
    return intersectlist


def diffpmi(pmifiles, outputfiles, pmithres=0.1, pthres=0.001, sortby='first'):
    """Report those words in each list most different in pmi from the other lists."""
    import csv

    nrlists = len(pmifiles)
    if nrlists != len(outputfiles):
        print "Error: different number of input and output filenames"
        return

    # Load the pmi data
    pmidicts = []
    for counter, pmifile in enumerate(pmifiles):
        pmidict = {}
        with open(pmifile, 'rU') as pmif:
            for pmidata in csv.reader(pmif):
                if len(pmidata) < 3:  # blank line sometimes happens
                    continue
                pmival = float(pmidata[1])
                pval = float(pmidata[2])
                if pmival < pmithres:
                    break
                if pval < pthres:
                    pmidict[pmidata[0]] = [pmival, pval]
        pmidicts.append(pmidict)

    # Generate sorted lists of pmi differences
    difflists = []
    for counter in range(nrlists):
        thisdata = []
        tripcounter = 0
        for word, vals in pmidicts[counter].iteritems():
            tripcounter += 1
            if tripcounter > 20:
                break
            wordval = vals[0]
            includeword = True
            comparedata = []
            for subcounter in range(nrlists):
                if subcounter != counter:
                    if word not in pmidicts[subcounter]:
                        comparedata.append([0, 0])
                    elif pmidicts[subcounter][word][0] < wordval:
                        comparedata.append(pmidicts[subcounter][word])
                    else:
                        # print counter, word, wordval, subcounter, pmidicts[subcounter][word]
                        includeword = False
            if includeword:
                # calculate mean and min diffs
                mindiff = wordval
                sumdata = 0
                for comparepmi, comparep in comparedata:
                    sumdata += comparepmi
                    if wordval - comparepmi <  mindiff:
                        mindiff = wordval - comparepmi
                avgdiff = wordval - (sumdata / len(comparedata))
                worddata = [word,] + vals + \
                           [x for compvals in comparedata for x in compvals] + \
                           [avgdiff, mindiff]
                thisdata.append(worddata)

        # write out and return results
        if len(thisdata) > 0:
            avgcol = 1 + nrlists * 2
            # Sort by mean or min (lowest) pmi difference
            sortcol = avgcol if sortby == 'mean' else avgcol + 1
            difflist = sorted(thisdata, key=lambda l: l[sortcol], reverse=True)
            with open(outputfiles[counter], 'wb') as outf:
                outwriter = csv.writer(outf)
                outwriter.writerows(difflist)
        else:
            difflist = []
        difflists.append(difflist)
    return difflists


def compvals(pmivalues):
    """Return mean and lowest pmi values in the list."""
    pmivals = pmivalues[::2]
    return [sum(pmivals)/len(pmivals), min(pmivals)]


def calc_pval(selection, baseline, subset=True):
    """Calculate chi-squared or Fisher's exact statistics.

    See if the selection frequency (tuple: keycount, totalcount)
    is comparable to the baseline frequency (tuple: keycount, totalcount).

    If subset is True, adjust baseline by subtracting selection.

    When our expected selection frequencies are small (< 5),
    calculate Fisher's exact statistic; otherwise calculate chi-squared.
    Fisher's stat is computationally intensive for large values,
    so only calculate it for those specific cases, not in general.
    """
    from scipy import stats
    from fisher import pvalue

    # adjust baseline as appropriate
    if subset:
        baseline = (baseline[0] - selection[0],
                    baseline[1] - selection[1])
    # adjust from (observed, total) to (observed, not observed)
    contingencyvals = (max(0, selection[0]), max(0, baseline[0]),
                       max(0, selection[1] - selection[0]), max(0, baseline[1] - baseline[0]))
    key_counts = contingencyvals[:2]
    nonkey_counts = contingencyvals[2:]
    contingencytable = [key_counts, nonkey_counts]
    # get cell expectations
    total = float(sum(key_counts + nonkey_counts))
    row1 = sum(key_counts)
    row2 = sum(nonkey_counts)
    col1 = key_counts[0] + nonkey_counts[0]
    col2 = key_counts[1] + nonkey_counts[1]
    minexpected = min([x / total for x in [row1 * col1, row1 * col2, row2 * col1, row2 * col2]])
    # calculate & return most appropriate of chi-squared and Fisher's exact p-values
    if minexpected <= 5:  # use fisher
        fisher_p = pvalue(*contingencyvals)
        # print "Fisher's exact p-value %f" % (fisher_p.two_tail)
        return fisher_p.two_tail
    else:
        chisquare, chi_p, _, __ = stats.chi2_contingency(contingencytable, correction=False)
        # print "Chi-squared statistic %f & p-value %f" % (chisquare, pvalue)
        return chi_p


def calcpmi(pxgiveny, px, py, pmitype=1):
    """pmitype = 0 for normalized pmi, 1 for standard pmi, 2 for squared pmi.

    Standard pmi: log(p(x,y)/(p(x)p(y))) = log(p(x|y)/p(x)
    Normalized pmi: standardpmi / -log(p(x,y))
    Squared pmi: log(p(x,y)^2/(p(x)p(y)))

    Equivalence: p(x,y) = p(x|y)*p(y)
    """
    import math
    if pmitype == 0:
        return math.log(pxgiveny/px) / -math.log(pxgiveny * py)
    elif pmitype == 1:
        return math.log(pxgiveny/px)
    else:  # pmitype == 2
        return math.log((pxgiveny * py)^2 / (px * py))


def corpusPMI(texts, topwords, window=10):
    """Calculate top co-occurring words for each of our topwords.

    window = size of window on either side of a word to consider.
    """
    from operator import itemgetter

    print "Nr. top words:", len(topwords)
    # initialize corpus- and word-specific occurrence dictionaries.
    corpusFD, corpusPres, topFDs, topCoWords = {}, {}, {}, {}
    for aWord in topwords:
        topFDs[aWord] = {}

    # run down texts
    for count, aText in enumerate(texts):
        # update overall corpus frequency distribution
        corpusFD, corpusPres = addFreqDist(corpusFD, corpusPres,
                                             freqDist(aText)[0])
        # update topword-specific freqency distributions
        for aWord in topwords:
            for aList in searchWindow(aText, aWord, window):
                # First option counts all occurrences in search window;
                # second option counts presence/absence only
                topFDs[aWord], dummy = addFreqDist(topFDs[aWord], {},
                                                     freqDistList(aList)[0])
                # dummy, topFDs[aWord] = addFreqDist({}, topFDs[aWord],
                #                                      freqDistList(aList)[0])
        count += 1
        if count % 10000 == 0:
            print "Processing document %d" % count
    return corpusFD, corpusPres, topFDs, len(texts)


def commonwords(corpusPres, textCount, thres=0.9):
    """Display all words with > thres presence rate across the corpus."""
    freqwords = []
    print "\nFound %d distinct words" % len(corpusPres)
    for key in corpusPres:
        rate = corpusPres[key]/float(textCount)
        if rate >= thres:
            print "%s is in %f of corpus" % (key, rate)
            freqwords.append(key)
    return freqwords


def topPMI_perword(topwords, topFDs, corpusFD, minthres=10):
    """Using co-occurrence dictionary for top words, divide by word count."""
    for aWord in topwords:
        topCoWords[aWord] = []
        for coWord in topFDs[aWord]:
            if corpusFD[coWord] > minthres:
                topCoWords[aWord].append(
                    (coWord, (topFDs[aWord][coWord] / float(corpusFD[coWord]))))
        topCoWords[aWord].sort(key=itemgetter(1), reverse=True)
        # Keep only top 10
        topCoWords[aWord] = topCoWords[aWord][:10]
    return topCoWords


def topPMI_valencewords(topFDs, corpusFD, topvalencewords,
                        minthres=20, newpos=100, newneg=100):
    """Get corpus words most associated w. +ve/-ve salience top words.

    Maximize the difference between +ve & -ve co-occurrence rates,
    averaged across all top valence words.
    """
    from operator import itemgetter

    sharediff = {}
    for coWord, freq in corpusFD.iteritems():
        if freq > minthres:
            poscount, posval, negcount, negval = 0, 0, 0, 0
            for aWord, val in topvalencewords.iteritems():
                if coWord in topFDs[aWord]:
                    if val == 1:
                            # poscount += 1
                        posval += topFDs[aWord][coWord] / float(corpusFD[coWord])
                    else:
                        # negcount += 1
                        negval += topFDs[aWord][coWord] / float(corpusFD[coWord])
            # posavg = 0 if poscount == 0 else posval / float(poscount)
            # negavg = 0 if negcount == 0 else negval / float(negcount)
            posavg = posval / len(topvalencewords)
            negavg = negval / len(topvalencewords)
            sharediff[coWord] = posavg - negavg
    sorteddiffs = sorted(sharediff.items(), key=itemgetter(1))
    newpositives = [w[0] for w in sorteddiffs[-newpos:]]
    newpositives.reverse()  # give strongest association first
    newnegatives = [w[0] for w in sorteddiffs[:newneg]]
    print "\nNew positives:", newpositives
    print "\nNew negatives:", newnegatives
    return newpositives, newnegatives


# ********************************* main program control *********************

def findPMI(infile):
    """For each of a set of top words find the most specific co-occurers."""
    import csv
    import topwords

    outfile = '.'.join(infile.split('.')[:-1]) + '_pmi.csv'
    mytopwords = topwords.get()
    with open(infile,'rU') as inf:
        corpusFD, corpusPres, topFDs, textcount = \
            corpusPMI([row[1].lower() for row in csv.reader(inf)],
                               mytopwords, window=20)
    # Calculate most frequent words in the corpus overall
    # freqwords = commonwords(corpusPres, textcount, thres=0.9)
    # Calculate most PMI-based frequent co-occurrers for each top-word
    topCoWords = topPMI_perword(mytopwords, topFDs, corpusFD, minthres=10)
    # Display & write results
    with open(outfile, 'wb') as outputfile:
        cowordOutput = csv.writer(outputfile)
        for aWord in mytopwords:
            # Display top 5
            print '===>', aWord
            print topCoWords[aWord][:5]
            # Flatten list and write to file
            cowordOutput.writerow(
                [aWord,] + \
                [item for sublist in topCoWords[aWord] for item in sublist])
    return outfile


def collocations_PMI(textfile, keywords, nrcollocs=50, window=20, thres=20):
    """Find words in corpus most associated with each of a list of keywords.

    Assume text file is of csv format: id,text

    Save top nrcollocs words for each word in keywords
    """
    import csv
    from lexica import writeValenceDict

    outfile = '.'.join(textfile.split('.')[:-1]) + '_collocs.csv'
    with open(textfile,'rU') as inf:
        corpusFD, corpusPres, topFDs, textcount = \
            corpusPMI([row[1].lower() for row in csv.reader(inf)],
                               keywords, window=window)
    for word in keywords:
        worddict = {w: -1 for w in keywords if w != word}
        worddict[word] = 1
        withword, notwithword = \
            topPMI_valencewords(topFDs, corpusFD, worddict, minthres=thres,
                                newpos=nrcollocs, newneg=nrcollocs)
        print "Words most associated with %s:" % word
        print withword
        with open(outfile, 'wb') as outf:
            output = csv.writer(outf)
            output.writerow([word, 'with'] + withword + \
                            ['not with',] + notwithword)
    return


def expandlexicon_PMI(textfile, topworddict, newpos=250, newneg=250, keyval=''):
    """Find words in corpus most associated with key +ve/-ve dict words.

    Use for corpus-specific expansion of lexicon for sentiment analysis,
    starting from initial kernel and using PMI to select.

    keyval is a string used to identify the source dictionary when saving
    the expanded dictionary.

    Assume text file is of csv format: id,text

    Save several versions of increasing size, in symmetric increments
    of 50 pos & 50 neg. This will work best if newpos == newneg
    Alternatively, can simply eliminate loop and simply save one new
    file with all the new positive & negative words
    """
    import csv
    from lexica import writeValenceDict

    with open(textfile,'rU') as inf:
        corpusFD, corpusPres, topFDs, textcount = \
            corpusPMI([row[1].lower() for row in csv.reader(inf)],
                               topworddict.keys(), window=10)

    newpositives, newnegatives = \
        topPMI_valencewords(topFDs, corpusFD, topworddict, minthres=100,
                            newpos=newpos, newneg=newneg)

    # Combine new positive & negative words into single dictionary
    for size in xrange(50, min(newpos, newneg) + 50, 50):
        newdict1 = {w: 1 for w in newpositives[:size]}
        newdict2 = {w: -1 for w in newnegatives[:size]}
        newdict = dict(newdict1, **newdict2)
        outfile = '.'.join(textfile.split('.')[:-1]) + '_corpusdict_' + \
                    keyval + '_' + str(2 * size) + '.csv'
        writeValenceDict(newdict, outfile)
    return newdict




# def keywordsPMI_old(corpusfile, corpusFDfile, keywordsets,
#                 window=(10,10), pmitype=1, mincount=10,
#                 includekeys=False,
#                 parallelmethod='sequential', nrjobs=1):

    # Older code -- more pythonic but slower and more of a memory hog

    # original approach: dictionary-based; pythonic but slow
    # freq_overall = {word: count/float(Cwords) for word, count in Cfrequency.iteritems()}
    # freq_cooccur = [{word: count/float(keycontextsize[x]) \
    #                     for word, count in keyFDs[x].iteritems() if count > mincount} \
    #                 for x in range(nrkeysets)]

        # The following is more pythonic, but gives no feedback and appears slower in implementation
        # pmi.append({word: (calcpmi(wordfreqs[word], freq_overall[word],
        #                            contextprob, pmitype),) + \
        #                   calc_pval((wordcounts[word], contextsize),
        #                             (Cfrequency[word], Cwords)) \
        #             for word in wordfreqs.keys() if word in freq_overall})

    # freq_cooccur = p(x|y), freq_overall = p(x), probcontext = p(y)

    # Calculate pmi
    # print "Calculating pmi & likelihood values at", datetime.now()
    # pmi = []
    # for keyset, wordcounts, wordfreqs, contextsize, contextprob in \
    #         zip(keywordsets, keyFDs, freq_cooccur, keycontextsize, probcontext):
    #     print "Working on keyword set: %s" % keyset[0]
    #     pmidict = {}
    #     for counter, word in enumerate(wordfreqs.keys()):
    #         if counter % 1000 == 0:
    #             print "Completed %d words" % counter
    #         if word in freq_overall:
    #             pmidict[word] = (calcpmi(wordfreqs[word], freq_overall[word],
    #                                      contextprob, pmitype),) + \
    #                             calc_pval((wordcounts[word], contextsize),
    #                                       (Cfrequency[word], Cwords))
        # convert pmi from dictionary to list, and sort
    #     pmisorted = sorted(pmidict.items(), key=lambda l: l[1][0], reverse=True)
    #     # write to csv file
    #     outfile = '.'.join(corpusfile.split('.')[:-1]) + '_pmi_' + keyset[0] + '.csv'
    #     with open(outfile, 'wb') as outf:
    #         outwriter = csv.writer(outf)
    #         outwriter.writerows([(x,) + y for x, y in pmisorted])
    #
    # # Return the final keyword set's sorted list
    # return pmisorted
