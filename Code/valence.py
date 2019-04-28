# encoding: utf-8

# valence.py
# by Maurits van der Veen
# last modified 2017-01-06

# Program to calculate positive/negative polarity of texts
# and to measure prominence of different word categories, including emotions.

# English version uses 7 different lexica: Lexicoder, Bing Liu's, WordStat, and labMT,
# plus the multi-category lexica LIWC, moral foundations (Graham & Haidt),
# and NRC (polarity as well as emotions).

# These dictionaries were processed from their original text format into
# a Python dictionary format (see the code in importLexica.py)
# They are all stored in the ValenceDicts folder

# Dutch version uses just 2 lexica so is simpler.


# **************************** Main program control **************************

def corpus_valence_NL(filestem, lexicafile, LIWCfile=''):
    """Calculate valence for a Dutch corpus.

    I/O wrapper for sent_valence_NL.
    Expects sentences in the file filestem_sents.txt and
    text ids in filestem_ids.txt.
    """
    from corpus import readwords

    sents = readwords(filestem + '_lines.txt')
    ids = readwords(filestem + '_lineids.txt')
    means, stdevs = sent_valence_NL(filestem, ids, sents, lexicafile, LIWCfile)
    return means, stdevs


def getValences(sentchunk, lexiconlist, wordlist, ignore,
                mods, negs):
    """Calculate valences for the sentence chunk."""
    chunknr, ids_sents = sentchunk
    valences = [[id,] + getValence(s, lexiconlist, wordlist, ignore,
                                   mods, negs,
                                   chunknr=chunknr, sentnr=count) \
                 for count, (id, s) in enumerate(ids_sents)]
    return (chunknr, valences)


def scalevals(valences):
    """Scale valences by the number of words in a sentence (stored as row[1])."""
    return [[x[1], ] + [val / float(x[1]) if x[1] > 0 else 0 for val in x[2:]] \
                  for x in valences]


def corpus_valence(filestem, lexicondir, lexica2use,
                   ids=list(), sents=list(), jobname='',
                   nrjobs=1, sents_per_job=1000000,
                   modify=True, negate=False, ignore=()):
    """Calculate sentence-level valences for entire corpus.

    Works with whatever lexica are returned by getlexica.
    Also calculates corpus-wide, sentence-level averages.

    Ignore valence of key terms passed in through ignore parameter

    Try to minimize memory load to avoid running out of application memory.
    Also run each of the two main computational stages in rounds, for same reason.
    """
    import multiprocessing as mp
    from functools import partial
    import numpy as np
    import csv
    from operator import itemgetter
    from corpus import readwords
    import gc
    from lexica import getlexica, lexicon_haswilds, allkeys, subsumed

    # 10 most common negaters, for use if negate == True
    negs = ('not', 'no', 'nor', 'nothing', 'none', 'never',
            'nowhere', 'noone', 'nobody', 'neither')

    # Load lexica
    lexiconlist, mods = getlexica(lexicondir, lexica2use)

    # Generate a union of lexiconkeys, using any wildcards included to keep length down
    allterms = allkeys(lexiconlist)
    # See which of our ignore terms are not in allterms, so need to be kept separately
    ignoreX = set(ignore) - allterms
    # Update allterms to ignore words in our ignore set
    allterms -= set(ignore)

    # Add a flag indicating whether a lexicon has wildcards
    lexiconlist = [(lex, lexicon_haswilds(lex)) for lex in lexiconlist]

    # Read in ids & sentences if we haven't done so yet
    if len(ids) == 0 or len(sents) == 0:
        sents = readwords(filestem + '_lines.txt')
        ids = readwords(filestem + '_lineids.txt')

    # Calculate various quantities
    nrsents = len(sents)
    print "Calculating valences for all %d sentences in the corpus" % (nrsents)
    if nrjobs > 1:  # parallel processing; calculate nr. rounds
        rounds = max(1, nrsents / (nrjobs * sents_per_job))  # at least 1 round!
        jobs_rounds = rounds * nrjobs
        chunksize = 1 + ((nrsents - 1) / jobs_rounds)
        print "Run in %d consecutive rounds of %d sentences per job to control memory use" % \
              (rounds, chunksize)

    if nrjobs == 1:
        # This is an elegant list comprehension, but not parallelizable and hence slow
        valences = [[id,] + getValence(s, lexiconlist, allterms, ignoreX,
                                       mods if modify else dict(),
                                       negs if negate else (), sentnr=count) \
                    for count, (id, s) in enumerate(zip(ids, sents))]
        ids, sents = [], []  # Free up memory

    else:  # parallelize
        # Divide sentences into smaller chunks.
        ids_sents = zip(ids, sents)
        sentchunks = [ids_sents[x * chunksize: (x + 1) * chunksize] for x in range(jobs_rounds)]
        ids_sents = []  # Free up memory
        # Define partial function to match with chunks
        partial_getValences = partial(getValences, lexiconlist=lexiconlist,
                                      wordlist=allterms, ignore=ignoreX,
                                      mods=mods if modify else dict(),
                                      negs=negs if negate else ())
        # Run parallel processes
        gc.collect()
        valences = []
        for round in range(rounds):
            print "Round %d" % round
            procPool = mp.Pool(processes = nrjobs)
            results = procPool.map(partial_getValences,
                                   enumerate(sentchunks[round * nrjobs: (round + 1) * nrjobs]))
            procPool.close()
            procPool.join()
            procPool.terminate()
            # Combine the results
            for resultnr, result in sorted(results, key=itemgetter(0)):
                valences += result
            results = []  # Empty out results, to free up memory
            gc.collect()

    # Save to output file (may be large!)
    with open(filestem + jobname + '_valences.csv', 'wt') as outf:
        outfile = csv.writer(outf)
        outfile.writerow(['id', 'nrwords'] + lexica2use)
        outfile.writerows(valences)
    return


def LIWCcounts(inputfile, outputfile, lexicafolder, lexica2use, ignorewords,
               textcol=1, header=True):
    """Like corpusvalence but for LIWC-style categories.

    Also not as parallelized &c.
    To apply to larger corpora, incorporate more of the corpusvalence structure.
    """
    import csv
    from lexica import getlexica

    # Load lexica
    lexiconlist, dummy = getlexica(lexicafolder, lexica2use)

    # Generate output header
    outheader = ['id', 'nrwords']
    for liwclex, liwcname in zip(lexiconlist, lexica2use):
        outheader += [liwcname + '_count',] + liwclex[2]

    # Load text data
    idtexts = []
    with open(inputfile, 'rU') as infile:
        inreader = csv.reader(infile)
        if header:
            dummy = next(inreader)

        for row in inreader:
            idtexts.append([row[0], row[textcol]])

    # Calculate word categories
    wordcategorycounts = [[id,] + calc_LIWC(text, lexiconlist, ignorewords, textnr=count) \
                          for count, (id, text) in enumerate(idtexts)]

    # Write out results
    with open(outputfile, 'wb') as outf:
        outfile = csv.writer(outf)
        outfile.writerow(outheader)
        outfile.writerows(wordcategorycounts)
    return


# def corpus_valence_old(filestem, valencedir, lexicafile,
#                    ids=list(), sents=list(), jobname='',
#                    nrjobs=1, sents_per_job=1000000,
#                    modify=True, negate=False):
#     """Calculate sentence-level valences for entire corpus.
#
#     Works with whatever lexica are returned by getlexica.
#     Also calculates corpus-wide, sentence-level averages.
#
#     Try to minimize memory load to avoid running out of application memory.
#     Also run each of the two main computational stages in rounds, for same reason.
#     """
#     import multiprocessing as mp
#     from functools import partial
#     import numpy as np
#     import csv
#     from operator import itemgetter
#     from corpus import readwords
#     import gc
#
#     # Read in ids & sentences if we haven't done so yet
#     if len(ids) == 0 or len(sents) == 0:
#         sents = readwords(filestem + '_lines.txt')
#         ids = readwords(filestem + '_lineids.txt')
#
#     # Calculate various quantities
#     nrsents = len(sents)
#     print "Calculating valences for all %d sentences in the corpus" % (nrsents)
#     if nrjobs > 1:  # parallel processing; calculate nr. rounds
#         rounds = max(1, nrsents / (nrjobs * sents_per_job))  # at least 1 round!
#         jobs_rounds = rounds * nrjobs
#         chunksize = 1 + ((nrsents - 1) / jobs_rounds)
#         print "Run in %d consecutive rounds of %d sentences per job to control memory use" % \
#               (rounds, chunksize)
#
#     # Read in lexica to use
#     stdlex, wildlex, liwclex, mods, lexheaders = \
#         getlexica(filestem, valencedir, lexicafile)
#
#     # 10 most common negaters
#     negs = ('not', 'no', 'nor', 'nothing', 'none', 'never',
#             'nowhere', 'noone', 'nobody', 'neither')
#
#     if nrjobs == 1:
#         # This is an elegant list comprehension, but not parallelizable and hence slow
#         valences = [[id,] + getValence(s, stdlex, wildlex, liwclex,
#                                        mods if modify else dict(),
#                                        negs if negate else (), sentnr=count) \
#                     for count, (id, s) in enumerate(zip(ids, sents))]
#         ids, sents = [], []  # Free up memory
#
#     else:  # parallelize
#         # Divide sentences into smaller chunks.
#         ids_sents = zip(ids, sents)
#         sentchunks = [ids_sents[x * chunksize: (x + 1) * chunksize] for x in range(jobs_rounds)]
#         ids_sents = []  # Free up memory
#         # Define partial function to match with chunks
#         partial_getValences = partial(getValences, stdlex=stdlex, wildlex=wildlex,
#                                       liwclex=liwclex, mods=mods, negs=negs)
#         # Run parallel processes
#         gc.collect()
#         valences = []
#         for round in range(rounds):
#             print "Round %d" % round
#             procPool = mp.Pool(processes = nrjobs)
#             results = procPool.map(partial_getValences,
#                                    enumerate(sentchunks[round * nrjobs: (round + 1) * nrjobs]))
#             procPool.close()
#             procPool.join()
#             procPool.terminate()
#             # Combine the results
#             for resultnr, result in sorted(results, key=itemgetter(0)):
#                 valences += result
#             results = []  # Empty out results, to free up memory
#             gc.collect()
#
#     # Save to output file (may be large!)
#     with open(filestem + jobname + '_valences.csv', 'wt') as outf:
#         outfile = csv.writer(outf)
#         outfile.writerow(['id',] + lexheaders)
#         outfile.writerows(valences)
#     return lexheaders


def avg_valences(filestem, lexiconnames, nrjobs=1):
    """Calculate corpus-level means & std. deviations for valence measures."""
    import csv
    import numpy as np
    import gc
    import multiprocessing as mp

    print "Reading valences from file"
    with open(filestem + '_valences.csv', 'rU') as inf:
        next(inf)  # skip header
        # This next line is inefficient, because most of these items are not floats
        valences = [[float(y) for y in x] for x in csv.reader(inf)]
    nrsents = len(valences)
    chunksize = 1 + ((nrsents - 1) / nrjobs)

    if nrjobs > 1:  # parallel processing; calculate nr. rounds
        sents_per_job = 1000000
        rounds = max(1, nrsents / (nrjobs * sents_per_job))  # at least 1 round!
        jobs_rounds = rounds * nrjobs
        chunksize = 1 + ((nrsents - 1) / jobs_rounds)
        print "Run in %d consecutive rounds of %d sentences per job to control memory use" % \
              (rounds, chunksize)

    # Calculate & display means and standard deviations
    print "\nCalculating sentence-level mean and std.dev. for 10 overall polarity measures:"
    # Scale valence by word length
    if nrjobs == 1:
        valsScaled = [[x[1],] + [val/float(x[1]) if x[1] > 0 else 0 for val in x[2:]] \
                      for x in valences]
        valences = []
    else:
        # Divide into smaller chunks
        valchunks = [valences[x * chunksize: (x + 1) * chunksize] \
                     for x in range(jobs_rounds)]
        valences = []
        gc.collect()  # minimize memory load prior to spawning processes
        # Run parallel processes
        valsScaled = []
        for round in range(rounds):
            print "Round %d" % round
            procPool = mp.Pool(processes = nrjobs)
            results = procPool.map(scalevals, valchunks[round * nrjobs: (round + 1) * nrjobs])
            procPool.close()
            procPool.join()
            procPool.terminate()
            # Combine the results
            for result in results:
                valsScaled += result
            results = []  # Not really necessary
            gc.collect()

    sentMeans = np.mean(valsScaled, axis=0)
    sentStds = np.std(valsScaled, axis=0)
    print ''  # print blank line
    for var_mean_stdev in zip(lexiconnames[1:], sentMeans, sentStds)[1:]:
        print var_mean_stdev

    # Write out aggregates for future recall
    write_corpus_valence(filestem, sentMeans, sentStds, lexiconnames)
    return sentMeans, sentStds


def sent_valence_NL(filestem, ids, sents, lexicafile, LIWCfile=''):
    """Calculate sentence-level valences for entire Dutch corpus."""
    import numpy as np
    import csv
    import pickle

    print "Calculating valences for all sentences in the corpus"
    with open(lexicafile, 'rb') as picklefile:
        moors, antwerpen, agreedict, krantlex = \
            pickle.load(picklefile)
    lexheaders = ['sentlength', 'Moors', 'Antwerpen', 'krantlex']
    stdlex = [moors, antwerpen]
    wildlex = [krantlex,]

    if LIWCfile != '':
        with open(LIWCfile, 'rb') as picklefile:
            LIWC_NL = pickle.load(picklefile)
        liwclex = [(LIWC_NL, len(LIWC_NL[1])),]
        lexheaders += LIWC_NL[2]
    else:
        liwclex = []

    # Hard-code SOCAL-style modifiers
    mods = {'uiterst': 0.4, 'uitermate': 0.4, 'hartstikke': 0.3,
            'ontzettend': 0.3, 'erg': 0.2, 'heel': 0.2, 'zeer': 0.2,
            'bijster': 0.2, 'nogal': 0.1}

    valences = [[id,] + getValence_NL(s, stdlex, wildlex, liwclex, mods) \
                for id, s in zip(ids, sents)]
    # Save to output file (may be large!)
    with open(filestem + '_valences.csv', 'wt') as outf:
        outfile = csv.writer(outf)
        outfile.writerow(['id',] + lexheaders)
        outfile.writerows(valences)

    # Calculate & display means and standard deviations
    valsScaled = [[x[1],] + [val/float(x[1]) if x[1] > 0 else 0 for val in x[2:]] \
                  for x in valences]
    sentMeans = np.mean(valsScaled, axis=0)
    sentStds = np.std(valsScaled, axis=0)
    print "\nSentence-level mean and std.dev. for 10 overall polarity measures:"
    for pair in zip(sentMeans, sentStds)[1:11]:
        print pair
    # Write out aggregates for future recall
    write_corpus_valence(filestem, sentMeans, sentStds, lexheaders)
    return sentMeans, sentStds


# def getlexica_old(filestem, valencedir, lexicafile):
#     """Return all the lexica to be used on the corpus specified.
#
#     Three lexicon sources/types:
#     1) general-purpose lexica
#     2) intersections of those, generated earlier and saved
#     3) corpus-specific lexica, derived from (2) using pmi (and saved under filestem - not currently used)
#     """
#     import pickle
#     from lexica import importValenceDicts, importCorpusDicts, \
#                         importIntersectLexica, recenter_lexicon
#     from operator import itemgetter
#
#     # Get external lexica
#     lexi, socal, liu, WS, labMT, LIWC, moral, NRC, SWN = importValenceDicts(valencedir)
#     # Recenter labMT, which is not 0-centered
#     labMT = recenter_lexicon(labMT, dval=-5)
#     # Combine all lexica into a list
#     stdlexica = [socal[0], liu, WS, labMT, NRC[3], SWN]
#     wildlexica = [lexi,]
#     # Skip LIWC & moral founds. for now; update valencecats below if change!
#     # liwclexica = [(LIWC, len(LIWC[1])), (moral, len(moral[1])),
#     #               (NRC[:2], len(NRC[1]))]
#     liwclexica = [(NRC[:2], len(NRC[1]))]
#
#     unionlex, intersect1, intersect2 = importIntersectLexica(lexicafile)
#     intersectnames = ['unionlex', 'inter4pure', 'inter5impure']
#     stdlexica += [unionlex, intersect1, intersect2]
#
#     # check values
#     # print "Lab MT"
#     # print sorted(labMT.items(), key=itemgetter(1), reverse=True)[:50]
#     # print "swn"
#     # print sorted(SWN.items(), key=itemgetter(1), reverse=True)[:50]
#     # print "unionlex"
#     # print sorted(unionlex.items(), key=itemgetter(1), reverse=True)[:50]
#     # print "lexicoder"
#     # print sorted(lexi.items(), key=itemgetter(1), reverse=True)[:50]
#
#     # Get corpus-specific lexica
#     # corpuslexica, corpusnames = importCorpusDicts(filestem)
#     # stdlexica += corpuslexica
#
#     # Combine intersected and corpus-specific lexica
#     # In testing on imdb_tok, these did not bring improvement, so skip
#     # combolexica, combonames = combinelexica(intersectlexica, corpuslexica)
#     # stdlexica += combolexica
#
#     # Make list of valence category names
#     valencecats = ['sentlength',
#                    'SOCAL', 'BingLiu', 'WordStat', 'labMT', 'NRC', 'SWN'] + \
#                   intersectnames + ['Lexicoder', 'NRCwords'] + NRC[2]
#     return stdlexica, wildlexica, liwclexica, socal[1], valencecats


def combinelexica(lex1, lex2):
    """Systematically combine the lexica in lists 1 and 2.

    Use simple dictionary combination; duplicate keys will get
    the value of the dictionary in the first list.
    """
    # simple dictionary combination is
    return [dict(d2, **d1) for d1 in lex1 for d2 in lex2], \
           ['combo' + str(c1) + '_' + str(c2) \
            for c1 in xrange(len(lex1)) \
            for c2 in xrange(len(lex2))]


# ************************** Apply all lexica to corpus **********************

def uselexica(textfile):
    """Apply lexica to a text file.

    Load lexica
    Calculate valences; calculate correlations
    Calculate optimal threshold for each lexicon; report correlation, accuracy
    Run logisticclassifier on inputs without binarizing
    Run logisticclassifier on inputs with binarizing to optimal threshold
    Run without & with extra dictionaries
    Run without & with extra Lexicoder processing of corpus

    So need:
    1. corpus, tokenized & Americanized
    2. corpus, with additional Lexicoder preprocessing
    3. std. lexica
    4. combined lexica (intersect, merge)
    5. corpus-specific lexica

    So for each corpus:
    - tokenize & Americanize
    - additional Lexicoder preprocessing
    - corpus-specific lexica

    """

    # Run corpus through all of them in parallel (faster)
    # Calculate correlations
    for corpustype in ['tok', 'lex']:
        filestart = filestem + '_' + corpustype
        corpus_valence(filestart)
    # Aggregate back up to doc level √
    # Scale to 0-1 range √
    # Find optimal threshold; report accuracy at that √
    # Save as raw and binarized features (i.e. get 2nd set of vals.) √
    # report mutual correlations within non-binarized & binarized vals √
    # Select subset by accuracy and/or correlation (by hand)
    # - specifically, compare size of extra dicts, and separate or combined
    # Use as inputs to logistic classifier & do std. train-test
    # Can also compare different aggregation methods
    return


# ***************************** Valence auxiliaries **************************

def getValence(sentence, lexiconlist, wordlist, ignore,
               modifiers={}, negaters=(),
               stopwords=('a', 'an', 'and', 'the', 'to', 'as'),
               sentnr=-1, chunknr=0, updateinterval=100000):
    """Get & sum valences for this sentence, optionally using modifiers or negaters.

    Lexica may have wild-cards.
    To handle LIWC-style lexica, see getValence_full (below)

    The modifier lexicon, if supplied, should have the form modifier:mod_fraction

    Note: Lexicoder has a language substitution process that will affect negation handling
    (see the function negation_subst() in language_subst.py)

    Each lexicon is expected to be a tuple: (lexicon, lexwild)
    where lexwild is a boolean indicating whether the lexicon contains wildcards.
    This is more efficient than re-determining this each time we're invoked.

    wordlist is the set of words for which lexica will be invoked.
    Ideally, it should be the union of all terms in the lexica. Keeping it a separate
    parameter also makes it possible to artificially restrict words considered in the
    valence calculation.
    """
    from lexica import lexiconmatch, lexiconmatch_wild, wordinset_wild
    from itertools import chain

    # Progress update
    if sentnr % updateinterval == 0:
        print "Processing sentence %d%s" % \
              (sentnr, ('' if chunknr == 0 else " of chunk %d" % chunknr))

    nrlexica = len(lexiconlist)

    # Add basic modifiers from language preprocessing, if modifiers will be used
    # (these are modifiers added by negation_subst() -- see opening comment)
    if len(modifiers) > 0:
        if 'minusx' not in modifiers:
            modifiers['minusx'] = -0.5
        if 'plusx' not in modifiers:
            modifiers['plusx'] = 1

    # Check each word in the sentence. Start out with no valence modification
    valences = [0, ] * (nrlexica)
    modifier = 1
    skipcounts = []
    sentwords = sentence.lower().split()
    nrwords = len(sentwords)
    for count, word in enumerate(sentwords):
        # Make sure word was not part of a modifier already handled
        # Note: this means that if we use modifiers, any of those words that are in
        # a valence lexicon will not get counted as valence words.
        if count not in skipcounts:

            # Handle modifiers, incl. multi-word modifying phrases
            # Longer phrases trump shorter ones; none is longer than 4 words
            if count < nrwords - 3:
                wordx = '_'.join(sentwords[count:count + 4])
                if wordx in modifiers:
                    skipcounts += [count + 1, count + 2, count + 3]
                    modifier *= 1 + modifiers[wordx]
                    continue
            if count < nrwords - 2:
                wordx = '_'.join(sentwords[count:count + 3])
                if wordx in modifiers:
                    skipcounts += [count + 1, count + 2]
                    modifier *= 1 + modifiers[wordx]
                    continue
            if count < nrwords - 1:
                wordx = '_'.join(sentwords[count:count + 2])
                if wordx in modifiers:
                    skipcounts += [count + 1,]
                    modifier *= 1 + modifiers[wordx]
                    continue
            if word in modifiers:
                modifier *= 1 + modifiers[word]
                continue

            # Check for negation next; note: we get here only if no modifiers caught
            if word in negaters:
                modifier *= -0.5
                continue

            # Look up valences; multiply by modifier value
            # Note 1: we get here only if neither modifiers nor negaters caught
            # Note 2: could simply skip wordinset test and always query each lexicon separately,
            # but this way is probably faster.
            if wordinset_wild(word, wordlist) and not wordinset_wild(word, ignore):
                lexmatches = [modifier * (lexiconmatch_wild(word, lexicon) if lexwild \
                                            else lexiconmatch(word, lexicon)) \
                              for lexicon, lexwild in lexiconlist]
                # Update sentence valences
                valences = [x + y for x, y in zip(valences, lexmatches)]

            # Finally, reset modifier, unless this was a stopword
            if word not in stopwords:
                modifier = 1
    return [nrwords,] + valences


def getValence_full(sent, stdlexica, wildlexica, liwclexica,
               modifiers={}, negaters=(),
               stopwords=('a', 'an', 'and', 'the', 'to', 'as'),
               sentnr=-1, chunknr=0):
    """Sum valences for this sentence, using 3 types of lexicon.

    Standard lexica are dictionaries of word:val
    Wildcard lexica may have words with a * wildcard as the last character
    - both of these can be modified by a modifier lexicon,
      of the form modifier:mod_fraction

    LIWC-style lexica contain several different word lists; no valence
    - to avoid repetitive calculations, each liwc lexicon
      is expected to be passed in as a tuple: (lexicon, nr categories)

    Note: LIWC has a 'long words' category which we could easily incorporate
    (simply add 1 to the values to be calculated, and add len(word > 6) as
    a test) but there is no reason to expect it to be useful for our purposes.

    Note that the negation set-up interacts heavily with the language
    substitution function language_subst.negation_subst()

    The first lexicon in the combined list of stdlexica and wildlexica
    is used to determine whether a word is a valence word. A lexicon that
    has valence values for just about every common word (labMT, sentiWordNet)
    is not a good idea here.
    """
    from lexica import lexiconmatch_multi, lexiconmatch_multiwild, LIWCmatch
    from itertools import chain

    # Progress update
    if sentnr % 100000 == 0:
        print "Processing sentence %d%s" % \
              (sentnr, ('' if chunknr == 0 else " of chunk %d" % chunknr))

    nr_stdlexica = len(stdlexica)
    nr_wildlexica = len(wildlexica)
    nr_liwclexica = len(liwclexica)

    # Add basic modifiers from language preprocessing, if called for
    if len(modifiers) > 0:
        if 'minusx' not in modifiers:
            modifiers['minusx'] = -0.5
        if 'plusx' not in modifiers:
            modifiers['plusx'] = 1

    # nr. values to be calculated: 1 for length of sentence, 1 for each
    # standard polarity lexicon, 1 for nr. of words captured by each
    # LIWC-style lexicon, plus sum(liwclens)
    valences = [0,] * (1 + nr_stdlexica + nr_wildlexica + nr_liwclexica + \
                       sum([x[1] for x in liwclexica]))

    # Check each word in the sentence. Start out with no valence modification
    lexmatches = [0,] * (nr_stdlexica + nr_wildlexica)
    modifier = 1
    skipcounts = []
    wordlist = sent.lower().split()
    nrwords = len(wordlist)
    for count, word in enumerate(wordlist):
        # Make sure word was not part of a modifier already handled
        if count not in skipcounts:

            # Handle modifiers, incl. multi-word modifying phrases
            # Longer phrases trump shorter ones; none is longer than 4 words
            if count < nrwords - 3:
                wordx = '_'.join(wordlist[count:count + 4])
                if wordx in modifiers:
                    skipcounts += [count + 1, count + 2, count + 3]
                    modifier *= 1 + modifiers[wordx]
                    continue
            if count < nrwords - 2:
                wordx = '_'.join(wordlist[count:count + 3])
                if wordx in modifiers:
                    skipcounts += [count + 1, count + 2]
                    modifier *= 1 + modifiers[wordx]
                    continue
            if count < nrwords - 1:
                wordx = '_'.join(wordlist[count:count + 2])
                if wordx in modifiers:
                    skipcounts += [count + 1,]
                    modifier *= 1 + modifiers[wordx]
                    continue
            if word in modifiers:
                modifier *= 1 + modifiers[word]
                continue

            # Check for negation next; note: we get here only if no modifiers caught
            if word in negaters:
                modifier *= -0.5
                continue

            # Look up valences; note: we get here only if neither modifiers nor negaters caught
            lexmatches = lexiconmatch_multi(word, stdlexica) + \
                            lexiconmatch_multiwild(word, wildlexica)
            # If a valence word (in union lexicon), multiply by modifier
            if len(lexmatches) > 0 and lexmatches[6] != 0:
                lexmatches = [modifier * x for x in lexmatches]

            # Finally, reset modifier, unless this was a stopword
            if word not in stopwords:
                modifier = 1

        # LIWC-style lexica may also include modifiers & negation words
        liwcmatches = \
            list(chain.from_iterable([LIWCmatch(word, lex[0], lex[1]) \
                                       for lex in liwclexica]))

        # Update sentence valences based on this word
        wordvalences = [1,] + lexmatches + liwcmatches
        valences = [x + y for x, y in zip(valences, wordvalences)]

    return valences


def calc_LIWC(text, liwclexica, ignore=set(), textnr=-1, chunknr=0):
    """Calculate counts for different LIWC-style categories.

    LIWC-style lexica contain several different word lists
    - to avoid repetitive calculations, each liwc lexicon
      is expected to be passed in as a tuple: (lexicon, nr categories)

    Note: the original LIWC has a 'long words' category which we could easily incorporate
    (simply add 1 to the values to be calculated, and add len(word > 6) as
    a test) but there is no reason to expect it to be useful for our purposes.
    """
    from lexica import LIWCmatch, wordinset_wild
    from itertools import chain

    # Progress update
    if textnr % 100 == 0:
        print "Processing text %d%s" % \
              (textnr, ('' if chunknr == 0 else " of chunk %d" % chunknr))

    nr_liwclexica = len(liwclexica)

    # nr. values to be calculated: 1 for length of sentence,
    # 1 for nr. of words captured by each LIWC-style lexicon, plus sum(liwclens)
    nrvalues = nr_liwclexica + sum([len(x[2]) for x in liwclexica])
    valences = [0,] * (1 + nrvalues)

    # Check each word in the sentence.
    wordlist = text.lower().split()
    for word in wordlist:
        if word == '.':  # skip periods
            continue
        elif wordinset_wild(word, ignore):
            wordvalences = [1,] + [0,] * nrvalues
            valences = [x + y for x, y in zip(valences, wordvalences)]

        else:
            liwcmatches = \
                list(chain.from_iterable([LIWCmatch(word, lex, len(lex[2])) \
                                           for lex in liwclexica]))
            # Update text counts based on this word
            wordvalences = [1,] + liwcmatches
            valences = [x + y for x, y in zip(valences, wordvalences)]

    return valences


def getValence_NL(sent, stdlexica, wildlexica, liwclexica, modifiers={},
                   negaters=('niet', 'nooit', 'geen', 'niets', 'nergens'),
                   stopwords=('de', 'het', 'een', 'en')):
    """Sum valences for this sentence, using 3 Dutch lexica."""
    from lexica import lexiconmatch_multi, lexiconmatch_multiwild, LIWCmatch
    from itertools import chain

    nr_stdlexica = len(stdlexica)
    nr_wildlexica = len(wildlexica)
    nr_liwclexica = len(liwclexica)

    # nr. values to be calculated: 1 for length of sentence, 1 for each
    # standard polarity lexicon, 1 for nr. of words captured by each
    # LIWC-style lexicon, plus sum(liwclens)
    valences = [0, ] * (1 + nr_stdlexica + nr_wildlexica + nr_liwclexica + \
                        sum([x[1] for x in liwclexica]))

    # Check each word in the sentence. Start out with no valence modification
    lexmatches = [0, 0, 0]
    modifier = 1
    hold_mod = ''
    skipcounts = []
    wordlist = sent.lower().split()
    nrwords = len(wordlist)
    # words that are both modifiers and (negative) valence words
    # hold in reserve, apply as modifier only if next word a valence word or another modifier
    mod_and_val = ['ontzettend', 'erg', 'zeer']
    for count, word in enumerate(wordlist):
        # Make sure word was not part of a modifier already handled
        if count not in skipcounts:

            # Handle modifiers, incl. multi-word modifying phrases
            # Longer phrases trump shorter ones; none is longer than 4 words
            if count < nrwords - 3:
                wordx = '_'.join(wordlist[count:count + 4])
                if wordx in modifiers:
                    skipcounts += [count + 1, count + 2, count + 3]
                    modifier *= 1 + modifiers[wordx]
                    continue
            if count < nrwords - 2:
                wordx = '_'.join(wordlist[count:count + 3])
                if wordx in modifiers:
                    skipcounts += [count + 1, count + 2]
                    modifier *= 1 + modifiers[wordx]
                    continue
            if count < nrwords - 1:
                wordx = '_'.join(wordlist[count:count + 2])
                if wordx in modifiers:
                    skipcounts += [count + 1, ]
                    modifier *= 1 + modifiers[wordx]
                    continue
            if word in modifiers:
                if word in mod_and_val:
                    hold_mod = word
                    continue
                modifier *= 1 + modifiers[word]
                continue

            # Check for negation next
            if word in negaters:
                modifier *= -0.5
                continue

            # Look up valences
            lexmatches = lexiconmatch_multi(word, stdlexica) + \
                         lexiconmatch_multiwild(word, wildlexica)
            # If a valence word (in union lexicon), multiply by modifier
            if len(lexmatches) > 0 and any([lex != 0 for lex in lexmatches]):
                if hold_mod != '':
                    modifier *= 1 + modifiers[hold_mod]
                    hold_mod = ''
                lexmatches = [modifier * x for x in lexmatches]

            # Finally, reset modifier, unless this was a stopword
            if word not in stopwords:
                if hold_mod != '':
                    lexmatches = lexiconmatch_multi(word, stdlexica) + \
                                 lexiconmatch_multiwild(word, wildlexica)
                    # If a valence word (in union lexicon), multiply by modifier
                    if len(lexmatches) > 0 and any([lex != 0 for lex in lexmatches]):
                        lexmatches = [modifier * x for x in lexmatches]
            modifier = 1
            hold_mod = ''

        # LIWC-style lexica may also include modifiers & negation words
        liwcmatches = \
            list(chain.from_iterable([LIWCmatch(word, lex[0], lex[1]) \
                                      for lex in liwclexica]))

        # Update sentence valences based on this word
        wordvalences = [1, ] + lexmatches + liwcmatches
        valences = [x + y for x, y in zip(valences, wordvalences)]

    return valences


def getValence_old(sent, lexi, liu, WS, labMT, LIWC, MF, NRC):
    """Calculate variety of valences for this sentence, using 7 lexica."""
    from lexica import lexiconmatch, lexiconmatch_wildcard

    LIWClen = 64  # hard-code nr. of LIWC, MF, and NRC categories
    MFlen = 11
    NRClen = 8
    # nr. values to be calculated: 1 for length of sentence; 5 for polarity;
    # 1 for long words, 1 for nr. words captured by each LIWC-style lexicon,
    # = 10 + plus the number of categories in each of these three lexica
    valences = [0,] * (10 + LIWClen + MFlen + NRClen)
    modifier = 1
    for word in sent.split():
        wordvalences = [1,
                        modifier * lexiconmatch_wildcard(word, lexi),
                        modifier * lexiconmatch(word, liu),
                        modifier * lexiconmatch(word, WS),
                        modifier * lexiconmatch(word, labMT),
                        modifier * lexiconmatch(word, NRC[3]),
                        1 if len(word) > 6 else 0] + \
                        LIWCmatch(word, LIWC, LIWClen) + \
                        LIWCmatch(word, MF, MFlen) + \
                        LIWCmatch(word, NRC[:2], NRClen)
        valences = [x + y for x, y in zip(valences, wordvalences)]
        if word == 'not':
            modifier *= -1
        elif word == 'minusx':
            modifier *= 0.5
        elif word == 'plusx':
            modifier *= 2
        else:
            modifier = 1
    return valences


def cutpoints(valences, vals, altcuts=[0.0, ], measure='percentcorrect'):
    """Calculate best cut-point, by % correct or by F1.

    Will return the valence value _after_ which to cut
        (going from lowest to highest) & associated performance
    Analogously, if alternate cut-points are passed in, will take each of those
        to be the last value to be classified as a 0 & report performance.
    """
    import numpy as np

    # sort by ascending valence; calculate % correct up to point N starting at 0
    sortedcombo_ascend = sorted(zip(vals, valences), key=lambda l: l[1])
    sc_plus = []
    correct = 0
    wrong = 0
    for sc in sortedcombo_ascend:
        if sc[0] == 0:
            correct += 1
        else:
            wrong += 1
        sc_plus.append([sc[0], sc[1], correct, wrong])

    # sort by descending valence; calculate % correct up to point N starting at 0
    sortedcombo_descend = sorted(sc_plus, key=lambda l: l[1], reverse=True)
    sc_plus = []
    prevperformance = [0, 0]
    correct = 0
    wrong = 0
    for sc in sortedcombo_descend:
        if sc[0] == 1:
            correct += 1
        else:
            wrong += 1
        sc_plus.append(sc + prevperformance)
        prevperformance = [wrong, correct]

    # add performance
    perfs = [x + calc_perf(x[2:]) for x in sc_plus]
    savedperfs = []
    sortcrit = 6 if measure == 'percentcorrect' else 7  # 7 is F1
    savedperfs.append(sorted(perfs, key=lambda l: l[sortcrit], reverse=True)[0][1:])
    valencesort = sorted(perfs, key=lambda l: l[1])
    for altcut in altcuts:
        # sort perfs by valence
        # original code: savedperfs.append([x for x in sorted(perfs, key=lambda l: l[1]) if x[1] >= altcut][0][1:])
        found = False
        for x in valencesort:
            if x[1] >= altcut:
                savedperfs.append(x[1:])
                found = True
                break
        if not found:
            # print "Alternative cut point exceeds max valence -> using that value."
            savedperfs.append(valencesort[-1][1:])
    return savedperfs


def calc_perf(confMatrix):
    """Calculate % correct and F1 metrics, given confusion matrix.

    Confusion matrix has form [is0guess0, is1guess0, is0guess1, is1guess1]
    """
    truepos = confMatrix[3]
    falsepos = confMatrix[2]
    falseneg = confMatrix[1]
    trueneg = confMatrix[0]
    precision = 0 if truepos + falsepos == 0 else \
        truepos / float(truepos + falsepos)
    recall = 0 if truepos + falseneg == 0 else \
        truepos / float(truepos + falseneg)
    F1score = 0 if precision + recall == 0 else \
        2 * precision * recall / float(precision + recall)
    accuracy = (truepos + trueneg) / float(sum(confMatrix))
    return [accuracy, F1score]


def assess_perf_imdb(infilename, suppliedmeans, nrlex=1, header=True, calc_avgs=(),
                     negval=0, neutval='', posval=1, valcol=1, keepcols=(0, 1, 3, 4, 5), metalen=5):
    """Assess performance of different lexica, as well as their combination.

    Defaults are set for imdb dataset. Note: partially changed to try to make more generic,
    but not completely followed through. Finish & double-check. Ideally this should be subsumed
    by the plain assess_perf
    """
    import numpy as np
    import csv

    valencecol1 = metalen + 1
    # read in all data
    data = []
    with open(infilename, 'rU') as inf:
        if header:
            headerdata = next(inf)[:-1]  # don't take the newline character
            lexnames = headerdata.split(',')[valencecol1:]
        else:
            lexnames = ['lex' + str(x) for x in range(nrlex)]
        inreader = csv.reader(inf)
        for row in inreader:
            data.append([int(row[x]) for x in keepcols] +
                        [float(row[x]) for x in range(valencecol1, valencecol1 + nrlex)])

    # calculate & display full corpus means
    means = np.mean(data, axis=0)[metalen:]
    # print "means", len(means), means
    # calculate & display full corpus stddevs
    stdevs = np.std(data, axis=0)[metalen:]
    # print stdevs

    # prepare means labeling
    suppliedmeans.append(('corpus', means, stdevs))
    # next 3 lines could be done with zip(*), but this is a bit clearer
    supplied_names = [x[0] for x in suppliedmeans]
    supplied_m = [x[1] for x in suppliedmeans]
    supplied_s = [x[2] for x in suppliedmeans]

    means_bycol = zip(*supplied_m)
    stds_bycol = zip(*supplied_s)
    labeleddata = [x for x in data if x[valcol] >= 0]
    display_perf(zip(*labeleddata), 'all labeled', means_bycol, supplied_names,
                 stds_bycol, lexnames, calc_avgs)
    testdata = [x for x in labeleddata if x[3] == 1]
    display_perf(zip(*testdata), 'test', means_bycol, supplied_names,
                 stds_bycol, lexnames, calc_avgs)


def assess_perf(infilename, suppliedmeans, nrlex=1, header=True, calc_avgs=(),
                negval=0, neutval='', posval=1,
                valcol=1, ratingscol=1, lengthcol=1,
                includecol='', valencecol1=2, measure='percentcorrect'):
    """Assess performance of different lexica, as well as their combination.

    Defaults are set for imdb dataset.
    """
    import numpy as np
    import csv

    # read in all data
    data = []
    posvals = 0
    negvals = 0
    with open(infilename, 'rU') as inf:
        if header:
            headerdata = next(inf)[:-1]  # don't take the newline character
            lexnames = headerdata.split(',')[valencecol1:]
        else:
            lexnames = ['lex' + str(x) for x in range(nrlex)]
        inreader = csv.reader(inf)
        for row in inreader:
            label = int(row[valcol])  # Assume labels are given as integers
            if label == negval:
                negvals += 1
                val = 0
            elif label == posval:
                posvals += 1
                val = 1
            else:
                continue  # For now, skip any neutral items
            # Assume rating and length are numeric
            if includecol == '' or int(row[includecol]) == 1:
                data.append([int(row[0]), val, float(row[ratingscol]), float(row[lengthcol])] +
                            [float(row[x]) for x in range(valencecol1, valencecol1 + nrlex)])

    # report number of positive and negative classes
    print "Working with %d negative and %d positive texts." % (negvals, posvals)

    firstvalencecol = 4  # after id, label, rating, length
    # calculate & display full corpus means
    means = np.mean(data, axis=0)[firstvalencecol:]
    # print "means", len(means), means
    # calculate & display full corpus stddevs
    stdevs = np.std(data, axis=0)[firstvalencecol:]
    # print stdevs

    # prepare means labeling
    suppliedmeans.append(('corpus', means, stdevs))
    # next 3 lines could be done with zip(*), but this is a bit clearer
    supplied_names = [x[0] for x in suppliedmeans]
    supplied_m = [x[1] for x in suppliedmeans]
    supplied_s = [x[2] for x in suppliedmeans]


    means_bycol = zip(*supplied_m)
    stds_bycol = zip(*supplied_s)
    display_perf(zip(*data), 'selected', means_bycol, supplied_names,
                 stds_bycol, lexnames, calc_avgs, firstvalencecol, measure)


def display_perf(cols, corpusname, means_bycol, means_names, stds_bycol, lexnames,
                 calc_avgs=(), firstvalencecol=4, measure='percentcorrect'):
    """Calculate & display performance cutting at means, optimal cutpoints.

    Hard-coded to assume 3 versions of each lexicon (full, negs only, no mods).
    """
    import numpy as np

    results = []
    labels = cols[1]
    ratings = cols[2]
    lengths = cols[3]
    metainfo = zip(labels, ratings, lengths)

    nrlex = len(lexnames)
    nrlexsub = nrlex/3

    for colindex in range(firstvalencecol, firstvalencecol + nrlex):
        results.append(cutpoints(cols[colindex], labels,
                                 altcuts=means_bycol[colindex - firstvalencecol],
                                 measure=measure))

    # Individual lexicon results
    print "\nResults for %s data" % corpusname
    headerstring = measure + " performance(cut-point), in order: optimal"
    for name in means_names:
        headerstring += ", %s" % (name)
    print headerstring

    for counter, (lexname, result) in enumerate(sorted(zip(lexnames, results))):
        if counter % 3 == 0:  # insert blank line between lexica
            print
        perfvalue = 100 * result[0][5] if measure == 'percentcorrect' else result[0][6]
        resultstring = "%26s: %5.2f(%5.3f)" % (lexname, perfvalue, result[0][0])
        for perfdata in result[1:]:
            perfvalue = 100 * perfdata[5] if measure == 'percentcorrect' else perfdata[6]
            resultstring += "; %5.2f(%5.3f)" % (perfvalue, perfdata[0])
        print resultstring

    # See if we need to display averages too
    if len(calc_avgs) > 0:

        # Standardize data (keep non-valence-cols untouched!)
        labeledvals = cols[firstvalencecol:]

        newmeans_bycol = []
        newstds_bycol = []
        for col, means, stds in zip(labeledvals, means_bycol, stds_bycol):
            newmeans_bycol.append([np.mean(col), ] + list(means))
            newstds_bycol.append([np.std(col), ] + list(stds))

        means_byset = zip(*newmeans_bycol)
        stds_byset = zip(*newstds_bycol)
        means_names = ['corpusmean', ] + means_names

        for name, means, stds in zip(means_names, means_byset, stds_byset):
            print "\n*** Average using set of means: %s ***" % name

            # Standardize all values based on the cut-offs ('means') specified
            labeledvals_std = []
            for col, colmean, colstd in zip(labeledvals, means, stds):
                labeledvals_std.append([(x - colmean) / colstd for x in col])

            # Calculate & display performance of different combinations specified
            print "\nResults for specified averages (3 conditions for each spec)"

            for thisavg in calc_avgs:  # loop over each column combination specified

                print "\nSpec: ", thisavg
                for x in range(3):  # loop over 3 sets of output files
                    col0 = x * nrlexsub
                    # average across the columns specified
                    avgvals = np.mean([labeledvals_std[col0 + x] for x in thisavg], axis=0)
                    cutoff = np.mean(avgvals)
                    result = cutpoints(avgvals, labels, altcuts=[cutoff, 0.0], measure=measure)

                    perfvalue = 100 * perfdata[5] if measure == 'percentcorrect' else perfdata[6]

                    print "\nCut (0): %5.2f(%5.3f); cut (mean): %5.2f(%5.3f); cut (opt): %5.2f(%5.3f)" % \
                          (100 * result[2][5] if measure == 'percentcorrect' else result[2][6], result[2][0],
                           100 * result[1][5] if measure == 'percentcorrect' else result[1][6], result[1][0],
                           100 * result[0][5] if measure == 'percentcorrect' else result[0][6], result[0][0])

                    # Calculate success data
                    combo = zip(avgvals, metainfo)

                    # collect valences, ratings, and lengths for confusion matrix entries
                    posgood = [(x, y[1], y[2]) for x, y in combo if (x > cutoff and y[0] == 1)]
                    neggood = [(x, y[1], y[2]) for x, y in combo if (x <= cutoff and y[0] == 0)]
                    poserror = [(x, y[1], y[2]) for x, y in combo if (x > cutoff and y[0] == 0)]
                    negerror = [(x, y[1], y[2]) for x, y in combo if (x <= cutoff and y[0] == 1)]

                    posgood_items = zip(*posgood)
                    neggood_items = zip(*neggood)
                    poserror_items = zip(*poserror)
                    negerror_items = zip(*negerror)

                    print "Correctly classified items:"
                    print "Label 1 & +ve: nr %d, mean %4.2f, max %4.2f, stdev %4.2f, rating %3.1f, length %6.2f" % \
                          (len(posgood), np.mean(posgood_items[0]), max(posgood_items[0]),
                           np.std(posgood_items[0]), np.mean(posgood_items[1]), np.mean(posgood_items[2]))
                    print "Label 0 & -ve: nr %d, mean %4.2f, min %4.2f, stdev %4.2f, rating %3.1f, length %6.2f" % \
                          (len(neggood), np.mean(neggood_items[0]), min(neggood_items[0]),
                           np.std(neggood_items[0]), np.mean(neggood_items[1]), np.mean(neggood_items[2]))

                    print "Incorrectly classified items:"
                    print "Label 0 but +ve: nr %d, mean %4.2f, max %4.2f, stdev %4.2f, rating %3.1f, length %6.2f" % \
                          (len(poserror), np.mean(poserror_items[0]), max(poserror_items[0]),
                           np.std(poserror_items[0]), np.mean(poserror_items[1]), np.mean(poserror_items[2]))
                    print "Label 1 but -ve: nr %d, mean %4.2f, min %4.2f, stdev %4.2f, rating %3.1f, length %6.2f" % \
                          (len(negerror), np.mean(negerror_items[0]), min(negerror_items[0]),
                           np.std(negerror_items[0]), np.mean(negerror_items[1]), np.mean(negerror_items[2]))


def display_perf_imdb(cols, corpusname, means_bycol, means_names, stds_bycol, lexnames, calc_avgs=()):
    """Calculate & display performance cutting at means, optimal cutpoints."""
    import numpy as np

    results = []
    labels = cols[1]
    ratings = cols[2]
    lengths = cols[4]
    metainfo = zip(labels, ratings, lengths)

    nrlex = len(lexnames)
    nrlexsub = nrlex/3

    for colindex in range(5, 5 + nrlex):
        results.append(cutpoints(cols[colindex], labels, altcuts=means_bycol[colindex - 5]))

    # Individual lexicon results
    print "\nResults for %s data" % corpusname
    headerstring = "Cut-points & associated performance, in order: optimal"
    for name in means_names:
        headerstring += ", %s" % (name)
    print headerstring

    for counter, (lexname, result) in enumerate(sorted(zip(lexnames, results))):
        if counter % 3 == 0:
            print
        resultstring = "%26s: %5.3f/%5.2f%%" % (lexname, result[0][0], 100 * result[0][5])
        for perfdata in result[1:]:
            resultstring += "; %5.3f/%5.2f%%" % (perfdata[0], 100 * perfdata[5])
        print resultstring

    # See if we need to display averages too
    if len(calc_avgs) > 0:

        # Standardize data (keep cols 0-4 untouched!)
        # labeledinfo = cols[:5]
        labeledvals = cols[5:]

        newmeans_bycol = []
        newstds_bycol = []
        for col, means, stds in zip(labeledvals, means_bycol, stds_bycol):
            newmeans_bycol.append([np.mean(col), ] + list(means))
            newstds_bycol.append([np.std(col), ] + list(stds))

        means_byset = zip(*newmeans_bycol)
        stds_byset = zip(*newstds_bycol)
        means_names = ['corpusmean', ] + means_names

        for name, means, stds in zip(means_names, means_byset, stds_byset):
            print "\n*** Average using set of means: %s ***" % name

            # Standardize all values based on the cut-offs ('means') specified
            labeledvals_std = []
            for col, colmean, colstd in zip(labeledvals, means, stds):
                labeledvals_std.append([(x - colmean) / colstd for x in col])

            # Calculate & display performance of different combinations specified
            print "\nResults for specified averages (3 conditions for each spec)"

            for thisavg in calc_avgs:  # loop over each column combination specified

                print "\nSpec: ", thisavg
                for x in range(3):  # loop over 3 sets of output files
                    col0 = x * nrlexsub
                    # average across the columns specified
                    avgvals = np.mean([labeledvals_std[col0 + x] for x in thisavg], axis=0)
                    cutoff = np.mean(avgvals)
                    result = cutpoints(avgvals, labels, altcuts=[cutoff, 0.0])

                    print "\nCut (0): %5.3f - %5.3f%%; cut (mean): %5.3f - %5.3f%%; cut (opt): %5.3f - %5.3f%%" % \
                          (result[2][0], 100 * result[2][5],
                           result[1][0], 100 * result[1][5],
                           result[0][0], 100 * result[0][5])

                    # Calculate success data
                    combo = zip(avgvals, metainfo)

                    # collect valences, ratings, and lengths for confusion matrix entries
                    posgood = [(x, y[1], y[2]) for x, y in combo if (x > cutoff and y[0] == 1)]
                    neggood = [(x, y[1], y[2]) for x, y in combo if (x <= cutoff and y[0] == 0)]
                    poserror = [(x, y[1], y[2]) for x, y in combo if (x > cutoff and y[0] == 0)]
                    negerror = [(x, y[1], y[2]) for x, y in combo if (x <= cutoff and y[0] == 1)]

                    posgood_items = zip(*posgood)
                    neggood_items = zip(*neggood)
                    poserror_items = zip(*poserror)
                    negerror_items = zip(*negerror)

                    print "Correctly classified items:"
                    print "Label 1 & +ve: nr %d, mean %4.2f, max %4.2f, stdev %4.2f, rating %3.1f, length %6.2f" % \
                          (len(posgood), np.mean(posgood_items[0]), max(posgood_items[0]),
                           np.std(posgood_items[0]), np.mean(posgood_items[1]), np.mean(posgood_items[2]))
                    print "Label 0 & -ve: nr %d, mean %4.2f, min %4.2f, stdev %4.2f, rating %3.1f, length %6.2f" % \
                          (len(neggood), np.mean(neggood_items[0]), min(neggood_items[0]),
                           np.std(neggood_items[0]), np.mean(neggood_items[1]), np.mean(neggood_items[2]))

                    print "Incorrectly classified items:"
                    print "Label 0 but +ve: nr %d, mean %4.2f, max %4.2f, stdev %4.2f, rating %3.1f, length %6.2f" % \
                          (len(poserror), np.mean(poserror_items[0]), max(poserror_items[0]),
                           np.std(poserror_items[0]), np.mean(poserror_items[1]), np.mean(poserror_items[2]))
                    print "Label 1 but -ve: nr %d, mean %4.2f, min %4.2f, stdev %4.2f, rating %3.1f, length %6.2f" % \
                          (len(negerror), np.mean(negerror_items[0]), min(negerror_items[0]),
                           np.std(negerror_items[0]), np.mean(negerror_items[1]), np.mean(negerror_items[2]))

            #         averages.append(avg)
            #
            # # get the corpus means for each of these averages
            # # avgmeans = np.mean(averages, axis=1)
            #
            # # display results
            # print "\nResults for specified averages (3 conditions for each spec)"
            #
            # for counter, avgvals in enumerate(averages):
            #     avgmean = np.mean(avgvals)
            #     results.append(cutpoints(avgvals, labels, altcuts=[avgmean, 0.0]))
            #
            # print "\nResults for specified averages (3 conditions for each spec)"
            # for counter, (result, average) in enumerate(zip(results, averages)):
            #     if counter % 3 == 0:
            #         print  # blank line between different average specs
            #     print "Cut (0): %5.3f - %5.3f%%; cut (mean): %5.3f - %5.3f%%; cut (opt): %5.3f - %5.3f%%" % \
            #           (result[2][0], 100 * result[2][5],
            #            result[1][0], 100 * result[1][5],
            #            result[0][0], 100 * result[0][5])
            #
            #     # Calculate success data
            #     combo = zip(average, metainfo)
            # cutoff = avgmeans[2]
            #
            # # collect valences, ratings, and lengths for confusion matrix entries
            # posgood = [(x, y[1], y[2]) for x, y in combo if (x > cutoff and y[0] == 1)]
            # neggood = [(x, y[1], y[2]) for x, y in combo if (x <= cutoff and y[0] == 0)]
            # poserror = [(x, y[1], y[2]) for x, y in combo if (x > cutoff and y[0] == 0)]
            # negerror = [(x, y[1], y[2]) for x, y in combo if (x <= cutoff and y[0] == 1)]
            #
            # posgood_items = zip(*posgood)
            # neggood_items = zip(*neggood)
            # poserror_items = zip(*poserror)
            # negerror_items = zip(*negerror)
            #
            # print "\nSuccess assessment for 10-lexicon average"
            # print "Label 1 & +ve: nr %d, mean %4.2f, max %4.2f, stdev %4.2f, rating %3.1f, length %6.2f" % \
            #       (len(posgood), np.mean(posgood_items[0]), max(posgood_items[0]),
            #        np.std(posgood_items[0]), np.mean(posgood_items[1]), np.mean(posgood_items[2]))
            # print "Label 0 & -ve: nr %d, mean %4.2f, min %4.2f, stdev %4.2f, rating %3.1f, length %6.2f" % \
            #       (len(neggood), np.mean(neggood_items[0]), min(neggood_items[0]),
            #        np.std(neggood_items[0]), np.mean(neggood_items[1]), np.mean(neggood_items[2]))
            #
            # print "\nError assessment for 10-lexicon average"
            # print "Label 0 but +ve: nr %d, mean %4.2f, max %4.2f, stdev %4.2f, rating %3.1f, length %6.2f" % \
            #       (len(poserror), np.mean(poserror_items[0]), max(poserror_items[0]),
            #        np.std(poserror_items[0]), np.mean(poserror_items[1]), np.mean(poserror_items[2]))
            # print "Label 1 but -ve: nr %d, mean %4.2f, min %4.2f, stdev %4.2f, rating %3.1f, length %6.2f" % \
            #       (len(negerror), np.mean(negerror_items[0]), min(negerror_items[0]),
            #        np.std(negerror_items[0]), np.mean(negerror_items[1]), np.mean(negerror_items[2]))


def create_testcase(textfile, valencefile, outputfile):
    """Combine texts & associated valences into a csv file. Supply ids."""
    import csv

    with open(textfile, 'rt') as intexts, open(valencefile, 'rt') as invals, \
            open(outputfile, 'wb') as outdata:
        outwriter = csv.writer(outdata)
        for counter, (intext, inval) in enumerate(zip(intexts, invals)):
            # if len(intext) == 0:
            #     print "text error on line %d" % counter
            # if len(val) == 0:
            #     print "valence error on line %d" % counter
            outwriter.writerow((counter, intext, inval))


# ****************************** corpus summary I/O **************************

def read_corpus_valence(valenceAvgFile):
    """Read mean & std.dev. valence info from existing file."""
    import csv

    print "Reading mean & std.dev. of valences from file."
    sentMeans = []
    sentStds = []
    with open(valenceAvgFile, 'rb') as valenceFile:
        for row in csv.reader(valenceFile):
            sentMeans.append(float(row[1]))
            sentStds.append(float(row[2]))
    return sentMeans, sentStds


def write_corpus_valence(filestem, means, stdevs, valencecats):
    """Save overall corpus valence data, along with name for each value."""
    import csv

    with open(filestem + '_valenceAvgs.csv', 'wb') as valenceFile:
        valenceData = csv.writer(valenceFile)
        valenceData.writerows(zip(valencecats, means, stdevs))


# ************************* Sentencecount-based features *********************

def nr_cond(ids, conds, truthval):
    """For each id, count the number of truthvals for cond."""
    curid = -999
    curcount = 0
    idcounts = []
    for id, cond in zip(ids, conds):
        if id <> curid:
            idcounts.append((curid, curcount))
            curcount = 1 if cond == truthval else 0
            curid = id
        elif cond == truthval:
            curcount += 1
    return idcounts[1:]


def frac_cond(ids, conds, truthval):
    """For each id, count the share of truthvals for cond."""
    curid = -999
    curcount = 0
    curnr = 0
    idcounts = []
    for id, cond in zip(ids, conds):
        if id <> curid:
            idcounts.append((curid, curcount/float(curnr)))
            curcount = 1 if cond == truthval else 0
            curnr = 1
            curid = id
        else:
            curnr += 1
            if cond == truthval:
                curcount += 1
    return idcounts[1:]


# ***************************** Generic auxiliaries **************************

def plothist(series1x, series2):
    """Plot histograms of 2 series."""
    import matplotlib.pyplot as plt

    # Try to get the same x scales
    series1 = [x/20 for x in series1x]
    ax1 = plt.subplot(111)
    ax1.hist(series1, bins=100, histtype='step', normed=True, color='b',
             label='articles')
    ax1.set_ylabel('#art.', color='b')
    for tl in ax1.get_yticklabels():
        tl.set_color('b')
    ax2 = ax1.twinx()
    ax2.hist(series2, bins=100, histtype='step', normed=True, color='r',
             label='sentences')
    ax2.set_ylabel('#sent.', color='r')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')
    plt.legend()
    plt.show()


# ****************************** Deprecated/unused ***************************

def scalebyNrwords(x):
    "Divide all entries except the first by the first (i.e. word count)."
    return [x[0],] + [val/float(x[0]) for val in x[1:]]


def valence2csv(valencefile):
    """Read pickle file with article-level valence data; write as csv."""
    import csv, pickle

    # Load from pickle file
    with open(valencefile, 'rb') as valenceF:
        textValences = pickle.load(valenceF)
    valenceData = [[x[0],] + x[1] for x in textValences]
    # Write to csv file
    outF = '.'.join(valencefile.split('.')[:-1]) + '_valences.csv'
    with open(outF, 'wb') as outFile:
        outValences = csv.writer(outFile)
        outValences.writerows(valenceData)


# Todo: make both of these internal to filterValence
# Global variable associated with reportcount
textCount = 0

def reportcount(val):
    """Increment and report on counter; just pass input parameter back."""
    global textCount
    textCount += 1
    if textCount % 200 == 0:
        print textCount
    return val


# ***************************** External invocation **************************

# For command-line invocation
if __name__ == "__main__":
    import argparse

    # Specify all the arguments and flags
    parser = argparse.ArgumentParser(
        description='text valence calculating module')
    parser.add_argument('filestem',
                        help="filestem for input files")
    # Extract the parameter and pass to the main function
    args = parser.parse_args()
    corpus_valence(args.filestem)
