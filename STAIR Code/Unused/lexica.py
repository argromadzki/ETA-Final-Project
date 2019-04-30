# encoding: utf-8

# lexica.py
# by Maurits van der Veen
# last modified 2017-05-09

# Functions to read in various sentiment analysis lexica
# and save in pickle format, after some minimal processing.

# Also include code to return processed lexica to sentiment analysis functions.

# TODO - deal with 'ire' in Lexicoder
# TODO - check through other dictionaries

# ******************************** Merging lexica ****************************

def mergeLexica(valencedir):
    """Intersect & union the various sentiment analysis lexica we have.

    Produce the intersection of words that appear (with the same valence!)
    in all of them. Flag words with opposing valences along the way.

    Returned lexicon will be a dictionary of format word: valence,
    where valence = 1 (positive) or -1 (negative). This requires rounding/
    truncating fractional values from SO-CAL, labMT, SentiWordNet.

    Lexica included:
    liu: dictionary with values of -1 or 1 (6786 words)
    WS: dictionary with values of -1 or 1 (15078 words)
    NRC: dictionary with values of -1, 0 or 1 (14182 words)
    Lexicoder: dictionary with values of -1 or 1, with wild-cards
    labMT: dictionary with fractional values from 1.3 to 8.5
           (in theory, 1 to 9), rescaled by subtracting 5, and
           filtered by removing everything between -1 and +1
           (this reduces list length from 10222 to 3731)
    SWN: dictionary with fractional values from -1 to 1
    SO-CAL: dictionary with integer values of -5 to +5 (10066 words)

    Not included:
    LIWC - has categories for positive/negative emotion,
            but not a polarity dictionary
    moral - not a polarity dictionary
    """
    from itertools import chain
    import pickle

    lexi, socal, liu, WS, labMT, LIWC, moral, NRC, SWN = \
        importValenceDicts(valencedir)

    # Ternarize the 2 dictionaries with a full range
    # labMT: recenter by -5; then dval 1 leaves 3731 (of 10222) valence words
    labMT = ternarize_lexicon(recenter_lexicon(labMT, dval=-5), dval=1)
    # SWN: dval 0.1 results in 24222 out of 83499 having a value of +1/-1
    SWNdict = filter_neutrals(ternarize_lexicon(SWN, dval=0.1))

    # Binarize the SO-CAL dictionary
    socal = binarize_lexicon(socal[0])

    # For NRC, take only the lexicon, not the other categories
    NRCdict = NRC[3]

    # For Lexicoder, rely on external word list to expand wildcards
    englishlist = globalEnglish()
    lexikeys = [key for key in lexi if key[-1] != '*']
    lexiwilds = list(chain.from_iterable(
            [wildcard_expand(key[:-1], englishlist) \
                 for key in lexi if key[-1] == '*']))
    lexikeys = set(lexikeys + lexiwilds)

    # Generate a union of all keys
    lexiconkeys = lexikeys.union(
                    set(socal.keys()), set(WS.keys()), set(labMT.keys()),
                    set(NRCdict.keys()), set(SWNdict.keys()), set(liu.keys()))

    lexiconlist = [socal, liu, WS, labMT, NRCdict, SWNdict]
    intersectlist, includecode, countercode, unionlist, confusedlist = \
        {}, {}, {}, {}, {}
    counts = [0, 0, 0, 0]
    offset = 4
    oneonelist = []

    # For each word, count positive & negative appearances in lexica
    for word in lexiconkeys:
        vals = lexiconmatch_multi(word, lexiconlist) + \
               [lexiconmatch_wild(word, lexi),]
        posvals = len([1 for v in vals if v == 1])
        negvals = len([1 for v in vals if v == -1])
        # Generate intersection lexicon for words in majority of lexica
        for nr in xrange(4, 8):
            if posvals == nr or negvals == nr:
                intersectlist[word] = 1 if posvals == nr else -1
                includecode[word] = nr
                countercode[word] = posvals if negvals == nr else negvals
                counts[nr - offset] += 1
        # Store conflicts/inconsistencies
        if (posvals >= 2 and negvals >= 2):
            confusedlist[word] = vals
        elif (posvals == 1 and negvals == 1):
            oneonelist.append(word)
        elif posvals + negvals > 0:
            # Generate union lexicon where there are no severe inconsistencies
            unionlist[word] = 1 if posvals > negvals else -1

    # Display results
    print "\nWords in all lists: %d" % len(lexiconkeys)
    print "Words in union: %d" % len(unionlist)
    print "Words with non-negligible conflict: %d" % len(confusedlist)
    print "Words with 1 positive & 1 negative: %d" % len(oneonelist)
    print "Words in intersection by stringency: 7 - %d, 6 - %d, 5 - %d, 4 - %d" % \
          (counts[3], counts[2], counts[1], counts[0])

    print "\nWords in that match in all 7 dictionaries (%d +ve):" % \
            len([1 for key, val in intersectlist.iteritems() \
               if includecode[key] == 7 and intersectlist[key] == 1])
    print "\nPositive:", [key for key, val in intersectlist.iteritems() \
                        if includecode[key] == 7 and intersectlist[key] == 1]
    print "\nNegative:", [key for key, val in intersectlist.iteritems() \
                        if includecode[key] == 7 and intersectlist[key] == -1]

    print "\nWords with conflict (valence order: socal, liu, WS, labMT, NRC, SWN, lexi):"
    for key, val in confusedlist.iteritems():
        print key, val, "pos:", len([1 for x in val if x == 1]), \
                'neg:', len([1 for x in val if x == -1])
    # Save results
    # with open('MergedLexicon.pkl', 'wb') as outf:
    #     pickle.dump((unionlist, intersectlist, includecode,
    #                  countercode, confusedlist), outf)
    return


def mergelexica2(valencedir, lexica2use):
    """Intersect & union the various sentiment analysis lexica we have.

    Like mergelexica, but newer, and saves metadictionaries directly,
    rather than saving a more complex set of information about words in lexica
    (see commented-out code at the end of mergelexica)

    Saves 3 output files:
    - agree4disagree0
    - agree5disagree1
    - uniondisagree1
    """
    from itertools import chain
    import pickle

    lexiconlist = []
    lexiconkeys = set()
    wildlexicon = []
    for lexicon in lexica2use:
        with open (valencedir + lexicon + '.pkl', 'rb') as lexpickle:
            lexicondict = pickle.load(lexpickle)
        lexiconlist.append(lexicondict)
        lexiconkeys |= set(lexicondict.keys())
        wildlexicon.append(lexicon_haswilds(lexicondict))

    nrlexica = len(lexica2use)
    intersectlist, includecode, countercode, unionlist, confusedlist = \
        {}, {}, {}, {}, {}
    counts = [0,] * (nrlexica + 1)
    oneonelist = []

    # For each word, count positive & negative appearances in lexica
    for word in lexiconkeys:
        vals = [lexiconmatch_wild(word, lexicon) if wildlex else lexiconmatch(word, lexicon) \
                for lexicon, wildlex in zip(lexiconlist, wildlexicon)]
        posvals = len([1 for v in vals if v > 0])
        negvals = len([1 for v in vals if v < 0])

        # Generate intersection lexicon for words in majority of lexica
        for nr in xrange((nrlexica + 1)/2, nrlexica + 1):
            if posvals == nr or negvals == nr:
                intersectlist[word] = 1 if posvals == nr else -1
                includecode[word] = nr
                countercode[word] = posvals if negvals == nr else negvals
                counts[nr] += 1

        # Store conflicts/inconsistencies
        if (posvals >= 2 and negvals >= 2):
            confusedlist[word] = vals
        elif (posvals == 1 and negvals == 1):
            oneonelist.append(word)
        elif posvals + negvals > 0:
            # Generate union lexicon where there are no severe inconsistencies
            unionlist[word] = 1 if posvals > negvals else -1

    # Remove entries in union list subsumed by wildcards
    todelete = [x for x in unionlist if subsumed(x, unionlist, report=False)]
    print "%d words in union subsumed by wildcards" % len(todelete)
    for subsumedword in todelete:
        del unionlist[subsumedword]

    # Display results
    # Note: here we have hard-coded the assumption that len(lexica2use) == 8
    print "\nWords/terms in all lists: %d" % len(lexiconkeys)
    print "Words in union: %d" % len(unionlist)
    print "Words with non-negligible conflict: %d" % len(confusedlist)
    print "Words with 1 positive & 1 negative: %d" % len(oneonelist)
    print "Words in intersection by stringency: 8 - %d, 7 - %d, 6 - %d, 5 - %d, 4 - %d" % \
          (counts[8], counts[7], counts[6], counts[5], counts[4])

    print "\nWords in that match in all 8 dictionaries (%d +ve):" % \
            len([1 for key, val in intersectlist.iteritems() \
               if includecode[key] == 8 and intersectlist[key] == 1])
    print "\nPositive:", [key for key, val in intersectlist.iteritems() \
                        if includecode[key] == 8 and intersectlist[key] == 1]
    print "\nNegative:", [key for key, val in intersectlist.iteritems() \
                        if includecode[key] == 8 and intersectlist[key] == -1]

    print "\nWords with conflict (valences in order supplied):"
    for key, val in confusedlist.iteritems():
        print key, val, "pos:", len([1 for x in val if x > 0]), \
                'neg:', len([1 for x in val if x < 0])

    # Save results
    with open(valencedir + 'uniondisagree1.pkl', 'wb') as outpickle:
        pickle.dump(unionlist, outpickle)
    int4pure = {key: val for key, val in intersectlist.iteritems() \
                if includecode[key] >= 4 and countercode[key] == 0}
    with open(valencedir + 'agree4disagree0.pkl', 'wb') as outpickle:
        pickle.dump(int4pure, outpickle)
    int5 = {key: val for key, val in intersectlist.iteritems() \
                if includecode[key] >= 5 and countercode[key] <= 1}
    with open(valencedir + 'agree5disagree1.pkl', 'wb') as outpickle:
        pickle.dump(int5, outpickle)

    # Code to save all info; that is then used below in intersectlexica()
    # with open('MergedLexicon.pkl', 'wb') as outf:
    #     pickle.dump((unionlist, intersectlist, includecode,
    #                  countercode, confusedlist), outf)
    return


def allkeys(lexiconlist):
    """Generate a list of all terms in any of the lexica.

    Filter out any terms subsumed by terms with wildcards
    """
    # Combine all keys into one large set
    lexiconkeys = set()
    for lexicon in lexiconlist:
        lexiconkeys |= set(lexicon.keys())
    # Remove entries subsumed by wildcards
    todelete = [x for x in lexiconkeys if subsumed(x, lexiconkeys, report=False)]
    for subsumedword in todelete:
        lexiconkeys.remove(subsumedword)
    # Return the resulting set
    return lexiconkeys


def negLexica(valencedir):
    """Find most negative words we have.

    Use the 3 graded lexica:
    labMT: dictionary with fractional values from 1.3 to 8.5
           (in theory, 1 to 9), rescaled by subtracting 5, and
           filtered by removing everything between -1 and +1
           (this reduces list length from 10222 to 3731)
    SWN: dictionary with fractional values from -1 to 1
    SO-CAL: dictionary with integer values of -5 to +5 (10066 words)
    """
    from operator import itemgetter

    lexi, socal, liu, WS, labMT, LIWC, moral, NRC, SWN = \
        importValenceDicts(valencedir)

    negwords1 = {word: val for word, val in labMT.iteritems() if val <= 2.1}
    negwords2 = {word: val for word, val in SWN.iteritems() if val <= -0.9}
    negwords3 = {word: val for word, val in socal[0].iteritems() if val <= -5}
    negwords3 = {(word if '\t' not in word else word[:-6] + 's'): val for word, val in negwords3.iteritems()}
    print "Number of negative words contributed: labMT - %d, SWN - %d, SO-CAL - %d" % \
          (len(negwords1), len(negwords2), len(negwords3))
    negwords = sorted(list(set(negwords1.keys() + negwords2.keys() + negwords3.keys())))
    print "Number total: %d" % len(negwords)
    print negwords


def posLexica(valencedir):
    """Find most positive words we have.

    Use the 3 graded lexica:
    labMT: dictionary with fractional values from 1.3 to 8.5
           (in theory, 1 to 9), rescaled by subtracting 5, and
           filtered by removing everything between -1 and +1
           (this reduces list length from 10222 to 3731)
    SWN: dictionary with fractional values from -1 to 1
    SO-CAL: dictionary with integer values of -5 to +5 (10066 words)
    """
    from operator import itemgetter

    lexi, socal, liu, WS, labMT, LIWC, moral, NRC, SWN = \
        importValenceDicts(valencedir)

    poswords1 = {word: val for word, val in labMT.iteritems() if val >= 7.7}
    poswords2 = {word: val for word, val in SWN.iteritems() if val >= 0.9}
    poswords3 = {word: val for word, val in socal[0].iteritems() if val >= 5}
    poswords3 = {(word if '\t' not in word else word[:-6] + 's'): val for word, val in poswords3.iteritems()}
    print "Number of negative words contributed: labMT - %d, SWN - %d, SO-CAL - %d" % \
          (len(poswords1), len(poswords2), len(poswords3))
    poswords = sorted(list(set(poswords1.keys() + poswords2.keys() + poswords3.keys())))
    print "Number total: %d" % len(poswords)
    print poswords


def intersectLexica():
    """Create intersected lexica."""
    import pickle
    with open('MergedLexicon.pkl') as lexipickle:
        unionlex, lexicon, codes, opposites, confusion = \
            pickle.load(lexipickle)
    int4 = {key: val for key, val in lexicon.iteritems() \
                      if codes[key] >= 4}
    print "\nAt least 4 agree: %d" % len(int4)
    int4pure = {key: val for key, val in lexicon.iteritems() \
                      if codes[key] >= 4 and opposites[key] == 0}
    print "At least 4 agree, none disagree: %d" % len(int4pure)
    int5 = {key: val for key, val in lexicon.iteritems() \
                      if codes[key] >= 5}
    print "At least 5 agree: %d" % len(int5)
    int5pure = {key: val for key, val in lexicon.iteritems() \
                      if codes[key] >= 5 and opposites[key] == 0}
    print "At least 5 agree, none disagree: %d" % len(int5pure)
    int6 = {key: val for key, val in lexicon.iteritems() \
                      if codes[key] >= 6}
    print "At least 6 agree: %d" % len(int6)
    int6pure = {key: val for key, val in lexicon.iteritems() \
                      if codes[key] >= 6 and opposites[key] == 0}
    print "At least 6 agree, none disagree: %d" % len(int6pure)
    int7 = {key: val for key, val in lexicon.iteritems() \
                      if codes[key] == 7}
    print "At least 7 agree: %d" % len(int7)
    with open('CombinedLexica.pkl', 'wb') as outf:
        pickle.dump((unionlex, int4, int4pure, int5, int5pure, int6,
                     int6pure, int7), outf)


def expand_lexicon(textfile, lexicafile):
    """Corpus-specific PMI-based expansion of valence lexicon.

    Expand lexicon based on most strongly positive/negative words
    (those on which >= 6 agree & none disagree), then see which
    of these:
    - are not in 4 impure
    - are not in union lexicon
    """
    from pmi import expandlexicon_PMI
    import pickle

    with open(lexicafile, 'rb') as lexipickle:
        unionlex, int4, int4pure, int5, int5pure, int6, int6pure, int7 = \
            pickle.load(lexipickle)
    # Get dictionary of new words
    newdict = expandlexicon_PMI(textfile, int6pure, newpos=100,
                                newneg=100, keyval='6pure')
    # Compare against other dictionaries
    notin4 = {key: val for key, val in newdict.iteritems() if key not in int4}
    print "New words in < 4 of our lexica with same val: %d" % len(notin4)
    for key, val in notin4.iteritems():
        print key, val
    notinunion = {key: val for key, val in newdict.iteritems() \
                  if key not in unionlex}
    print "New words not in union of our lexica: %d" % len(notinunion)
    for key, val in notinunion.iteritems():
        print key, val
    return newdict, notin4, notinunion


def mergeLexica_NL(moorsfile, antwerpenfile, mergedfile):
    """Combine the 2 Dutch sentiment analysis lexica we have."""
    from itertools import chain
    import pickle

    # Read & process the Moors et al. lexicon
    with open(moorsfile, 'rb') as moorspickle:
        moors_raw = pickle.load(moorspickle)
    # recenter by -4; filter 0.2 removes 756 words (about 18%) leaving 3543
    moors = filter_near_neutrals(recenter_lexicon(moors_raw, dval=-4), dval=0.2)

    # Read & process the Pattern (UvAntwerpen) lexicon
    with open(antwerpenfile, 'rb') as antwerpenpickle:
        antwerpen_raw = pickle.load(antwerpenpickle)
    # filter 0.1 removes 593 words (about 18%) leaving 2630
    antwerpen = filter_near_neutrals(antwerpen_raw, dval=0.1)

    #
    # Generate a union of all keys
    moorskeys = set(moors.keys())
    antwerpenkeys = set(antwerpen.keys())
    allkeys = moorskeys.union(antwerpenkeys)
    moorsonly = allkeys - antwerpenkeys
    antwerpenonly = allkeys - moorskeys
    bothlex = allkeys - moorsonly - antwerpenonly
    ambiglist = [(key, moors[key], antwerpen[key]) for key in bothlex \
                 if (moors[key] < 0 and antwerpen[key] > 0) or \
                    (moors[key] > 0 and antwerpen[key] < 0)]
    agreelist = [word for word in bothlex \
                 if (moors[word] > 0 and antwerpen[word] > 0) or \
                    (moors[word] < 0 and antwerpen[word] < 0)]
    agreedict = {word: (1 if moors[word] > 0 else -1) for word in agreelist}
    krantlex = {}  # placeholder for later

    # Display results
    print "\nTotal nr. words: %d" % len(allkeys)
    print "Words with conflict: %d\n" % len(ambiglist)
    for ambig in ambiglist:
        print "%s: %5.2f vs. %5.2f" % (ambig[0], ambig[1], ambig[2])
    with open(mergedfile, 'wb') as outf:
        pickle.dump((moors, antwerpen, agreedict, krantlex), outf)
    return


def add_lexicon_NL(newlex, lexicafile):
    """Add third (corpus-specific) lexicon to the Dutch lexicon set."""
    import pickle

    with open(lexicafile, 'rb') as lexipickle:
        moors, antwerpen, agreedict, dummy = pickle.load(lexipickle)
    with open(lexicafile, 'wb') as lexipickle:
        pickle.dump((moors, antwerpen, agreedict, newlex), lexipickle)


def expand_lexicon_NL(textfile, lexicafile):
    """Corpus-specific PMI-based expansion of valence lexicon.

    Expand lexicon based on most strongly positive/negative words
    (those on which both agree), then see which of these:
    - are not in either lexicon
    """
    from pmi import expandlexicon_PMI
    import pickle

    with open(lexicafile, 'rb') as lexipickle:
        moors, antwerpen, agreedict, krantlex = pickle.load(lexipickle)
    # Get dictionary of new words
    newdict = expandlexicon_PMI(textfile, agreedict, newpos=250,
                                newneg=250, keyval='2agree')
    # Compare against other dictionaries
    notineither = {key: val for key, val in newdict.iteritems() \
                   if key not in moors and key not in antwerpen}
    print "New words not in either of our lexica: %d" % len(notineither)
    for key, val in notineither.iteritems():
        print key, val
    return newdict, notineither



# ***************************** I/O & loading lexica *************************

def importCorpusDicts(filestem):
    """Read in corpus-specific dictionaries of various lengths.

    Hard-coded to expect 3 different threshold values (4-6)
    and 5 different dictionary sizes (100-500).
    """
    thresholds = [6,7]
    sizes = [100,]
    fstart = filestem + '_corpusdict_'
    return [readValenceDict(fstart + str(thres) + '_' + str(size) + '.csv') \
            for thres in thresholds for size in sizes], \
           ['corpus' + str(thres) + '_' + str(size) \
            for thres in thresholds for size in sizes]


def importIntersectLexica(lexicafile):
    """Read in the intersected lexica."""
    import pickle
    with open(lexicafile, 'rb') as infile:
        unionlex, int4, int4pure, int5, int5pure, int6, int6pure, int7 = \
            pickle.load(infile)
    return unionlex, int4pure, int5


def lexiconinfo():
    """Provide 1-line descriptions of the available lexica."""
    print "Available lexica (10):"
    print
    print "BingLiu - "
    print "labMT - "
    print "LexicoderSD - "
    print "NRC - "
    print "SOCAL - "
    print "SentiWordNet - "
    print "WordStat - "
    print
    print "agree4disagree0 - at least 4 of 7 lexica agree on +/- valence; none disagree (xx pos, xx neg)"
    print "agree5disagree1 - at least 5 of 7 agree; no more than 1 disagrees (xx pos, xx neg)"
    print "unionconsensus - "
    print
    print "Specify any combination of these to use in the sentiment analysis."


def getlexica(lexicondir, lexica2use):
    """Return all the lexica to be used, plus modifier lexicon.

    Each lexicon is specified by name, which should correspond to
    a file in lexicondir with that name and the extension '.pkl'
    """
    import pickle

    lexiconlist = []
    for lexicon in lexica2use:
        with open(lexicondir + lexicon + '.pkl', 'rb') as lexpickle:
            lexiconlist.append(pickle.load(lexpickle))

    with open(lexicondir + 'modifiers.pkl', 'rb') as lexpickle:
        mods = pickle.load(lexpickle)

    return lexiconlist, mods


def getlexica_old(lexicondir, lexica2use):
    """Return the lexica specified in lexica2use."""
    import pickle

    stdlexica, wildcardlexica = [], []

    # Load the lexica requested
    for lexicon in lexica2use:
        with open(lexicondir + lexicon + '.pkl', 'rb') as lexiconpickle:
            lexi = pickle.load(lexiconpickle)
        if lexicon in ['BingLiu', 'labMT', 'NRC', 'SentiWordNet',
                       'SOCAL', 'WordStat', 'agree4disagree0']:
            stdlexica.append(lexi)
        elif lexicon in ['LexicoderSD',]:
            wildcardlexica.append(lexicon)

    # Load the modification dictionary from SO-CAL
    with open(lexicondir + 'modifiers.pkl', 'rb') as modpickle:
        mods = pickle.load(modpickle)

    return stdlexica, wildcardlexica, mods


def importValenceDicts(valencedir):
    """Read in the various lexica; should all be in the same folder.

    Note: hardcoded so that folder is expected to be in pwd
    """
    import pickle
    # 1. Lexicoder
    with open(valencedir + 'LexicoderDictionary.pkl', 'rb') as LSDpickle:
        lexi = pickle.load(LSDpickle)
    # 2. Lexicoder
    with open(valencedir + 'SO-CAL.pkl', 'rb') as socalpickle:
        socal = pickle.load(socalpickle)
    # 3. Bing Liu
    with open(valencedir + 'BingLiuDictionary.pkl', 'rb') as liupickle:
        liu = pickle.load(liupickle)
    # 4. WordStat
    with open(valencedir + 'WordStatDictionary.pkl', 'rb') as WSpickle:
        WS = pickle.load(WSpickle)
    # 5. labMT: center on 0, and delete values between -1 and +1
    with open(valencedir + 'labMT.pkl', 'rb') as labMTpickle:
        labMT = pickle.load(labMTpickle)
    # 6. LIWC
    with open(valencedir + 'LIWC2007dictionary.pkl', 'rb') as LIWCpickle:
        LIWC = pickle.load(LIWCpickle)
    # 7. moral foundations
    with open(valencedir + 'MFdictionary.pkl', 'rb') as moralpickle:
        moral = pickle.load(moralpickle)
    # 8. NRC emotions
    with open(valencedir + 'NRCdictionary.pkl', 'rb') as NRCpickle:
        NRC = pickle.load(NRCpickle)
    # 9. SentiWordNet
    with open(valencedir + 'SWNdictionary.pkl', 'rb') as SWNpickle:
        SWN = pickle.load(SWNpickle)
    return lexi, socal, liu, WS, labMT, LIWC, moral, NRC, SWN[0]


def readValenceDict(filename):
    """Read csv-format valence dictionary."""
    import csv
    adict = {}
    with open(filename, 'rU') as infile:
        for row in csv.reader(infile):
            adict[row[0].strip()] = float(row[1].strip())
    return adict


def writeValenceDict(adict, filename):
    """Write csv-format valence dictionary."""
    import csv
    with open(filename, 'wt') as outfile:
        outwriter = csv.writer(outfile)
        outwriter.writerows(adict.items())
    return


# ******************************* Accessing lexica ***************************

def filter_lexicon(fullDict, dval):
    """Delete entries with absolute value < dval.

    Make a copy first, to avoid destroying original dictionary."""
    wordDict = dict(fullDict)
    for key in fullDict:
        if fullDict[key] > -dval and fullDict[key] < dval:
            del wordDict[key]
    return wordDict


def recenter_lexicon(adict, dval):
    """Add dval to every entry in adict."""
    return {key: val + dval for key, val in adict.iteritems()}


def filter_neutrals(adict):
    """Remove entries with a valence of 0."""
    return {key: val for key, val in adict.iteritems() if val != 0}


def filter_near_neutrals(adict, dval):
    """Remove entries with a valence near 0."""
    return {key: val for key, val in adict.iteritems() if abs(val) >= dval}


def binarize_lexicon(adict):
    """Convert negative entries to -1 and positive ones to +1."""
    return {word: (-1 if adict[word] < 0 else 1) for word in adict}


def ternarize_lexicon(adict, dval):
    """Convert negative entries to -1, positive to +1, and close-to-0 to 0."""
    return {word: (0 if abs(adict[word]) < dval else
                    (-1 if adict[word] < 0 else 1)) for word in adict}


def lexiconmatch_multi(word, lexiconlist):
    """Run lexiconmatch on a list of lexica."""
    return [lexiconmatch(word, l) for l in lexiconlist]


def lexiconmatch(word, lexicon):
    """Return 0 if word not in lexicon; lexicon valence otherwise."""
    return lexicon[word] if word in lexicon else 0


def lexiconmatch_multiwild(word, lexiconlist, wild='*'):
    """Run lexiconmatch_wild on a list of lexica."""
    return [lexiconmatch_wild(word, l, wild) for l in lexiconlist]


def lexiconmatch_wild(word, lexicon, wild='*'):
    """Return 0 if word not in lexicon; lexicon valence otherwise.

    Note: accepts a wildcard (default '*') for '0 or more letters'.
    """
    if word in lexicon:
        return lexicon[word]
    else:
        if word[-1] != wild:
            word += wild
        while len(word) > 2:
            if word in lexicon:
                return lexicon[word]
            else:
                word = word[:-2] + wild
    return 0


def wordinset_wild(word, wordset, wild='*'):
    """Return True if word not in wordset, nor matched by wildcard.

    Note: accepts a wildcard (default '*') for '0 or more letters'.
    """
    if word in wordset:
        return True
    else:
        if word[-1] != wild:
            word += wild
        while len(word) > 2:
            if word in wordset:
                return True
            else:
                word = word[:-2] + wild
    return False


def lexicon_haswilds(lexicon):
    return any([x[-1] == '*' for x in lexicon.keys()])


def wildcard_expand(wildword, alist):
    """Return all entries in alist that begin with wildword."""
    ind = len(wildword)
    return [w for w in alist if w[:ind] == wildword]


def LIWCmatch(word, lexicon, nrcats):
    """Calculate category memberships for a word in a LIWC-style lexicon."""
    valencelist = [0,] * nrcats
    wordscaptured = 0
    lexmatch = lexiconmatch_wild(word, lexicon[0])
    if lexmatch != 0:
        wordscaptured += 1
        for x in lexmatch:
            valencelist[lexicon[1][x]] = 1
    return [wordscaptured,] + valencelist


def words_in_LIWC_cat(LIWCfilename, LIWCcat):
    """Return the words in a particular LIWC category."""
    import csv

    words = []
    with open(LIWCfilename,'rU') as infile:
        indata = csv.reader(infile, delimiter='\t')
        for row in indata:
            if len(row) > 0 and row[0] <> '' and \
                    row[0][0] <> '$' and row[0][0] <> '%':  # not a comment or separator
                if row[0][0] not in ('0', '1', '2', '3', '4', '5', '6', '7', '8','9'):
                    # These are the words, with a list of categories after them
                    aWord = row[0].rstrip()
                    if aWord <> '' and str(LIWCcat) in row[1:]:
                        words.append(aWord)
    return words


# ************************** Lexicoder lexica & scripts **********************

def importLexicoder(infileName1, infileName2):

    import pickle

    # Read in the non-negated values
    words = {}
    with open(infileName1,'rU') as inFile:
        for inString in inFile.readlines():
            aWord = inString[:-3].strip().lower()
            if aWord <> '':
                # if aWord[-1] == '*':  # adjust for wildcard
                #     aWord = aWord[:-1] + '[a-z]*'
                if ' ' not in aWord:  # skip multi-word entries
                    words[aWord] = int(inString[-2]) * 2 - 1

    # Read in the negated values; see which, if any, are new
    # Removed from the original file: better not, best not, no damag*
    # Changed in the original file: no no to no_no
    negwords = {}
    with open(infileName2,'rU') as inFile:
        for inString in inFile.readlines():
            if inString[:4] == 'NOT ':
                aWord = inString[4:-3].strip().lower()
                # if aWord[-1] == '*':  # adjust for wildcard
                #     aWord = aWord[:-1] + '[a-z]*'
                if ' ' not in aWord:
                    negwords[aWord] = 1 - 2 * int(inString[-2])
                    if aWord not in words:
                        words[aWord] = 1 - 2* int(inString[-2])
                        print "New one from negatives: ", aWord, 1 - 2* int(inString[-2])
            else:
                aWord = inString[:-3].strip().lower()
                if aWord <> '':
                    words[aWord] = int(inString[-2]) * 2 - 1
                    print "New one from negatives, not beginning 'NOT ': ", aWord

    # Remove overlaps due to wildcards
    todelete = [x for x in words if subsumed(x, words)]
    print "Subsumed: %d words" % len(todelete)
    print todelete
    for subsumedword in todelete:
        del words[subsumedword]
    # Identify the ones that are in the positive file but not in the negative one
    print "Words not negated with 'NOT ':"
    print [x for x in words if x not in negwords]
    # Results: 1616 pos + 2761 neg. = 4377 tot. (pre-subsumption: 1678, 2811, 4489)
    print "Total number of terms in dictionary:", len(words)
    print "Number of positive words:", sum([words[x] for x in words if words[x] == 1])
    print "Number of negative words:", -sum([words[x] for x in words if words[x] == -1])
    # Save Lexicoder dictionary in a pickle file
    with open('LexicoderDictionary.pkl', 'wb') as LSDout:
        pickle.dump(words, LSDout)


def subsumed(origx, words, report=True):
    """See whether origx is subsumed by a wildcarded entry in words."""
    if origx[-1] == '*':
        x = origx[:-2] + '*'
    else:
        x = origx + '*'
    while len(x) > 1:
        if x in words:
            if report:
                print x, 'subsumes', origx
            return True
        else:
            x = x[:-2] + '*'
    return False


# The Lexicoder conversion script in text-format is not entirely clear, so rely on Applescript version instead.
# Copy all the Applescript files into a single files, and pre-process with TextEdit

# Replace as follows (not including quotes):
# (Note: the strings to search for will differ if TextWrangler is available)
# "	«event R*chRepl» " by ""
# " given «class Repl»:" by ","
# There are two options used in the scripts: whole words only, and case-specificity
# Replace these by booleans, the first for words only, the second for case-specificity
# True, True:   ", «class savo»:no, «class FnIn»:{every «class TxtD»}, «class Opts»:{«class SMod»:«constant SModGrep», «class STop»:true, «class Wrap»:false, «class Rvrs»:false, «class Case»:true, «class Wrds»:true, «class ExSl»:false, «class Orsl»:false}
# True, False:  ", «class savo»:no, «class FnIn»:{every «class TxtD»}, «class Opts»:{«class SMod»:«constant SModGrep», «class STop»:true, «class Wrap»:false, «class Rvrs»:false, «class Case»:false, «class Wrds»:true, «class ExSl»:false, «class Orsl»:false}
# False, True:  ", «class savo»:no, «class FnIn»:{every «class TxtD»}, «class Opts»:{«class SMod»:«constant SModGrep», «class STop»:true, «class Wrap»:false, «class Rvrs»:false, «class Case»:true, «class Wrds»:false, «class ExSl»:false, «class Orsl»:false}
# False, False: ", «class savo»:no, «class FnIn»:{every «class TxtD»}, «class Opts»:{«class SMod»:«constant SModGrep», «class STop»:true, «class Wrap»:false, «class Rvrs»:false, «class Case»:false, «class Wrds»:false, «class ExSl»:false, «class Orsl»:false}
# Replace \" by quote, and then unreplace in our python script

# Wildcards used in these commands are BBedit/TextWrangler wildcards, mostly standard
# Note: we have blank lines in the text file separating the original Applescript files

# Consult LSDpreprocessnotes.txt file for changes made to the actual regex commands

def importLexicoderScripts(infileName):

    import csv, re

    # Read in Apple scripts from the conversion dictionary
    with open(infileName,'rU') as scriptFile:
        with open("LexiREs.txt", 'wb') as LexiOut:
            LexiOut.write("aText = unidecode(aText)\n")
            scriptData = csv.reader(scriptFile)
            for row in scriptData:
                if len(row) > 0:
                    # need to replace backslash quote so that it does not muck up things
                    searchRE = re.sub('quote', '\\"', row[0])
                    thisRE = "re.sub("
                    if row[2] == 'True': # words-only
                        searchRE = '|'.join(["\\\\b" + x + "\\\\b" for x in searchRE.split('|')])
                    thisRE += "\"" + searchRE + "\", \"" + row[1] + "\", aText"
                    if row[3] == 'False':
                        thisRE += ", flags=re.IGNORECASE)\n"
                    else:
                        thisRE += ")\n"
                    LexiOut.write(thisRE)


# ************************** Standard sentiment lexica ***********************

def globalEnglish(infilename='/Users/maurits/STAIR/Corpora/Lexica/English/Auxiliary/globalenglish15000.txt'):
    """Import word list to use to expand Lexicoder wildcards.

    Expects a list where the word is the first item on each line.
    """
    words = []
    with open(infilename, 'rt') as infile:
        for wordinfo in infile.readlines():
            words.append(wordinfo.split()[0])
    return words


def importBingLiu(infileName1, infileName2):
    """Import sentiment lexica from Bing Liu; pickle dump results

    Used files positive-words.txt and negative-words.txt inside
    opinion-lexicon-English folder (minor edits from original)."""
    import pickle

    # Read in the positive words
    words = {}
    words['a\+'] = 1 # special case (a+), removed from the input file because it would mess up regex
    with open(infileName1,'rU') as inFile:
        for aWord in inFile.readlines():
            if aWord[0] <> ';' and aWord[0] <> ' ':
                words[aWord.strip()] = 1

    # Read in the negative words
    with open(infileName2,'rU') as inFile:
        for aWord in inFile.readlines():
            if aWord[0] <> ';' and aWord[0] <> ' ':
                words[aWord.strip()] = -1

    del words['']
    print words.items()[:20]
    print len(words)
    # Save BingLiu dictionary in a pickle file
    with open('BingLiuDictionary.pkl', 'wb') as outFile:
        pickle.dump(words, outFile)


def importMPQA(infilename, outfilename):
    """Import the length-1 subjectivity clues files from MPQA; convert to dictionary.

    Original sentiment assessments are strongpos, weakpos, weakneg, strongneg
    Assign values 1, 0.5, -0.5, -1.
    Average across different word usages (parts-of-speech)
    """
    import pickle

    opinionvals = {'strongpos': 1, 'weakpos': 0.5, 'weakneg': -0.5, 'strongneg': -1,
                   'neutral': 0, 'both': 0}

    with open(infilename, 'rU') as in1:
        cluesdata = in1.readlines()

    lexicon = {}
    comments, wordcount = 0, 0
    for counter, wordinfo in enumerate(cluesdata):
        if wordinfo[0] != '#':  # skip comment lines (should not be present)
            wordsplit = wordinfo.split()
            termlength = [x[-1] for x in wordsplit if x[:3] == 'len']
            if len(termlength) > 0 and termlength[0] == '1':
                theword = [x[6:] for x in wordsplit if x[:5] == 'word1']
                thesent = [x[13:] for x in wordsplit if x[:12] == 'mpqapolarity']
                if len(theword) > 0 and len(thesent) > 0:
                    if theword[0] in lexicon:
                        lexicon[theword[0]].append(opinionvals[thesent[0]])
                    else:
                        lexicon[theword[0]] = [opinionvals[thesent[0]],]
                        wordcount += 1
    newlexicon = {key: sum(val)/float(len(val)) for key, val in lexicon.iteritems() if abs(sum(val)) > 0}
    print "Total lines: %d; comments: %d; words: %d, in lexicon: %d" % \
          (counter, comments, wordcount, len(newlexicon))
    with open(outfilename, 'wb') as outfile:
        pickle.dump(newlexicon, outfile)
    return newlexicon


def importWordStat(infileName):
    """Import WordStat sentiment dictionary; pickle dump results

    Used WordStatSentiments.txt file in the WSD folder
    (file has minor edits from original; among others, changed spelling of 'bareley' (sic)"""
    import pickle, re

    # initialize with the 4 good/bad expressions at the front of the WordStat Sentiments file
    words = {'not good': -1, 'real bad': -1, 'not bad': 1, 'real good': 1}
    negationwords = []
    doublenegwords = []
    curCat = 'negs'

    with open(infileName,'rU') as inFile:
        for aWord in inFile.readlines():
            aWord = aWord.lower().strip()
            if aWord == 'negations':
                curCat = 'negs'
            elif aWord == 'double_negation':
                curCat = 'doublenegs'
            elif aWord == 'positive words':
                curCat = 'positive'
            elif aWord == 'negative words':
                curCat = 'negative'
            else:
                aWord = aWord[:-4]    # remove ' (1)' at end of each line
                # if aWord[-1] == '*':  # adjust for wildcard
                #     aWord = aWord[:-1] + '[a-z]*'
                if curCat == 'negs':
                    negationwords.append(aWord)
                elif curCat == 'doublenegs':
                    doublenegwords.append(re.sub("_", " ", aWord))
                elif curCat == 'positive':
                    words[aWord] = 1
                else: # curCat == 'negative'
                    words[aWord] = -1

    # print doublenegwords
    print words.items()[:20]
    print len(words)
    # Save WordStat dictionary and negation phrases in a pickle file
    with open('WordStatDictionary.pkl', 'wb') as outFile:
        pickle.dump(words, outFile)


def importSOCAL(infilestem):
    """Import SO-CAL sentiment lexicon; pickle dump results

    Expects 5 separate files based on the filestem passed in:
    -adjectives.txt
    -adverbs.txt
    -nouns.txt
    -verbs.txt
    -modifiers.txt

    Each file contains 1 word per line, separated by a space from its valence.
    For the first 4, these are words to be added to the lexicon; for the last,
    the 'valence' is the multiplier to use when the word precedes a valence
    word.

    Note 1: A number of these 'words' are multi-word phrases, some with
    generic wildcards (e.g. "(bowl)_#PER?#_over" ). These are all separated
    by hyphens or underscores, so they will get read in correctly (if there
    are spaces, these phrases will be skipped in the input), but
    will not work correctly in the parsing. For now, just ignore.

    Note 2: Words may occur in more than 1 list (for example: when a noun
    is also an adjective, as in the case of 'evil'). If so, simply
    average their valences.

    Note 3: For nouns, only the singular is given. Add an -s, and handle
    nouns ending in -y  or -s correctly. Note that this will produce odd results
    for words that are generally not countable (e.g. 'salt' would become
    'salts' which is exceedingly rare as a noun plural, but quite common
    as a verb form)

    Note 4: For verbs, only the infinitive is given. Use Nodebox to get
    other verb forms (in separate function) and add all.

    Issues untouched:
    • spelling errors in adjectives list:
    - see email to Taboada
    • entries in verb dictionary twice:
    - ameliorate, at 1 and 2
    - appall, at -3 and -5
    - befriend (both at 1)
    - belie, at -2 and -3
    - bug, at -1 and -2
    - enthrall (both at 3)
    - extol, at 2 and 3
    - gladden, at 2 and 3
    - loathe at -4 and loath at -5
    - misunderstand (both at -1), plus misunderstood, also at -1
    - quibble (both at -1)
    - uplift, at 2 and 3

    Issues fixed in modifiers:
    - a_mutltidue_of -> a_multitude_of
    - visable -> visible
    - collossal -> colossal

    Changes made in modifiers:
    - added 'not_very' as -1.5, just like 'not_too'
    """
    import pickle
    from operator import itemgetter

    words, counts = {}, {}

    # Read in the nouns; handle plurals
    with open(infilestem + '-nouns.txt', 'rt') as infile:
        for word in infile.readlines():
            wordsplit = word.split()
            if len(wordsplit) == 2:
                # singular
                theword = wordsplit[0]
                theval = int(wordsplit[1])
                words, counts = \
                    updatewordscounts(theword, theval, words, counts)
                # plural
                if theword[-1] == 'y':
                    wordplural = theword[:-1] + 'ies'
                elif theword[-1] == 's':
                    wordplural = theword + 'es'
                else:
                    wordplural = theword + 's'
                words, counts = \
                    updatewordscounts(wordplural, theval, words, counts)
    lensofar = len(words)
    print "Nouns:", lensofar

    # Read in the verbs; handle conjugation
    with open(infilestem + '-verbs.txt', 'rt') as infile:
        for word in infile.readlines():
            wordsplit = word.split()
            if len(wordsplit) == 2:
                theval = int(wordsplit[1])
                for verbtense in alltenses(wordsplit[0]):
                    words, counts = \
                        updatewordscounts(verbtense, theval, words, counts)
    print "Verbs:", len(words) - lensofar
    lensofar = len(words)

    # Read in adjectives & adverbs: no special treatment
    for wordtype in ('adjectives', 'adverbs'):
        with open(infilestem + '-' + wordtype + '.txt', 'rt') as infile:
            for word in infile.readlines():
                wordsplit = word.split()
                if len(wordsplit) == 2:
                    theword = wordsplit[0]
                    theval = int(wordsplit[1])
                    words, counts = \
                        updatewordscounts(theword, theval, words, counts)
    print "Adjectives & adverbs:", len(words) - lensofar

    # Now read in the modifiers
    mods = {}
    with open(infilestem + '-modifiers.txt', 'rt') as infile:
        for word in infile.readlines():
            wordsplit = word.split()
            if len(wordsplit) == 2:
                mods[wordsplit[0]] = float(wordsplit[1])
    print "Modifiers:", len(mods)

    # Print some output to double-check it worked right
    print words.items()[:20]
    print "Total nr. of words", len(words)
    print mods.items()[:20]
    print "Total nr. of modifiers", len(mods)
    dupes = [(word, val) for word, val in counts.items() if val > 1]
    print sorted(dupes, key=itemgetter(1), reverse=True)
    print "Total nr. of words encountered more than once", len(dupes)
    # Save SO-CAL dictionary in a pickle file
    with open('SO-CAL.pkl', 'wb') as outFile:
        pickle.dump((words, mods, counts), outFile)


def updatewordscounts(word, val, words, counts):
    """Update words & counts dictionaries"""
    if word != '':
        if word in words:
            words[word] = words[word] * counts[word] + val
            counts[word] += 1
            words[word] /= float(counts[word])
        else:
            words[word] = val
            counts[word] = 1
    return words, counts


def alltenses(v):
    """Return all different verb forms, given infinitive.

    Uses the Linguistics module from Nodebox (called 'en').
    """
    import en
    # filter out multi-word phrases, which we just ignore
    if '_' in v:
        return []
    # filter out words not in the dictionary (misspellings &c)
    try:
        en.verb.present(v, person=1)
    except KeyError:
        return [v,]
    return list({en.verb.present(v, person=1),
                 en.verb.present(v, person=2),
                 en.verb.present(v, person=3),
                 en.verb.present(v, person='*'),
                 en.verb.present_participle(v),
                 en.verb.past(v, person=1),
                 en.verb.past(v, person=2),
                 en.verb.past(v, person=3),
                 en.verb.past(v, person='*'),
                 en.verb.past_participle(v)})


# ******************************* LIWC-style lexica **************************

def importLIWC(infileName, outfilename):
    """Import LIWC-style dictionary; pickle dump results

    Produce 3 items:
    - words = list of category numbers to which word belongs
    - indices = a dictionary translating from category number to category position
    - cats = an ordered list of category names
    So if indices[34] == 4 and cats[4] = 'happiness',
        then LIWC category nr. 34 is 'happiness'

    Languages (original dicts in LIWC folder):
    - English: LIWC2007.txt (small edits from the RIOT scan version).
    - Dutch: LIWC2007_Dutch.txt (= .dic, with extension renamed)
    """
    import csv, pickle
    from unidecode import unidecode

    words = {}
    cats = []
    indices = {}
    curInd = 0
    with open(infileName,'rU') as inFile:
        inData = csv.reader(inFile, delimiter='\t')
        for row in inData:
            if len(row) > 0 and row[0] <> '' and \
                    row[0][0] <> '$' and row[0][0] <> '%':  # not a comment or separator
                if row[0][0] in ('0', '1', '2', '3', '4', '5', '6', '7',
                                 '8','9'): # a LIWC category
                    cats.append(row[1])
                    indices[int(row[0])] = curInd
                    curInd += 1
                else: # words
                    aWord = unidecode(row[0].rstrip())
                    # if aWord[-1] == '*':  # adjust for wildcard, regex style
                    #     aWord = aWord[:-1] + '[a-z]*'
                    if aWord <> '':
                        words[aWord] = [int(x) for x in row[1:] if x != '']

    print "Categories: ", cats
    print words.items()[:50]
    # Save LIWC dictionary and categories in a pickle file
    with open(outfilename, 'wb') as outFile:
        pickle.dump((words, indices, cats), outFile)
    return words


def importNRC(infileName):
    """Import NRC-style dictionary; pickle dump results

    Produce a plain sentiment dictionary plus a LIWC-style set-up.
    (since the NRC categories do not have numbers, do not need 'cats'.)

    Use the file 'ValenceDicts/NRC-Emotion-Lexicon-v0.92/NRC_v0.92.txt'
    (same as original file, but with header text removed)
    """
    import csv, pickle

    words = {}
    wordcats = {}
    # Hardcode category names & positions
    cats = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
    indices = {x: y for x, y in zip(cats, xrange(len(cats)))}

    with open(infileName,'rU') as inFile:
        for row in csv.reader(inFile, delimiter='\t'):
            if len(row) > 0 and row[0] <> '':
                word = row[0].strip()
                cat = row[1].strip()
                val = int(row[2])
                if cat == 'positive':
                    if word in words:
                        if val == 1:
                            words[word] = 1
                    else:
                        words[word] = val
                elif cat == 'negative':
                    if word in words:
                        if val == 1:
                            words[word] = -1
                    else:
                        words[word] = 0 - val
                else:  # types of emotion
                    if not word in wordcats:
                        wordcats[word] = []
                    if val == 1:
                        wordcats[word].append(cat)

    print "Words: %d, positive: %d, negative: %d, neutral: %d" \
          % (len(words), len([1 for w in words if words[w] == 1]),
             len([1 for w in words if words[w] == -1]),
             len([1 for w in words if words[w] == 0]))
    print wordcats.items()[:50]
    # Save NRC dictionary and categories in a pickle file
    with open('NRCdictionary.pkl', 'wb') as outFile:
        pickle.dump((wordcats, indices, cats, words), outFile)


def importSWN(infileName):
    """Import SentiWordNet; pickle dump results

    Produce a plain sentiment dictionary by separating out members of each synset.
    For poly-semous words, simply average. In the long run, might try to weight by
    common-ness of each meaning. Save separate dictionary for POS & ID data

    Use the file ValenceDicts/SentiWordNet/SentiWordNet_3.0.0_20130122.txt

    File format: tab-separated
    - POS (a, n, ...)
    - WordNet ID
    - positive score (0-1)
    - negative score (0-1)
    - synset (list of members)
    - gloss (verbalization of meaning)

    Synset members: space-separated; each word with the suffix '#n' where n is the sequence
    number for the synset memberships of the same word.

    Ignore multi-word phrases. The dataset contains 83499 words, of which 20099 positive,
    20698 negative (but 9783 of the positive/negative words are both!)
    If we include multi-word phrases, there are 147700 'words' in the list, of which
    24764 positive, 26240 negative (11061 both !)
    """
    import csv, pickle
    from operator import itemgetter

    # Initialize dictionaries
    synsets = {}
    wordpos = {}
    posvals = {}
    negvals = {}
    polarity = {}
    with open(infileName,'rU') as inFile:
        # Run down synsets
        for row in csv.reader(inFile, delimiter='\t'):
            if len(row) > 0 and row[0] <> '' and row[0][0] <> '#':
                pos = row[0].strip()
                id = row[1].strip()
                posval = float(row[2])
                negval = float(row[3])
                synsetraw = row[4].split()
                # Run down members of each synset
                for term in synsetraw:
                    word = term[:-2]
                    if '_' not in word:  # ignore multi-word phrases
                        if word not in synsets:  # first or only meaning of word
                            synsets[word] = [id,]
                            wordpos[word] = [pos,]
                            posvals[word] = posval
                            negvals[word] = negval
                        else:  # polysemous word
                            nrmeanings = len(synsets[word])
                            oldposvals = posvals[word] * nrmeanings
                            oldnegvals = negvals[word] * nrmeanings
                            synsets[word].append(id)
                            if pos not in wordpos[word]:
                                wordpos[word].append(pos)
                            posvals[word] = (oldposvals + posval)/  \
                                            float(nrmeanings + 1)
                            negvals[word] = (oldnegvals + negval)/  \
                                            float(nrmeanings + 1)
    # Combine positive and negative
    for word, val in posvals.iteritems():
        polarity[word] = val - negvals[word]
    # Report basic data
    print "Words: %d, positive: %d, negative: %d, both: %d" \
          % (len(synsets),
             len([1 for w in posvals if posvals[w] > 0]),
             len([1 for w in negvals if negvals[w] > 0]),
             len([1 for w in posvals if posvals[w] > 0 and negvals[w] > 0]))
    print "Most positive words:"
    print sorted(posvals.items(), key=itemgetter(1), reverse=True)[:500]
    print "Most negative words:"
    print sorted(negvals.items(), key=itemgetter(1), reverse=True)[:500]
    # Save separate dictionaries in a pickle file
    with open('SWNdictionary.pkl', 'wb') as outFile:
        pickle.dump((polarity, posvals, negvals, synsets, wordpos), outFile)


# ***************************** External invocation **************************

if __name__ == "__main__":
    import sys
    from distutils.util import strtobool

    # See if we were called with the right number of arguments
    nrArgs = len(sys.argv) -1       # Arguments to python (includes function name itself)
    if nrArgs == 0:
        mergeLexica()
    elif nrArgs == 1:
        # importLexicoderScripts(sys.argv[1])
        importSWN(sys.argv[1])
        # importWordStat(sys.argv[1])
    else: # nrArgs == 2:
        expandLexicon(sys.argv[1], sys.argv[2])
        # importLexicoder(sys.argv[1],sys.argv[2])
        # importBingLiu(sys.argv[1], sys.argv[2])
