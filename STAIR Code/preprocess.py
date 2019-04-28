# encoding: utf-8

# textPreprocess.py
# by Maurits van der Veen
# last modified 2017-04-07
# Program to apply various kinds of preprocessing to a text
# A text is taken to consist of an ID and the text itself.
# An initial preprocessing stage can merge multiple pieces of text
# and split off class labels for later use.

# The end product is a pickle file containing a dataset (split into
# training and test subsets) of feature vectors ready for classification
# (ids1, ids2, features1, features2, featureNames)

# ****************************************************************************

# Preprocessing parameters that can/could be tested in grid search:
# (For more details, see at individual preprocessing stages)
# Stage 1
# - Americanize (True, False)

# Stage 2
# - enormous # of individual word/phrase substitutions
#     (skip for now, but could bundle substs. into groups)
# - strip possessive pronouns (True, False)
# - strip directional pronouns (True, False)

# Stage 3
# - choice of stemmer (0, 1, 2, 3)
# - concatenation of not / removal of plusx & minusx (True, False)

# Stage 4
# - binary, count, or tfidf vectorization (0, 1, 2)
# - bigrams & trigrams (1, 2, 3)
# - minFreq & maxFreq

# Stage 5
# - dimensionality reduction (nr features)
# - normalization (True, False)

# Filter stage
#     (the latter two are internal parameters for now)
# - filtering condition (0, 1, 2, 3, 4)
# - valencefilter threshold (0.33, 0.43, etc.)
# - sentence aggregation procedure (means, avg)
# - binarize sentiment valences (True, False)

# Cluster stage

# Wordvector stage


# ************************* preprocessing controllers ************************
# ****************************************************************************

def preprocess_std(dirname, textname, dictname=''):
    """Take csv-format raw text, and produce 2 cleaned versions:

    1. Sentence-tokenized, Americanized, corpus-specific word substitution
    2. valence-related word substitution, stopword removal
    """

    infile = dirname + '/' + textname
    filestem = dirname + '/' + '.'.join(textname.split('.')[:-1])
    tokfile = filestem + '_tok.csv'
    lexfile = filestem + '_lex.csv'
    replacefile = '' if dictname == '' else dirname + '/' + dictname
    # de-accent, sentence tokenize, Americanize, word substitute
    preprocess_init(infile, tokfile, replacefile)
    # additional word substitutions to improve valence calculations
    preprocess_semantic(tokfile, lexfile)
    return tokfile, lexfile


# Add min-occur as option; then run tests of vectChoices and bestFeatures
# - make sure features is functional again
# Test clustering in all its glory
# Reintroduce SVD later just to see if it helps (no rush)

def fullParams():
    """Return full set of parameters (*** will cause internal conflict ***)"""
    return {'prep0': {'header': True, 'nrTexts': 2, 'nrClasses': 1},
            'prep1': {'Americanize': [True,]},
            'prep2': {'mods': [True,], 'substs': [True,],
                      'stripPoss': [False,], 'stripDir': [False,], 'stripNLTK': [False,]},
            'prep3': {'stemChoice': [1,], 'concat': [False,]},
            'prep4': {'ngrams': [1, 3],
                      'counttype': [0,1,2], 'featureweight': [0,1,2],
                      'minF': [0.002,], 'maxF': [0.95,]},
            'prep5': {'SVDnr': [1000, 2000, -1],
                      'norm': [True, False]},
            # Separate steps without a fixed place in the above sequence
            'filter':  {'cond': [1,], 'thres': [0.33,]},
            'cluster': {'nclusters': [2000,], 'level': [0,]},
            'vector':  {'vectors': ['Lebret', 'GloVe', 'word2vec'],
                        'combine': ['concat', 'avg'],
                        'cluster': [-1, 2000],
                        'use': ['replace', 'dimreduce']}}


def defaultParams():
    """Return default set of parameters. Outdated -> fix!"""
    return {'steps': ['prep0', 'prep1', 'prep2', 'filter', 'prep3', 'cluster', 'BOW', 'combine'],
            'prep0': {'header': True, 'nrTexts': 2, 'nrClasses': 1},
            'prep1': {'Americanize': [True,]},
            'prep2': {'mods': [True,], 'substs': [True,],
                      'stripPoss': [False,], 'stripDir': [False,], 'stripNLTK': [False,]},
            'filter': {'cond': [1,], 'thres': [0.33,]},
            'prep3': {'stemChoice': [1,], 'concat': [False,]},
            'cluster': {'nclusters': [2000,], 'level': [0,]},
            'BOW': {'ngrams': [1, 3],
                      'counttype': [0,1,2], 'featureweight': [0,1,2],
                      'minF': [0.002,], 'maxF': [0.95,]},
            'prep5': {'SVDnr': [1000, 2000, -1],
                     'norm': [True, False]}}


def preprocess(fileStem, params={}, featureslabelslist='_PPfiles.txt'):
    """Preprocess, in 6 steps (0..5), plus a filtering and a clustering step.

    The filtering step is best put between steps 2 and 3.

    The clustering step makes further word analysis impossible, and must go
    after step 0 and before step 4. Experiment with various placements, but
    begin with the last possible position.

    Preprocessing gets us all the way from raw input texts to texts
    represented as vectors ready to be processed by ML algorithms.

    Keep track of the name suffixes of the files generated along the way.
    Keep these separate from fileStem until the very end, for flexibility.
    Return actual filenames (fileStem + suffixes) produced in last stage run.
    """
    import os
    # from valence import filter_on_valence
    from brown_cluster import brown_cluster
    from autocodingIO import readItem, writeItem

    # Check for presence of file containing names of feature & label files
    if os.path.isfile(fileStem + featureslabelslist):
        # If so, bypass preprocessing stage
        return readItem(fileStem + featureslabelslist)

    elif params == {}:  # parameters not passed in
        if os.path.isfile(fileStem + '_PPparams.csv'):
            # Find file containing preprocessing parameters
            params = readParams(fileStem + '_PPparams.csv')
        else:
            return [], []
            # Alternative: params = defaultParams()

    # Save all preprocessing text files in a subfolder
    fileStemT = fileStem + 'T/'
    if not os.path.isdir(fileStemT):  # directory may already exist
        os.makedirs(fileStemT)

    if 'prep0' in params:
        print "\nPreparing raw data for text preprocessing..."
        texts, classes = preprocess0(fileStem, params['prep0']['header'],
                                               params['prep0']['nrTexts'],
                                               params['prep0']['nrClasses'])
        fileSeqs = ['_1', '_2']
        fileExts = [texts[len(fileStem):],]
    else: # This may not work well; haven't really tested it
        classes = []
        fileExts, fileSeqs = ['',], ['',]

    if 'prep1' in params:
        print "\nPreprocessing stage 1: accents, punctuation, dialect, substitution..."
        newFileExts = []
        for x in fileExts:
            for Americanize in params['prep1']['Americanize']:
                for y in fileSeqs:
                    newExt = preprocess1(fileStem, fileStemT, x, y,
                                         'Dict.csv', Americanize)
                newFileExts.append(x + newExt)
        fileExts = newFileExts

    if 'prep2' in params:
        print "\nPreprocessing stage 2: word/phrase substitution & stopword removal..."
        newFileExts = []
        for x in fileExts:
            for mods in params['prep2']['mods']:
                for substs in params['prep2']['substs']:
                    for stripPoss in params['prep2']['stripPoss']:
                        for stripDir in params['prep2']['stripDir']:
                            for stripNLTK in params['prep2']['stripNLTK']:
                                for y in fileSeqs:
                                    newExt = preprocess2(fileStemT + x, y, mods,
                                                         substs, stripPoss,
                                                         stripDir, stripNLTK)
                                newFileExts.append(x + newExt)
        fileExts = newFileExts

    if 'filter' in params:
        newFileExts = []
        for x in fileExts:
            for thres in params['filter']['thres']:
                newExts = filter_on_valence(fileStemT + x, fileSeqs, thres,
                                            params['filter']['cond'])
                newFileExts += [x + y for y in newExts]
        fileExts = newFileExts

    if 'prep3' in params:
        print "\nPreprocessing stage 3: stemming & concatenation..."
        newFileExts = []
        for x in fileExts:
            for stemChoice in params['prep3']['stemChoice']:
                for concat in params['prep3']['concat']:
                    for y in fileSeqs:
                        newExt = preprocess3(fileStemT + x, y,
                                              stemChoice, concat)
                    newFileExts.append(x + newExt)
        fileExts = newFileExts

    if 'cluster' in params:
        newFileExts = []
        for x in fileExts:
            for nclusters in params['cluster']['nclusters']:
                for level in params['cluster']['level']:
                    newExt = brown_cluster(fileStemT + x, fileSeqs,
                                            nclusters, level)
                newFileExts.append(x + newExt)
        fileExts = newFileExts

    if 'prep4' in params:
        print "\nPreprocessing stage 4: vectorize, scale, add valence data..."
        newFileExts = []
        for x in fileExts:
            for ngrams in params['prep4']['ngrams']:
                for counttype in params['prep4']['counttype']:
                    for featureweight in params['prep4']['featureweight']:
                        for minF in params['prep4']['minF']:
                            for maxF in params['prep4']['maxF']:
                                newExt = makeBOW(fileStemT + x, fileSeqs,
                                            ngrams[0], ngrams[1], counttype,
                                            featureweight, minF, maxF)
                                newFileExts.append(x + newExt)
        fileExts = newFileExts

    if 'prep5' in params:
        print "\nPreprocessing stage 5: dimensionality reduction..."
        newFileExts = []
        for x in fileExts:
            for SVDnr in params['prep5']['SVDnr']:
                normset = ['False',] if SVDnr == -1 \
                    else params['prep5']['norm']
                for normalize in normset:
                    newExt = preprocess5(fileStemT + x, SVDnr, normalize)
                    newFileExts.append(x + newExt)

    # Wrap up & save 'results'
    print "\nPreprocessing complete."
    preprocessedfiles = ([fileStemT + x for x in newFileExts], classes)
    writeItem(fileStem + '_PPfiles.txt', preprocessedfiles)
    return preprocessedfiles


def boolStr(aBool):
    return 'T' if aBool else 'F'


def readParams(paramFile):
    # Read preprocessing parameters from file. No error-checking."""
    import csv

    params = {}
    with open(paramFile,'rU') as paramF:
        paramData = csv.reader(paramF)
        for stage in paramData:   # one preprocessing stage per line
            stageNr = stage[0].strip()
            if stageNr == '':
                break
            params[stageNr] = {}
            for x in xrange(1, len(stage)-1, 2):
                if stage[x].strip() == '':
                    break
                params[stageNr][stage[x].strip()] = \
                    my_literal_eval(stage[x+1].strip())
    return params


def my_literal_eval(aString):
    # Wrapper around ast_literal_eval to catch capitalizations of T/F."""
    from ast import literal_eval

    if aString in ('TRUE', 'True', 'true'):
        return True
    elif aString in ('FALSE', 'False', 'false'):
        return False
    else:
        return literal_eval(aString)


# ************************ preparation for preprocessing *********************
# ****************************************************************************

# The file Muslim_all.csv contains all articles, class info, etc.

# The file Muslim_coded.csv contains all coded articles. From these,
# using random number generation and selection in Excel, we identified 200
# articles to serve as the test set (subdivisible into 2, fairly
# balanced, subsets).

# The file Muslim_IDs.txt contains the IDs of these 200 articles. They
# should be split off from the overall dataset to make our process truly
# 'clean' (they should not even be included in any of the preprocessing
# decisions that depend on word frequencies, valences, etc.)

def preprocess0(fileStem, headerRow=True, nrTexts=2, nrLabels=1):
    """Extract texts & classes, merge texts, split out testing data.

    Go from a raw file containing IDs, texts, class labels, other stuff
    to a text file containing merged texts (usually title + main text)
    and one file for each class label, encoded and pickled.
    No clean-up of intermediate files, which may be useful later.
    Little or no error-checking.
    """
    import os
    print "Separating text data from label data..."
    texts, classes = extractTextandClasses(fileStem, nrTexts, nrLabels)
    if nrTexts > 1:
        print "Merging text components into single text..."
        texts = mergeTexts(texts)
    if nrLabels > 1:
        print "Splitting class labels into separate files..."
        classes = splitLabels(classes, nrLabels)
    else:
        classes = [classes,]
    print "Separating out test data..."
    splitbyID([texts,] + classes, fileStem + '_IDs.txt', headerRow)
    print "Processing label information..."
    for x in xrange(nrLabels):
        processLabels(fileStem + 'L' + str(x))
    return texts, classes


def splitbyID(contentfiles, idfile, headerRow=True):
    """Split the contentfiles on the specified IDs.

    contentfiles must exist & be in csv format (with name extension '.csv')
    idfile must exist & be plain text, with one ID per line.
    Output file 1 gets entries with IDs _not_ in the list
    Strip header row, if there is one.
    No error-checking.
    """
    import csv
    import os
    import sys

    csv.field_size_limit(1000000000)

    # read IDs into a list
    with open(idfile, 'rU') as selectIDs:
        IDs = [IDline.strip() for IDline in selectIDs.readlines()]

    # process each contentfile in turn
    for afile in contentfiles:
        print afile
        curheaderRow = headerRow
        out1 = afile + '_1.csv'
        out2 = afile + '_2.csv'
        if os.path.isfile(out1) and os.path.isfile(out2):
            print "Test set already split out for %s; step skipped" \
                  % afile
        else:
            with open(afile + '.csv','rU') as f:
                allData = csv.reader(f)
                with open(out1, 'wb') as outFile1:
                    data1 = csv.writer(outFile1)
                    with open(out2, 'wb') as outFile2:
                        data2 = csv.writer(outFile2)
                        # read each line; filter on presence/absence of ID in list
                        for row in allData:
                            if curheaderRow:
                                curheaderRow = False
                            else:
                                if row[0].strip() in IDs:
                                    data2.writerow(row)
                                else:
                                    data1.writerow(row)
    return contentfiles  # return input file list for convenience


def extractTextandClasses(fileStem, nrTexts, nrLabels):
    """Extract texts and classes into separate files.

    Assumes csv format: ID, text1, ..., textn, label1, ..., labeln
    (ignores anything after labeln). Expects <fileStem>.csv to exist.
    Little or no error-checking.
    """
    import csv
    import os
    import sys

    csv.field_size_limit(1000000000)

    outT = fileStem + 'T.csv'
    outL = fileStem + 'L0.csv'
    if os.path.isfile(outT) and os.path.isfile(outL):
        print "Texts and classes already extracted; step skipped."
    else:
        with open(fileStem + '.csv','rU') as inFile:
            allData = csv.reader(inFile)
            with open(outT, 'wb') as outText:
                texts = csv.writer(outText)
                with open(outL, 'wb') as outFile:
                    labels = csv.writer(outFile)
                    for row in allData:
                        texts.writerow(row[:nrTexts+1])
                        labels.writerow([row[0],] + \
                                         row[nrTexts+1:nrTexts+nrLabels+1])
    return outT[:-4], outL[:-4]  # return filenames without extension


def mergeTexts(fileStem):
    """Convert csv file in format ID,text1,text2,... to format ID, fulltext.

    Little or no error-checking. Expects <fileStem>.csv to exist.
    Note: merging text components may increase text length beyond what Excel
    will handle per csv cell => Don't save changes after opening in Excel!
    """
    import csv
    import os
    import sys

    csv.field_size_limit(1000000000)

    outfile = fileStem + 'M.csv'
    if os.path.isfile(outfile):
        print "Texts already merged; step skipped."
    else:
        with open(fileStem + '.csv','rU') as textFile:
            multiTexts = csv.reader(textFile)
            with open(outfile, 'wb') as outFile:
                mergedTexts = csv.writer(outFile)
                textCount = 0
                for row in multiTexts:
                    mergedTexts.writerow((row[0], ' . '.join(row[1:])))
                    textCount += 1
                    if textCount % 500 == 0:
                        print textCount
    return outfile[:-4]


# **************************** Class label management ************************

def splitLabels(fileStem, nrlabels):
    """Split class information in multiple labels into multiple files.

    fileStem.csv must exist & be of csv format: ID, label1, ...., labeln
    It is left untouched for the first label (so that file still has all
    class info in it; may be useful later on!).
    """
    import csv
    import os

    labelValues = [[] for _ in xrange(nrlabels)]
    with open(fileStem + '.csv','rU') as labelF:
        labelData = csv.reader(labelF)
        for row in labelData:
            theID = row[0].strip()
            for labelnr, thisLabel in enumerate(row[1:nrlabels+1]):
                labelValues[labelnr].append([theID, thisLabel])
    outfiles = []
    for x in xrange(nrlabels):
        outfile = fileStem[:-1] + str(x) + '.csv'
        outfiles.append(outfile[:-4])
        if os.path.isfile(outfile):
            print "Label %d already extracted; step skipped." % x
        else:
            with open(outfile, 'wb') as f:
                labelData = csv.writer(f)
                labelData.writerows(labelValues[x])
    return outfiles


def processLabels(fileStem):
    """Encode class values; remove entries without class label.

    Assumes 2 label files, for training/new data & for test data.
    In the former, assumes coded entries all appear at the top.
    Stops reading in data upon encountering the first '-1' label value.
    TODO : handle missing rather than -1
    """
    import csv
    import os
    import pickle
    from sklearn.preprocessing import LabelEncoder

    outfile = fileStem + '.pkl'
    if os.path.isfile(outfile):
        print "Label %s already processed; step skipped." % fileStem
    else:
        ids1, ids2, labels1, labels2 = [], [], [], []
        needEncoding = False
        print "Processing class", fileStem

        with open(fileStem + '_1.csv','rU') as labelF:
            labeldata = csv.reader(labelF)
            for row in labeldata:
                if row[1] == '-1':  # not a label
                    break
                ids1.append(int(row[0]))
                labels1.append(row[1])
                if row[1] != '0' and row[1] != '1':
                    needEncoding = True
        print "\nTraining set contains %d articles" % len(ids1)

        with open(fileStem + '_2.csv','rU') as labelF:
            labeldata = csv.reader(labelF)
            for row in labeldata:
                ids2.append(int(row[0]))
                labels2.append(row[1])
                # Allow test set to have a -1 -> these don't count in assessment
                # if row[1] != '0' and row[1] != '1':
                #     needEncoding = True
        print "\nTest set contains %d articles" % len(ids2)

        if needEncoding:
            le = LabelEncoder()
            labels1 = le.fit_transform(labels1)
            labels2 = [] if len(labels2) == 0 else le.transform(labels2)
            print "Classes %s converted to internal labels of 0 and 1" % \
                  str(le.classes_)
        else:
            le = "none"
            labels1 = [int(x) for x in labels1]
            labels2 = [int(x) for x in labels2]

        # Display some summary data & save pickle file
        zeroes = len([1 for x in labels1 if x == 0])
        print "Nr. training texts with labels 0: %d, 1: %d" % \
              (zeroes, len(labels1) - zeroes)
        with open(outfile, 'wb') as outFile:
            pickle.dump((ids1, ids2, labels1, labels2, le), outFile)
    return outfile[:-4]


def processLabelsX(infile, idfile, outfile):
    """Encode class values; remove entries without class label.

    Assumes 2 label files, for training/new data & for test data.
    In the former, assumes coded entries all appear at the top.
    Stops reading in data upon encountering the first '-1' label value.
    TODO : handle missing rather than -1
    """
    import csv
    import os
    import pickle
    from sklearn.preprocessing import LabelEncoder
    from corpus import splitbyID

    # Get ids & labels
    ids1, ids2, labels1, labels2 = splitbyID(infile, idfile)
    # Remove ids & labels where labels == '-1' (should never be in test set)
    ids1, labels1 = \
        zip(*[(id, l) for id, l in zip(ids1, labels1) if l != '-1'])
    needEncoding = len([1 for x in labels1 + labels2 \
                        if x != '0' and x != '1']) > 0

    print "\nTraining set contains %d articles" % len(ids1)
    print "\nTest set contains %d articles" % len(ids2)

    if needEncoding:
        le = LabelEncoder()
        labels1 = le.fit_transform(labels1)
        labels2 = [] if len(labels2) == 0 else le.transform(labels2)
        print "Classes %s converted to internal labels of 0 and 1" % \
              str(le.classes_)
    else:
        le = "none"
        labels1 = [int(x) for x in labels1]
        labels2 = [int(x) for x in labels2]

    # Display some summary data & save pickle file
    zeroes = len([1 for x in labels1 if x == 0])
    print "Nr. training texts with labels 0: %d, 1: %d" % \
          (zeroes, len(labels1) - zeroes)
    with open(outfile, 'wb') as outf:
        pickle.dump((ids1, ids2, labels1, labels2, le), outf)


def convertMulticlass(labels):
    """Convert multiclass label to tuple of binary classes."""
    classLabels = list(set(labels))
    if len(classLabels) == 2: # just a binary classification
        classLabels.pop()
    return [(int(i==j) for j in classLabels) for i in labels]


# ************************** preprocessing, stage 1 **************************
# ****************************************************************************


def preprocess1(fileStem, fileStemT, fileExt, fileSeq,
                replacefile='', Americanize=True):
    """First preprocessing stage: accents, punctuation, dialect, replace."""
    import csv, os, re
    import sys
    import translate

    csv.field_size_limit(1000000000)

    newExt = '_' + boolStr(Americanize)
    outfile = fileStemT + fileExt + newExt + fileSeq + '.csv'
    if os.path.isfile(outfile):
        print "Preprocessing stage 1 already done; step skipped."
    else:
        infile = fileStem + fileExt + fileSeq + '.csv'
        translate.translate_text(infile, outfile,
                                 fileStem + replacefile,
                                 'UK-to-US.csv' if Americanize else '',
                                 translate.translate_B2A if Americanize else None)
    return newExt


def preprocess_init(infile, outfile, replacefile='',
                    Americanize=True):
    """Initial preprocessing: accents, punctuation, word substitutions.

    Two types of word substitutions:
    - Americanize, to make British word forms American
    - make corpus-specific substitutions as specified in replacedict.
    """
    import csv
    import os
    from os.path import isfile
    import re

    # Do punctuation and accent preprocessing, followed by translation.
    outfile_temp = infile + '_temporary.csv'
    filestem = '.'.join(infile.split('.')[:-1])

    # Do accent- and punctuation preprocessing
    with open(infile,'rU') as textFile:
        allTexts = csv.reader(textFile)
        with open(outfile_temp, 'wb') as cleanedfile:
            cleanedtexts = csv.writer(cleanedfile)
            for count, row in enumerate(allTexts):
                cleanedText = punctuationPreprocess(accentPreprocess(row[1]))
                cleanedtexts.writerow((row[0], cleanedText))
                if count % 50000 == 0:
                    print count

    # Do translation
    translate.translate_text(outfile_temp, outfile,
                             fileStem + replacefile,
                             'UK-to-US.csv' if Americanize else '',
                             translate.translate_B2A if Americanize else None)
    # Remove temporary file
    os.remove(outfile_temp)


def striphtml(atext):
    """Remove html markup from text.

    Note: BeautifulSoup will strip also < used as 'less than',
    plus all the text that follows.
    For that reason, better to strip explicitly, if possible."""
    from bs4 import BeautifulSoup
    return BeautifulSoup(atext).get_text()


# ************************ preprocessing: punctuation ************************

def punctuationPreprocess(aText):
    """Handle a number of common punctuation situations.

    Starting point for the regexes here is the punctuation preprocessing
    in Lexicoder, but it has been considerably revised.

    Punctuation characters to deal with include (see string.punctuation):
    - !, ?, and ; which become periods;
    - brackets (), [], and {} which become commas;
    - colon, single and double quotes (:, ', ") which become commas
    - a single dash inside a word, which we leave untouched
    - ibid. following a space, which becomes a comma (like a double dash)
    - special characters &, %, and / which we spell out: and, percent, or
    - special characters +, = which we spell out: plus, is equal to
    - special char $ which we spell out and move after the dollar value
    - remaining special chars #, *, <, >, @, \, ^, _, |, ~ which we surround
      by spaces (may think about better things to do later)
    """
    import re

    # 1a. Handle websites (very simplistic right now: xx prefix and periods
    # become spaces). Just deals with web address (e.g. www.wm.edu),
    # not with specific files (e.g. www.wm.edu/index.html), as the latter
    # will rarely be mentioned in newspaper articles
    aText = re.sub(r"www\d{0,3}[.]([a-zA-Z0-9])+[.]([a-z]{2,4})",
                   "xx\\1 xx\\2", aText)
    aText = re.sub(r"([a-zA-Z0-9])+[.]([a-zA-Z0-9])+[.]([a-z]{2,4})",
                   "xx\\1 xx\\2 xx\\3", aText)

    # 1b. Remove phone numbers in 4-3-4 (UK), 3-3-4 (US), 3-4 (both) formats
    # Could think about handling the +44 (0) format for the UK and
    # the 1- or +1 formats for the US
    aText = re.sub("\\bd4\[ \t\n\r\f\v-.]d3[ \t\n\r\f\v-.]d4", "", aText)
    aText = re.sub("\\bd3[ \t\n\r\f\v-.]d3[ \t\n\r\f\v-.]d4", "", aText)
    aText = re.sub("\\b\(d3\)[ \t\n\r\f\v-.]d3[ \t\n\r\f\v-.]d4", "", aText)
    aText = re.sub("\\bd3[ \t\n\r\f\v-.]d4", "", aText)

    # 1c. Remove various special strings at the end of a news article
    aText = re.sub(r"\[ps\]", "", aText)
    aText = re.sub(r"\[es\]", "", aText)

    # 2a. Handle dollar values
    aText = re.sub(r"([0-9.,]+) ?bn\b", "\\1 billion", aText)
    aText = re.sub(r"([0-9.,]+) ?mn\b", "\\1 million", aText)
    aText = re.sub(r"\$([0-9.,]+[0-9])", "\\1 dollar", aText)
    aText = re.sub(r"\bdollar billion\b", "billion dollar", aText)
    aText = re.sub(r"\bdollar million\b", "million dollar", aText)

    # 2b. Handle pound sterling values
    aText = re.sub(r"\bps ?([0-9.,]+[0-9])b\b", "\\1 billion pound sterling", aText)
    aText = re.sub(r"\bps ?([0-9.,]+[0-9])m\b", "\\1 million pound sterling", aText)
    aText = re.sub(r"\bps ?([0-9.,]+[0-9])", "\\1 pound sterling", aText)

    # 2b. Handle times of day
    aText = re.sub(r"[0-9]{1,2} ?a\.?m\.?\b", "time_val time_am", aText)
    aText = re.sub(r"[0-9]{1,2} ?p\.?m\.?\b", "time_val time_pm", aText)
    aText = re.sub(r"[0-9]{1,2}:[0-9]{2} ?a\.?m\.?\b", "time_val time_am", aText)
    aText = re.sub(r"[0-9]{1,2}:[0-9]{2} ?p\.?m\.?\b", "time_val time_pm", aText)
    aText = re.sub(r"[0-9]{1,2}:[0-9]{2}", "time_val", aText)

    # 2x. Handle special characters used instead of words
    aText = re.sub("&", " and ", aText)
    aText = re.sub("%", " percent", aText)
    aText = re.sub(r"\+", " plus ", aText)
    aText = re.sub(r"=", " is equal to ", aText)
    aText = re.sub("/", " or ", aText)  # will also split up URLs
    # the 3/4 character (which is something else but shows up as 3/4 in our csv)
    # becomes 'I3 or 4' after the above substitution
    aText = re.sub("I3 or 4", "", aText)
    # remaining special characters just get surrounded by spaces,
    # except underscores which we assume to be deliberate concatenators
    aText = re.sub(r"([#*<>@\\^|~])", " \\1 ", aText)

    # 3. Expand contractions
    aText = re.sub("\\bit's\\b", "it is", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bhe's\\b", "he is", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bshe's\\b", "she is", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bain't\\b", "is not", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bisn't\\b", "is not", aText, flags=re.IGNORECASE)
    aText = re.sub("\\baren't\\b", "are not", aText, flags=re.IGNORECASE)
    aText = re.sub("\\blet's\\b", "let us", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bwon't\\b", "will not", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bcan't\\b", "can not", aText, flags=re.IGNORECASE)
    aText = re.sub("n't", " not", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bcannot\\b", "can not", aText, flags=re.IGNORECASE)
    # 'd can be either had or would; could look at context to disambiguate:
    # is next word an infinitive (would) or a participle/object (had)?
    # for now, pick had (more neutral).
    # Also, don't ignore case on the D, as then we'll catch O'Donnell, etc.!
    aText = re.sub("([A-Za-z]+)'d", "\\1 had", aText)
    aText = re.sub("([a-z]+)'ll", "\\1 will", aText, flags=re.IGNORECASE)
    aText = re.sub("([a-z]+)'m", "\\1 am", aText, flags=re.IGNORECASE)
    aText = re.sub("([a-z]+)'ve", "\\1 have", aText, flags=re.IGNORECASE)
    aText = re.sub("([a-z]+)'re", "\\1 are", aText, flags=re.IGNORECASE)

    # Specific expressions w. possessive apostrophes
    aText = re.sub("\\bbull's eye\\b", "bullseye", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bno man's land\\b", "nomansland", aText,
                   flags=re.IGNORECASE)
    aText = re.sub("\\bpandora's box\\b", "pandoras box", aText,
                   flags=re.IGNORECASE)
    # Remove general possessive 's (very common, no useful info)
    aText = re.sub("\\b([a-z]+)'s\\b", "\\1", aText,
                   flags=re.IGNORECASE)

    # too followed by punctuation means it is used in the meaning of 'also'
    aText = re.sub("\\btoo ?([.,:;)'\"\\]])", "also \\1", aText,
                   flags=re.IGNORECASE)

    # un- starting a word means 'not x'
    aText = re.sub("\\bun-", "not ", aText, flags=re.IGNORECASE)

    # Arab words with internal '
    aText = re.sub("\\bba'ath", 'baath', aText, flags=re.IGNORECASE)

    # 4. Simplify punctuation, step 1: sentence breaks become periods
    #    (including semi-colon)
    aText = re.sub("!", ".", aText)
    aText = re.sub(r"\?", ".", aText)
    aText = re.sub(";", ".", aText)
    # Sequences of periods (ellipsis) become just 1
    aText = re.sub("[\\.]{2,}", ".", aText, flags=re.IGNORECASE)

    # 5. Simplify punctuation, step 2: internal breaks become commas
    aText = re.sub(":", ",", aText)  # Note: this will also break up time-of-day
    aText = re.sub(r'"', ", ", aText)
    aText = re.sub(r"'", ", ", aText)
    aText = re.sub("--", ",", aText)
    aText = re.sub(" - ", ",", aText)
    aText = re.sub(r"\[", ",", aText)
    aText = re.sub(r"\]", ",", aText)
    aText = re.sub(r"\(", ",", aText)
    aText = re.sub(r"\)", ",", aText)
    aText = re.sub(r"\{", ",", aText)
    aText = re.sub(r"\}", ",", aText)
    aText = re.sub(r"\.,", ",", aText)
    # 6. Simplify punctuation, step 3: consecutive commas or periods
    aText = re.sub(r",( ?,)+", ",", aText)
    aText = re.sub(r"\.( ?\.)+", ".", aText)

    # 7. Simplify punctuation, step 4: underscores become spaces
    aText = re.sub("_", " ", aText)

    # 8. Modify expressions that might otherwise get categorized erroneously
    # due to use of valence word in non-valence (or different-valence) context
    # identified by punctuation -> prepend x to the word
    aText = re.sub("-like", " xlike", aText, flags=re.IGNORECASE)
    aText = re.sub(", well,", " xwell,", aText, flags=re.IGNORECASE)
    aText = re.sub(r"[\.]{2,} well", " xwell", aText, flags=re.IGNORECASE)
    aText = re.sub(r"\bWell,", "xwell,", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bOK\\b", "okay", aText)
    aText = re.sub(", okay,", " xokay", aText, flags=re.IGNORECASE)
    # aText = re.sub("\" okay,", " xokay", aText, flags=re.IGNORECASE)
    # This would capture something like "he felt OK, after all"
    aText = re.sub("[\\.]{2,} okay", " xokay", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bOkay,", "xokay,", aText, flags=re.IGNORECASE)

    # 9. Remove non-sentence-ending periods
    # In numbers: No. 1, Nr. 2, etc.
    aText = re.sub("\\b([Nn][or])(\\. )([0-9]+)\\b", "#\\3", aText)
    # After single upper- or lower-case letter (e.g. in a list)
    aText = re.sub("( [a-zA-Z])\\. ", "\\1 ", aText)
    # After Canadian province abbreviations
    aText = re.sub("\\bQue\\.", "Quebec", aText)
    aText = re.sub("\\bOnt\\.", "Ontario", aText)
    aText = re.sub("\\bNfld\\.", "Newfoundland", aText)
    aText = re.sub("\\bAlta\\.", "Alberta", aText)
    aText = re.sub("\\bMan\\.", "Manitoba", aText)
    aText = re.sub("\\bSask\\.", "Saskatchewan", aText)
    # After other abbreviations (esp. forms of address)
    aText = re.sub("\\bSt\\.", "St", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bSte\\.", "Ste", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bi\\.e\\.", "that is", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bet al\\.", "et alii", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bmr\\.", "Mr", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bms\\.", "Ms", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bmrs\\.", "Mrs", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bprof\\.", "Prof", aText, flags=re.IGNORECASE)
    aText = re.sub("\\ba\\. ?m\\. ", "am ", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bp\\. ?m\\.", "pm ", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bdr\\.", "Dr", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bgen\\.", "gen", aText, flags=re.IGNORECASE)
    aText = re.sub("\\be\\. coli\\b", "ecoli", aText, flags=re.IGNORECASE)
    aText = re.sub("\\bvs\\.", "versus", aText, flags=re.IGNORECASE)
    aText = re.sub("\\binc\\.", "incorporated", aText, flags=re.IGNORECASE)

    # Month abbreviations
    aText = re.sub("\\bJan\\.", "January", aText)
    aText = re.sub("\\bFeb\\.", "February", aText)
    aText = re.sub("\\bMar\\.", "March", aText)
    aText = re.sub("\\bApr\\.", "April", aText)
    aText = re.sub("\\bJun\\.", "June", aText)
    aText = re.sub("\\bJul\\.", "July", aText)
    aText = re.sub("\\bAug\\.", "August", aText)
    aText = re.sub("\\bSept\\.", "September", aText)
    aText = re.sub("\\bOct\\.", "October", aText)
    aText = re.sub("\\bNov\\.", "November", aText)
    aText = re.sub("\\bDec\\.", "December", aText)
    # Remove periods from acronyms
    aText = re.sub("(\\.)([A-Z])(\\.) ", "\\1\\2 ", aText)
    aText = re.sub("([A-Z])(\\.)([A-Z])", "\\1\\3", aText)
    aText = re.sub("([A-Z])(\\.)([A-Z])", "\\1\\3", aText)
    aText = re.sub("([A-Z])(\\.)([A-Z])", "\\1\\3", aText)
    aText = re.sub("([A-Z])(\\.)([A-Z])", "\\1\\3", aText)
    aText = re.sub("([A-Z])(\\.)([A-Z])", "\\1\\3", aText)
    # Remove decimal periods
    aText = re.sub("([0-9])(\\.)([0-9])", "\\1dot\\3", aText)
    # Precede periods by a space
    aText = re.sub(r"\.", " . ", aText)

    # Finally, remove commas
    # (these were used above to identify usage of 'too', 'well', etc.,
    # which is why we could not remove them earlier
    aText = re.sub(",", " ", aText)

    # In Lexicoder, proper names are disambiguated from words by inserting
    # an underscore. However, this misses too many non-name situations,
    # while the likelihood of a real name being a valence word is small enough
    # that we can accept it.
    # aText = re.sub("([a-z0-9’,] )([A-Z])([a-zA-Z]+)",
    #                "\\1(name) \\2_\\3", aText)

    # nltk might catch some remaining sentence punctuation issues
    # import nltk
    # sent_tok = nltk.data.load('tokenizers/punkt/english.pickle')
    # aText = ' . '.join(sent_tok.tokenize(aText, realign_boundaries=True))

    # Finally, remove multiple spaces and any empty 'sentences'
    aText = re.sub(" {2,}", " ", aText)
    return re.sub(r"\.( \.)+", ".", aText)


# ******************** preprocessing: accented characters ********************

def accentPreprocess(aText):
    """Replace accented vowels &c by their non-accented version.

    Call unidecode first, then the more brute-force ascii decoding."""
    from unidecode import unidecode
    return unidecode(aText).decode('ascii', 'ignore')



# ************************** preprocessing, stage 2 **************************
# ****************************************************************************

# MuslimTexts_prep1.csv contains the output of preprocessing stage 1.
# In stage 2, do language preprocessing

def preprocess2(fileStem, fileSeq, mods, substs, stripPoss, stripDir, stripNLTK):
    """wrapper around preprocess_semantic to handle file naming."""
    import os

    infile = fileStem + fileSeq + '.csv'
    newExt = '_' + boolStr(mods) + boolStr(substs) + \
             boolStr(stripPoss) + boolStr(stripDir)
    outfile = fileStem + newExt + fileSeq + '.csv'
    if os.path.isfile(outfile):
        print "Preprocessing stage 2 already done; step skipped."
    else:
        preprocess_semantic(infile, outfile, mods, substs,
                            stripPoss, stripDir, stripNLTK)
    return newExt


def preprocess_semantic(infile, outfile, mods=True, substs=True,
                        stripPoss=False, stripDir=False, stripNLTK=False):
    """Semantic preprocessing: word substitutions & stopword removal.

    Goal is to simplify negation phrases to 'not' and put them close to the
    negated thing, to disambiguate expressions, and to reword non-valence
    usage of valence words.

    Starting point and inspiration was the preprocessing list from Lexicoder.
    This list was converted to regex format, and then commented, corrected,
    modified, and reorganized, and expanded as seemed appropriate.
    Steps:
    1. handle modification phrases (especially negation)
    2. substitute to disambiguate once we remove context (bag of words)
    3. remove basic stopwords (as specified)
    """
    import csv
    import sys
    if mods or substs:  # conditional import
        from language_subst import substitute_val, negation_subst

    csv.field_size_limit(1000000000)
    with open(infile,'rU') as textFile:
        allTexts = csv.reader(textFile)
        with open(outfile, 'wb') as preppedFile:
            preppedTexts = csv.writer(preppedFile)
            for count, row in enumerate(allTexts):
                text = row[1].lower()
                if mods:
                    text = negation_subst(text)
                if substs:
                    text = substitute_val(text)
                text = stripStopwords(text, stripPoss, stripDir, stripNLTK)
                preppedTexts.writerow((row[0], text))
                if count % 5000 == 0:
                    print count


# *********************** preprocessing: strip stopwords *********************

def stripStopwords(aText, stripPoss=False, stripDir=False, stripNLTK=False):
    """Strip stopwords from a text.

    Do this only after language preprocessing, as that step relies on
    some of these stopwords.
    """
    aText = stripPronouns(stripBasics(aText))
    if stripPoss:
        aText = stripPossessives(aText)
    if stripDir:
        aText = stripDirPronouns(aText)
    if stripNLTK:
        aText = stripNLTKstopwords(aText)
    return aText


def stripBasics(aText):
    """Strip the most basic stopwords from a text.

    With % presence in our corpus:
    Articles: a, an, the
    Basic conjunctions: and, or
    Basic verbs: is, was, be, are, were, has, have, had
    Basic pronoun/referents: this, that"""
    import re
    return re.sub("\\b(a|an|the|and|or|that) ", "", aText)


def stripPronouns(aText):
    """Strip common general pronouns from a text."""
    import re
    return re.sub("\\b(at|by|for|from|in|into|of|on|to|with) ", "", aText)


def stripPossessives(aText):
    """Strip possessive pronouns from a text.

    Since contrast between our and their may be important in media coverage
    of 'other' groups, these words are probably better kept in."""
    import re
    return re.sub("\\b(my|your|his|her|its|our|their) ", "", aText)


def stripDirPronouns(aText):
    """Strip common directional pronouns from a text.

    May indicate something meaningful about the nouns they apply to
    (including valence), so hold off on stripping these at first."""
    import re
    aText = re.sub("\\b(above|below|before|after|over|under) ", "", aText)
    return re.sub("\\b(onto|until|since|between) ", "", aText)


def stripNLTKstopwords(aText):
    """Remove standard English stopwords as specified in nltk.

    Use hard-coded version of nltk.corpus.stopwords.words('english'),
    since lazy-loading from nltk.corpus causes problems in threading.
    One addition: 're' (from you're, etc.)
    """
    import re
    aText = re.sub("\\b(i|me|my|myself|we|our|ours)\\b", "", aText)
    aText = re.sub("\\b(ourselves|you|your|yours|yourself)\\b", "", aText)
    aText = re.sub("\\b(yourselves|he|him|his|himself|she)\\b", "", aText)
    aText = re.sub("\\b(her|hers|herself|it|its|itself|they)\\b", "", aText)
    aText = re.sub("\\b(them|their|theirs|themselves|what)\\b", "", aText)
    aText = re.sub("\\b(which|who|whom|this|that|these|those)\\b", "", aText)
    aText = re.sub("\\b(am|is|are|was|were|be|been|being)\\b", "", aText)
    aText = re.sub("\\b(have|has|had|having|do|does|did)\\b", "", aText)
    aText = re.sub("\\b(doing|a|an|the|and|but|if|or)\\b", "", aText)
    aText = re.sub("\\b(because|as|until|while|of|at|by)\\b", "", aText)
    aText = re.sub("\\b(for|with|about|against|between|into)\\b", "", aText)
    aText = re.sub("\\b(because|as|until|while|of|at|by)\\b", "", aText)
    aText = re.sub("\\b(through|during|before|after|above)\\b", "", aText)
    aText = re.sub("\\b(below|to|from|up|down|in|out|on|off)\\b", "", aText)
    aText = re.sub("\\b(over|under|again|further|then|once)\\b", "", aText)
    aText = re.sub("\\b(here|there|when|where|why|how|all)\\b", "", aText)
    aText = re.sub("\\b(any|both|each|few|more|most|other)\\b", "", aText)
    aText = re.sub("\\b(some|such|no|nor|not|only|own)\\b", "", aText)
    aText = re.sub("\\b(same|so|than|too|very|s|t|can)\\b", "", aText)
    aText = re.sub("\\b(will|just|don|re|should|now)\\b", "", aText)
    return aText



# ************************** preprocessing, stage 3 **************************
# ****************************************************************************

# Between stages 2 and 3, do valence calculation and sentence filtering.
# This is also a good point for PMI (though can do PMI after stemming too).

def preprocess3(fileStem, fileSeq, stemChoice=1, concat=True):
    """Third preprocessing stage: stem and concatenate."""
    import csv, nltk, os

    infile = fileStem + fileSeq + '.csv'
    newExt = '_' + str(stemChoice) + boolStr(concat)
    outfile = fileStem + newExt + fileSeq + '.csv'
    if os.path.isfile(outfile):
        print "Preprocessing stage 3 already done; step skipped."
    else:
        if stemChoice == 0:
            stemmer = None
        elif stemChoice == 1:
            stemmer = nltk.PorterStemmer()
        elif stemChoice == 2:
            stemmer = nltk.SnowballStemmer("english")
        else: # stemChoice == 3
            stemmer = nltk.LancasterStemmer()
        stem_concat(infile, outfile, stemmer, concat)
    return newExt


def stem_concat(infile, outfile, stemmer, concat=True):
    """Stem and concatenate."""
    import csv
    import sys
    csv.field_size_limit(1000000000)

    with open(infile,'rU') as textFile, open(outfile, 'wb') as preppedFile:
        preppedTexts = csv.writer(preppedFile)
        for counter, row in enumerate(csv.reader(textFile)):
            if stemmer == None:  # no stemming
                preppedText = row[1].strip()
            else:
                preppedText = stemText(row[1], stemmer)
            if concat:
                preppedText = concatText(preppedText)
            preppedTexts.writerow((row[0], preppedText))
            if counter % 5000 == 0:
                print counter


def concatText(aText):
    """Concatenate not phrases & hyphenated words."""
    import re
    # strip out plusx and minusx first
    aText = re.sub("\\b(plusx|minusx)\\b", "", aText)
    # attach not to the following word; do same for hyphenations
    return re.sub("\\bnot ([a-z]+)\\b", "not_\\1", aText.replace('-', '_'))


def stemText(aText, stemmer):
    """Stem a text with given choice of stemmer."""
    sentences = aText.decode('utf-8').encode('ascii', 'ignore').split('.')
    return ' . '.join([' '.join([stemmer.stem(w) for w in sent.split()]) \
                     for sent in sentences])


# ************************** preprocessing, stage 4 **************************
# ****************************************************************************

# TODO: take feature combination out of here into separate preprocessing fn.
# Then set it up so multiple feature files can be combined, not just valences.

def makeBOW(fileStem, fileSeqs, ngramsfrom, ngramsto, counttype, featureweight,
                minF, maxF, addvalences=True):
    """Fourth preprocessing stage: vectorize, scale & augment.

    Starting with this stage, results across different text sets become
    interdependent (run test sets with same vectorizer). This is
    also the last stage in which article codes do not matter.

    At some future point might want to have class labels available at
    this stage, e.g. to implement BNS vectorization. If so, also build in option
    to vectorize on training set only.

    Parameters in vectorization:
    1. count type: binary (0), count (1), augmented (2)
    2. feature weight: none (0), idf (1), term variance (2)
    (3. supervised feature weighting is done in features.py)
    Implement these independently here, but have vectChoice for specific
    combinations (to constrain excessive parameter dimensionality)
    """
    from corpus import readidtextfile
    import os
    import cPickle as pickle
    import numpy as np
    from scipy.sparse import csr_matrix, hstack
    from sklearn.feature_extraction.text import CountVectorizer, \
                                                TfidfTransformer

    print "\nVectorization: %d-%d-grams, count %d, feature %d, %f-%f" % \
          (ngramsfrom, ngramsto, counttype, featureweight, minF, maxF)
    newExt = '_' + str(ngramsfrom) + str(ngramsto) + str(counttype) + \
             str(featureweight) + str(minF) + '-' + str(maxF)
    outfile = fileStem + newExt + '.pkl'
    if os.path.isfile(outfile):
        print "Preprocessing stage 4 already done; step skipped."
    else:
        # Assume just 2 files. Could generalize for more
        f1 = fileSeqs[0]
        f2 = fileSeqs[1]
        # Process input data
        print "Reading text data"
        ids1, texts1 = readidtextfile(fileStem + f1 + '.csv')
        ids2, texts2 = readidtextfile(fileStem + f2 + '.csv')
        features1, features2, featurenames = \
            doVectorize(texts1, texts2,
                        ngramsfrom, ngramsto, minF, maxF, counttype, featureweight)
        # Save training & test features
        with open(outfile, 'wb') as outf:
            pickle.dump((ids1, ids2, features1, features2, featurenames),
                        outf, pickle.HIGHEST_PROTOCOL)
    return newExt


def vectorize(infile, idfile, outfolder, ngramsfrom, ngramsto, minF, maxF,
              counttype, featureweight):
    """Split corpus into train & test, and vectorize on former."""
    from corpus import splitbyID
    from sklearn.externals import joblib
    import os

    # Create folder to hold vectorized info
    try:
        os.mkdir(outfolder)
        outfolder += '/'
    except OSError:
        print "Error: folder %s already exists" % outfolder
        return
    # Split into training & test data
    ids1, ids2, texts1, texts2 = splitbyID(infile, idfile)
    # Call actual vectorizer
    features1, features2, featurenames = \
        doVectorize(texts1, texts2,
                    ngramsfrom, ngramsto, minF, maxF, counttype, featureweight)
    # Save training & test features
    save_features(outfolder + '/', ids1, ids2, features1, features2, featurenames)
    # with open(outfile, 'wb') as outf:
    #     pickle.dump((ids1, ids2, features1, features2, featurenames), outf)


def save_features(outfolder, ids1, ids2, features1, features2, featurenames):
    """Save feature data. Break training & test features into smaller pieces.

    Pickle in Python 2.7.x has an issue with very large files.
    To avoid this, break it down.
    """
    import pickle
    import gzip

    maxprod = 100000000  # 100 * 10^6
    nrfeatures = len(featurenames)
    maxobs = maxprod / nrfeatures  # integer division results in integer
    nrtrain = len(ids1) / maxobs
    if len(ids1) % maxobs != 0:
        nrtrain += 1
    nrtest = len(ids2) / maxobs
    if len(ids2) % maxobs != 0:
        nrtest += 1

    with open(outfolder + 'ids1.txt', 'wt') as outf:
        outf.writelines([str(x) + '\n' for x in ids1])
    with open(outfolder + 'ids2.txt', 'wt') as outf:
        outf.writelines([str(x) + '\n' for x in ids2])
    with open(outfolder + 'featurenames.txt', 'wt') as outf:
        outf.writelines([x + '\n' for x in featurenames])
    # COO matrices do not permit indexing; features 1 & 2 may be in that format
    # (depends on version of sklearn). Convert to CSR first.
    for sub in xrange(nrtrain):
        range_end = min((sub+1) * maxobs, features1.shape[0])
        with gzip.open(outfolder + 'features1_' + str(sub) + '.pklz', 'wb') as outf:
            pickle.dump(features1.tocsr()[sub * maxobs:range_end], outf, protocol=-1)
    for sub in xrange(nrtest):
        range_end = min((sub+1) * maxobs, features2.shape[0])
        with gzip.open(outfolder + 'features2_' + str(sub) + '.pklz', 'wb') as outf:
            pickle.dump(features2.tocsr()[sub * maxobs:range_end], outf, protocol=-1)


def load_features(infolder):
    """Load feature data; assumes IDs are integers."""
    import os
    import pickle
    import gzip
    import numpy as np

    infolder += '/'
    with open(infolder + 'ids1.txt', 'rU') as inf:
        ids1 = [int(x.strip()) for x in inf.readlines()]
    with open(infolder + 'ids2.txt', 'rU') as inf:
        ids2 = [int(x.strip()) for x in inf.readlines()]
    with open(infolder + 'featurenames.txt', 'rU') as inf:
        features = [x.strip() for x in inf.readlines()]
    with gzip.open(infolder + 'features1_0.pklz') as inf:
        features1 = pickle.load(inf)
    x = 1
    while os.path.isfile(infolder + 'features1_' + str(x) + '.pklz'):
        with gzip.open(infolder + 'features1_' + str(x) + '.pklz') as inf:
            features1X = pickle.load(inf)
        features1 = np.vstack((features1, features1X))
        x += 1
    with gzip.open(infolder + 'features2_0.pklz') as inf:
        features2 = pickle.load(inf)
    x = 1
    while os.path.isfile(infolder + 'features2_' + str(x) + '.pklz'):
        with gzip.open(infolder + 'features2_' + str(x) + '.pklz') as inf:
            features2X = pickle.load(inf)
        features2 = np.vstack((features2, features2X))
        x += 1
    return ids1, ids2, features1, features2, features


def doVectorize(texts1, texts2, ngramsfrom, ngramsto, minF, maxF, counttype, featureweight):
    """Perform vectorization process."""
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer, \
                                                TfidfTransformer
    print "Vectorizing texts"

    # Count options: binary, count, or augmented (could add log)
    vectorizer = CountVectorizer(decode_error='ignore',
        ngram_range=(ngramsfrom, ngramsto), max_df=maxF, min_df=minF,
        binary=counttype==0)
    features1 = vectorizer.fit_transform(texts1)
    features2 = vectorizer.transform(texts2)
    featureNames = vectorizer.get_feature_names()
    if counttype == 1:  # scaled basic count
        maxCount = max_sparse(features1)
        features1 = np.true_divide(features1, maxCount)
        features2 = np.true_divide(features2, maxCount)
    elif counttype == 2:  # augmented count
        features1 = augment_count(features1)
        features2 = augment_count(features2)

    # Feature weighting options: none, idf, or tvs
    if featureweight == 1:  # tf-dif
        tfidf = TfidfTransformer()
        features1 = tfidf.fit_transform(features1)
        features2 = tfidf.transform(features2)
    elif featureweight == 2:  # term variance score
        myTVS = TermVarianceScore()
        features1 = myTVS.fit_transform(features1)
        features2 = myTVS.transform(features2)

    print "Total feature count is %d" % len(featureNames)
    return features1, features2, featureNames


def combinefeatures(file1, file2, outfile):
    """Combine 2 feature sets.

    Note: a more extensive version of this is in features.py"""
    import pickle
    from scipy.sparse import hstack

    # load feature sets
    with open(file1, 'rb') as inf1:
        idsA1, idsA2, featA1, featA2, featAnames = pickle.load(inf1)
    with open(file2, 'rb') as inf2:
        idsB1, idsB2, featB1, featB2, featBnames = pickle.load(inf2)

    # check ids match
    if (sum([1 for x, y in zip(idsA1, idsB1) if x != y]) > 0) or \
            (sum([1 for x, y in zip(idsA2, idsB2) if x != y]) > 0):
        print "ids do not match for feature sets"
        return

    # put feature sets together
    features1 = hstack((featA1, featB1), 'csr')
    features2 = hstack((featA2, featB2), 'csr')
    featureNames = featAnames + featBnames

    # Save training & test features
    with open(outfile, 'wb') as outf:
        pickle.dump((idsA1, idsA2, features1, features2, featureNames), outf)


# *********************** stage 4 add in valence data ************************

def getValences(infile, binarize=False):
    """Return the valence data associated with this file.

    Note: in an ideal set-up, should scale based only on min and max in
    the training set. To do so, could take in testIDs and split.
    """
    import csv
    import numpy as np
    from scipy.sparse import csr_matrix

    print "Reading valence data"
    ids1x, valences1 = readValenceData(fileStem[:offset] + f1 + '.pkl')
    ids2x, valences2 = readValenceData(fileStem[:offset] + f2 + '.pkl')
    # Make sure ids match; note: need to pass in ids1 & ids2 to do so
    # if len([1 for x,y in zip(ids1,ids1x) if x != y]) != 0:
    #     print "IDs don't match (text-valence) for master file"
    #     return
    # if sum([1 for x,y in zip(ids2,ids2x) if x != y]) != 0:
    #     print "IDs don't match (text-valence) for input file 2"
    #     return

    print "Rescaling & adding valence info"
    valences1min = list(np.amin(valences1, axis=0))
    valences1max = list(np.amax(valences1, axis=0))
    valences1range = [x - y for x, y in zip(valences1max, valences1min)]
    # If vectorization is binary, make valences too: round
    # Alternate option for sentiment valences (posns. 1-5):
    #   negative -> 0, positive -> 1 (handle this in valences.py)
    valences1X = scaleValences(valences1, valences1min, valences1range,
                               binarize)
    valences2X = scaleValences(valences2, valences1min, valences1range,
                               binarize)

    # Read valence names from file, to add to feature names
    valenceoffset = -7 if offset == None else offset -7
    valenceNames = []
    with open(fileStem[:valenceoffset] + '_valenceAvgs.csv', 'rU') as f:
        valInfo = csv.reader(f)
        for row in valInfo:
            valenceNames.append(row[0].strip())
    return csr_matrix(valences1X), csr_matrix(valences2X), valenceNames


def readValenceData(picklefile):
    import pickle
    with open(picklefile, 'rb') as valpickle:
        valenceinfo = zip(*pickle.load(valpickle))
    return [int(x) for x in valenceinfo[0]], valenceinfo[1]


def scaleValences(valences, mins, ranges, binarize):
    return [[doscale(val, vmin, vrange, binarize) for val, vmin, vrange in \
                zip(textVals, mins, ranges)] \
            for textVals in valences]


def doscale(val, vmin, vrange, binarize):
    """Rescale (and, if necessary, round) valences. Truncate at 0 and 1."""
    scaled = val if vrange == 0 else (val-vmin)/vrange
    binscaled = round(scaled) if binarize else scaled
    return min(max(binscaled, 0), 1)


# ****************** stage 4 class for term variance scores ******************

class TermVarianceScore():
    """Converts term counts to term variance scores.

    Term variance is an index introduced in Liu et al. 2005 to get a measure
    of how informative the presence of a term is. Liu et al. define the
    variance of term i in a corpus of j documents as the sum across all
    docs of: (count of feature i in doc j - avg. count of feature i) ^2
    = sum over j of: count^2 - 2*count*avg + avgcount^2
    =`sum over j of: (count^2 - 2*count*avg)   plus: j*avgcount^2
    = first part is zero when count = 0 -> can keep sparse

    Note: better to translate this to a rate, to control for doc length
        (same derivation for keeping it sparse applies)

    No error-checking on parameters or values passed in.
    """

    def __init__(self, scale=(0,1)):
        self.scale = scale


    def fit_transform(self, X):

        """Prepare & convert feature counts to term variance scores."""
        import numpy as np
        from scipy.sparse import csr_matrix

        # Sum counts across articles; divide by tot. nr. words to get rates
        feature_counts = X.sum(axis=0)
        total_count = np.sum(feature_counts)
        feature_rates = feature_counts / float(total_count)
        featrate_sq = np.square(feature_rates)
        adj_featrate_sq = X.shape[0] * featrate_sq

        # Now measure variance
        word_counts = np.array(X.sum(axis=1))[:,0]
        word_counts_inv = \
            np.reshape(1. / np.array([max(xi, 1) for xi in word_counts]),
                       (X.shape[0], 1))
        doc_rates = csr_matrix(X.multiply(word_counts_inv))
        docrate_sq = doc_rates.copy()
        docrate_sq.data **= 2
        docrate_prod = doc_rates.multiply(-2 * feature_rates)
        # docrate_prod = np.multiply(-2 * doc_rates, feature_rates)
        docrate_diff = docrate_sq - docrate_prod
        variance = docrate_diff.sum(axis=0) + adj_featrate_sq

        # Rescale to max at 1
        varmax = variance.max()
        variance /= varmax
        # Multiply all count measures by the rescaled variance
        self.variance = variance
        return X.multiply(variance)


    def transform(self, X):
        return X.multiply(self.variance)


# **************************** stage 4 auxiliaries ***************************

def augment_count(X):
    """Augment feature count: 0.5 + (0.5 * count / docmax)"""
    import numpy as np
    from scipy.sparse import csr_matrix

    # Find count of most frequent word in each article
    docmax = np.array(X.max(axis=1).todense())[:,0]
    docmax_inv = np.reshape(0.5 / np.array([max(xi, 1) for xi in docmax]),
                            (X.shape[0], 1))
    aug_vals = csr_matrix(X.multiply(docmax_inv))
    aug_vals.data += 0.5
    return aug_vals


def min_sparse(aSparseMatrix):
    # min. value of a sparse matrix; assumes matrix is not empty
    if len(aSparseMatrix.data) == 0:
        return 0
    m = aSparseMatrix.data.min()
    return m if aSparseMatrix.getnnz() == aSparseMatrix.size else min(m, 0)


def max_sparse(aSparseMatrix):
    # max. value of a sparse matrix; assumes matrix is not empty
    if len(aSparseMatrix.data) == 0:
        return 0
    m = aSparseMatrix.data.max()
    return m if aSparseMatrix.getnnz() == aSparseMatrix.size else max(m, 0)


# ************************** preprocessing, stage 5 **************************
# ****************************************************************************

def preprocess5(fileStem, SVDnr, norm):
    """Final preprocessing stage: dimensionality reduction.

    Reduce number of features with SVD (a version of PCA).
    TruncatedSVD is a way to do 'unsupervised' feature selection
    on sparse matrices. The algorithm 'arpack' performs much better
    than 'randomized' (the other option).

    Variance in Muslim dataset explained @ 500 features: 47%; 1000 - 58%,
        1500 - 66%, 2000 - 72%, 2500 - 76% (processing slows down dramatically
        as n rises)
    This value is found in selector.explained_Variance_ratio_.sum()

    Note: SVDnr must be smaller than the number of articles in features1!
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize, MinMaxScaler
    import scipy as sp
    import pickle

    print "\nRunning SVD: %d features, normalize: %s" % \
          (SVDnr, 'T' if norm else 'F')
    if SVDnr == -1:
        return ''  # no extensions added
    newExt = '_' + str(SVDnr) + boolStr(norm)
    outfile = fileStem + newExt + '.pkl'
    if os.path.isfile(outfile):
        print "Preprocessing stage 5 already done; step skipped."
    else:
        with open(fileStem + '.pkl', 'rb') as infile:
            ids1, ids2, features1, features2, featureNames = \
                pickle.load(infile)
        features1X = selector.fit_transform(features1)
        features2X = selector.transform(features2)

        # SVD produces negative values, which may cause problems
        # in the training set (for fitting functions later)
        scale0_1 = True
        if scale0_1:
            scaler = MinMaxScaler(copy=False)
            features1X = scaler.fit_transform(features1X)
            features2X = scaler.transform(features2X)

        # Output of truncated SVD is not normalized; might wish to do so
        if norm:
            features1X = normalize(features1X)
            features2X = normalize(features2X)

        # write new output file (note: feature names no longer applicable)
        with open(outfile, 'wb') as outFile:
            pickle.dump((ids1, ids2, features1X, features2X, []), outFile)
    return newExt


# ************************** unused (old) auxiliaries ************************

def divideNonZeroes(featureMatrix, maxVal):
    """Divide every non-zero value in csr matrix featureMatrix by maxVal."""
    from scipy.sparse import lil_matrix, csr_matrix
    featMatrix = lil_matrix(featureMatrix)
    for row, col in zip(*featMatrix.nonzero()):
        featMatrix[row,col] = featMatrix[row,col]/maxVal
    return csr_matrix(featMatrix)


def subsetIndices(anArray, indices):
    """Subset array based on indices specified."""
    newArray  = []
    for x in indices:
        newArray.append(anArray[x])
    return newArray


def csr_vappend(a,b):
    """Append 2 csr_matrices.

    Take in 2 csr_matrices and append the second one to the bottom
    of the first one. Overwrites the original.
    """
    import numpy as np
    a.data = np.hstack((a.data, b.data))
    a.indices = np.hstack((a.indices, b.indices))
    a.indptr = np.hstack((a.indptr, (b.indptr + a.nnz)[1:]))
    a._shape = (a.shape[0] + b.shape[0], b.shape[1])
    return a


# ************************** command-line invocation *************************
# ****************************************************************************


# The following code is for batch operation from outside python,
# for testing purposes
if __name__ == "__main__":
    import sys
    from distutils.util import strtobool

    # See if we were called with the right number of arguments
    nrArgs = len(sys.argv) -1  # Arguments to python include program name
    if nrArgs > 3:
        print "Error in nr. of arguments"
    elif nrArgs == 0:
        # preprocess1('Muslim/Muslim', 'Muslim/Muslim', 'TM', '', 'Dict.csv')
        splitLabels('Muslim/MuslimL0', 7)
        splitbyID(['Muslim/MuslimL6'], 'Muslim/Muslim_IDs.txt', headerRow=True)
    elif nrArgs == 1:
        # mergeTexts(sys.argv[1])
        # preprocess(sys.argv[1])
        # reduce_BAdict(sys.argv[1])
        # extractTextandClasses(sys.argv[1], 2, 3)
        # splitLabels(sys.argv[1], 4)
        # processLabels(sys.argv[1])
        mergeTexts(sys.argv[1])
    elif nrArgs == 2:
        # removeValids(sys.argv[1],sys.argv[2])
        # makeBOW(sys.argv[1],sys.argv[2])
        preprocess_std(sys.argv[1], sys.argv[2])
        # stemTexts(sys.argv[1], int(sys.argv[2]))
    elif nrArgs == 3:
        preprocess_std(sys.argv[1], sys.argv[2], sys.argv[3])


