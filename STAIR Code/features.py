# encoding: utf-8

# features.py
# by Maurits van der Veen
# last modified 2015-07-22

# Functions to handle reading features from pickle file into memory
# and to reduce dimensionality, either by decomposition or by selection

# TODO: Add term variance as feature ranking fn. (see TVS vect choice)
# TODO: Add support for information gain as selector (low priority)

__author__ = 'maurits'

import globals


# **************************** Feature selection *************************

from sklearn.feature_selection import chi2, f_classif

def emptyParams():
    return {'select': {'bestNr': [], 'bestFn': []}}


def fullParams():
    from sklearn.feature_selection import chi2, f_classif

    return {'select': {'bestNr': [500, 1000, 2000, 0, -1],
                       'bestFn': [chi2, f_classif, bns, logcount, label_tfidf]}}


def initParams():
    """Return feature selection specs for this run."""
    from sklearn.feature_selection import chi2, f_classif

    return {'select': {'bestNr': [-1,],
                       'bestFn': [bns, logcount, label_tfidf]}}
    # return fullParams()


def paramCombos(params, nrfeatures):
    """Generate possible combinations of parameters given feature dict."""
    combos = []
    for bestNr in params['select']['bestNr']:
        if (bestNr < nrfeatures or bestNr <= 0):
            fnlist = [chi2] if bestNr == 0 else \
                     params['select']['bestFn']
            for bestFn in fnlist:
                combos.append([bestNr, bestFn])
    return combos


# *************************** Reading & writing features *********************

def writefeatures(outputstem, idlist, featurelist, featurenames):
    """Write features to csv file, with feature names in header row."""
    import csv
    outfilename = outputstem + '.csv'
    with open(outfilename, 'wb') as outfile:
        outdata = csv.writer(outfile)
        outdata.writerow(['id',] + featurenames)
        outdata.writerows([[id,] + list(feats) for id, feats in \
                           zip(idlist, featurelist)])


def readfeatures(infilename, nrfeatures=1, firstfeatcol=1, header=True):
    """Read features from csv file (id, feature1, ...)."""
    import csv
    ids = []
    features = []
    featurenames = ['feature' + str(x) for x in xrange(nrfeatures)]
    with open(infilename, 'rU') as infile:
        for row in csv.reader(infile):
            if len(row) > 0:
                if header:
                    featurenames = row[firstfeatcol:firstfeatcol+nrfeatures]
                    header = False
                elif 'id' not in row[0].lower():  # Remnant of old header
                    ids.append(int(row[0].strip()))
                    features.append([float(x) for x in row[firstfeatcol:firstfeatcol+nrfeatures]])
    return ids, features, featurenames


def read1feature(infilename, header=True):
    """Read a single feature from csv file (id, feature1, ...)."""
    ids, features, featurenames = readfeatures(infilename, 1, header)
    return ids, [x[0] for x in features], featurenames


def readfeature_subset(infilename, featureset, firstfeatcol=1, header=True):
    """Read a subset of features from csv file (id, feature1, ...)."""
    maxindex = max(featureset)
    ids, features, featurenames = \
        readfeatures(infilename, maxindex +1, firstfeatcol, header)
    features = [[x[i] for i in xrange(maxindex + 1) if i in featureset] \
                for x in features]
    featurenames = [featurenames[i] for i in xrange(maxindex + 1) \
                    if i in featureset]
    return ids, features, featurenames


# *************************** Reading & writing features *********************

def loadFeatures(fileStem, classFile, IDfile):
    """Load features and class labels into memory.

    Features split into training and testing set, but not (yet)
    into training and validation sets.
    """
    import csv, pickle

    # Read in features & classes
    # Note that idsc1 contains a subset of ids1 if some articles are uncoded
    with open(fileStem + 'pkl', 'rb') as pfile:
        ids1, ids2, features1, features2, featureNames = pickle.load(pfile)
    idsc1, idsc2, classes1, classes2 = readCodes(classFile, IDfile)
    nrcoded = len(idsc1)

    # double-check that ids match as expected
    if len([1 for x,y in zip(ids1[:nrcoded],idsc1) if x <> y]) <> 0:
        print "IDs don't match (features-codes) for training samples"
        return
    if len([1 for x,y in zip(ids2,idsc2) if x <> y]) <> 0:
        print "IDs don't match (features-codes) for test samples"
        return

    # store in namedtuple, so can access repeatedly and easily
    return globals.ClassifierData(idsc1, ids2, ids1[nrcoded:],
                          features1[:nrcoded], features2, features1[nrcoded:],
                          classes1, classes2, featureNames,
                          nrcoded, len(idsc2), len(ids1) - nrcoded)


def buildFeatureSet(sourcelist):
    """Given list of files containing features, produce single feature set.

    Each file should be in pickle format, containing the tuple
    trainids, testids, trainfeatures, testfeatures, featurenames
    Save combined pickle file & return the 5-tuple too

    Assume files exist. If the first featureset in the list is sparse,
    will make the final result sparse too. Double-check that sample IDs match.
    No other checking (duplicate feature names possible).
    """
    import pickle
    from scipy.sparse import issparse

    if len(sourcelist) == 0:
        return []
    with open(sourcelist[0], 'wb') as featSource:
        ids1a, featuresa, ids1b, featuresb, allnames = pickle.load(featSource)
    if len(sourcelist) > 1:
        for file in sourcelist[1:]:
            with open(file, 'wb') as featSource:
                ids2a, features2a, ids2b, features2b, names2 = \
                    pickle.load(featSource)
            if len([1 for x,y in zip(ids1a, ids2a) if x != y]) != 0 or \
                    len([1 for x,y in zip(ids1b, ids2b) if x != y]) != 0:
                print "IDs don't match for feature concatenation."
                return []
            allnames += names2
            featuresa, featuresb = \
                concatSparseFeatures(featuresa, featuresb,
                                     features2a, features2b) \
                if issparse(features) else \
                concatFeatures(featuresa, featuresb,
                               features2a, features2b)

    # write out combined results & return
    outfile = "combined" + str(len(sourcelist)) + "features.pkl"
    with open(outfile, 'wb') as outFile:
        pickle.dump((ids1a, ids1b, featuresa, featuresb, allnames), outFile)
    return ids1a, ids1b, featuresa, featuresb, allnames


def concatFeatures(feat1a, feat1b, feat2a, feat2b):
    """Concatenate two dense feature-sets."""
    from scipy import hstack

    return hstack([feat1a, feat2a]), hstack([feat1b, feat2b])


def concatSparseFeatures(feat1a, feat1b, feat2a, feat2b):
    """Concatenate two sparse feature-sets."""
    from scipy.sparse import hstack, csr_matrix

    return hstack([feat1a, csr_matrix(feat2a)], 'csr'), \
           hstack([feat1b, csr_matrix(feat2b)], 'csr')


# ****************************** Feature modification ************************

def rescale_0_1(val):
    """Scale a feature (i.e. a list) to the 0...1 range."""
    minv = min(val)
    rangev = max(val) - minv
    return [(x - minv) / float(rangev) for x in val]


def pos_to_1(val):
    """Convert negative values to 0 and positive to 1."""
    return [1 if x >= 0 else 0 for x in val]


def binarizethres(val, thres):
    """Convert values below thres to 0 and above to 1."""
    return [1 if x >= thres else 0 for x in val]


# ******************** Stacking: turn predictions into features **************

# TODO: make sure it all works, extracted out from textAutocoder
def stackFeatureSet(filestem, suffix, stackType):
    """Read classification results in for use as features for stacking.

    Read input data from preds file saved in earlier round.
    Save as pickle file, along with auto-generated feature names.

    Add to features or replace features, keeping names
    Make sure round output results are identifiable as special
    - simple pass-through of features, classifiers (thres = 1)

    Read predictions / scores (depending on parameter)
    Go back to the most recent file that has an _avg and a _preds
    for the round.
    """
    target = stackType

    # Find file with predictions, working with filestem and suffix
    tempStem = fileStem[:-1]
    tempRound = roundNr - 1
    while not os.path.isfile(tempStem + str(tempRound) + \
            '_avg.csv') or \
          not os.path.isfile(tempStem + str(tempRound) + \
            '_fullpreds.csv'):
        tempRound -= 1
        if tempRound < 0:
            print "No averages & preds saved for any round"
            return

    # Read in predictions
    filename = tempStem + str(tempRound) + '_fullpreds.csv'
    print "Stacking round, based on %s from round %d." % \
          (target, tempRound)
    ids, predictions = readRoundPredictions(filename, target)

    # Skip over all specs & performance data
    predfeatures = [[float(y) for y in x[headersLen:]] \
                    for x in predictions]
    predfeaturesT = zip(*predfeatures)

    # Split predictions into train, test, new sets
    trainFnew = predfeaturesT[:myData.nrtrain]
    testFnew = \
        predfeaturesT[myData.nrtrain:myData.nrtrain+myData.nrtest]
    newFnew = predfeaturesT[myData.nrtrain+myData.nrtest:]

    # Training data read in is ordered according to a previous
    # round's fold order (stratification, possibly shuffle).
    # Re-sort it according to original order.
    trainIDsnew = [ids[ids.index(x)] for x in myData.trainIDs]
    trainFnewX = [trainFnew[ids.index(x)] for x in myData.trainIDs]

    # Check that IDs match on train set, test set, new set
    if len([1 for x,y in zip(myData.trainIDs, trainIDsnew) \
            if x != y]) != 0:
        print "IDs don't match (texts-preds) for training data"
        return
    if len([1 for x,y in
            zip(myData.testIDs,
                ids[myData.nrtrain:myData.nrtrain+myData.nrtest]) \
            if x != y]) != 0:
        print "IDs don't match (texts-preds) for test data"
        return
    if len([1 for x,y in zip(myData.newIDs,
                             ids[myData.nrtrain+myData.nrtest:]) \
            if x != y]) != 0:
        print "IDs don't match (texts-preds) for new data"
        return

    # Augment each article by the additional features
    newtrainFeatures = hstack([myData.trainFeatures,
                                    csr_matrix(trainFnewX)], 'csr')
    newtestFeatures = hstack([myData.testFeatures,
                                   csr_matrix(testFnew)], 'csr')
    newnewFeatures = hstack([myData.newFeatures,
                                  csr_matrix(newFnew)], 'csr')

    # Generate feature names from all 8 parameter settings:
    # - SVDnr, norm, bestNr, bestFn, classifier, param1, param2, param3
    i = featuresIndex
    prednames = [x[i] + x[i+1][0] + x[i+2] + x[i+3].split()[1] + \
                    x[i+4] + x[i+5] + \
                    ('x' if x[i+6] == '' else \
                         (x[i+6] if len(x[i+6].split()) == 1 else
                         x[i+6].split()[1])) + \
                    x[i+7] for x in predictions]
    newfeatureNames = myData.featureNames + prednames

    return globals.ClassifierData(myData.trainIDs, myData.testIDs,
                myData.newIDs, newtrainFeatures, newtestFeatures,
                newnewFeatures, myData.trainLabels, myData.testLabels,
                newfeatureNames, myData.nrtrain, myData.nrtest, myData.nrnew)


# *********************** Feature management: train/test set *****************


def matchIDs(idsA1, idsA2, idsB1, idsB2):
    """Verify that ids in training and test sets match."""
    if len([1 for x, y in zip(idsA1, idsB1) if x != y]) > 0:
        print "IDs don't match for first (training) set"
        return False
    if len([1 for x, y in zip(idsA2, idsB2) if x != y]) > 0:
        print "IDs don't match for second (test) set"
        return False
    return True


# def organizeData(trainids, testids, newids,
#                  trainfeatures, testfeature, newfeatures,
#                  trainlabels, testlabels, featurenames):
#     """Compile classifier data into a single namedtuple."""
#     import numpy as np
#
#     theData = globals.ClassifierData(trainids, testids, ids1[newids],
#                           trainfeatures, testfeatures, newfeatures,
#                           np.array(trainlabels), np.array(testlabels),
#                           featureNames, nrtrain, len(idsc2),
#                           len(ids1) - nrtrain)
#     return theData


def splitTrainValid(myData, validIDlist):
    """Split training data into train- and valid- subsets."""
    import scipy as sp
    import numpy as np

    y = np.array([-1 if y in validIDlist else 1 for y in myData.trainIDs])
    y_val = np.where(y < 0)[0]
    y_tr = np.where(y > 0)[0]
    trainIDs = y[y_tr]
    validIDs = y[y_val]
    trainFeatures = myData.trainFeatures[y_tr]
    validFeatures = myData.trainFeatures[y_val]
    trainLabels = myData.trainLabels[y_tr]
    validLabels = myData.trainLabels[y_val]

    # Calculate fraction of training labels that is 1.
    # Note: this assumes a binary class
    fraction1 = sum(trainLabels)/float(len(trainLabels))

    return pareFeatures(
        globals.TrainingData(trainIDs, validIDs, myData.testIDs,
                                myData.newIDs, trainFeatures, validFeatures,
                                myData.testFeatures, myData.newFeatures,
                                trainLabels, validLabels, myData.testLabels,
                                myData.featureNames, len(trainIDs),
                                len(validIDs), myData.nrtest,
                                myData.nrnew, fraction1))


def nosplitTrainValid(myData):
    """No CV validation set -> just set to empty lists."""

    # Calculate fraction of training labels that is 1.
    # Note: this assumes a binary class
    fraction1 = sum(myData.trainLabels)/float(len(myData.trainLabels))

    return pareFeatures(
        globals.TrainingData(myData.trainIDs, [], myData.testIDs,
                                myData.newIDs, myData.trainFeatures, [],
                                myData.testFeatures, myData.newFeatures,
                                myData.trainLabels, [], myData.testLabels,
                                myData.featureNames, myData.nrtrain, 0,
                                myData.nrtest, myData.nrnew, fraction1))


def pareFeatures(myData):
    """Remove all features that are 0 across the entire training set."""
    import numpy as np

    feature_sums = myData.trainFeatures.sum(axis=0)
    to_keep = feature_sums != 0
    if len(to_keep.shape) > 1:  # squeeze from matrix down to vector
        temp = np.array(to_keep)
        to_keep = np.squeeze(temp)
    featureNamesX = np.array(myData.featureNames)[to_keep]
    if len(myData.featureNames) > len(featureNamesX):
        print "Removing %d features (don't occur in training set)" % \
              (len(myData.featureNames) - len(featureNamesX))
    return globals.TrainingData(myData.trainIDs, [], myData.testIDs,
                                myData.newIDs,
                                myData.trainFeatures[:,to_keep],
                                [] if myData.validFeatures == [] else \
                                    myData.validFeatures[:,to_keep],
                                myData.testFeatures[:,to_keep],
                                [] if myData.newFeatures == [] else \
                                    myData.newFeatures[:,to_keep],
                                myData.trainLabels,
                                myData.validLabels,
                                myData.testLabels, featureNamesX,
                                myData.nrtrain, myData.nrvalid, myData.nrtest,
                                myData.nrnew, myData.fraction1)


def subdrawData(myData, subdrawids):
    """Return new TrainingData tuple with subdraw data."""
    import scipy as sp
    import numpy as np

    y = np.array([1 if y in subdrawids else -1 for y in myData.trainIDs])
    y_tr = np.where(y > 0)[0]
    trainIDs = y[y_tr]
    trainFeatures = myData.trainFeatures[y_tr]
    trainLabels = myData.trainLabels[y_tr]

    # Calculate fraction of training labels that is 1.
    # Note: this assumes a binary class
    fraction1 = sum(trainLabels)/float(len(trainLabels))
    print "in subdraw, fraction 1 =", fraction1

    return pareFeatures(
        globals.TrainingData(trainIDs, myData.validIDs, myData.testIDs,
                             myData.newIDs, trainFeatures,
                             myData.validFeatures, myData.testFeatures,
                             myData.newFeatures, trainLabels,
                             myData.validLabels, myData.testLabels,
                             myData.featureNames, len(trainIDs),
                             len(myData.validIDs), myData.nrtest,
                             myData.nrnew, fraction1))



# ************************** Dimensionality reduction ***********************

def dimensionReduce(TTdata, SVDnr, norm):
    """Reduce number of features with SVD (a version of PCA).

    TruncatedSVD is a way to do 'unsupervised' feature selection
    on sparse matrices. The algorithm 'arpack' performs much better
    than 'randomized' (the other option).

    Variance explained @ 500 features: 47%; 1000 - 58%, 1500 - 66%,
        2000 - 72%, 2500 - 76% (processing slows down dramatically as n rises)
    This value is found in selector.explained_Variance_ratio_.sum()

    Note: SVDnr must be smaller than the number of articles used to fit it!
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize
    from preprocess import min_sparse, max_sparse
    import scipy as sp

    if SVDnr == -1:
        return TTdata
    else:
        # Do SVD on training & new set together, thus incorporating knowledge
        # about the full dataset (but not about the validation or test sets!)
        trainandnewFeatures = sp.sparse.vstack((TTdata.trainFeatures,
                                                TTdata.newFeatures),
                                               format='csr')
        selector = TruncatedSVD(n_components=SVDnr, algorithm='arpack',
                                random_state=42)
        trainandnewFeaturesX = selector.fit_transform(trainandnewFeatures)
        validFeaturesX = [] if TTdata.nrvalid == 0 else \
                         selector.transform(TTdata.validFeatures)
        testFeaturesX = selector.transform(TTdata.testFeatures)

        # SVD produces negative values, which may cause problems
        # in the training set (for fitting functions later)
        # Note: had sklearn's MinMaxScaler here, but it does it at the
        # feature-level, which eliminates relative feature weighting
        minval = min_sparse(trainandnewFeaturesX)
        range = max_sparse(trainandnewFeaturesX) - minval
        trainandnewFeaturesX -= minval
        trainandnewFeaturesX /= range
        if TTdata.nrvalid != 0:
            validFeaturesX -= minval
            validFeaturesX /= range
        testFeaturesX -= minval
        testFeaturesX /= range

        # Output of truncated SVD is also not normalized; might wish to do so
        if norm:
            trainandnewFeaturesX = normalize(trainandnewFeaturesX)
            validFeaturesX = [] if TTdata.nrvalid == 0 else \
                             normalize(validFeaturesX)
            testFeaturesX = normalize(testFeaturesX)

        # Split back into training and new sets
        trainFeaturesX = trainandnewFeaturesX[:TTdata.nrtrain]
        newFeaturesX = trainandnewFeaturesX[TTdata.nrtrain:]

    return globals.TrainingData(TTdata.trainIDs, TTdata.validIDs,
                    TTdata.testIDs, TTdata.newIDs, trainFeaturesX,
                    validFeaturesX, testFeaturesX, newFeaturesX,
                     TTdata.trainLabels, TTdata.validLabels,
                     TTdata.testLabels, [],  # no more feature names
                     TTdata.nrtrain, TTdata.nrvalid, TTdata.nrtest,
                     TTdata.nrnew, TTdata.fraction1)


# ************************** Feature ranking/selection ***********************

def bestFeatures(TTdata, bestSpecs):
    """Supervised feature ranking selection.

    Note: the function f_classif will make the matrices dense
          (this is OK for 2500 features or fewer).
    """
    bestNr, bestFn = bestSpecs
    if bestNr == 0:
        return TTdata
    elif bestNr == -1:
        return multiplybyrank(TTdata, bestFn)
    else:
        return selectbyrank(TTdata, bestFn, bestNr)


def selectbyrank(TTdata, rankingFn, k):
    """Select k best features in TTdata, by ranking function."""
    from sklearn.feature_selection import SelectKBest
    import numpy as np
    import scipy as sp

    selector = SelectKBest(rankingFn, k=k)
    # Select best features based on training set alone (preferred),
    # or training and validation set combined (usually not a good idea).
    # Never include test set!
    trainonly = True
    if trainonly or TTdata.nrvalid == 0:
        trainFeaturesX = selector.fit_transform(TTdata.trainFeatures,
                                                TTdata.trainLabels)
        validFeaturesX = [] if TTdata.nrvalid == 0 else \
                         selector.transform(TTdata.validFeatures)
    else:
        trainandvalidFeatures = \
            sp.sparse.vstack((TTdata.trainFeatures,
                              TTdata.validFeatures), format='csr')
        trainandvalidLabels = np.hstack((TTdata.trainLabels,
                                         TTdata.validLabels))
        trainandvalidFeaturesX = \
            selector.fit_transform(trainandvalidFeatures,
                                   trainandvalidLabels)
        trainFeaturesX = trainandvalidFeaturesX[:TTdata.nrtrain]
        validFeaturesX = trainandvalidFeaturesX[TTdata.nrtrain:]

    # Update other lists with trained selector
    testFeaturesX = selector.transform(TTdata.testFeatures)
    newFeaturesX = [] if TTdata.nrnew == 0 else \
                   selector.transform(TTdata.newFeatures)
    featureNamesX = [w for w, sel in zip(TTdata.featureNames,
                                         selector.get_support()) if sel]

    # Return new feature set. Feature names
    return globals.TrainingData(TTdata.trainIDs, TTdata.validIDs,
                    TTdata.testIDs, TTdata.newIDs, trainFeaturesX,
                    validFeaturesX, testFeaturesX, newFeaturesX,
                    TTdata.trainLabels, TTdata.validLabels,
                    TTdata.testLabels, featureNamesX,
                    TTdata.nrtrain, TTdata.nrvalid, TTdata.nrtest,
                    TTdata.nrnew, TTdata.fraction1)


def multiplybyrank(TTdata, rankingFn):
    """Multiply feature value by ranking fn value for feature."""
    import numpy as np
    from preprocess import min_sparse, max_sparse
    from scipy.sparse import issparse, csr_matrix

    # sklearn's rankingFns return a ranking value and a probability.
    # We ignore the latter (and ours don't return anything useful anyway)
    featVals, dummy = rankingFn(TTdata.trainFeatures, TTdata.trainLabels)

    # rescale feature values to 0..1 range
    featMin = min(featVals)
    featRange = max(featVals) - featMin
    featVals -= featMin
    featVals /= featRange

    # Multiply every non-zero value in feature sets by its associated
    #   score function value.
    if issparse(TTdata.trainFeatures):
        trainFeaturesX = csr_matrix(
            TTdata.trainFeatures.copy().multiply(featVals))
        validFeaturesX = [] if TTdata.validFeatures == [] else \
            csr_matrix(TTdata.validFeatures.copy().multiply(featVals))
        testFeaturesX = [] if TTdata.testFeatures == [] else \
            csr_matrix(TTdata.testFeatures.copy().multiply(featVals))
        newFeaturesX = [] if TTdata.newFeatures == [] else \
            csr_matrix(TTdata.newFeatures.copy().multiply(featVals))
    else: # numpy nd-array
        trainFeaturesX = \
            TTdata.trainFeatures.copy() * featVals
        validFeaturesX = [] if TTdata.validFeatures == [] else \
            TTdata.validFeatures.copy() * featVals
        testFeaturesX = [] if TTdata.testFeatures == [] else \
            TTdata.testFeatures.copy() * featVals
        newFeaturesX = [] if TTdata.newFeatures == [] else \
            TTdata.newFeatures.copy() * featVals


    return globals.TrainingData(TTdata.trainIDs, TTdata.validIDs,
                    TTdata.testIDs, TTdata.newIDs, trainFeaturesX,
                    validFeaturesX, testFeaturesX, newFeaturesX,
                    TTdata.trainLabels, TTdata.validLabels,
                    TTdata.testLabels, TTdata.featureNames,
                    TTdata.nrtrain, TTdata.nrvalid, TTdata.nrtest,
                    TTdata.nrnew, TTdata.fraction1)


# ****************************** Ranking functions ***************************

def bns(X, y):
    """Bi-normal separation ranking for feature selection.

    BNS is defined as abs(Finv(truepos/totalpos) - Finv(falsepos/totalneg)),
    where Finv is the inverse of the cumulative distribution function for the
    standard normal distribution (implemented in scipy.special.ndtri).

    For now, assume no negative input values, and simply convert any non-zero
    values to 1 (i.e. convert to presence-absence).
    Also assume y has both 0s and 1s

    The function used by selectKBest needs to return scores (BNS for us) as
    well as p-values (basically, probability of independence, given score).
    However, selectKBest does not use the p-values, so just return dummies.

    Note: can also use bns to generate list of key words.
    """
    from scipy.special import ndtri

    totpos, totneg, truepos, falsepos = posnegcounts(X, y)


    trueposR = truepos/float(totpos) if totpos > 0 else 0
    falseposR = falsepos/float(totneg) if totneg > 0 else 0
    # Adjust to prevent over/underflow
    trueposRadj = (trueposR + 0.000001) * 0.999999
    falseposRadj = (falseposR + 0.000001) * 0.999999

    # Calculate the inverse of the std. normal cdf
    invcdfT = ndtri(trueposRadj)
    invcdfF = ndtri(falseposRadj)
    binormalsep = abs(invcdfT - invcdfF)

    return binormalsep, [0,] * X.shape[1]


def label_tfidf(X, y):
    """Label-doc.frequency (delta tfidf in Martineau and Finin 2009).

    A simple measure, defined as freq. of term i in doc j. * log2 of
    ((# docs. w. label 0 & term i / # docs. w. label 1 & term i) *
    (# docs with label 1 / # docs w. label 0))

    Paltoglou & Thelwall 2010 suggest adding 0.5 to top and bottom
    inside the log for smoothing purposes, and replacing the term
    frequency by the augmented term frequency (sublinear):
    0.5 + 0.5tf/maxt(tf), where maxt(tf) is the frequency of the
    most frequent term in the document. This is vectChoice 4 for us
    """
    import numpy as np

    totpos, totneg, truepos, falsepos = posnegcounts(X, y)
    numerator = falsepos * totpos + 0.5
    denominator = truepos * totneg + 0.5

    return np.log2(numerator/denominator), [0,] * X.shape[1]


def logcount(X,y):
    """Log count ratio, as used in naive Bayes (see Wang, Manning 2012)."""
    import numpy as np

    totpos, totneg, truepos, falsepos = posnegcounts(X, y)
    # add smoothing parameter
    alpha = 0.5
    truepos = truepos + alpha
    falsepos = falsepos + alpha
    # divide by sum; take abs() just in case alpha is negative
    truepos /= abs(truepos).sum()
    falsepos /= abs(falsepos).sum()
    return np.log(truepos / falsepos), [0,] * X.shape[1]


def posnegcounts(X, y):
    """Helper to feature ranking functions (bns, label_tfidf, logcount).

    Assumes feature values are between 0 and 1.
    For BOW mode (sparse features), assume any non-0 value is a 1.
    For word-vector mode, work with rounded values. Assume dense matrix
    implies word-vector mode.
    """
    import numpy as np
    from scipy.sparse import issparse

    # Get the number of 1s and 0s in y
    totpos = sum(y)
    totneg = len(y) - totpos

    y1 = np.array(y).reshape(len(y),1)
    if issparse(X):  # any non-zero value is a '1'
        Xtruepos = X + y1 > 1
        Xfalsepos = X - y1 > 0
    else:  # round values
        roundedX = np.around(X)
        Xtruepos = roundedX + y1 > 1
        Xfalsepos = roundedX - y1 > 0

    # Sum down columns (per feature); squeeze dimensions if necessary
    truepos = Xtruepos.sum(axis=0)
    if len(truepos.shape) > 1:
        temp = np.array(truepos)
        truepos = np.squeeze(temp)
    falsepos = Xfalsepos.sum(axis=0)
    if len(falsepos.shape) > 1: # force Nx1 matrix into N vector
        temp = np.array(falsepos)
        falsepos = np.squeeze(temp)

    return totpos, totneg, truepos, falsepos


# TODO: implement
def infogain(X, y):
    """Information gain ranking for feature selection.

    Note: can also be used to generate lexicon of key words.

    Uses subfunction e(x,y); see Forman for description.
    """

# Code found online at http://stackoverflow.com/questions/25462407/
#                                   fast-information-gain-computation
# (claim: works but is not fast)

    def _entropy(values):
        counts = np.bincount(values)
        probs = counts[np.nonzero(counts)] / float(len(values))
        return - np.sum(probs * np.log(probs))

    def _information_gain(feature, y):
        feature_set_indices = np.nonzero(feature)[1]
        feature_not_set_indices = [i for i in feature_range \
                                   if i not in feature_set_indices]
        entropy_x_set = _entropy(y[feature_set_indices])
        entropy_x_not_set = _entropy(y[feature_not_set_indices])

        return entropy_before - (((len(feature_set_indices) /
                                   float(feature_size)) * entropy_x_set)
                                 + ((len(feature_not_set_indices) /
                                     float(feature_size)) *
                                    entropy_x_not_set))

    feature_size = X.shape[0]
    feature_range = range(0, feature_size)
    entropy_before = _entropy(y)
    information_gain_scores = []

    for feature in X.T:
        information_gain_scores.append(_information_gain(feature, y))
    return information_gain_scores, []
