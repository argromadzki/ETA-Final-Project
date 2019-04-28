# encoding: utf-8

# classifiers.py
# by Maurits van der Veen
# last modified 2015-06-22

__author__ = 'maurits'

import globals

# TODO: Add sklearn.ensemble.GradientBoostingClassifier

calcNewPreds = False
USLdata = []

# ************************ classifier parameter specs ************************

def paramNames():
    return {'multiNB': [],
            'kNN': ['metric', 'weights', 'k'],
            'RNG': ['metric', 'weights'],
            'linSVM': ['class_weight', 'C', 'loss'],
            'rbfSVM': ['class_weight', 'C', 'gamma'],
            'logit': ['class_weight', 'C', 'penalty'],
            'SGD': ['class_weight','alpha'],
            'Ada': ['n_est', 'crit', 'maxF'],
            'randForest': ['n_est', 'crit', 'maxF'],
            'extraTrees': ['n_est', 'crit', 'maxF']
            }


def fullParams():
    """Return all possible classifier-parameter combinations."""

    return {'multiNB': {},

            'kNN': {'metric': ['euclidean',],
                    'weights': ['uniform', 'distance',
                                dudaniweights, dualweights],
                    'k': [1, 5, 9, 15, 25]},
            'RNG': {'metric': ['euclidean',],
                    'weights': ['uniform', 'distance',
                                dudaniweights, dualweights]},

            'linSVM': {'class_weight': [None, 'auto', 2, 3, 4],
                       'C': [0.32, 1, 3.2, 10, 32, 100],
                       'loss': ['hinge', 'squared_hinge']},
            'rbfSVM': {'class_weight': [None, 'auto', 2, 3, 4],
                       'C': [0.32, 1, 3.2, 10, 32, 100],
                       'gamma': [0.001, 0.032, 0.01, 0.32, 1, 3.2]},
            'logit': {'class_weight': [None, 'auto', 2, 3, 4],
                      'C': [0.32, 1, 3.2, 10, 32, 100],
                      'penalty': ['l1', 'l2']},
            'SGD': {'loss': ['modified_huber', 'squared_hinge'],
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'alpha': [0.0000001, 0.000001, 0.00001, 0.0001,
                              0.01, 0.1, 1],
                    'learning_rate': ['constant',],
                    'eta0': [0.0000001, 0.000001, 0.00001, 0.0001, 0.01],
                    'class_weight': [None, 'auto', 2, 3, 4],},

            'Ada': {'n_est': [10,20,50,100],
                    'crit': ['gini','entropy'],
                    'maxF': ['auto', 0.1, 0.2, 0.5, None]},
            'randForest': {'n_est': [10,20,50,100],
                           'crit': ['gini','entropy'],
                           'maxF': ['auto', 0.1, 0.2, 0.5, None]},
            'extraTrees': {'n_est': [10,20,50,100],
                           'crit': ['gini','entropy'],
                           'maxF': ['auto', 0.1, 0.2, 0.5, None]}
            }


def initParams():
    """Return classifier specs for this run."""
    from RNG import dudaniweights, dualweights

    return {'linSVM': {'class_weight': [None,],
                       'C': [0.32, 1, 3.2],
                       'loss': ['l2',]},
            'rbfSVM': {'class_weight': [None,],
                       'C': [3.2,],
                       'gamma': [1,]},
            'logit': {'class_weight': [None,],
                      'C': [3.2, 10],
                      'penalty': ['l2',]}}
    # return fullParams()


def minParams():
    """Return minimal classifier specs."""
    return {'linSVM': {'class_weight': [None,],
                       'C': [1,],
                       'loss': ['squared_hinge',]},
            'logit': {'class_weight': [None,],
                      'C': [32, 100],
                      'penalty': ['l2',]}}


# ************************** classifier operation ****************************


def doClassify(myData):
    """Comment string"""
    from globals import USLtuple
    return runAll(myData, minParams(),
                  USLtuple([0,0], [-1] * myData.nrvalid), True)


def runAll(myData, classifDict, USLresults, newPreds):
    """Generate unique tuples for each classifier-parameter combination to
    facilitate averaging performance across multiple draws of training- and
    valid-sets.
    """
    import numpy as np
    from sklearn.naive_bayes import MultinomialNB

    global USLdata
    global calcNewPreds
    USLdata = USLresults
    calcNewPreds = newPreds

    results = []
    if 'kNN' in classifDict:
        results = runNN(myData, classifDict['kNN'])
    if 'RNG' in classifDict:
        results += runRNG(myData, classifDict['RNG'])
    if 'multiNB' in classifDict:
        results.append([['multiNB','', '', ''],] +
                       classify(myData, MultinomialNB(),
                                getWords=True))
    if 'linSVM' in classifDict:
        results += runSVMlin(myData, classifDict['linSVM'])
    if 'rbfSVM' in classifDict:
        results += runSVMrbf(myData, classifDict['rbfSVM'])
    if 'logit' in classifDict:
        results += runLogit(myData, classifDict['logit'])
    if 'SGD' in classifDict:
        results += runSGD(myData, classifDict['SGD'])
    if 'Ada' in classifDict:
        results += runAda(makeDense(myData), classifDict['Ada'])
    if 'randForest' in classifDict:
        results += randomForest(makeDense(myData), classifDict['randForest'])
    if 'extraTrees' in classifDict:
        results += extraTrees(makeDense(myData), classifDict['extraTrees'])
    if 'DBN':  # Deep learning
        results += runDBN(myData, classifDict)
    # Results is a list of 3-item lists: [specs, performance, classification]
    return results


def runNN(myData, params):
    """k-NN algorithm"""
    from sklearn.neighbors import KNeighborsClassifier
    from RNG import dudaniweights, dualweights
    results = []
    for metric in params['metric']:
        for weights in params['weights']:
            for k in params['k']:
                results.append(
                    [['kNN', metric, weights, k],] +
                    classify(myData,
                             KNeighborsClassifier(n_neighbors=k,
                                weights=weights, metric=metric),
                             getWords=False))
    return results


def runRNG(myData, params):
    """Relative neighborhood graph."""
    from RNG import RNGClassifier, dudaniweights, dualweights

    results = []
    for metric in params['metric']:
        for weights in params['weights']:
            results.append(
                [['RNG', metric, weights, ''],] +
                classify(myData,
                         RNGClassifier(weights=weights, metric=metric,
                                       pureOnly=False),
                         getWords=False))
    return results


def runSVMlin(myData, params):
    """Linear SVM."""
    from sklearn.svm import LinearSVC
    results = []
    for cweight in params['class_weight']:
        cweightX = cweight if cweight not in (2, 3, 4) \
            else overweight(myData.fraction1, cweight)
        for C in params['C']:
            for loss in params['loss']:
                results.append(
                    [['linSVM', cweight, C, loss],] +
                    classify(myData,
                             LinearSVC(loss=loss, C=C, class_weight=cweightX),
                             getWords=True))
    return results


def runSVMrbf(myData, params):
    """rbf kernel SVM."""
    from sklearn.svm import SVC
    results = []
    for cweight in params['class_weight']:
        cweightX = cweight if cweight not in (2, 3, 4) \
            else overweight(myData.fraction1, cweight)
        for C in params['C']:
            for gamma in params['gamma']:
                results.append(
                    [['rbfSVM', cweight, C, gamma],] +
                    classify(myData,
                             SVC(gamma=gamma, C=C, class_weight=cweightX),
                             getWords=False))
    return results


def runLogit(myData, params):
    """Multinomial logistic regression."""
    from sklearn.linear_model import LogisticRegression
    results = []
    for cweight in params['class_weight']:
        cweightX = cweight if cweight not in (2, 3, 4) \
            else overweight(myData.fraction1, cweight)
        for C in params['C']:
            for penalty in params['penalty']:
                results.append(
                    [['logit', cweight, C, penalty],] +
                    classify(myData,
                             LogisticRegression(penalty=penalty, C=C,
                                                class_weight=cweightX),
                             getWords=True))
    return results


def runSGD(myData, params):
    """Stochastic gradient descent classifier.

    Note: several additional parameters to play with.
    Note 2: tends to crash unless feature values rescaled
    """
    from sklearn.linear_model import SGDClassifier
    results = []
    for loss in ['modified_huber',]:
        for penalty in ['l2',]:
            for lrate in ['optimal',]:
                for cweight in params['class_weight']:
                    cweightX = cweight if cweight not in (2, 3, 4) \
                        else overweight(myData.fraction1, cweight)
                    for alpha in params['alpha']:
                        results.append(
                            [['SGD', cweight, alpha, ''],] +
                            classify(myData,
                                SGDClassifier(loss=loss, penalty=penalty,
                                    alpha=alpha, learning_rate=lrate,
                                    class_weight=cweightX, random_state =42),
                                getWords=True))
    return results


def runAda(myData, params):
    """Adaboost algorithm (requires dense data)."""
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    results = []
    for n in params['n_est']:
        for crit in params['crit']:
            for maxF in params['maxF']:
                results.append(
                    [['Ada', n, crit, maxF],] +
                    classify(myData,
                             AdaBoostClassifier(
                                 DecisionTreeClassifier(criterion = crit,
                                                        max_features=maxF,
                                                        random_state=42),
                                 n_estimators=trees, random_state=42),
                             getWords=False))
    return results


def randomForest(myData, params):
    """Random forest classifier (requires dense data)."""
    from sklearn.ensemble import RandomForestClassifier
    results = []
    for n in params['n_est']:
        for crit in params['crit']:
            for maxF in params['maxF']:
                results.append(
                    [['randForest', n, crit, maxF],] +
                    classify(myData,
                             RandomForestClassifier(n_estimators=trees,
                                criterion=crit, max_features=maxF,
                                random_state=42),
                             getWords=False))
    return results


def extraTrees(myData, params):
    """Extra trees classifier (requires dense data)."""
    from sklearn.ensemble import ExtraTreesClassifier
    results = []
    for n in params['n_est']:
        for crit in params['crit']:
            for maxF in params['maxF']:
                results.append(
                    [['extraTrees', n, crit, maxF],] +
                    classify(myData,
                             ExtraTreesClassifier(n_estimators=trees,
                                criterion=crit, max_features=maxF,
                                random_state=42),
                             getWords=False))
    return results


def runDBN(myData, params):
    """Deep belief network (from nolearn package).

    Parameters to vary: # hidden layers, # of nodes in each hidden layer,
    learning rates & decays (overall or per layer), and # epochs.

    Note: DBN is getting deprecated in nolearn, and will be replaced
    by Lasagne. Look into either using the nolearn.lasagne wrappers,
    or lasagne directly."""
    return []


def classify(myData, myClassifier, getWords):
    """Run specified classifier and evaluate performance."""
    from operator import itemgetter
    from evaluate import performance

    nrTopWords = 50
    # Fit the classifier on the training data
    myClassifier.fit(myData.trainFeatures, myData.trainLabels)
    # Get the top words/features (only meaningful if we did not do SVD)
    if len(myData.featureNames) > 0:
        topWords = [''] * 2 * nrTopWords
        if getWords:
            try:
                allwords = sorted(zip(myData.featureNames, myClassifier.coef_[0]),
                                  key=itemgetter(1))
                topWords = zip(*allwords[:nrTopWords])[0] + \
                                zip(*allwords[-nrTopWords:])[0][::-1]
                # make printable
                topWords = [x.encode('ascii', 'ignore') for x in topWords]
            except AttributeError:
                topWords = [''] * 2 * nrTopWords
    else:
        topWords = []

    # Get predictions and scores for the validation set
    if myData.nrvalid > 0:
        validLabelsP = list(myClassifier.predict(myData.validFeatures))
        validScoresP = \
            list(getPredictScores(myClassifier, myData.validFeatures,
                                  validLabelsP))
        # Replace predictions by unsupervised learning results
        # if sum(USLdata.predictions) != -1 * len(USLdata.predictions):
        #     validLabelsP = [x if y == -1 else y for x, y in
        #                      zip(validLabelsP, USLdata.predictions)]
        validPerformance = performance(myData.validLabels, validLabelsP)
    else:
        validLabelsP = []
        validScoresP = []
        validPerformance = [0,0,0,0,0,0,0,0,0,0]

    # 3. Get predictions and scores for the test set
    if myData.nrtest > 0:
        testLabelsP = list(myClassifier.predict(myData.testFeatures))
        testScoresP = \
            list(getPredictScores(myClassifier, myData.testFeatures,
                                  testLabelsP))
        testPerformance = performance(myData.testLabels, testLabelsP)
    else:
        testLabelsP = []
        testScoresP = []
        testPerformance = [0,0,0,0,0,0,0,0,0,0]

    # 4. Get predictions and scores for the new set, if any
    if myData.nrnew > 0 and calcNewPreds:
        newLabelsP = list(myClassifier.predict(myData.newFeatures))
        newScoresP = list(getPredictScores(myClassifier, myData.newFeatures,
                                           newLabelsP))
    else:
        newLabelsP = []
        newScoresP = []

    # Return [performance (add USLdata.info if applicable), predictions]
    return [validPerformance + testPerformance,
            globals.Classification(validLabelsP, validScoresP, testLabelsP,
                                   testScoresP, newLabelsP, newScoresP,
                                   topWords, myClassifier)]


# ************************** classifier auxiliaries **************************

def overweightX(labellist, factor):
    """Make minority class weight a multiple of its inverse proportion.

    Example: if 25% of labels are 1 and 75% 0, and the factor parameter is 2,
    then make weight of class 1 equal to 2* 1/(1/3) = 6. This is most likely
    to be useful especially if we suspect that the minority class is
    over-represented in the training & valid sets.
    """
    onestrainvalid = sum(labellist)
    zerostrainvalid = len(labellist) - onestrainvalid
    if onestrainvalid > zerostrainvalid: # more 1s
        oneweight = 1
        zeroweight = factor*onestrainvalid / float(zerostrainvalid)
    else: # more 0s (or equal numbers)
        zeroweight = 1
        oneweight = factor*zerostrainvalid / float(onestrainvalid)
    return {0 : zeroweight, 1 : oneweight}


def overweight(fraction1, factor):
    """Make minority class weight a multiple of its inverse proportion.

    Example: if 25% of labels are 1 and 75% 0, and the factor parameter is 2,
    then make weight of class 1 equal to 2* 1/(1/3) = 6. This is most likely
    to be useful especially if we suspect that the minority class is
    over-represented in the training & valid sets.
    """
    if fraction1 > 0.5:
        oneweight = 1
        zeroweight = factor * fraction1 / float(1 - fraction1)
    else: # more 0s (or equal numbers)
        zeroweight = 1
        oneweight = factor * (1 - fraction1) / float(fraction1)
    return {0 : zeroweight, 1 : oneweight}


def getPredictScores(myClassifier, features, predictions):
    """Get scores (probabilities) associated with class predictions.

    For decision function scores, rescale to range of 0-1 so as to
    get ersatz probability values.
    """
    from itertools import chain
    import sklearn
    try:
        predictscores = [x[1] for x in \
                         myClassifier.predict_proba(features)]
    except (AttributeError, TypeError):
        try:
            predictscores = myClassifier.decision_function(features)
            maxScore = max(predictscores)
            minScore = min(predictscores)
            if maxScore == minScore: # no usable scores -> use predictions
                return predictions
            else:  # scale decision fn. to use as probability
                return [scaleScore(x, minScore, maxScore) for x \
                        in predictscores]
        except (AttributeError, NotImplementedError):
            # no usable scores -> use predictions
            return predictions
    return predictscores


def scaleScore(x, minscore, maxscore):
    # Scale decision function score to be usable as ersatz probability.
    #
    # Positive decision function values are scaled to the range 0.5-1;
    # negative ones to the range 0-0.5. A bit sub-optimal,
    # but serviceable.
    if x >=0:
        return 0.5 + x / (2 * maxscore)
    else:
        return (x - minscore) / (-2 * minscore)


def weightedNN(distances,labels,n):
    """Calculate distance-weighted values for each class."""
    import math
    minDist = distances[0]
    maxDist = distances[n-1]
    sumweight0 = 0
    sumweight1 = 0

    for thisEdge in xrange(n):
        if labels[thisEdge] == 1:
            sumweight1 += dualweight(minDist,maxDist,distances[thisEdge])
        else: # label == 0
            sumweight0 += dualweight(minDist,maxDist,distances[thisEdge])
    return(sumweight0,sumweight1)


def makeDense(myData):
    """Make text feature arrays dense & rescale."""
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    myScaler = StandardScaler()
    dense1 = myScaler.fit_transform(texts1.toarray()).astype(np.float32)
    return dense1, myScaler.transform(texts2.toarray()).astype(np.float32), \
           myScaler.transform(texts3.toarray()).astype(np.float32), \
           myScaler.transform(texts4.toarray()).astype(np.float32)
