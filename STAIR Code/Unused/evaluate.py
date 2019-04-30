# encoding: utf-8

# evaluate.py
# by Maurits van der Veen
# last modified 2015-08-23

# Functions to evaluate the quality of a prediction, given a target

__author__ = 'maurits'


# *************************** notes, thoughts, etc. **************************

# Extension: handle multiple predictions against single target
#            (could list by feature name)


# *********************** matching predictions to targets ********************


def feature_setup(filestem):
    """Steps used to assess lexical features.

    For Muslim dataset, starting point is Muslim/MuslimTM.csv
    For imdb dataset, it is imdb/imdbT.csv
    """
    from preprocess import preprocess_std
    from corpus import texts2lines_file
    from lexica import expandLexicon
    from valence import corpus_valence
    from filter import sent2doc_valences, sent2doc_valences_filter

    # Take raw text file, sentence-tokenize (producing _tok file),
    # and do phrase substitution/reordering (producing _lex file)
    # TODO: improve Brit2Amer translation
    # TODO: filter out Islamabad, Mr. Muslimi
    # preprocess_std('Muslim', 'MuslimTM.csv', 'MuslimDict.csv')
    filestem += '_tok'

    # The steps below need to be done once each for the _tok and _lex files

    # Split texts into sentences (periods as sentence separators)
    # texts2lines_file('Bicycle/BicycleM_tok')
    # texts2lines_file('Bicycle/BicycleM_lex')

    # Generate corpus-specific lexica, based on seed lexicon
    # expandLexicon('Bicycle/BicycleM_tok.csv', 'Lexicon.pkl')
    # expandLexicon('Bicycle/BicycleM_lex.csv', 'Lexicon.pkl')
    # Performance on imdb_tok: 500 > 400 > 300 > 200 > 100 and
    # source intersect 4 > 5 > 6
    # -> use intersect 3 as source as well, and do 500, 600, 700, 800

    # Calculate sentence-level valences for entire corpus
    # corpus_valence(filestem)

    # Aggregate sentence-level valences to doc-level
    # May need to set read_features to skip any id that is not numeric
    # May want to change aggmethod parameter passed by sent2doc_valences
    # sent2doc_valences(filestem)
    # sent2doc_valences_filter(filestem)

    # Scale features and binarize as appropriate
    scale_features(filestem)
    scale_features(filestem + 'F1')
    scale_features(filestem + 'F2')
    # Binarize features that are not yet 0 or 1, selecting best threshold for
    #   training labels (add threshold to feature name), add 'B' to filestem
    # binarize_features(filestem + 'X')

    # Assess feature quality against training labels
    # assess_features(filestem + 'XB')


def logistic_shortcut(trainfeatures, trainlabels, testfeatures, testlabels,
                      newfeatures):
    """Run logistic regression classifier."""
    from sklearn.linear_model import LogisticRegression
    from classifiers import classify, overweight
    results = []
    for cweight in [None, 'auto', 2, 3, 4]:
        cweightX = cweight if cweight not in (2, 3, 4) \
            else overweight(myData.fraction1, cweight)
        for C in [0.1, 0.32, 1, 3.2, 10, 32, 100]:
            for penalty in ['l1', 'l2']:
                results.append(
                    [['logit', cweight, C, penalty],] +
                    classify(myData,
                             LogisticRegression(penalty=penalty, C=C,
                                                class_weight=cweightX),
                             getWords=True))
    return results


def extract_locs(textfile, outfilestem):
    """Generate a location-feature file corresponding text file's ids."""
    from corpus import readidtextfile

    ids, texts = readidtextfile(textfile)
    locs_features(outfilestem, ids, texts)


def locs_features(filestem, ids, texts):
    """Generate features for mention of different locations.

    Hardcoded for UK context: major cities in UK, other countries, and
    aggregates there-of into UK-relevant regions.
    """
    from features import writefeatures
    from corpus import presence_flags

    regions = ['UK', 'EU', 'ME_Arab', 'ME_nonArab', 'Arab_nonME',
               'Muslim_Comm', 'Muslim_other', 'Commonwealth_nonMus']

    countries = [['Britain', 'United Kingdom', 'England', 'Scotland',
                  'Wales', 'Northern Ireland',
                  'London', 'Manchester', 'Midlands', 'Yorkshire', 'Glasgow',
                  'Liverpool', 'Hampshire', 'Tyneside', 'Nottingham',
                  'Sheffield', 'Bristol', 'Belfast', 'Leicester', 'Edinburgh',
                  'Brighton', 'Hove', 'Bournemouth' ,'Poole', 'Cardiff',
                  'Teesside', 'Stoke-on-Trent', 'Coventry', 'Sunderland',
                  'Birkenhead', 'Reading', 'Kingston', 'Preston', 'Newport',
                  'Swansea', 'Southend', 'Derby', 'Plymouth', 'Luton',
                  'Farnborough', 'Aldershot', 'Medway', 'Blackpool',
                  'Milton Keynes', 'Barnsley', 'Dearne', 'Northampton',
                  'Norwich', 'Aberdeen', 'Swindon', 'Crawley', 'Ipswich',
                  'Wigan', 'Mansfield', 'Oxford', 'Warrington', 'Slough',
                  'Peterborough', 'Cambridge', 'Doncaster', 'Dundee', 'York',
                  'Gloucester', 'Burnley', 'Telford', 'Blackburn', 'Basildon',
                  'Grimsby', 'Hastings', 'Wycombe', 'Thanet',
                  'Accrington', 'Rossendale', 'Burton-upon-Trent',
                  'Colchester', 'Eastbourne', 'Exeter', 'Cheltenham',
                  'Paignton', 'Torquay', 'Lincoln', 'Chesterfield',
                  'Chelmsford', 'Basingstoke', 'Maidstone', 'Bedford',
                  'Worcester'],

                 ['Austria', 'Belgium', 'Bulgaria', 'Cyprus', 'Croatia',
                  'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France',
                  'Germany', 'Greece', 'Hungary', 'Ireland', 'Italy',
                  'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Netherlands',
                  'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia',
                  'Spain', 'Sweden'],

                 ['Bahrain', 'Egypt', 'Iraq', 'Jordan', 'Kuwait',  'Lebanon',
                  'Oman', 'Palestine', 'Qatar', 'Saudi Arabia', 'Syria',
                  'United Arab Emirates', 'Yemen'],

                 ['Cyprus', 'Iran', 'Israel', 'Turkey'],

                 ['Algeria', 'Comoros', 'Djibouti', 'Libya', 'Mauritania',
                  'Morocco', 'Somalia', 'Sudan', 'Tunisia'],

                 ['Bangladesh', 'Brunei', 'Malaysia', 'Pakistan',
                  'Sierra Leone', ],

                 ['Afghanistan', 'Albania', 'Azerbaijan',
                  'Burkina Faso', 'Chad', 'Gambia', 'Guinea',
                  'Indonesia', 'Kazakhstan', 'Kosovo',
                  'Kyrgyzstan', 'Maldives', 'Mali',
                  'Mauritania', 'Mayotte', 'Niger',
                  'Senegal', 'Somalia', 'Sudan',
                  'Tajikistan', 'Turkmenistan', 'Uzbekistan',
                  'Western Sahara'],

                 # Look separately for Antigua, Barbuda; St. Christopher,
                 # Nevis; St. Vincent, Grenadines; Trinidad, Tobago;
                 # remove 'The' from Bahamas, Maldives; note that Cyprus
                 # is also in EU & ME_nonArab
                 ['Antigua', 'Barbuda', 'Australia', 'Bahamas', 'Barbados',
                  'Belize', 'Botswana', 'Cameroon', 'Canada', 'Cyprus',
                  'Dominica', 'Fiji', 'Ghana', 'Grenada', 'Guyana',
                  'India', 'Jamaica', 'Kenya', 'Kiribati', 'Lesotho',
                  'Malawi', 'Maldives', 'Malta', 'Mauritius', 'Mozambique',
                  'Namibia', 'Nauru', 'New Zealand', 'Nigeria',
                  'Papua New Guinea', 'Rwanda ', 'St. Christopher', 'Nevis',
                  'St. Lucia', 'St. Vincent', 'Grenadines', 'Samoa',
                  'Seychelles', 'Singapore', 'Solomon Islands',
                  'South Africa', 'Sri Lanka', 'Swaziland', 'Tanzania',
                  'Tonga', 'Trinidad', 'Tobago', 'Tuvalu', 'Uganda',
                  'Vanuatu', 'Zambia']]

    writefeatures(filestem, ids, presence_flags(texts, countries), regions)


def calibrate_features(featurefilename, featureset, neutralscaler='', firstfeatcol=1):
    """Calibrate features by standardizing; return means & stdevs"""
    from sklearn.preprocessing import StandardScaler
    from features import writefeatures, readfeature_subset

    ids, features, featurenames = \
        readfeature_subset(featurefilename, featureset,
                           firstfeatcol=firstfeatcol, header=True)
    print "Nr. features: %d, nr. items: %d" % (len(features[0]), len(features))

    # Rescale & recenter features
    if neutralscaler == '':
        neutralscaler = StandardScaler()
        neutralscaler.fit(features)
    # else:  # set the mean of the corpus-specific dimension (nr. 9 of 10) to 0
    #     neutralscaler.mean_[8] = 0
    featuresX = neutralscaler.transform(features)
    featurenamesX = [x + 'S' for x in featurenames]

    # Reset 0s in features to 0s in featuresX?
    # (alternatively, filter out the rescaled (new-value) 0s in Stata)

    # Calculate average
    denominator = float(len(featureset))
    avgvalence = [sum(x)/denominator for x in featuresX]

    # Write results
    featurelist = [orig + list(scaled) + [avg,] for orig, scaled, avg \
                   in zip(features, featuresX, avgvalence)]
    featurenames = featurenames + featurenamesX + ['avg_valence',]
    writefeatures(featurefilename.split('.')[0] + 'X', ids,
                  featurelist, featurenames)
    return neutralscaler


def assess_features(filestem):
    """Evaluate quality of a set of features.

    Compare against labels (matched by ids):
    - feature file = filestem + '_features.csv'
    - label file = filestem + '_lab.csv
    """
    import csv
    from corpus import readidtextfile
    from features import readfeatures, writefeatures

    # Load labels, convert to value
    labelids, labels = readidtextfile(filestem + '_lab.csv')
    labels = [int(l) for l in labels]

    # Load features; keep only those ids for which we have a label
    ids, features, featnames = readfeatures(filestem + '_featuresX.csv',
                                            nrfeatures=999, header=True)
    idsfeatures = sorted(subset_by_id(zip(ids, features), labelids))
    featuresX = zip(*[x[1] for x in idsfeatures])
    obsids = [x[0] for x in idsfeatures]
    if len([1 for x, y in zip(labelids, obsids) if x != y]) > 0:
        print "ids don't match"
        return

    # print len(featuresX), len(featuresX[0])
    print "Number of labeled observations: %d" % len(idsfeatures)

    # 1. Get performance of each feature, assume cut = 0
    featperfs = [[name, ] + \
                 performance(labels, rescale_0_1(feat, True)) for \
                 name, feat in zip(featnames, featuresX)]
    with open(filestem + '_featurequality.csv', 'wt') as outfile:
        featqualfile = csv.writer(outfile)
        featqualfile.writerows(featperfs)

    # 2. Get performance of each feature allowing threshold change
    newthres = [best_thres(labels, feat) for feat in featuresX]
    featperfs = [[name + str(thres), ] + \
                 performance(labels, rescale_0_1(feat, True), thres) \
                 for name, feat, thres in zip(featnames, featuresX, newthres)]
    with open(filestem + '_featurequalityX.csv', 'wt') as outfile:
        featqualfile = csv.writer(outfile)
        featqualfile.writerows(featperfs)


def correlation_matrix(featurelist):
    """Produce matrix of Pearson's r across features.

    np.corrcoef expects each sublist to be a variable, rather than
    an observation for all variables -> need to transpose.
    """
    import numpy as np
    correls = np.corrcoef(map(list, zip(*featurelist)))
    # display for user to eyeball
    print correls


def subset_by_id(datalist, targetids):
    """Return subset of datalist for ids in targetids."""
    targetset = set(targetids)
    return [obs for obs in datalist if obs[0] in targetset]


def calc_posfraction(features):
    """Calculate average positive fraction across features."""
    nrobs = len(features)
    posnrs = [len([1 for x in feat if x > 0]) for feat in zip(*features)]
    posfracs = [x/float(nrobs) for x in posnrs]
    calcposfrac = sum(posfracs)/float(len(posfracs))
    print "Average positive fraction: %f" % calcposfrac
    return calcposfrac


def scale_features(filestem):
    """Scale all features in the input file to mean 0, stdev 1.

    First calculate fraction with positive valence.
    Lots here could be improved (sped up) by using scipy and/or panda
    """
    from sklearn.preprocessing import StandardScaler
    from features import readfeatures, writefeatures, \
                         binarizethres, pos_to_1

    ids, features, featnames = readfeatures(filestem + '_features.csv',
                                            nrfeatures=999, header=True)
    calcposfrac = calc_posfraction(features)

    # standardize features
    featnamesX = [name + 'S' for name in featnames]
    features = StandardScaler().fit_transform(features)

    # calculate average across features
    nrfeatures = float(len(features[0]))
    featavg = [sum(feat)/nrfeatures for feat in features]
    features = [list(feat) + [avg,] \
                for feat, avg in zip(features, featavg)]
    featnamesX += ['avg',]
    # Write all features
    writefeatures(filestem + 'S', ids, features, featnamesX)

    # For each feature, calculate 3 binarization versions
    estposfrac = 0.5
    binarizedfeats = []
    binarizednames = []
    for feature, name in zip(zip(*features), featnamesX):
        binaries, binnames = calcbinaries(feature, calcposfrac,
                                          estposfrac, name)
        binarizedfeats += binaries
        binarizednames += binnames
    # Write binarized values
    writefeatures(filestem + 'SX', ids, zip(*binarizedfeats),
                  binarizednames)
    # Combine with other features

    # Alternative option: scale to 0...1 range
    # Note: not rescaling a feature here causes some problem whereby values
    #   appear to become strings somehow; until resolved, just rescale all
    # featstoscale = [range_0_1(feat) for feat in featuresX]
    # featstoscale = [False for feat in featuresX]
    # featnamesX = [name if scaled else name + 'X' \
    #               for name, scaled in zip(featnames, featstoscale)]
    # features = [feat if scaled else rescale_0_1(feat) \
    #             for feat, scaled in zip(featuresX, featstoscale)]
    # writefeatures(filestem + 'X', ids, zip(*features), featnamesX)


def calcbinaries(feature, calcposfrac, estposfrac, name):
    """Binarize feature, using 3 different thresholds.

    Calculate thres to get desired fractions avgposfrac and estposfrac.
    """
    nrobs = len(feature)
    sortedfeat = sorted(feature)
    calcthres = sortedfeat[int(nrobs * (1 - calcposfrac))]
    print "Threshold for average positive fraction (%f): %f" % \
          (calcposfrac, calcthres)
    estthres = sortedfeat[int(nrobs * (1 - estposfrac))]
    print "Threshold for estimated positive fraction (%f): %f" % \
          (estposfrac, estthres)
    return [[1 if val >= 0 else 0 for val in feature], \
            [1 if val >= calcthres else 0 for val in feature], \
            [1 if val >= estthres else 0 for val in feature]], \
           [name + '_0', name + '_' + str(calcthres)[:5],
            name + '_' + str(estthres)[:5]]


def binarize_features(filestem):
    """Binarize non-binary features; append threshold to feature name."""
    import pickle
    from features import readfeatures, writefeatures

    # get training labels
    labelfile = filestem + '_lab.pkl'
    with open(labelfile, 'rb') as labels:
        trainids, testids, trainlabels, testlabels, encoding = \
            pickle.load(labels)
    # sort these, ids and labels in parallel
    trainidslabels = sorted(zip(trainids, trainlabels))
    trainlabelsX = [x[1] for x in trainidslabels]

    ids, features, featnames = readfeatures(filestem + 'X_features.csv',
                                            nrfeatures=999, header=True)
    # find out which features to binarize
    featuresX = zip(*features)
    featsbinary = [isbinary(feat) for feat in featuresX]
    # get values for training set only
    trainidsfeatures = sorted(subset_by_id(zip(ids, features), trainids))
    trainfeaturesX = zip(*[obs[1] for obs in trainidsfeatures])
    binarythres = [0.5 if binary else binarize(trainlabelsX, feat) \
                   for binary, feat in zip(featsbinary, trainfeaturesX)]
    featnamesX = [name if binary else name + '_' + str(thres)[:5] \
                  for name, thres in zip(featnames, binarythres)]
    featuresX = [[1 if lab >= thres else 0 for lab in predictlabels] \
                 for predictlabels, thres in zip(featuresX, binarythres)]
    writefeatures(filestem + 'XB', ids, zip(*featuresX), featnamesX)


def compare_features(filestem):
    """Check into multi-collinearity,
    maybe run logistic regression (why again?)

    Split off from assess_features.
    """
    # 2. display correlations between features; filter manually
    # correlation_matrix(features)

    # 3. run logistic regression classifier on remaining features
    # train on training set, monitor test set; try std. parameter vals.
    # trainidslabels = sorted(zip(trainids, trainlabels))
    # testidslabels = sorted(zip(testids, testlabels))
    #
    # trainidsfeatures = sorted(subset_by_id(idsfeatures, trainids))
    # testidsfeatures = sorted(subset_by_id(idsfeatures, testids))


# Have training set (look up functions): trainids, trainlabels
# Have set of predictions: trainids, trainfeatures
# For each of trainfeatures, if not binary, find optimal threshold, report it, report accuracy
def evaluate_features(trainfeatures, targetlabels):
    """Get quality of each feature.

    Assume features & labels are correctly sorted (matched).
    """




# **************************** binarization functions ************************

def isbinary(feature):
    """See if feature is binary, with only values 0 and 1."""
    return len([1 for x in feature if x != 1 and x != 0]) == 0


def isbinary2(feature):
    """See if feature is binary, with only values 0 and 1."""
    featset = set(feature)
    return len(featset) == 2 and 0 in featset and 1 in featset


def binarize_features_simple(features):
    """Binarize any feature columns with values other than 0 & 1."""
    return [feat if isbinary(feat) else binarize(feat) for feat in zip(*features)]


def range_0_1(feature):
    """Check if feature values fall within 0...1 range."""
    return len([1 for x in feature if x < 0 or x > 1]) == 0


def rescale_0_1(feature, midwas0=True):
    """Rescale feature to the 0...1 range."""
    if midwas0: # rescale to have mean at 0.5
        maxfrom0 = float(max(abs(min(feature)), max(feature)))
        return [x / (2 * maxfrom0) + 0.5 for x in feature]
    else:  # simple rescaling
        minval = min(feature)
        maxval = max(feature)
        range = float(maxval - minval)
        return [(x-minval)/range for x in feature]


def best_thresX(targetlabels, predictlabels):
    """Find best threshold for binarizing to 0 and 1 (criterion: Krippendorff)

    Brute force search in 0.01 increments
    Deprecated in favour of using scipy.optimize.minimize_scalar
    (see best_thres)
    """
    best, bestK = 0, -1
    for x in xrange(1, 100):
        thres = x/100.0
        predictlabelsX = [1 if lab >= thres else 0 for lab in predictlabels]
        K = krippendorff(targetlabels, predictlabelsX)
        if K > bestK:
            best = thres
            bestK = K
            # bestlabels = predictlabelsX
    # print best, bestK
    return best


def performance_fn(x, targetlabels, predictlabels):
    """Calculate performance of threshold x."""
    return -krippendorff(targetlabels, [1 if lab >= x else 0 for lab in predictlabels])


def best_thres(targetlabels, predictlabels):
    """Find best threshold for optimizing Krippendorff alpha."""
    from scipy import optimize

    bestthres = optimize.minimize_scalar(performance_fn, bounds=(0,1),
                                         args=(targetlabels, predictlabels),
                                         method='bounded' )
    # When bound seems to make no difference, returns 1-e; pick 0.5 instead
    return bestthres['x'] if bestthres['x'] < 0.9999 else 0.5


def getthres(obs, positive_fraction):
    """Calculate binarization threshold to achieve positive_fraction."""
    if positive_fraction == 0:
        return max(obs) + 1
    else:
        nr0 = int(len(obs) * (1 - positive_fraction))
        return sorted(obs)[nr0]


# **************************** evaluation functions **************************

def performance(targetLabels, predictLabels, thres=0):
    """Calculate various performance metrics."""
    from itertools import chain
    from sklearn.metrics import confusion_matrix, roc_auc_score

    if targetLabels == [] or predictLabels == []:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # Check that we have the right label sets
    # print "Targets", list(set(targetLabels))
    # print "Predicted", list(set(predictLabels))

    # Area under curve is the only item to calculate pre-binarization
    if sum(targetLabels) == len(targetLabels) or sum(targetLabels) == 0:
        auc = -1
    else:
        auc = roc_auc_score(targetLabels, predictLabels)

    if not isbinary(predictLabels):
        predictLabels = [1 if lab >= thres else 0 for lab in predictLabels]
    kripscore = krippendorff(targetLabels, predictLabels)
    confMatrix = list(chain.from_iterable(confusion_matrix(targetLabels,
                                                           predictLabels)))
    if len(confMatrix) == 9: # error - 3 vals. because prediction is -1s
        return [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif len(confMatrix) == 1: # all 0s or 1s in both valid and prediction
        if targetLabels[0] == 1: # all 1s
            return [kripscore, 1, auc, 1, 1, 1, 0, 0, 0, len(targetLabels)]
        else:
            return [kripscore, 1, auc, 1, 1, 1, len(targetLabels), 0, 0, 0]

    truepos = confMatrix[3]
    falseneg = confMatrix[2]
    falsepos = confMatrix[1]
    trueneg = confMatrix[0]
    precision = 0 if truepos + falsepos == 0 else \
        truepos / float(truepos + falsepos)
    recall = 0 if truepos + falseneg == 0 else \
        truepos / float(truepos + falseneg)
    F1score = 0 if precision + recall == 0 else \
        2 * precision * recall / float(precision + recall)
    accuracy = (truepos + trueneg) / float(len(targetLabels))
    return [kripscore, accuracy, auc, F1score, precision, recall] + confMatrix


def krippendorff(targetlabels, predictlabels):
    """Return Krippendorff's alpha for target-prediction agreement."""
    if predictlabels[0] == -1: # intest labels assigned
        return -9
    numlabels = 2 * len(targetlabels)
    numlabels1 = sum(targetlabels) + sum(predictlabels)
    if numlabels1 == 0 or numlabels1 == numlabels:
        return 0
    else:
        return 1 - float(numlabels - 1) * \
                   sum([1 for label,prediction in \
                        zip(targetlabels, predictlabels) \
                        if label != prediction]) / \
                    float(numlabels1 * (numlabels - numlabels1))


def krippendorff2(conf_matrix):
    """Calculate Krippendorff's alpha from confusion matrix.

    See Krippendorff's 2011 "Computing Krippendorff's Alpha-Reliability."
    """
    tp, fn, fp, tn = conf_matrix
    n = 2 * (tp + fp + fn + tn)
    o01 = fp + fn
    n0 = 2 * tp + o01
    n1 = 2 * tn + o01
    return 1 - (n - 1) * o01 / float(n0 * n1)


def cohen_kappa(conf_matrix):
    """Calculate Cohen's kappa from confusion matrix."""
    tp, fn, fp, tn = conf_matrix
    # po = relative observed agreement among raters
    n = tp + fp + fn + tn
    po = (tp + tn) / float(n)
    # pe = hypothetical probability of chance agreement, using observed data
    p1 = (tp + fp) / float(n)
    p2 = (tp + fn) / float(n)
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    return 1 - (1 - po) / (1 - pe)


# ********************************* deprecated *******************************

def predictionQuality(targetlabels, predictlabels):
    """Return binary classification results plus Krippendorf's alpha.

    Returns true positive, false positive, false negative, true negative.
    Note: won't work if no positive labels in valid set (div. by 0).
    Note2: not currently used (use performance() instead)
    """
    if predictlabels == []:
        return (0, 0, 0, 0, 0)
    else:
        targetandpredict = zip(targetlabels, predictlabels)
        numlabels = 2 * len(targetlabels)
        numlabels1 = sum(targetlabels) + sum(predictlabels)
        return (sum((1 for target, prediction in targetandpredict \
                     if target == 1 and prediction == 1)),
                sum((1 for target, prediction in targetandpredict \
                     if target == 0 and prediction == 1)),
                sum((1 for target, prediction in targetandpredict \
                     if target == 1 and prediction == 0)),
                sum((1 for target, prediction in targetandpredict \
                     if target == 0 and prediction == 0)),
                1 - float(numlabels - 1) * \
                    sum([1 for target, prediction in targetandpredict \
                         if target != prediction]) /
                    float(numlabels1 * (numlabels - numlabels1)))


# ************************** external invocation (i/o) ***********************

if __name__ == "__main__":
    # See how well prediction predicts target
    import sys

    # See if we were called with the right number of arguments
    if len(sys.argv) == 2:
        feature_setup(sys.argv[1])
        # scale_features(sys.argv[1])
        # binarize_features(sys.argv[1])
        # assess_features(sys.argv[1] + 'SX')
    elif len(sys.argv) == 3:
        # read ids, labels from input files & sort by id
        ids, labels, _ = read1feature(sys.argv[1])
        targets = sorted(zip(ids, labels))
        ids, labels, _ = read1feature(sys.argv[2])
        predicts = sorted(zip(ids, labels))
        # verify match before calling performance function
        if len([1 for x,y in zip(targets, predicts) if x[0] <> y[0]]) <> 0:
            print "IDs don't match (target - prediction)"
        else:
            print "Performance (kripp, accuracy, auc, F1, " + \
                  "precision, recall, conf. matrix):"
            print performance([x[1] for x in targets], [y[1] for y in predicts])


