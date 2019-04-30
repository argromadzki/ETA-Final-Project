# encoding: utf-8

# scaling.py
# by Maurits van der Veen
# last modified 2015-11-16

# Functions for ideology scaling project:

__author__ = 'maurits'


# *************************** notes, thoughts, etc. **************************

# 1. Need to find a more elegant way to deal with uncoded material.

# Specifically, right now it has to be appended at the bottom of the labels &
# text files. This means that once we start adding uncoded material, we would
# need to start from scratch if we wanted to add coded material.

# Better to have two separate files: coded and uncoded.
# These could be combined together only at the vectorization stage.
# This should be a fairly straightforward extension

# This requires changing combine_sources and add_extra
# - different numbering scheme: start uncoded at 1000000
# - different files: Coded, CodedX, Uncoded, UncodedX
# - flag for coded or uncoded


# ****************************************************************************

def add_extra(datafolder, suffix):
    """Add extra files to the combined file."""
    import csv
    import sys

    csv.field_size_limit(1000000000)
    combinedfile =  datafolder + 'Combined/Combined' + suffix
    extrafile = datafolder + 'Combined/Extra' + suffix

    with open(combinedfile, 'ab') as outfile, open(extrafile, 'rb') as infile:
        outwriter = csv.writer(outfile)
        for row in csv.reader(infile):
            outwriter.writerow(row)


def add_extraT(datafolder, suffix):
    """Add extra files to the combined file."""
    combinedfile =  datafolder + 'Combined/Combined' + suffix
    extrafile = datafolder + 'Combined/Extra' + suffix

    with open(combinedfile, 'at') as outfile, open(extrafile, 'rt') as infile:
        extraids = infile.read()
        outfile.write(extraids)


def combine_sources(datafolder, sourcedict, offset=1000000, replace=True, combine='Combined'):
    """Combine several sources of ideological texts.

    Each entry in sourcedict should be of form:
    <name> : (label, train_fraction, test_fraction)
    """
    import csv
    from os.path import isfile
    import os
    import random
    import sys

    csv.field_size_limit(1000000000)
    random.seed(42)
    stem = datafolder + combine + '/' + combine
    if replace:
        if isfile(stem + '.csv'):
            os.remove(stem + '.csv')
        if isfile(stem + '_labels.csv'):
            os.remove(stem + '_labels.csv')
        if isfile(stem + '_testIDs.txt'):
            os.remove(stem + '_testIDs.txt')

    # now do the actual combining
    with open(stem + '.csv', 'ab' \
              if isfile(stem + '.csv') else 'wb') as outfile, \
            open(stem + '_labels.csv', 'ab' \
                 if isfile(stem + '_labels.csv') else 'wb') as labelfile, \
            open(stem + '_testIDs.txt', 'at' \
                 if isfile(stem + '_testIDs.txt') else 'wt') as idfile:

        outwriter = csv.writer(outfile)
        labelwriter = csv.writer(labelfile)

        obstot, traintot, testtot, codetot = 0, 0, 0, 0
        sourcenr = 0

        for sourcename, specs in sourcedict.iteritems():
            path = datafolder + sourcename + '/' + sourcename
            label = specs[0]
            train_fraction = specs[1]
            use_fraction = train_fraction + specs[2]

            obs = 0
            testids = 0
            omitids = 0

            with open(path + '_ready.csv', 'rb') as infile:
                for row in csv.reader(infile):
                    randnr = random.random()
                    oldid = int(row[0])
                    newid = sourcenr + oldid
                    if use_fraction == 0 and label == -1:  # to be coded
                        outwriter.writerow((newid, row[1]))
                        labelwriter.writerow((newid, label, sourcename, oldid))
                    elif randnr > use_fraction:  # don't use this obs
                        omitids += 1
                    else:
                        outwriter.writerow((newid, row[1]))
                        labelwriter.writerow((newid, label, sourcename, oldid))
                        # See if to be used as test
                        if randnr >= train_fraction:
                            idfile.write(str(newid) + '\n')
                            testids += 1
                    obs += 1
            if label == -1:
                print "%s texts: %d total, to be coded" % (sourcename, obs)
                codetot += obs
            else:
                print "%s texts: %d total, %d training, %d test" % \
                      (sourcename, obs, obs - omitids - testids, testids)
                traintot += (obs - omitids - testids)
                testtot += testids
            obstot += obs
            sourcenr += offset
        print "Overall: %d texts, %d training, %d test, %d to be coded." % \
              (obstot, traintot, testtot, codetot)
    return obstot


def combine_sources_LR(datafolder, sourcedicts, offset=1000000, replace=True,
                       ext = '_stemmed.csv', combine='Combined'):
    """Combine several sources of ideological texts.

    Sourcedicts should have 3 entries, for 0 and 1 classes, and to-be-classified.
    Each of these should contain the class (0, 1, or -1), the subfolder containing that class,
        and the sourcedict associated with the class
    Each entry in sourcedict should be of form:
    <name> : (train_fraction, test_fraction)
    Note: for the to-be-classified data, these values are ignored
    """
    import csv
    from os.path import isfile
    import os
    import random
    import sys

    csv.field_size_limit(1000000000)
    random.seed(42)
    stem = datafolder + combine + '/' + combine
    if replace:
        if isfile(stem + '.csv'):
            os.remove(stem + '.csv')
        if isfile(stem + '_labels.csv'):
            os.remove(stem + '_labels.csv')
        if isfile(stem + '_testIDs.txt'):
            os.remove(stem + '_testIDs.txt')

    # now do the actual combining
    with open(stem + '.csv', 'ab' \
              if isfile(stem + '.csv') else 'wb') as outfile, \
            open(stem + '_labels.csv', 'ab' \
                 if isfile(stem + '_labels.csv') else 'wb') as labelfile, \
            open(stem + '_testIDs.txt', 'at' \
                 if isfile(stem + '_testIDs.txt') else 'wt') as idfile:

        outwriter = csv.writer(outfile)
        labelwriter = csv.writer(labelfile)

        obstot, traintot, testtot, codetot = 0, 0, 0, 0
        sourcenr = 0

        for sourcedictinfo in sourcedicts:
            label = sourcedictinfo[0]
            subfolder = datafolder + sourcedictinfo[1] + '/'

            for sourcename, specs in sourcedictinfo[2].iteritems():
                path = subfolder + sourcename + '/' + sourcename
                train_fraction = specs[0]
                use_fraction = train_fraction + specs[1]

                obs = 0
                testids = 0
                omitids = 0

                with open(path + ext, 'rb') as infile:
                    for row in csv.reader(infile):
                        randnr = random.random()
                        oldid = int(row[0])
                        newid = sourcenr + oldid
                        if use_fraction == 0 and label == -1:  # to be coded
                            outwriter.writerow((newid, row[1]))
                            labelwriter.writerow((newid, label, sourcename, oldid))
                        elif randnr > use_fraction:  # don't use this obs
                            omitids += 1
                        else:
                            outwriter.writerow((newid, row[1]))
                            labelwriter.writerow((newid, label, sourcename, oldid))
                            # See if to be used as test
                            if randnr >= train_fraction:
                                idfile.write(str(newid) + '\n')
                                testids += 1
                        obs += 1
                if label == -1:
                    print "%s texts: %d total, to be coded" % (sourcename, obs)
                    codetot += obs
                else:
                    print "%s texts: %d total, %d training, %d test" % \
                          (sourcename, obs, obs - omitids - testids, testids)
                    traintot += (obs - omitids - testids)
                    testtot += testids
                obstot += obs
                sourcenr += offset

        print "Overall: %d texts, %d training, %d test, %d to be coded." % \
              (obstot, traintot, testtot, codetot)
    return obstot


def classify_texts(scoredfile, score=1, header=False):
    """Split a scored file into parts based on source & assign class.

    Most efficient if all texts from a particular source are contiguous.
    """
    import csv
    scored_folder = '/'.join(scoredfile.split('/')[:-1]) + '/'
    current_source = ''
    outfile = 'dummy.csv'  # should not be used, but define just in case
    with open(scoredfile, 'rU') as infile:
        for row in csv.reader(infile):
            if header:
                header = False
            else:
                if row[2] != current_source:
                    outfile = scored_folder + row[2] + '_classified.csv'
                    current_source = row[1]
                with open(outfile, 'ab') as outf:
                    csv.writer(outf).writerow([row[0], row[1], round(float(row[score + 2]))])


def add_texts(corpusfile, resultfile, outputfile):
    """Add class prediction info to the corpusfile.

    corpusfile contains: newid, text
    resultfile contains: newid, sourcecorpus, scores, avg. score
    """
    import csv

    # Get a dictionary with scores for all the new items
    with open(resultfile, 'rU') as classfile:
        classifications = {row[0]: row[1:] for row in csv.reader(classfile)}
    # Add scores; filter out test items
    with open(corpusfile, 'rU') as articlefile, open(outputfile, 'wb') as outfile:
        outwriter = csv.writer(outfile)
        for row in csv.reader(articlefile):
            if row[0] in classifications:
                outwriter.writerow(row + classifications[row[0]])


def write_new(filename, ids, specs, scoreslist, sources):
    """Write prediction results to file."""
    import csv
    nrspecs = len(specs)
    if nrspecs == 0:
        print "no classifiers run -> no output to write"
        return
    with open(filename, 'wb') as outfile:
        outwriter = csv.writer(outfile)
        outwriter.writerow(['id', 'source'] + specs + ['avg',])
        for id, scores, source in zip(ids, zip(*scoreslist), sources):
            outwriter.writerow([id, source] + list(scores) + [sum(scores)/float(nrspecs),])


def combine_sourcesX(datafolder, sourcedict, coded=True, extra=False, replace=True):
    """Combine several sources of ideological texts.

    Each entry in sourcedict should be of form:
    <name> : (label, train_fraction, test_fraction)
    """
    import csv
    from os.path import isfile
    import random
    import sys

    csv.field_size_limit(1000000000)
    random.seed(394)
    folder = datafolder + 'Combined/'

    # if extra, need to set correct starting id,
    # so run down the existing labels file
    if extra:
        filestem = folder + 'Extra'
        # Find next id to use
        with open(folder + 'Combined' + '_labels.csv', 'rb') as xfile:
            for row in csv.reader(xfile):
                id = int(row[0])
        id += 1
    else:  # If not extra, replace any existing Combined.csv
        id = 0  # Start ids at 0
        filestem = folder + 'Combined'
        # if os.path.isfile(filestem + '.csv'):
        #     os.remove(drawfile)

    # now do the actual combining
    with open(filestem + '.csv', 'ab' \
                 if isfile(filestem + '.csv') else 'wb') as outfile, \
            open(filestem + '_labels.csv', 'ab' \
                 if isfile(filestem + '_labels.csv') else 'wb') as labelfile, \
            open(filestem + '_testIDs.txt', 'at' \
                 if isfile(filestem + '_testIDs.txt') else 'wt') as idfile:

        outwriter = csv.writer(outfile)
        labelwriter = csv.writer(labelfile)

        for sourcename, specs in sourcedict.iteritems():
            path = datafolder + sourcename + '/' + sourcename
            label = specs[0]
            train_fraction = specs[1]
            use_fraction = train_fraction + specs[2]

            startid = id
            testids = 0
            omitids = 0

            with open(path + 'X.csv', 'rb') as infile:
                for counter, row in enumerate(csv.reader(infile)):
                    randnr = random.random()
                    oldid = int(row[0])
                    outwriter.writerow((id, row[1]))
                    # Store source & old ID info in labelfile
                    if randnr > use_fraction:
                        label = -1
                        omitids += 1
                    labelwriter.writerow((id, label, sourcename, oldid))
                    # See if to be used as test
                    if randnr >= train_fraction and randnr < use_fraction:
                        idfile.write(str(id) + '\n')
                        testids += 1
                    id += 1
            obs = id - startid
            print "%s texts: %d total, %d training, %d test" % \
                  (sourcename, obs, obs - omitids - testids, testids)
    return id


def source_to_val(sources, sourcelist):
    """Convert source name to a value.

    Hard-code rather than sending to sklearn's label encoder
    (for example), so we can control the ordering through sourcelist.

    Not the most efficient implementation, but clear and straightforward.
    """
    return [sourcelist.index(x) for x in sources]


def plot_sources(name, scores, sources, sourcelist):
    """Plot left-right scores of several sources of ideological texts.

    Each separate source gets its own line.
    Also save the data plotted.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # set data up in dataframe
    val = source_to_val(sources, sourcelist)
    df = pd.DataFrame(dict(x=scores, y=val, label=sources))
    df.to_csv(name)  # save data
    groups = df.groupby('label')

    # set up plotting parameters
    # plt.rcParams.update(pd.tools.plotting.mpl_stylesheet)
    # colors = pd.tools.plotting._get_standard_colors(len(groups), color_type='random')

    # generate plot
    fig, ax = plt.subplots()
    # ax.set_color_cycle([(0, 0, 1), (1, 0, 0), (0, 1, 0), (1, 1, 0),
    #                     (1, 0, 1), (1, 0.5, 0.5), (0.5, 0.5, 0.5),
    #                     (0.5, 0, 0), (1, 0.5, 0), (0, 1, 1),
    #                     (0.5, 0.5, 1) ])

    ax.set_color_cycle([(0, 0, 1), (1, 0, 0), (0, 1, 0),
                        (1, 1, 0), (1, 0, 1), (0, 1, 1),
                        (1, 0.5, 0.5), (0.5, 0.5, 1), (0.5, 1, 0.5),
                        (1, 1, 0.5), (0.5, 1, 1), (1, 0.5, 1),
                        (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5),
                        (0.5, 0.5, 0), (0, 0.5, 0.5), (0.5, 0, 0.5),
                        (1, 0.5, 0), (1, 0, 0.5), (0.5, 1, 0), (0.5, 0, 1),
                        (0, 0.5, 1), (0, 1, 0.5), (0.5, 0.5, 0.5)])

    ax.margins(0.05)
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=name)
    ax.legend(numpoints=1, loc='best', bbox_to_anchor=(1.05, 1),
              fancybox=True, shadow=True)
    ax.axes.get_yaxis().set_ticks([])
    # display
    plt.show()


def graph_sources(namestem, scores, sources, sourcelist):
    """Graph scores of several sources as stacked density plots.

    Each separate source gets its own plot.
    Also save the data plotted.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.stats import gaussian_kde

    # set data up in dataframe
    df = pd.DataFrame(dict(x=scores, label=sources))
    # save data in csv file
    df.to_csv(namestem + '.csv')


    groups = df.groupby('label')

    # set up plotting parameters
    # plt.rcParams.update(pd.tools.plotting.mpl_stylesheet)
    # colors = pd.tools.plotting._get_standard_colors(len(groups), color_type='random')

    # generate plot
    plt.figure(1)  # optional statement, but cleaner to have it here
    nrplots = len(groups)
    for counter, (name, group) in enumerate(groups):
        plt.subplot(nrplots, 1, counter + 1)
        plt.hist(group.x.values, 100, range=(0, 1))
        plt.title('Scores for ' + name)
        plt.tick_params(bottom='off', left='off', labelbottom='off', labelleft='off')
    plt.tick_params(bottom='on', labelbottom='on')
    plt.savefig(namestem + '.png', bbox_inches='tight')
    # display
    plt.show()


def plot1d(scores):
    """Plot scores in 1D; deprecated"""
    import numpy as np
    import matplotlib.pyplot as pp

    _, __, ___, labels = \
    splitbyID(ideologystem + '_labels.csv',
              ideologystem + '_testIDs.txt')
    labelvals = list(set(labels))

    val = 0.  # y-axis location of plot
    for score in scores:

        for l in labelvals:
            lscore = [x for x, y in zip(scores, labels) if y == l]
            ar = np.array(lscore)
            pp.plot(ar, np.zeros_like(ar) + val, '|')
        pp.show()
