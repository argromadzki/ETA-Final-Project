# encoding: utf-8

# sentiment.py
# by Maurits van der Veen
# last modified 2017-03-13

# Functions to handle sentiment analysis process

# ToDo: remove LIWC-style dictionaries still in there (though ignored)
# ToDo: reorganize code file more logically
# ToDo: remove deprecated matching functions

# ************************** Text selection functions **********************

def idarticles(corpus, searchspecs):
    """Return ids of all articles that match the searchspecs."""
    import csv

    idlist = []
    with open(corpus, 'rU') as articlefile:
        for counter, row in enumerate(csv.reader(articlefile)):
            if matchspecs(row[1], searchspecs):
                idlist.append(row[0])
            if counter % 10000 == 0:
                print 'Processing article %d' % counter
    return idlist


def extractarticles(corpus, resultcorpus, searchspecs):
    """Extract all articles that match the searchspecs."""
    import csv

    nrfound = 0
    with open(corpus, 'rU') as articlefile, open(resultcorpus, 'wb') as outfile:
        outwriter = csv.writer(outfile)
        for counter, row in enumerate(csv.reader(articlefile)):
            if matchspecs(row, searchspecs):
                nrfound += 1
                outwriter.writerow(row)
            if counter % 10000 == 0:
                print 'Processing article %d' % counter
    print 'Extracted %d articles' % nrfound
    return nrfound


def extractkeywords(inputfile, outputfile, searchspecs):
    """Extract all keywords that match the searchspecs."""
    import csv

    with open(inputfile, 'rU') as articlefile, open(outputfile, 'wb') as outfile:
        outwriter = csv.writer(outfile)
        for counter, row in enumerate(csv.reader(articlefile)):
            outwriter.writerow([row[0], caughtspecs(row[1], searchspecs)])
            if counter % 10000 == 0:
                print 'Processing article %d' % counter
    return


def extractitems_id(corpus, resultcorpus, idlist):
    """Extract all items with the specified ids."""
    import csv

    nrfound = 0
    idset = set(idlist)
    with open(corpus, 'rU') as articlefile, open(resultcorpus, 'wb') as outfile:
        outwriter = csv.writer(outfile)
        for counter, row in enumerate(csv.reader(articlefile)):
            if row[0] in idset:
                nrfound += 1
                outwriter.writerow(row)
            if counter % 10000 == 0:
                print 'Processing article %d' % counter
    return nrfound


def caughtspecs(text, searchspecs):
    """Find words in the text that match the search specs."""
    specstring = ''
    for spec in searchspecs:
        specstring += matchspec(text if spec[2] else text.lower(), spec[0], thres=-2) + ' '
    return specstring


def countspecs(text, searchspecs):
    """Count how often text in row matches the search specs."""
    count = 0
    for spec in searchspecs:
        count += matchspec(text if spec[2] else text.lower(), spec[0], thres=-1)
    return count


def matchspecs(row, searchspecs):
    """See if text in row matches any of the search specs."""
    for spec in searchspecs:
        text = row[1] if len(spec) < 4 else row[spec[3]]
        if matchspec(text if spec[2] else text.lower(), spec[0], spec[1]):
            return True
    return False


def matchspec(text, termlist, thres):
    """See if number of matches for any of the terms in the termlist exceeds threshold.

    Handle multi-word phrases & wildcards.
    If thres >= 1, look for that number of matches
    If thres < 1, look for frequency ratio (occurrences / wordcount)
    If thres == -1, return count of matches
    If thres == -2, return string of matched words only

    Assumes that words are separated by spaces (will not recognize a match
    if immediately followed by punctuation) or hyphens (to capture 'anti-Catholic' &c.)

    *** Needs to be deprecated in favour of using wordsinwindow ***
    """
    matches = 0
    matchwords = []
    wordbreaks = [i for i, x in enumerate(text) if x == ' ' or x == '-'] + [len(text), ]
    spaces = [i for i, x in enumerate(text) if x == ' '] + [len(text), ]
    if 0 not in wordbreaks:
        wordbreaks = [-1, ] + wordbreaks
    for i, ind in enumerate(wordbreaks):
        if ind == len(text):
            break
        match = 0
        for term in termlist:
            if term[-1] == '*':  # wildcard
                match = 1 if text[ind + 1:ind + len(term)] == term[:-1] else 0
            else:  # match full word
                match = 1 if (text[ind + 1:ind + len(term) + 1] == term and \
                              ind + len(term) + 1 in wordbreaks) else 0
            if match == 1:
                if thres == -2:  # find what matched
                    startindex = ind + 1 if text[ind] == ' ' else wordbreaks[i - 1] + 1
                    endindex = ind + len(term) + 1 if term[-1] != '*' else \
                        nextspace(ind + len(term) - 1, spaces)
                    matchwords.append(text[startindex:endindex])
                break  # don't count same text twice!
        matches += match
        if matches == thres:
            return True
    if thres == -2:
        return ' '.join(matchwords)
    if thres == -1:
        return matches
    elif thres < 1 and (matches / float(len(text.split()))) > thres:
        return True
    else:
        return False


def nextspace(ind, spacelist):
    """Given list of indices into a string that represent spaces, find next after ind."""
    for space in spacelist:
        if space > ind:
            return space
    return 9999999


# ***************** Basic processing & cleaning of raw text ****************

def punctuationPrep(filestem, fileend, header=True, lang='english',
                       stripspecial=True, stripcomma=True,
                       textcols=(10, 12), keepcols=(0,), append=False):
    """Take csv file from LexisNexis output & create clean id,text format.

    Merge text fields; punctuation preprocess, and strip header.
    """
    import csv
    from unidecode import unidecode

    csv.field_size_limit(1000000000)
    outfile = filestem + '_clean.csv'
    with open(filestem + fileend, 'rU') as inf, \
            open(outfile, 'ab' if append else 'wb') as outf:
        output = csv.writer(outf)
        for counter, article in enumerate(csv.reader(inf)):
            if header:
                header = False
            else:
                article_text = ' . '.join([article[col] for col in textcols])
                if stripspecial:
                    article_text = unidecode(article_text)
                        # previously included the following: .decode('ascii', 'ignore')
                preppedtext = punctuationPreprocess(article_text, lang)
                if stripcomma:
                    preppedtext = stripcommas(preppedtext)
                # Write text plus identifying info (in keepcols)
                output.writerow([article[col] for col in keepcols] + [preppedtext,])
            if counter % 50000 == 0:
                print "Processing article %d" % counter


# ***************** Deriving features from text / article data ****************

def simplegeo_section(inputname, outputname, citynames,
                      header=True, sectioncol=12):
    """Add domestic/foreign flags based on section information."""
    import csv

    # Section specs for determining domestic vs. local
    localsections = ['city', 'local', 'region', 'metro']
    localwords = localsections + citynames
    foreignwords = ['foreign', 'world', 'global', 'international']
    domcounter_sec = 0
    forcounter_sec = 0

    # Do the actual geocoding
    with open(inputname, 'rU') as inf, open(outputname, 'wb') as outf:
        output = csv.writer(outf)
        for counter, article in enumerate(csv.reader(inf)):
            if header:
                article += ['domestic_section', 'foreign_section']
                header = False
            else:
                section = article[sectioncol].lower()
                domestic_sec = any([x in section for x in localwords])
                if domestic_sec:
                    domcounter_sec += 1
                foreign_sec = any([x in section for x in foreignwords])
                if foreign_sec:
                    forcounter_sec += 1

                article += [1 if domestic_sec else 0,
                            1 if foreign_sec else 0]
            output.writerow(article)
            if counter % 50000 == 0:
                print "Processing article", counter

    # Give summary data
    print "\nOut of %d articles, marked %d as domestic vs. %d as foreign based on section info" % \
          (counter, domcounter_sec, forcounter_sec)


def simplegeo_text_faster(inputname, domesticnames, worldnames, textcol=1,
                          capsmatter=True, header=False):
    """Simple geocoding based on text contents. Best run on cleaned text.

    Separate out single- from multi-word names to speed things up.
    """
    import csv

    # Set-up
    domcounter_txt, forcounter_txt = 0, 0
    # Prefer to do a search for single words; otherwise use re
    domnames = {key for key in domesticnames if ' ' not in key}
    fornames = {key for key in worldnames if ' ' not in key}
    domkeys = [key for key in domesticnames if ' ' in key]
    forkeys = [key for key in worldnames if ' ' in key]
    print "Searching for %d single-word domestic geolocations and %d multi-word ones" % \
          (len(domnames), len(domkeys))
    print "Searching for %d single-word foreign geolocations and %d multi-word ones" % \
          (len(fornames), len(forkeys))

    # Loop through file
    geocodes = []
    with open(inputname, 'rU') as inf:
        infile = csv.reader(inf)
        if header:
            headerline = next(infile)
            geocodes.append([headerline[0], 'domestic_text', 'foreign_text'])
        for counter, article in enumerate(infile):
            text = article[textcol] if capsmatter else article[textcol].lower()
            domestic_txt = 0

            if any([x in domnames for x in text.split()]) or \
                    any([x in text for x in domkeys]):
                domestic_txt = 1
                domcounter_txt += 1
            foreign_txt = 0
            if any([x in fornames for x in text.split()]) or \
                    any([x in text for x in forkeys]):
                foreign_txt = 1
                forcounter_txt += 1

            geocodes.append([article[0], domestic_txt, foreign_txt])
            if counter % 50000 == 0:
                print "Processing article", counter

    # Give summary data
    print "\n%d articles total" % (counter + (0 if header else 1))
    print "Domestic articles, by text: %d" % domcounter_txt
    print "Foreign articles, by text: %d" % forcounter_txt
    return geocodes


def simplegeo_text(inputname, domesticnames, worldnames, textcol=1,
                   capsmatter=True, header=False):
    """Simple geocoding based on text contents. Best run on cleaned text.

    Separate out single- from multi-word names to speed things up.
    """
    import csv
    import re

    # Set-up
    domcounter_txt, forcounter_txt = 0, 0
    # Prefer to do a search for single words; otherwise use re
    domnames = {key for key in domesticnames if ' ' not in key}
    fornames = {key for key in worldnames if ' ' not in key}
    domkeys = ['\\b' + key + '\\b' for key in domesticnames if ' ' in key]
    forkeys = ['\\b' + key + '\\b' for key in worldnames if ' ' in key]
    print "Searching for %d single-word domestic geolocations and %d multi-word ones" % \
          (len(domnames), len(domkeys))
    print "Searching for %d single-word foreign geolocations and %d multi-word ones" % \
          (len(fornames), len(forkeys))

    # Loop through file
    geocodes = []
    with open(inputname, 'rU') as inf:
        infile = csv.reader(inf)
        if header:
            headerline = next(infile)
            geocodes.append([headerline[0], 'domestic_text', 'foreign_text'])
        for counter, article in enumerate(infile):
            text = article[textcol] if capsmatter else article[textcol].lower()
            domestic_txt = 0

            if any([x in domnames for x in text.split()]) or \
                    any([re.search(pattern, text) != None for pattern in domkeys]):
                domestic_txt = 1
                domcounter_txt += 1
            foreign_txt = 0
            if any([x in fornames for x in text.split()]) or \
                    any([re.search(pattern, text) != None for pattern in forkeys]):
                foreign_txt = 1
                forcounter_txt += 1

            geocodes.append([article[0], domestic_txt, foreign_txt])
            if counter % 50000 == 0:
                print "Processing article", counter

    # Give summary data
    print "\n%d articles total" % (counter + (0 if header else 1))
    print "Domestic articles, by text: %d" % domcounter_txt
    print "Foreign articles, by text: %d" % forcounter_txt
    return geocodes


def simplegeo_assess(inputname, outputname, domesticnames, worldnames,
                     textcol=1, capsmatter=True, header=False):
    """Simple geocoding based on text contents. Best run on cleaned text.

    Use full toponym extraction & compare to simple geocodes caught to get
    a sense of quality of simple geocoding.
    """
    import csv
    import re
    import geocoder
    import corpus

    # Loop through file
    resultlist = []
    foundalllocs, foundsomelocs, foundboth, foundpossiblelocs = 0, 0, 0, 0
    with open(inputname, 'rU') as inf:
        infile = csv.reader(inf)
        if header:
            headerline = next(infile)
        for counter, article in enumerate(infile):
            text = article[textcol] if capsmatter else article[textcol].lower()
            locsfound, _ = geocoder.extractToponyms(text)
            # remove compass directions
            locsfound = [x if x.split()[0] not in ('East', 'West', 'North', 'South') \
                             else ' '.join(x[1:]) for x in locsfound]
            domlocs, otherlocs = corpus.split_on_list(locsfound, domesticnames)
            forlocs, otherlocs = corpus.split_on_list(otherlocs, worldnames)
            if len(domlocs) + len(forlocs) > 0:
                if len(otherlocs) == 0:
                    foundalllocs += 1
                else:
                    if len(domlocs) > 0 and len(forlocs) > 0:
                        foundboth += 1
                    else:
                        foundsomelocs += 1
            elif len(otherlocs) > 0:
                foundpossiblelocs += 1
            domlocs = list(set(domlocs))
            forlocs = list(set(forlocs))
            otherlocs = list(set(otherlocs))
            resultlist.append((article[0], ', '.join(domlocs),
                               ', '.join(forlocs),
                               ', '.join(otherlocs) if (len(domlocs) == 0 or len(forlocs) == 0) else ''))
            if counter % 100 == 0:
                print "Processing article", counter

    # Give summary data
    nrarticles = counter + (0 if header else 1)
    nrartfloat = float(nrarticles)
    print "\n%d articles; with known only %d (%4.2f%%), both %d (%4.2f%%) also unknown %d (%4.2f%%), unknown possibles only %d (%4.2f%%)" % \
          (nrarticles, foundalllocs, 100*foundalllocs/nrartfloat,
           foundboth, 100*foundboth/nrartfloat,
           foundsomelocs, 100*foundsomelocs/nrartfloat,
           foundpossiblelocs, 100*foundpossiblelocs/nrartfloat)
    # Write results
    with open(outputname, 'wb') as outf:
        csv.writer(outf).writerows(resultlist)
    return


def simplegeo_text_old(inputname, domesticnames, worldnames, textcol=1,
                   capsmatter=True, header=False):
    """Simple geocoding based on text contents. Best run on cleaned text."""
    import csv

    domcounter_txt, forcounter_txt = 0, 0

    # Loop through file
    geocodes = []
    with open(inputname, 'rU') as inf:
        infile = csv.reader(inf)
        if header:
            headerline = next(infile)
            geocodes.append([headerline[0], 'domestic_text', 'foreign_text'])
        for counter, article in enumerate(infile):
            text = article[textcol] if capsmatter else article[textcol].lower()
            domestic_txt = 0
            if any([x in domesticnames for x in text.split()]) or \
                    any([x in text for x in domesticnames if ' ' in x]):
                domestic_txt = 1
                domcounter_txt += 1
            foreign_txt = 0
            if any([x in worldnames for x in text.split()]) or \
                    any([x in text for x in worldnames if ' ' in x]):
                foreign_txt = 1
                forcounter_txt += 1

            geocodes.append([article[0], domestic_txt, foreign_txt])
            if counter % 50000 == 0:
                print "Processing article", counter

    # Give summary data
    print "\n%d articles total" % (counter + (0 if header else 1))
    print "Domestic article, by text: %d" % domcounter_txt
    print "Foreign article, by text: %d" % forcounter_txt
    return geocodes


def textkeywords(corpusstem, inputsuffix, outputsuffix,
                 searchspecs, specnames, textcol=1,
                 capsmatter=False, header=False,
                 savezeroes=False, savewindow=False,
                 includeself=True, windowleft=0, windowright=0):
    """Add flags based on text contents. Best run on cleaned text.

    Searchspecs contains one or more search specifications.
    """
    import csv
    from wordsinwindow import findindices, indicesinwindow

    csv.field_size_limit(1000000000)

    inputname = corpusstem + inputsuffix + '.csv'
    outputname = corpusstem + outputsuffix + '.csv'

    # Initialize variables - arrays with one entry for each searchspec
    nrspecs = len(specnames)
    windows2save = [[] for x in range(nrspecs)]
    zeroes2save = [[] for x in range(nrspecs)]
    mincount = [99,] * nrspecs
    maxcount, totcount, zerocount, nonzerocount = \
        [0,] * nrspecs, [0,] * nrspecs, [0,] * nrspecs, [0,] * nrspecs

    # Loop through file
    with open(inputname, 'rU') as inf, open(outputname, 'wb') as outf:
        infile = csv.reader(inf)
        outfile = csv.writer(outf)

        if header:
            headerline = next(infile)
            origheader = headerline
            for specname in specnames:
                headerline.append(specname)
            outfile.writerow(headerline)

        for articlecount, article in enumerate(infile):
            text = article[textcol]
            wordlist = text.split()

            for speccount, searchspec in enumerate(searchspecs):

                specsfound = findindices(wordlist, searchspec[0],
                                         capsmatter=capsmatter, skipperiods=True)
                keycount = len(specsfound)
                if len(searchspec[1]) > 0:
                    specsfound2 = findindices(wordlist, searchspec[1],
                                              capsmatter=capsmatter, skipperiods=True,)
                    keycount -= len(specsfound2)
                    specsfound = [x for x in specsfound if x not in specsfound2]
                if savewindow:
                    # Get the text within the indices specified.
                    inwindow, outsidewindow, _ = \
                        indicesinwindow(text, searchspec[0], window=(windowleft, windowright),
                                        indices=specsfound, capsmatter=capsmatter,
                                        includekeys=includeself, returntext=True)
                    windows2save[speccount].append([article[0], inwindow])
                if keycount <= 0:  # -ve shouldn't happen, but just in case...
                    zerocount[speccount] += 1
                    if savezeroes:
                        zeroes2save[speccount].append(article)
                else:  # update counts for this spec
                    nonzerocount[speccount] += 1
                    if keycount < mincount[speccount]:
                        mincount[speccount] = keycount
                    if keycount > maxcount[speccount]:
                        maxcount[speccount] = keycount
                    totcount[speccount] += keycount

                article.append(keycount)

            # write out article data updated with counts
            outfile.writerow(article)
            if articlecount % 50000 == 0:
                print "Processing article", articlecount

        # Write out zero-mention results & filtered window text if required
        if savezeroes:
            for speccount, specname in enumerate(specnames):
                zeromentions = corpusstem + '_' + specname + '_zeromentions.csv'
                with open(zeromentions, 'wb') as zerof:
                    zerowriter = csv.writer(zerof)
                    if header:
                        zerowriter.writerow(origheader)
                    zerowriter.writerows(zeroes2save[speccount])

        if savewindow:
            for speccount, specname in enumerate(specnames):
                windowtext = corpusstem + '_' + specname + '_wind.csv'
                with open(windowtext, 'wb') as windowf:
                    windowwriter = csv.writer(windowf)
                    if header:
                        windowwriter.writerow(('id', 'windowtext'))
                    windowwriter.writerows(windows2save[speccount])

    # Give summary data
    print "\n%d articles total" % (articlecount + (0 if header else 1))
    for speccount, specname in enumerate(specnames):
        print "Nr. mentions for %s: 0 - %d, min - %d, max - %d, avg. %5.2f" % \
              (specname, zerocount[speccount], mincount[speccount],
               maxcount[speccount], totcount[speccount] / float(nonzerocount[speccount]))
    return


def calcmentions(inputname, searchspec, textcol=1,
                 capsmatter=False, header=False,
                 savezeroes=False, savewindow=False,
                 includeself=True, windowleft=0, windowright=0):
    """Add flags based on text contents. Best run on cleaned text."""
    import csv
    import os
    from wordsinwindow import findindices, indicesinwindow

    # Set-up
    mincount, maxcount, totcount, zerocount, nonzerocount = 99, 0, 0, 0, 0
    zeromentions = '.'.join(inputname.split('.')[:-1]) + '_zeromentions.csv'
    filteredtext = '.'.join(inputname.split('.')[:-1]) + '_windowtext.csv'

    nrmentions = []
    # Loop through file
    with open(inputname, 'rU') as inf, open(zeromentions, 'wb') as zerof, \
            open(filteredtext, 'wb') as filteredf:
        infile = csv.reader(inf)
        zeroes = csv.writer(zerof)
        filtered = csv.writer(filteredf)

        if header:
            headerline = next(infile)
            zeroes.writerow(headerline)
            filtered.writerow(headerline)
            nrmentions.append([headerline[0], 'nrmentions'])
        for counter, article in enumerate(infile):
            text = article[textcol]
            wordlist = text.split()
            specsfound = findindices(wordlist, searchspec[0],
                                     capsmatter=capsmatter, skipperiods=True)
            keycount = len(specsfound)
            if len(searchspec[1]) > 0:
                specsfound2 = findindices(wordlist, searchspec[1],
                                          capsmatter=capsmatter, skipperiods=True,)
                keycount -= len(specsfound2)
                specsfound = [x for x in specsfound if x not in specsfound2]
            if savewindow:
                # Get the text within the indices specified.
                inwindow, outsidewindow, _ = \
                    indicesinwindow(text, searchspec[0], window=(windowleft, windowright),
                                    indices=specsfound, capsmatter=capsmatter,
                                    includekeys=includeself, returntext=True)
                filtered.writerow([article[0], inwindow])
            if keycount <= 0:  # -ve shouldn't happen, but just in case...
                zerocount += 1
                if savezeroes:
                    zeroes.writerow(article)
            else:  # update counts
                nonzerocount += 1
                if keycount < mincount:
                    mincount = keycount
                if keycount > maxcount:
                    maxcount = keycount
                totcount += keycount

            nrmentions.append([article[0], keycount])
            if counter % 50000 == 0:
                print "Processing article", counter

    # Clean up
    if not savezeroes:
        os.remove(zeromentions)
    if not savewindow:
        os.remove(filteredtext)

    # Give summary data
    print "\n%d articles total" % (counter + (0 if header else 1))
    print "Nr. mentions: 0 - %d, min - %d, max - %d, avg. %5.2f" % \
          (zerocount, mincount, maxcount, totcount / float(nonzerocount))
    return nrmentions


def calcmentions_old(inputname, searchspec, textcol=1, capsmatter=False, header=False):
    """Add flags based on text contents. Best run on cleaned text."""
    import csv

    # Set-up
    mincount, maxcount, totcount, zerocount, nonzerocount = 99, 0, 0, 0, 0
    zeromentions = '.'.join(inputname.split('.')[:-1]) + '_zeromentions.csv'

    nrmentions = []
    # Loop through file
    with open(inputname, 'rU') as inf, open(zeromentions, 'wb') as zerof:
        infile = csv.reader(inf)
        zeroes = csv.writer(zerof)
        if header:
            headerline = next(infile)
            nrmentions.append([headerline[0], 'nrmentions'])
        for counter, article in enumerate(infile):
            text = article[textcol] if capsmatter else article[textcol].lower()
            # count number of acceptable matches
            # (the `1` parameter to countspecs is ignored)
            relcount = countspecs(text, ([searchspec[0], 1, capsmatter],)) - \
                       countspecs(text, ([searchspec[1], 1, capsmatter],))
            if relcount <= 0:  # -ve shouldn't happen, but just in case...
                zerocount += 1
                zeroes.writerow(article)
            else:
                nonzerocount += 1
                if relcount < mincount:
                    mincount = relcount
                if relcount > maxcount:
                    maxcount = relcount
                totcount += relcount

            nrmentions.append([article[0], relcount])
            if counter % 50000 == 0:
                print "Processing article", counter

    # Give summary data
    print "\n%d articles total" % (counter + (0 if header else 1))
    print "Nr. mentions: 0 - %d, min - %d, max - %d, avg. %5.2f" % \
          (zerocount, mincount, maxcount, totcount / float(nonzerocount))
    return nrmentions


def savefeatures(inputname, outputname, featureslists):
    """Append features to the input data, save as outputname.

    Minimal error-checking
    (in particular, assume header status is the same for all,
     and lists have same length as input file)
    """
    import csv

    # Set-up
    newfeatureslist = zip(*featureslists)
    with open(inputname, 'rU') as inf, open(outputname, 'wb') as outf:
        indata = csv.reader(inf)
        output = csv.writer(outf)
        for counter, newfeatures in enumerate(newfeatureslist):
            article = next(indata)
            id = article[0]
            for featureset in newfeatures:
                if featureset[0] == id:
                    article += featureset[1:]
                else:
                    print "Error: ids don't match"
                    print article
                    print newfeatures
                    break
            # Write article data, expanded with new features
            output.writerow(article)
            if counter % 50000 == 0:
                print "Processing article", counter


def reify_keys(keylist):
    """Convert a search specification with a possible wild card to one for the
    built-in regular expression module: put '\b' at the start and end, and add
    a re-format wildcard specification as appropriate.
    """
    import re

    keylist = [re.sub(r'\*', '[\w-]*', key) for key in keylist]
    return ['\\b' + key + '\\b' for key in keylist]


def addfeatures(filename, featurelist, searchtype='full',
                textcol=1, capsmatter=True, header=False):
    """Add flags based on text contents.

    Should be run on cleaned text for best results.

    Note: may want to allow list of avoidstrings for each feature too in the future

    Note 2: we run separate for loops through the file,
    depending on searchtype. This is not the most elegant, code-wise, but
    it is more efficient, as we need to check for searchtype just once.
    """
    import csv
    import re
    from wordsinwindow import findindices

    # Set-up
    ids_features = []
    counts = [0, ] * len(featurelist)

    # Loop through file
    with open(filename, 'rU') as inf:
        infile = csv.reader(inf)
        if header:
            headerline = next(infile)
            ids_features.append([headerline[0],] + [x[0] for x in featurelist])

        if searchtype == 'count':  # count number of occurrences
            keys = [featurespec[1] for featurespec in featurelist]
            for counter, article in enumerate(csv.reader(inf)):
                id_features = [article[0], ]
                text = article[textcol] if capsmatter else article[textcol].lower()
                wordlist = text.split()
                for featurecount, keylist in enumerate(keys):
                    feature = len(findindices(wordlist, keylist,
                                              skipperiods=False, capsmatter=capsmatter))
                    if feature > 0:
                        counts[featurecount] += 1
                    id_features.append(feature)
                ids_features.append(id_features)
                if counter % 50000 == 0:
                    print "Processing article", counter

        elif searchtype == 'firstword':  # Check for presence in first word only
            keys = [featurespec[1] for featurespec in featurelist]
            for counter, article in enumerate(csv.reader(inf)):
                id_features = [article[0], ]
                text = article[textcol] if capsmatter else article[textcol].lower()
                for featurecount, keylist in enumerate(keys):
                    feature = 1 if text.split()[0] in keylist else 0
                    if feature > 0:
                        counts[featurecount] += 1
                    id_features.append(feature)
                ids_features.append(id_features)
                if counter % 50000 == 0:
                    print "Processing article", counter

        elif searchtype == 'firstsentence':  # Check for presence in first sentence only
            rekeys = [reify_keys(featurespec[1]) for featurespec in featurelist]
            for counter, article in enumerate(csv.reader(inf)):
                id_features = [article[0], ]
                text = article[textcol] if capsmatter else article[textcol].lower()
                firstsent = text.split('.')[0]
                for featurecount, keylist in enumerate(rekeys):
                    feature = 1 if any([re.search(pattern, firstsent) != None \
                                        for pattern in keylist]) else 0
                    if feature > 0:
                        counts[featurecount] += 1
                    id_features.append(feature)
                ids_features.append(id_features)
                if counter % 50000 == 0:
                    print "Processing article", counter

        else:  # searchtype == 'full' (default) Check for presence in full text
            rekeys = [reify_keys(featurespec[1]) for featurespec in featurelist]
            for counter, article in enumerate(csv.reader(inf)):
                id_features = [article[0], ]
                text = article[textcol] if capsmatter else article[textcol].lower()
                for featurecount, keylist in enumerate(rekeys):
                    feature = 1 if any([re.search(pattern, text) != None \
                                        for pattern in keylist]) else 0
                    if feature > 0:
                        counts[featurecount] += 1
                    id_features.append(feature)
                ids_features.append(id_features)
                if counter % 50000 == 0:
                    print "Processing article", counter

    # Give summary data & return features
    print "\n%d articles total" % (counter + (0 if header else 1))
    for featurecount, featurespec in enumerate(featurelist):
        print "Feature %s - %d" % (featurespec[0], counts[featurecount])
    return ids_features


def mergefeatures(featureslists):
    """Merge together feature lists of row format (id, feature1, feature2, ...)"""
    mergedlist = featureslists[0]
    for flist in featureslists[1:]:
        mergedlist = [list(x) + y[1:] for x, y in zip(mergedlist, flist) if x[0] == y[0]]
        if len(mergedlist) != len(flist):
            print "Error - id mismatch"
            return
    return mergedlist


def mergeparts(featureslist):
    """Merge together feature info for multiple parts of individual articles.

    Assume features to be combined are "inclusive or". Note that this is incorrect
    for any features specified as being first-word only, since a subsequent part's
    first word is not the first word of the article. The chance of this making a
    difference is vanishingly small, but not zero.
    """
    partcounter = 0
    mergedids = []
    mergeddata = []
    curid = -1
    for row in featureslist:
        if row[0] != curid:  # new id
            mergedids.append(row[0])
            mergeddata.append(row[1:])
            curid = row[0]
        else:  # Combine with previous row
            prevrow = mergeddata.pop()
            newrow = [1 if x == 1 or y == 1 else 0 \
                      for x, y in zip(prevrow, row[1:])]
            mergeddata.append(newrow)
            partcounter += 1
    print "Number of multiple parts: %d" % (partcounter)
    return mergedids, mergeddata


# ****************** Corpus information & manipulation functions **************

def count_sentences(infile, textcol=1):
    """Count # sentences and # words in the corpus."""
    import csv
    nrwords, nrsents = [], []
    counter = -1
    with open(infile, 'rU') as inf:
        for counter, article in enumerate(csv.reader(inf)):
            nrsents.append(len(article[textcol].split('.')))
            nrwords.append(len([x for x in article[textcol].split() if x != '.']))
            if counter % 50000 == 0:
                print "Processing article %d" % counter
    nrarticles = counter + 1
    totalwords = sum(nrwords)
    totalsents = sum(nrsents)
    print "\nArticles: %d, sentences: %d (mean %f), words: %d (mean %f)" % \
          (nrarticles, totalsents, totalsents/float(nrarticles),
           totalwords, totalwords/float(nrarticles))


def sort_bydate(inputname, outputname, header=True):
    """Sort articles by date.

    Assume year is in column 4, month number in column 6,
    and day number in column 7. Within days sort by source (2) & id (0),
    and make sure to keep parts in order (13).
    """
    import csv
    from operator import itemgetter

    with open(inputname, 'rU') as infile:
        allarticles = [a for a in csv.reader(infile)]

    firstobs = 1 if header else 0

    articlestosort = [a for a in allarticles[firstobs:]]
    for counter, a in enumerate(articlestosort):
        try:
            dummy = [int(a[0]), int(a[4]), int(a[6]), int(a[7]),
                     (0 if a[13].lower() == 'false' else 1 if a[13].lower() == 'true' else int(a[13]))]
        except:
            print counter
            print a
            return
    articlestosort = [[int(a[0]),] + a[1:4] + [int(a[4]), a[5], int(a[6]), int(a[7])] + \
                      a[8:13] + [0 if a[13].lower() == 'false' else 1 if a[13].lower() == 'true' else int(a[13]), ] + a[14:] \
                      for a in allarticles[firstobs:]]

    with open(outputname, 'wb') as outfile:
        outwriter = csv.writer(outfile)
        if header:
            outwriter.writerow(allarticles[0])
        outwriter.writerows(sorted(articlestosort,
                                   key=itemgetter(4, 6, 7, 2, 0, 13)))
    return


def deduplicate(inputname, outputname, dupname, header=True, checksource=False):
    """Deduplicate, using Levenshtein similarity on title & text.

    Columns of interest: year - 4, monthnr - 6, day - 7, part - 13
                         title - 10, text - 12
    Note that extra-long articles split over multiple entries
    have the same title.

    Note, also, that we do not look at sourcename, so if 2 papers have
    identical articles (e.g. from a wire service) on the same day,
    one of them is going to be deleted. This is good in terms of not
    overweighting wire articles, but less so in that it may affect
    a newspaper's overall results in unpredictable ways.

    We look at sourcename by pre-pending it to the article title
    """
    import csv
    import Levenshtein

    # Chunks of title/text to compare
    # If title is too short, may be generic
    # Sometimes title is the same but text isn't -> compare titles only secondarily
    title_len = 60 if not checksource else 80  # longer to accommodate source name
    titleonly_len = 60
    text_len = 300
    Lev_thres_text = 0.9
    Lev_thres_title = 0.8

    dupecounter = 0

    curyear = 0
    curmonth = 0
    curday = 0
    ids = []
    titles = []
    texts = []
    csv.field_size_limit(1000000000)
    with open(inputname, 'rU') as inf, \
            open(outputname, 'wb') as outf, open(dupname, 'wb') as dupf:
        output = csv.writer(outf)
        dupes = csv.writer(dupf)
        for counter, article in enumerate(csv.reader(inf)):
            if header:  # write header to both output files
                output.writerow(article)
                dupes.writerow(article)
                header = False
                continue
            thisyear = int(article[4])
            thismonth = int(article[6])
            thisday = int(article[7])
            if curyear == thisyear and curmonth == thismonth \
                    and curday == thisday:
                if article[0] not in ids:  # new article
                    text_compare = min(text_len, len(article[12]))
                    title_compare = min(titleonly_len, len(article[10]))
                    titleprefix = article[2] if checksource else ''
                    title = titleprefix + article[10][:title_compare]
                    if text_compare == 0:
                        # empty article -> compare titles as though they were the text
                        match = any([Levenshtein.ratio(title, prevtitle[:len(title)]) \
                                        > Lev_thres_text for prevtitle in titles])
                    elif "reports earnings for Qtr" in article[10]:
                        # for quarterly earnings reports, demand exact match
                        match = any([title == prevtitle[:len(title)] for prevtitle in titles])
                    else:
                        text = article[12][:text_compare]
                        # Check if texts very similar; if so, make sure titles are also similar
                        match = any([Levenshtein.ratio(text, prevtext[:text_compare]) \
                                        > Lev_thres_text for prevtext in texts]) and \
                                any([Levenshtein.ratio(title, prevtitle[:len(title)]) \
                                        > Lev_thres_title for prevtitle in titles])
                    if match:
                        dupes.writerow(article)
                        dupecounter += 1
                    else:  # no match but same day -> add title & text info to lists
                        output.writerow(article)
                        titles.append(titleprefix + article[10])
                        texts.append(article[12])
                        ids.append(article[0])
                else:  # same day, but additional part of a very long article
                    output.writerow(article)
                    texts.append(article[12])
            else:  # different day -> update date & reset lists
                output.writerow(article)
                curyear = thisyear
                curmonth = thismonth
                curday = thisday
                ids = []
                titles = []
                texts = []
            if counter % 20000 == 0:
                print "Done with article %d" % counter
        print "Expunged %d duplicates" % dupecounter
    return


def keepfeatures(inputname, outputname, featurelist, flag=False, header=True):
    """Keep specified column indices only. Add optional flag feature."""
    import csv

    with open(inputname, 'rU') as inf, open(outputname, 'wb') as outf:
        output = csv.writer(outf)
        for article in csv.reader(inf):
            newarticle = [article[x] for x in featurelist]
            if header:
                if flag is not False:
                    newarticle.append('flag')
                header = False
            else:
                if flag is not False:
                    newarticle.append(flag)
            output.writerow(newarticle)


def idrange(filename, header=True):
    """Find smallest and largest id numbers in corpus.

    Assumes integer ids.
    """
    import csv
    minval = -1
    maxval = -1
    with open(filename, 'rU') as infile:
        for article in csv.reader(infile):
            if header:
                header = False
            else:
                idnr = int(article[0])
                if minval == -1 or idnr < minval:
                    minval = idnr
                if maxval == -1 or idnr > maxval:
                    maxval = idnr
    # print "\nSmallest and largest id numbers: %d, %d" % \
    # (minval, maxval)
    return minval, maxval


def translate_text (infile, outfile, replacefile='', dialectfile='',
                    translationfunction=None):
    """Initial preprocessing: word substitutions.

    Two types of word substitutions:
    - make corpus-specific substitutions as specified in replacedict.
    - translate across dialects
    """
    import csv
    from os.path import isfile
    import re

    # Load corpus-specific & dialect-specific substitution dictionaries.
    repDict = getReplaceDict(replacefile) if isfile(replacefile) else {}
    repKeys = re.compile('|'.join(repDict.keys()), flags=re.IGNORECASE)

    dialectDict = getReplaceDict(dialectfile) if isfile(dialectfile) else {}
    dialectKeys = re.compile('\\b(' + '|'.join(dialectDict.keys()) + ')\\b',
                         flags=re.IGNORECASE)

    # Process each text in infile
    with open(infile,'rU') as textFile:
        allTexts = csv.reader(textFile)
        with open(outfile, 'wb') as preppedFile:
            preppedTexts = csv.writer(preppedFile)
            for count, row in enumerate(allTexts):
                preppedText = row[1]
                if translationfunction is not None:
                    preppedText = translationfunction(preppedText)
                if len(dialectDict) > 0:
                    preppedText = replaceTerms(preppedText, dialectDict, dialectKeys)
                if len(repDict) > 0:
                    preppedText = replaceTerms(preppedText, repDict, repKeys)
                preppedTexts.writerow((row[0], preppedText))
                if count % 50000 == 0:
                    print count


def getReplaceDict(infileName):
    """Import a dictionary conversion table from a csv file."""
    import csv
    from os.path import isfile
    if not isfile(infileName):
        print "No substitution dictionary found."
        return {}
    else:
        replaceDict = {}
        with open(infileName, 'rU') as dictFile:
            dictData = csv.reader(dictFile)
            for line in dictData:
                replaceDict[line[0].strip()] = line[1].strip()
        return replaceDict


def replaceTerms(aText, replaceDict, searchPatterns):
    """Replace substrings as specified in replaceDict/searchPatterns."""
    import re
    return aText if replaceDict == {} else \
        searchPatterns.sub(lambda x: replaceDict[x.group().lower()], aText)


# *************** preprocessing: dialect (Americanize/Britishise) ************

def translate_B2A(aText):
    """Convert British spellings in a text to American spellings."""
    return replace_is(replace_lys(replace_our(aText)))


def replace_is(aString):
    """Replace British -is verb spellings by American -iz versions."""
    import re
    return re.sub(r"\w{2}\w+is(able|ably|ability|ance|ation|ational|ations|e|ed|ement|er|ers|es|ing|ingly)\b",
                  is_to_iz, aString, flags=re.IGNORECASE)


def is_to_iz(matchobj):
    """Replace 'is' by 'iz' in a verb-form, except for some special cases.

    specialstems (mostly verbs, but not only) are those that do not
        get -is (usually -ise) translate to -iz. This includes words
        constructed with -cise (to cut), -rise (including -prise),
        and -wise (likewise, etc.), as well as mostly French-derived
        words in -aise, -oise, and -uise

    superstems = stems listed because they end in -rise/-cise
        (i.e. special stems) but _do_ get the z in American English

    This list is likely incomplete; additions welcomed

    See also: http://www.oxforddictionaries.com/us/words/
                               verbs-ending-in-ize-ise-yze-and-yse-american

    """
    #TODO: check check against http://www.wordbyletter.com/suffixe.php for -ise/-ize words
    origword = matchobj.group(0)
    word = origword.lower()
    specialstems = ['ais', 'cis', 'ois', 'ris', 'uis', 'wis',
                    'advertis', 'advis', 'anis', 'avis', 'bis', 'chastis',
                    'chemis', 'compromis', 'demis', 'denis', 'despis', 'devis',
                    'franchis', 'improvis', 'paradis', 'practis', 'premis',
                    'promis', 'revis', 'surmis', 'supervis', 'televis',
                    'treatis']
    superstems = ['bowdleris', 'cauteris', 'characteris', 'computeris',
                  'criticis', 'garis', 'iaris', 'laris', 'mesmeris',
                  'militaris', 'oris', 'pulveris', 'satiris',
                  'summaris', 'tenderis', 'uris']
    if not any(x in word for x in specialstems) or \
                any(x in word for x in superstems):
        iz = word.rfind('is')
        return origword[:iz] + 'iz' + origword[iz + 2:]
    else:
        return origword


def replace_lys(aString):
    """Replace British -lys verb spellings by American -lyz versions."""
    import re
    return re.sub(r"\w{2}\w+ys(able|ably|ability|e|ed|er|ers|es|ing)\b",
                  lys_to_lyz, aString, flags=re.IGNORECASE)


def lys_to_lyz(matchobj):
    """Replace 'ys' by 'yz' in a verb-form, for limited # of verbs.:

    All end in -lyse: -analyse, breathalyse, catalyse, dialyse, electrolyse,
        hydrolyse, paralyse
    """
    origverb = matchobj.group(0)
    theVerb = origverb.lower()
    verbstems = ['analys', 'breathalys', 'catalys', 'dialys', 'electrolys',
                 'hydrolys', 'paralys']
    if any(x in theVerb for x in verbstems):
        lys = theVerb.rfind('lys')
        return origverb[:lys] + 'lyz' + origverb[lys + 3:]
    else:
        return origverb


def replace_our(aString):
    """British-to-American translation for -our words."""
    import re
    return re.sub(r"\b(\w\w+?our\w*?)\b", our_to_or, aString, flags=re.IGNORECASE)


def our_to_or(matchobj):
    """Replace -our words by the corresponding -or version.

    Note: 'glamour' is -our in American English too.
    """
    origword = matchobj.group(0)
    word = origword.lower()
    our_or = ['arbour', 'ardour', 'armour', 'behaviour', 'labour', 'candour', 'clamour',
        'clangour', 'colour', 'demeanour', 'enamour', 'endeavour', 'favour',
        'fervour', 'flavour', 'harbour', 'honour', 'humour', 'labour', 'neighbour',
        'odour', 'parlour', 'rancour', 'rigour', 'rumour', 'saviour', 'savour',
        'splendour', 'succour', 'tumour', 'valour', 'vapour', 'vigour']
    if any(x in word for x in our_or):
        our = word.rfind('our')
        return origword[:our] + 'or' + origword[our + 3:]
    else:
        return origword


def reduce_BAdict(filename):
    """Remove -ise/ize, -lyse/lyze and -our/or from dictionary."""
    thedict = getReplaceDict(filename)
    newdict = {}
    for Bword, Aword in thedict.iteritems():
        if verbBtoA(Bword) ==  Bword and our_to_or(Bword) == Bword:
            newdict[Bword] = Aword
    writeReplaceDict(newdict,
                     '.'.join(filename.split('.')[:-1]) + '_reduced.csv')







def makelower_all(textfile, header=False):
    """Make all text lower-case."""
    import csv

    outfilestem = '.'.join(textfile.split('.')[:-1])
    outfile = outfilestem[:-1] + 'Y.csv' if outfilestem[-1] == 'X' else \
        outfilestem + '_lower.csv'
    with open(textfile, 'rU') as inf, open(outfile, 'wb') as outf:
        output = csv.writer(outf)
        for article in csv.reader(inf):
            if header:
                header = False
            else:
                output.writerow([x.lower() for x in article])


def idrange(filename, header=True):
    """Find smallest and largest id numbers in corpus.

    Assumes integer ids.
    """
    import csv
    minval = -1
    maxval = -1
    with open(filename, 'rU') as infile:
        for article in csv.reader(infile):
            if header:
                header = False
            else:
                idnr = int(article[0])
                if minval == -1 or idnr < minval:
                    minval = idnr
                if maxval == -1 or idnr > maxval:
                    maxval = idnr
    # print "\nSmallest and largest id numbers: %d, %d" % \
    # (minval, maxval)
    return minval, maxval


def init_file(pathname):
    """Make sure a directory path exists,
    and initialize/overwrite any existing file at the location.
    """
    import os
    ensure_dir(pathname)
    with open(pathname, 'wb') as initf:
        x = 0  # dummy command


def ensure_dir(file_path):
    """Make sure a directory exists."""
    import os
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def combine_corpora_multi(corpuslist, outputcorpus, header=True):
    """Combine two or more corpora, making sure not to have overlapping ids."""
    import csv

    if len(corpuslist) <= 1:
        print "Need multiple corpora to combine - exiting without action."
        return
    init_file(outputcorpus)

    # First get the largest id number in any of the corpora
    maxid = max([idrange(corpus)[1] for corpus in corpuslist])
    counts = [0, ] * len(corpuslist)
    offset = int('1' + '0' * len(str(maxid)))  # feels like a bit of a hack, but it works
    print "\nAdding %d to each consecutive corpus, to set ids apart." % offset
    for counter, corpus in enumerate(corpuslist):
        corpheader = header
        with open(corpus, 'rU') as infile, \
                open(outputcorpus, 'ab') as outfile:
            outwriter = csv.writer(outfile)
            for artcount, article in enumerate(csv.reader(infile)):
                if corpheader:
                    if counter == 0:  # for first corpus only
                        outwriter.writerow(article)
                    corpheader = False
                else:
                    id = counter * offset + int(article[0])
                    outwriter.writerow([id, ] + article[1:])
            if not header:
                artcount += 1
            print "%d articles in %s" % (artcount, corpus)
            counts[counter] = artcount
    print "Total number of articles in combined corpus: %d" % sum(counts)
    return offset


def texts2lines_file(filestem):
    """As texts2lines, but streaming in/out (for large corpora)."""
    import csv

    linefile = filestem + '_lines.txt'
    idfile = filestem + '_lineids.txt'
    with open(filestem + '.csv', 'rU') as intexts, \
            open(linefile, 'wt') as outlines, \
            open(idfile, 'wt') as outids:
        textData = csv.reader(intexts)
        for id, atext in textData:
            if 'id' not in id.lower():  # skip header (if there is one)
                idsaved = False
                for aline in atext.split('.'):
                    aline = aline.replace('\n', ' ').strip()
                    if len(aline) > 0:
                        idsaved = True
                        outlines.write(aline + '\n')
                        outids.write(id + '\n')
                if not idsaved:  # save an empty line for this id
                    outlines.write(' ' + '\n')
                    outids.write(id + '\n')
    return linefile, idfile


def prep_neutral_for_Stata(metafile, valenceinfo, outputname):
    """Combine meta-data we wish to keep with extra features & valences.

    Meta-data to keep and some extra features are in the metafile,
    valences are in valenceinfo.

    Merge multiple parts of single article where applicable
    """
    import csv

    headers = ['id', 'pubname', 'year', 'month', 'day',
               'part', 'religion']
    metafilecols = len(headers)

    # Read metadata plus dummy religion measure
    with open(metafile, 'rU') as infile:
        inreader = csv.reader(infile)
        next(inreader)  # skip header
        datalist = [[row[0], ] + row[2:6] + [row[10], row[14]] \
                    for row in inreader]
        partcounter1 = 0
        data = [headers, ]
        curid = -1
        for article in datalist:
            if article[0] != curid:  # new id
                data.append(article)
                curid = article[0]
            else:
                prevarticle = data.pop()
                newarticle = article[:7]
                data.append(newarticle)
                partcounter1 += 1

    # Read valence measure
    valencefile = valenceinfo[0]
    valencecol = valenceinfo[1]
    valencename = valenceinfo[2]
    with open(valencefile, 'rU') as infile:
        data = [obs + [row[0], row[valencecol]] \
                for obs, row in zip(data, csv.reader(infile))]
    # Replace column headers just added
    data[0][metafilecols] = 'id'
    data[0][metafilecols + 1] = valencename

    # Save to output file
    with open(outputname, 'wb') as outfile:
        csv.writer(outfile).writerows(data)
    print "Number of multiple parts: %d" % (partcounter1)
    return


def prep_sents_for_Stata(sentfile, valencefile, outputname, keepcols=(17,)):
    """Combine sentences with valences."""
    import csv

    # Read id & sentence data -- clean file, no headers
    headers = ['id', 'sentence']
    with open(sentfile, 'rU') as infile:
        inreader = csv.reader(infile)
        next(inreader)  # skip header
        sentlist = [headers, ] + [row[:2] for row in inreader]

    # Append valence data, double-checking id
    # (note: multiple sentences will have the same id, so this is not perfect)
    data = []
    firstrow = True
    with open(valencefile, 'rU') as infile:
        for obs, vals in zip(sentlist, csv.reader(infile)):
            if firstrow:
                firstrow = False
            elif int(obs[0]) != int(vals[0]):
                print "error: id mismatch - had %s, got %s" % (obs[0], vals[0])
                break
            data.append(obs + [vals[x] for x in keepcols])

    # Save to output file
    with open(outputname, 'wb') as outfile:
        csv.writer(outfile).writerows(data)
    return


def prep_for_Stata(metafile, featurefile, featurenames, measurelist, outputname):
    """Combine meta-data we wish to keep with extra features & valences.

    Meta-data to keep and some extra features are in the metafile,
    other extra features are based on the cleaned text and are in featurefile.

    Merge multiple parts of single article where applicable
    """
    import csv

    headers1 = ['id', 'pubname', 'year', 'month', 'day', 'part',
                'religion', 'domestic_section', 'foreign_section']
    metafilecols = len(headers1)
    headers2 = ['id',] + featurenames
    nrfeatures = len(featurenames)
    headers = headers1 + headers2
    allcols = metafilecols + nrfeatures + 1

    # Read metadata plus 3 features based on metadata
    with open(metafile, 'rU') as infile:
        inreader = csv.reader(infile)
        next(inreader)  # skip header
        datalist = [[row[0], ] + row[2:6] + \
                    [row[10], row[14], int(row[15]), int(row[16])] \
                    for row in inreader]
        partcounter1 = 0
        data = []
        curid = -1
        for article in datalist:
            if article[0] != curid:  # new id
                data.append(article)
                curid = article[0]
            else:
                prevarticle = data.pop()
                newarticle = article[:7] + \
                             [1 if x == 1 or y == 1 else 0 \
                              for x, y in zip(prevarticle[7:], article[7:])]
                data.append(newarticle)
                partcounter1 += 1

    # Read text-based features, starting in row firstfeatcol
    firstfeatcol = 2  # hard-code for now; assume row[1] is the text
    with open(featurefile, 'rU') as infile:
        dataextra = [[row[0], ] + [int(x) for x in row[firstfeatcol:firstfeatcol + nrfeatures]] \
                     for row in csv.reader(infile)]
        partcounter2 = 0
        dataX = []
        curid = -1
        for article in dataextra:
            if article[0] != curid:  # new id
                dataX.append(article)
                curid = article[0]
            else:
                prevarticle = dataX.pop()
                newarticle = [article[0], ] + \
                             [1 if x == 1 or y == 1 else 0 \
                              for x, y in zip(prevarticle[1:], article[1:])]
                dataX.append(newarticle)
                partcounter2 += 1

    # Check that lengths of the two datasets match
    if len(data) != len(dataX):
        print "Error: length mismatch %d - %d" % (len(data), len(dataX))
        return
    data = [headers, ] + [x + y for x, y in zip(data, dataX)]

    # Read valence measures
    for measurefile, col, varname in measurelist:
        with open(measurefile, 'rU') as infile:
            data = [obs + [row[0], row[col]] \
                    for obs, row \
                    in zip(data, csv.reader(infile))]
    # Replace column headers just added
    for measurenr in range(len(measurelist)):
        data[0][allcols + measurenr * 2] = 'id'
        data[0][allcols + measurenr * 2 + 1] = measurelist[measurenr][2]

    # Save to output file
    with open(outputname, 'wb') as outfile:
        csv.writer(outfile).writerows(data)
    print "Number of multiple parts: %d" % (partcounter1)
    return


def combine_files(inputdata, outputname, varnames=(), adjustvarnames=True):
    """Combine multiple data files with same ids, keeping specified columns.

    Keeps all data in memory -- don't try with very large files!

    For each input file, specify filename, data columns to keep,
    and how to combine multiple parts if applicable

    Merge method is specified as:
    'none' - simply use the first value
    'max'  - use max value (equivalent to inclusive OR for binary flags)
    'sum'  - sum individual values

    Note: Columns to be combined with 'max' or 'sum' are assumed to hold integer values
    """
    import csv
    csv.field_size_limit(1000000000)

    # process each input file separately, merging multiple parts per item if necessary
    collected = []
    for inputspec in inputdata:

        # elegantly handle an optional 4th inputspec
        inputfile, inputcols, mergespec = inputspec[:3]
        if len(inputspec) == 4 and inputspec[3] == 'noheader':
            header = False
        else:
            header = True

        newcollected = []
        with open(inputfile, 'rU') as inputf:
            curid = -999
            firstrow = True
            for row in csv.reader(inputf):
                cols2keep = [row[col] for col in inputcols]
                if curid != row[0]:  # new article
                    if firstrow:
                        if not header:
                            # generate dummy header line
                            newcollected.append(['id',] + ['var' + str(x) for x in range(len(cols2keep))])
                            idval = row[0]
                        else:
                            idval = 'id'
                        firstrow = False
                    else:
                        idval = row[0]
                    newcollected.append([idval,] + cols2keep)
                    curid = row[0]
                else:  # combine with previous row
                    lastadded = newcollected.pop()
                    if mergespec == 'max':
                        replaceval = [max(int(x),int(y)) for x, y in zip(lastadded[1:], cols2keep)]
                    elif mergespec == 'sum':
                        replaceval = [int(x) + int(y) for x, y in zip(lastadded[1:], cols2keep)]
                    else:  # combination spec == 'none'
                        replaceval = lastadded[1:]  # i.e. do nothing
                    newcollected.append([curid,] + replaceval)
        collected.append(newcollected)

    # combine the data from each input file, double-checking id match
    combineddata = []
    for entry in zip(*collected):
        combined = entry[0]
        for subentry in entry[1:]:
            if combined[0] != subentry[0]:
                print "Error: non-matching ids (had %s, got %s)" % \
                      (combined[0], subentry[0])
                return
            else:
                combined += subentry[1:]
        combineddata.append(combined)

    # Use specified variable names, or else
    # add suffix to variable names, to avoid duplicate column names
    if len(varnames) == len(combineddata[0]):
        combineddata = [varnames,] + combineddata[1:]
    elif adjustvarnames:
        headers = combineddata[0]
        newheaders = [headers[0],]
        curindex = 1
        for counter, inputspec in enumerate(inputdata):
            inputfile, inputcols, mergespec = inputspec[:3]
            if counter == 0:  # treat first inputfile differently
                for inputcol in inputcols:
                    newheaders.append(headers[curindex])  # no change
                    curindex += 1
            else:
                # inputfilename = basename(inputfile)
                for inputcol in inputcols:
                    # newheaders.append(inputfilename + ' ' + headers[curindex])
                    newheaders.append(headers[curindex] + '_' + str(counter))
                    curindex += 1
        combineddata = [newheaders,] + combineddata[1:]

    with open(outputname, 'wb') as outfile:
        csv.writer(outfile).writerows(combineddata)


def check_ids_match(valencefile, outputfile, idlocs, headerline=True):
    """Make sure the id numbers of items pulled from different files match."""
    import csv
    with open(valencefile, 'rU') as infile, \
            open(outputfile, 'wb') as outfile:
        outwriter = csv.writer(outfile)
        for row in csv.reader(infile):
            if headerline:
                headerline = False
                outwriter.writerow(['id', ] + [x for x in row if x != 'id'])
            elif any([row[0] != row[x] for x in idlocs]):
                print "id mismatch:", [row[0], ] + [row[x] for x in idlocs]
                break
            else:
                outwriter.writerow([row[x] for x in range(len(row)) if x not in idlocs])


# ******************************** Valence calculation ****************************

# TODO:
# specify lexica to use:
# tuples: filename, lextype
# generate union on the spot to use as threshold for pushing down
# separate LIWC valencing from others into 2 functions

# TODO
# write unionlex (union of all lexicon words to determine whether word has valence
# write loadlex (update of getlexica, using lexicondata)

# def corpus_valence_new(filestem, lexicondata,
#                        ids=list(), sents=list(), nrjobs=1):
#     """Calculate sentence-level valences for entire corpus.
#
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
#         sents_per_job = 1000000
#         rounds = nrsents / (nrjobs * sents_per_job)
#         jobs_rounds = rounds * nrjobs
#         chunksize = 1 + ((nrsents - 1) / jobs_rounds)
#         print "Run in %d consecutive rounds of %d sentences per job to control memory use" % \
#               (rounds, chunksize)
#
#     # Read in lexica to use, by type
#     stdlex, wildlex, liwclex, mods, lexheaders = loadlexica(lexicondata)
#     # Generate a union of all lexica, to use in determining whether a word has valence
#     stdlex = [unionlex(stdlex, wildlex)] + stdlex
#
#     if nrjobs == 1:
#         # This is an elegant list comprehension, but not parallelizable and hence slow
#         valences = [[id,] + getValence(s, stdlex, wildlex, liwclex, mods, ignorewords, sentnr=count) + [s,] \
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
#                                       liwclex=liwclex, mods=mods, ignore=ignorewords)
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
#     with open(filestem + '_valences.csv', 'wt') as outf:
#         outfile = csv.writer(outf)
#         outfile.writerow(['id',] + lexheaders)
#         outfile.writerows(valences)
#     return lexheaders


# def corpus_valence(filestem, valencedir, lexicafile,
#                    ids=list(), sents=list(), nrjobs=1):
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
#         sents_per_job = 1000000
#         rounds = nrsents / (nrjobs * sents_per_job)
#         jobs_rounds = rounds * nrjobs
#         chunksize = 1 + ((nrsents - 1) / jobs_rounds)
#         print "Run in %d consecutive rounds of %d sentences per job to control memory use" % \
#               (rounds, chunksize)
#
#     # Read in lexica to use
#     stdlex, wildlex, liwclex, mods, lexheaders = \
#         getlexica(filestem, valencedir, lexicafile)
#     # Make this more generalizable; also, do on-the-spot creation of unionlex to use as
#     # test for whether need to look at valence
#
#     if nrjobs == 1:
#         # This is an elegant list comprehension, but not parallelizable and hence slow
#         valences = [[id,] + getValence(s, stdlex, wildlex, liwclex, mods, ignorewords, sentnr=count) + [s,] \
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
#                                       liwclex=liwclex, mods=mods, ignore=ignorewords)
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
#     with open(filestem + '_valences.csv', 'wt') as outf:
#         outfile = csv.writer(outf)
#         outfile.writerow(['id',] + lexheaders)
#         outfile.writerows(valences)
#     return lexheaders


# def getValences(sentchunk, stdlex, wildlex, liwclex, mods, ignorewords):
#     """Calculate valences for the sentence chunk."""
#     chunknr, ids_sents = sentchunk
#     valences = [[id,] + getValence(s, stdlex, wildlex, liwclex, mods, ignorewords,
#                                    chunknr=chunknr, sentnr=count) \
#                  for count, (id, s) in enumerate(ids_sents)]
#     return (chunknr, valences)


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
        rounds = nrsents / (nrjobs * sents_per_job)
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
    for var_mean_stdev in zip(lexiconnames[1:], sentMeans, sentStds)[1:11]:
        print var_mean_stdev

    # Write out aggregates for future recall
    write_corpus_valence(filestem, sentMeans, sentStds, lexiconnames)
    return sentMeans, sentStds


def scalevals(valences):
    """Scale valences by the number of words in a sentence (stored as row[1])."""
    return [[x[1], ] + [val / float(x[1]) if x[1] > 0 else 0 for val in x[2:]] \
                  for x in valences]


def getlexica(filestem, valencedir, lexicafile):
    """Return all the lexica to be used on the corpus specified.

    Three lexicon sources/types:
    1) general-purpose lexica
    2) intersections of those, generated earlier and saved
    3) corpus-specific lexica, derived from (2) using pmi (and saved under filestem - not currently used)
    """
    import pickle
    from operator import itemgetter

    # Get external lexica
    lexi, socal, liu, WS, labMT, NRC, SWN = importValenceDicts(valencedir)
    # Recenter labMT, which is not 0-centered
    labMT = recenter_lexicon(labMT, dval=-5)
    # Combine all lexica into a list
    stdlexica = [socal[0], liu, WS, labMT, NRC[3], SWN]
    wildlexica = [lexi,]
    # Skip LIWC & moral founds. for now; update valencecats below if change!
    # liwclexica = [(LIWC, len(LIWC[1])), (moral, len(moral[1])),
    #               (NRC[:2], len(NRC[1]))]
    liwclexica = [(NRC[:2], len(NRC[1]))]

    unionlex, intersect1, intersect2 = importIntersectLexica(lexicafile)
    intersectnames = ['unionlex', 'inter4pure', 'inter5impure']
    stdlexica += [unionlex, intersect1, intersect2]

    # check values
    # print "Lab MT"
    # print sorted(labMT.items(), key=itemgetter(1), reverse=True)[:50]
    # print "swn"
    # print sorted(SWN.items(), key=itemgetter(1), reverse=True)[:50]
    # print "unionlex"
    # print sorted(unionlex.items(), key=itemgetter(1), reverse=True)[:50]
    # print "lexicoder"
    # print sorted(lexi.items(), key=itemgetter(1), reverse=True)[:50]

    # Get corpus-specific lexica
    # corpuslexica, corpusnames = importCorpusDicts(filestem)
    # stdlexica += corpuslexica

    # Combine intersected and corpus-specific lexica
    # In testing on imdb_tok, these did not bring improvement, so skip
    # combolexica, combonames = combinelexica(intersectlexica, corpuslexica)
    # stdlexica += combolexica

    # Make list of valence category names
    valencecats = ['sentlength',
                   'SOCAL', 'BingLiu', 'WordStat', 'labMT', 'NRC', 'SWN'] + \
                  intersectnames + ['Lexicoder', 'NRCwords'] + NRC[2]
    return stdlexica, wildlexica, liwclexica, socal[1], valencecats


# def getValence(sent, stdlexica, wildlexica, liwclexica,
#                modifiers={}, ignore=(),
#                negaters=('not', 'no', 'nor', 'nothing', 'never', 'nowhere'),
#                stopwords=('a', 'an', 'and', 'the', 'to', 'as'),
#                sentnr=-1, chunknr=0):
#     """Sum valences for this sentence, using 3 types of lexicon.
#
#     Standard lexica are dictionaries of word:val
#     Wildcard lexica may have words with a * wildcard as the last character
#     - both of these can be modified by a modifier lexicon,
#       of the form modifier:mod_fraction
#
#     LIWC-style lexica contain several different word lists; no valence
#     - to avoid repetitive calculations, each liwc lexicon
#       is expected to be passed in as a tuple: (lexicon, nr categories)
#
#     Note: original LIWC has a 'long words' category which we could easily incorporate
#     (simply add 1 to the values to be calculated, and add len(word > 6) as
#     a test) but there is no reason to expect it to be useful for our purposes.
#
#     Note that the negation set-up interacts heavily with the language
#     substitution function language_subst.negation_subst()
#
#     The first lexicon in the combined list of stdlexica and wildlexica
#     is used to determine whether a word is a valence word. We should generate
#     a union of all lexica used (incl. wildcards) for that purpose.
#     """
#     from itertools import chain
#
#     # Progress update
#     if sentnr % 100000 == 0:
#         print "Processing sentence %d%s" % \
#               (sentnr, ('' if chunknr == 0 else " of chunk %d" % chunknr))
#
#     nr_stdlexica = len(stdlexica)
#     nr_wildlexica = len(wildlexica)
#     nr_liwclexica = len(liwclexica)
#     # Add basic modifiers from language preprocessing
#     if 'minusx' not in modifiers:
#         modifiers['minusx'] = -0.5
#     if 'plusx' not in modifiers:
#         modifiers['plusx'] = 1
#
#     # nr. values to be calculated: 1 for length of sentence, 1 for each
#     # standard polarity lexicon, 1 for nr. of words captured by each
#     # LIWC-style lexicon, plus sum(liwclens)
#     valences = [0,] * (1 + nr_stdlexica + nr_wildlexica + nr_liwclexica + \
#                        sum([x[1] for x in liwclexica]))
#
#     # Check each word in the sentence. Start out with no valence modification
#     lexmatches = [0,] * (nr_stdlexica + nr_wildlexica)
#     modifier = 1
#     skipcounts = []
#     wordlist = sent.lower().split()
#     nrwords = len(wordlist)
#     for count, word in enumerate(wordlist):
#         # Make sure word was not part of a modifier already handled
#         if count not in skipcounts:
#
#             # Handle modifiers, incl. multi-word modifying phrases
#             # Longer phrases trump shorter ones; none is longer than 4 words
#             if count < nrwords - 3:
#                 wordx = '_'.join(wordlist[count:count + 4])
#                 if wordx in modifiers:
#                     skipcounts += [count + 1, count + 2, count + 3]
#                     modifier *= 1 + modifiers[wordx]
#                     continue
#             if count < nrwords - 2:
#                 wordx = '_'.join(wordlist[count:count + 3])
#                 if wordx in modifiers:
#                     skipcounts += [count + 1, count + 2]
#                     modifier *= 1 + modifiers[wordx]
#                     continue
#             if count < nrwords - 1:
#                 wordx = '_'.join(wordlist[count:count + 2])
#                 if wordx in modifiers:
#                     skipcounts += [count + 1,]
#                     modifier *= 1 + modifiers[wordx]
#                     continue
#             if word in modifiers:
#                 modifier *= 1 + modifiers[word]
#                 continue
#
#             # Check for negation next
#             if word in negaters:
#                 modifier *= -0.5
#                 continue
#
#             # Look up valences
#             lexmatches = lexiconmatch_multi(word, stdlexica) + \
#                             lexiconmatch_multiwild(word, wildlexica)
#             # If a valence word (in union lexicon), multiply by modifier
#             if len(lexmatches) > 0 and lexmatches[6] != 0:
#                 lexmatches = [modifier * x for x in lexmatches]
#
#             # Finally, reset modifier, unless this was a stopword
#             if word not in stopwords:
#                 modifier = 1
#
#         # LIWC-style lexica may also include modifiers & negation words
#         liwcmatches = \
#             list(chain.from_iterable([LIWCmatch(word, lex[0], lex[1]) \
#                                        for lex in liwclexica]))
#
#         # Update sentence valences based on this word
#         wordvalences = [1,] + lexmatches + liwcmatches
#         valences = [x + y for x, y in zip(valences, wordvalences)]
#
#     return valences


def aggregate_vals(filestem, ids, vals, aggmethod='wordscale'):
    """Aggregate meta info to the document level.

    Need to keep article length from affecting meta measures, so when
    aggregating meta info (valences), select 1 of 3 options for adjusting
    the summed info:
    wordscale - divide by total word count
    sentscale - divide by nrsentences
    sentmeans - subtract corpus sentence means * nrsentences
    First option generally seems best (and is default).
    For the third option, need to read in corpus averages from file
    """
    docids = []
    valences = []
    for id, docvals in zip(*aggregate_to_sublists(ids, vals)):
        nrsents = len(docvals)
        docids.append(id)
        valencesums = [sum(val) for val in zip(*docvals)]
        if aggmethod == 'wordscale':
            wordcount = valencesums[0]
            valencesumsAdj = valencesums if wordcount == 0 else \
                 [wordcount/float(nrsents),] + \
                 [vsum/float(wordcount) for vsum in valencesums[1:]]
        elif aggmethod == 'sentscale':
            valencesumsAdj = \
                [vsum/float(nrsents) for vsum in valencesums]
        else: # aggmethod == 'sentmeans':
            sentMeans, _ = read_corpus_valence(filestem + '_valenceAvgs.csv')
            valencesumsAdj = \
                [vsum - nrsents * amean for vsum, amean in \
                      zip(valencesums, sentMeans)]
        valences.append(valencesumsAdj)
    return docids, valences


def aggregate_to_sublists(ids, items):
    """Aggregate doc data into sublists by id."""
    curid = -999
    docids = []
    docitems = []
    curitems = []
    for id, item in zip(ids, items):
        if id == curid:
            curitems.append(item)
        else:
            docids.append(curid)
            docitems.append(curitems)
            curitems = [item, ]
            curid = id
    # Close out last sublist
    docids.append(curid)
    docitems.append(curitems)
    # Return all entries but the first (for id == -999)
    return docids[1:], docitems[1:]


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


# ********************** Input / output functions ******************

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
            if header:
                featurenames = row[firstfeatcol:firstfeatcol+nrfeatures]
                if nrfeatures == 999:
                    nrfeatures = len(featurenames)
                header = False
            elif len(row) == 0:  # empty line -- should not happen, but may happen on PCs
                continue  # skip line
            elif 'id' not in row[0].lower():  # Remnant of old header
                ids.append(int(row[0].strip()))
                features.append([float(x) for x in row[firstfeatcol:firstfeatcol+nrfeatures]])
    return ids, features, featurenames


def readfeature_subset(infilename, featureset, firstfeatcol=1, header=True):
    """Read a subset of features from csv file (id, feature1, ...)."""

    maxindex = max(featureset)
    minindex = min(featureset)
    firsttoread = firstfeatcol + minindex
    nrtoread = maxindex - minindex + 1

    ids, features, featurenames = \
        readfeatures(infilename, nrtoread, firsttoread, header)

    # read features in order, even if specified out of order
    features = [[x[i] for i in xrange(maxindex + 1) if i in featureset] \
                for x in features]
    featurenames = [featurenames[i] for i in xrange(maxindex + 1) \
                    if i in featureset]
    return ids, features, featurenames


def readwords(wordfile):
    """Extract word list from file with 1 word per line."""
    wordlist = []
    with open(wordfile, 'rU') as words:
        for w in words:
            wordlist.append(w.strip())
    return wordlist


# ********************** Aggregating sentence-level valences ******************

# TODO: remove the cases now handled in _vals_filter below
def sent2doc_valences_filter(filestem, catname='all', offset=100000000):
    """Aggregate sentence-level text- & feature files to doc-level.

    Hard-coded filtering step.
    """
    sents = [x.lower() for x in readwords(filestem + '_lines.txt')]
    featureset = set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ids, features, featurenames = \
            readfeature_subset(filestem + '_valences.csv', featureset,
                               header=True)
    cats = [id/offset for id in ids]
    print "Aggregating sentence valences for filtered sentences only."

    # Specify narrow and broad word sets to look for, by religion
    # Note: for Islam, excluded negatively valenced words that might bias
    # results: ('fatwa', 'fidayeen', 'islamophobia', 'islamophobic',
    #  'islamist', 'islamists', 'jihad', 'jihadi', 'jihadis',
    #  'mujaheddin', 'sharia', 'taliban')
    words = {'Catholics': [('catholic', 'catholics', 'catholicism'),
                           ('bishop', 'cardinal', 'communion', 'confession', 'dominican',
                            'franciscan', 'jesuit', 'papal', 'pope', 'popes')],
             'Jews': [('jew', 'jews', 'jewish', 'judaic', 'judaism',
                       'jewishness', 'jewry', 'jewess', 'jewesses'),
                      ('hanukkah', 'kaddish', 'kippah', 'kosher', 'mitzvah',
                       'passover', 'purim', 'rabbi', 'rabbis', 'roshhashanah',
                       'shabbat', 'synagogue', 'torah', 'torahs',
                       'yarmulke', 'yeshiva', 'yomkippur')],
             'Muslims': [('islam', 'islamic', 'muslim', 'muslims', 'muslima', 'muslimas'),
                         ('allah', 'ayatollah', 'ayatollahs',
                          'burqa', 'burqas', 'chador', 'chadors',
                          'halal', 'headscarf', 'headscarves',
                          'hijab', 'hijabs', 'imam', 'imams',
                          'koran', 'korans', 'madrassa', 'madrassas',
                          'mosque', 'mosques', 'muhammad',
                          'mullah', 'mullahs', 'niqab', 'niqabs',
                          'shia', 'shias', 'sunni', 'sunnis')]}
             # 'refugees': [('refugee', 'refugees', 'asylum', 'migrant', 'migrants',
             #               'immigrant', 'immigrants'),
             #              ('deport', 'deports', 'deported', 'deporting', 'deportation',
             #               'emigrant', 'emigrants', 'emigrate', 'emigrated', 'emigrating',
             #               'illegal alien', 'illegal aliens', 'illegal entry',
             #               'leave to remain')],
             # 'refugee': [('refugee', 'refugees', 'asylum'),
             #             ('seeker',)],
             # 'migrant': [('migrant', 'migrants', 'immigrate', 'immigrated',
             #              'immigrating', 'immigrant', 'immigrants'),
             #             ('emigrant', 'emigrants', 'emigrate', 'emigrated', 'emigrating')]}
             # Note: the multi-word refugee ones probably won't hit

    if catname == 'high-valence':  # Filter based on high sentence-level valence
        filter_and_combine(filestem, '_highvalence', ids, sents, features, featnames,
                           neutralfilter(features))

    elif type(catname) is str and catname in words:
        # Filter based on religion-specific words.
        # Set up word lists
        catwords = words[catname][0]
        extrawords = catwords + words[catname][1]
        # Do the actual filtering, narrow and broad
        filter_and_combine(filestem, '_valnarrow', ids, sents,
                           features, featurenames,
                           wordfilter(catwords, sents), savesents=True)
        filter_and_combine(filestem, '_valbroad', ids, sents,
                           features, featurenames,
                           wordfilter(extrawords, sents))

    elif type(catname) is list and all([subcat in words for subcat in catname]):
        # catname is a list of cats, should be in offset order matching the corpus!
        wordlists = [words[subcat][0] for subcat in catname]
        wordlistsX = [words[subcat][0] + words[subcat][1] for subcat in catname]
        filter_and_combine(filestem, '_valnarrow', ids, sents,
                           features, featurenames,
                           wordfilter_multi(wordlists, cats, sents), savesents=True)
        filter_and_combine(filestem, '_valbroad', ids, sents,
                           features, featurenames,
                           wordfilter_multi(wordlistsX, cats, sents))
    else:
        print "Sorry, category name(s) not recognized."
    return featurenames


# Sample narrow/broad lists for refugees, migrants

# 'refugees': [('refugee', 'refugees', 'asylum', 'migrant', 'migrants',
#               'immigrant', 'immigrants'),
#              ('deport', 'deports', 'deported', 'deporting', 'deportation',
#               'emigrant', 'emigrants', 'emigrate', 'emigrated', 'emigrating',
#               'illegal alien', 'illegal aliens', 'illegal entry',
#               'leave to remain')],
# 'refugee': [('refugee', 'refugees', 'asylum'),
#             ('seeker',)],
# 'migrant': [('migrant', 'migrants', 'immigrate', 'immigrated',
#              'immigrating', 'immigrant', 'immigrants'),
#             ('emigrant', 'emigrants', 'emigrate', 'emigrated', 'emigrating')]}
# Note: the multi-word refugee ones probably won't hit


def sent2doc_vals_filter(filestem, filterdata, savesentences,
                         lowered=True, sentsuffix='_lines', valsuffix='_valences'):
    """Aggregate sentence-level text- & feature files to doc-level.

    Filter on catdata, which should have the format [list1, list2, ...]
    where each list has the format [name, termlist, avoidlist]
    The name will be used in saving the valence data;
    termlist contains the terms that will determine whether a sentence is included,
        except when the same terms are captured by an item in avoidlist
    savesentences is a flag to specify whether to save these sentences in a separate file
    """

    print "Aggregating sentence valences for filtered sentences only."
    # read in & (if necessary) make lower-case all sentences
    if lowered:
        sents = readwords(filestem + sentsuffix + '.txt')
    else:
        sents = [x.lower() for x in readwords(filestem + sentsuffix + '.txt')]

    # read sentence valences
    # assume we want to include all valences in the valence file (i.e. not just a subset)
    ids, features, featurenames = readfeatures(filestem + valsuffix + '.csv',
                                               nrfeatures=999, header=True)

    print "Data read in; now filtering & aggregating"
    for filterlist in filterdata:
        filter_and_combine(filestem, filterlist[0], ids, sents,
                           features, featurenames,
                           wordfilter((filterlist[1], filterlist[2]), sents),
                           savesents=savesentences)
    return featurenames


def filter_and_combine(filestem, filtername, ids, sents,
                       features, featnames, cond, savesents=False):
    """Filter based on word list and combine back into document."""
    import csv

    idlist = list(unique_everseen(ids))
    # Get filtered list of ids, sentences, features (i.e. valences)
    idsx, sentsx, featuresx = \
        filter_sents(cond, ids, sents, features)

    if savesents:
        # Save sentences left
        print "Saving list of filtered sentences"
        with open(filestem + '_' + filtername + '_sentences.csv', 'wb') as filteredsents:
            filteredwriter = csv.writer(filteredsents)
            filteredwriter.writerow(['id', 'sentence'] + featnames[1:])
            featuresy = [[0 if feats[0] == 0 else val/float(feats[0]) for val in feats[1:]] \
                         for feats in featuresx]
            filteredwriter.writerows([[idx, sentx] + feat for \
                                      idx, sentx, feat in zip(idsx, sentsx, featuresy)])

    # Aggregate from sentence-level back up to article-level
    aggids, aggvalences = \
        aggregate_vals(filestem, idsx, featuresx, aggmethod='wordscale')
    idval_dict = {id: val for id, val in zip(aggids, aggvalences)}
    docvalences = [idval_dict.get(id, [0 for x in featnames]) \
                   for id in idlist]
    writefeatures(filestem + '_' + filtername + '_sent_valfull', idlist, docvalences, featnames)


def unique_everseen(iterable, key=None):
    """List unique elements, preserving order. Remember all elements ever seen."""
    # unique_everseen('AAAABBBCCDAABBB') --> A B C D
    # unique_everseen('ABBCcAD', str.lower) --> A B C D
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in filterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element


def filterfalse(predicate, iterable):
    # filterfalse(lambda x: x%2, range(10)) --> 0 2 4 6 8
    if predicate is None:
        predicate = bool
    for x in iterable:
        if not predicate(x):
            yield x


def wordfilter(wordlist, sents, capsmatter=False):
    """For each sentence in sents, see if it matches searchspecs.

    Search specs can be individual words (possibly with wildcards) to be matched,
    optionally combined with a list of words to be avoided. For multi-word phrases,
    will need to invoke wordsinwindow.
    """
    if len(wordlist) ==  2 and isinstance(wordlist[0], (list, tuple)):
        # Have search strings & avoid strings
        searchterms = wordlist[0]
        avoidterms = wordlist[1]
        return [countmatches(sent.split(), searchterms, avoidterms, capsmatter) > 0 \
                for sent in sents]
    else:
        # Assume search strings only
        return [countmatches(sent.split(), wordlist, [], capsmatter) > 0 \
                for sent in sents]


def countmatches(sentwords, searchterms, avoidterms, capsmatter=False):
    # count matches in sentwords, subtracting those in avoidterms
    from wordsinwindow import findindices

    specsfound = findindices(sentwords, searchterms,
                             capsmatter=capsmatter, skipperiods=True)
    keycount = len(specsfound)
    if len(avoidterms) > 0:
        # Look for avoid strings
        specsfound2 = findindices(sentwords, avoidterms,
                                  capsmatter=capsmatter, skipperiods=True)
        keycount -= len(specsfound2)
    return keycount


def wordfilter_multi(wordlists, cats, sents):
    """As wordfilter, but with wordlists for multiple categories."""
    return [wordmatch(wordlists[x], sent) for sent, x in zip(sents, cats)]


def wordmatch(wordlist, sent):
    """See if sent contains >= 1 word in wordlist w. wildcards."""
    sentwords = sent.split()
    for word in wordlist:
        if word[-1] == '*':
            wordlen = len(word) - 1
            if any([word[:wordlen] == sentwd[:wordlen] \
                    for sentwd in sentwords]):
                return True
        elif word in sentwords:
            return True
    return False


def filter_sents(cond, ids, sents, vals):
    """Filter list based on condition."""
    print "number of sentences", len(ids)
    filtered = [(id, sent, val) for x, id, sent, val in zip(cond, ids, sents, vals) if x]
    print "number of filtered sentences", len(filtered)
    return zip(*filtered)
    # previous version: filter one by one
    # filteredids = [id for x, id in zip(cond, ids) if x]
    # filteredsents = [sent for x, sent in zip(cond, sents) if x]
    # filteredvals = [val for x, val in zip(cond, vals) if x]
    # print "number of filtered sentences", len(filteredids)
    # return filteredids, filteredsents, filteredvals


def calibrate_features(featurefilename, featureset, neutralscaler='', stdev_adj=1,
                       firstfeatcol=1, filtercol=-1, adjustvals=False, header=True,
                       outsuffix='_cal'):
    """Calibrate features by standardizing; return means & stdevs.

    Optionally filter out (replace by -999) any items with a value of 0
    in the filter column."""

    from sklearn.preprocessing import StandardScaler
    import numpy as np

    ids, features, featurenames = \
        readfeature_subset(featurefilename, featureset,
                           firstfeatcol=firstfeatcol, header=header)
    print "Nr. features: %d, nr. items: %d" % (len(features[0]), len(features))

    # Optionally read filter data from the same file
    if filtercol != -1:
        filterfeature = [0,]
        ids, filterdata, filtername = \
            readfeature_subset(featurefilename, filterfeature,
                               firstfeatcol=filtercol, header=header)
        # extract from single-item sublists
        filterdata = [x[0] for x in filterdata]
        print "Nr. items to be filtered out: %d" % (len([x for x in filterdata if x == 0]))
    else:
        filteroption = 'adjustdenominator'

    # Optionally adjust valences
    if adjustvals and filtercol != -1:  # Use filter column value as a divisor as well
        features = [[(x/float(filtval) if filtval > 0 else 0) \
                        for x in feats] \
                    for feats, filtval in zip(features, filterdata)]

    # Rescale & recenter features
    if neutralscaler == '':
        neutralscaler = StandardScaler()
        neutralscaler.fit(features)
    featuresX = neutralscaler.transform(features)
    featurenamesX = [x + 'S' for x in featurenames]
    # See what the new standard deviations & means are
    # means = np.mean(featuresX, axis=0)
    # print means
    # stdevs = np.std(featuresX, axis=0)
    # print stdevs

    # Now handle the various options for treating missing info & calculating averages
    denominator = float(len(featureset))
    divisor = denominator * stdev_adj

    if filtercol != -1:
        # Average, option 1: use calibrated values except in case of filter
        avgvalence = [-999 if filt == 0 else sum(vals)/divisor \
                      for vals, filt in zip(featuresX, filterdata)]

    else:
        # - individual-level adjustment: for original values that were 0,
        #   set to -999 for missing, rather than use adjusted value
        featuresX = [[-999 if x == 0 else y for x, y in zip(uncalib, calib)] \
                     for uncalib, calib in zip(features, featuresX)]
        nonzerofeaturesX = [[xi for xi in x if xi != -999] for x in featuresX]

        if filteroption == 'adjustdenominator':
            # Average, option 2: divide by the number of non-'zero' values
            avgvalence = [-999 if len(x) == 0 else sum(x)/(len(x) * stdev_adj) for x in nonzerofeaturesX]
        else:
            # Average, option 3: divide by the number of lexica used
            # (i.e. count 'zero' values as 0, instead of the calibrated equivalent)
            avgvalence = [-999 if len(x) == 0 else sum(x)/divisor for x in nonzerofeaturesX]

    # Write results
    featurelist = [orig + list(scaled) + [avg,] for orig, scaled, avg \
                   in zip(features, featuresX, avgvalence)]
    featurenames = featurenames + featurenamesX + ['avg_valence',]

    # Optionally, specify how many non-zero items there were:
    # nr_nonzero = [len(x) for x in nonzerofeaturesX]
    # featurelist = [orig + list(scaled) + [avg, nr] for orig, scaled, avg, nr \
    #                in zip(features, featuresX, avgvalence, nr_nonzero)]
    # featurenames = featurenames + featurenamesX + ['avg_valence', 'nr_in_avg']

    outputstem = '.'.join(featurefilename.split('.')[:-1]) + outsuffix
    writefeatures(outputstem, ids, featurelist, featurenames)
    return neutralscaler


def create_scalers(featurefilename, outfilestem, featuresets, firstfeatcol=1):
    """Generate scalers, storing mean & std. dev. for sets of features"""

    import numpy as np
    import pickle
    from sklearn.preprocessing import StandardScaler

    # collect indices of all features used
    featuresused = set()
    for scalername, featureset, descriptor in featuresets:
        featuresused |= set(featureset)
    featuresused = sorted(list(featuresused))

    # read in all the features used
    ids, features, featurenames = \
        readfeature_subset(featurefilename, featuresused,
                           firstfeatcol=firstfeatcol, header=True)
    print "Nr. features: %d, nr. items: %d" % (len(features[0]), len(features))
    print "Features:", featurenames

    for scalername, featureset, descriptor in featuresets:
        # adjust feature index by firstfeature
        relindices = [featuresused.index(x) for x in featureset]
        relfeatures = [[xi for i, xi in enumerate(x) if i in relindices] \
                       for x in features]
        relnames = [x for ind, x in enumerate(featurenames) if ind in relindices]
        print "\nScaler for:", relnames

        # generate scaler & print info
        neutralscaler = StandardScaler()
        relfeaturesX = neutralscaler.fit_transform(relfeatures)
        print "Means:", neutralscaler.mean_
        print "Std. devs.:", neutralscaler.scale_

        # generate adjustment to scaling to produce an average that has std. deviation of 1
        denominator = len(featureset)
        avgvalence = [sum(vals)/denominator for vals in relfeaturesX]
        stdev_avg = np.std(avgvalence)
        print "Std. dev. of average:", stdev_avg

        # Save scaler
        with open(outfilestem + scalername + '.pkl', 'wb') as outpickle:
            pickle.dump((neutralscaler, stdev_avg, relnames, descriptor), outpickle)

    return


def inspect_scaler(filename):
    """Display contents of scaler"""
    import pickle

    with open(filename, 'rb') as inpickle:
        neutralscaler, stdev_avg, featurenames, descriptor = pickle.load(inpickle)
    print "Descriptor:", descriptor
    print "Lexica used:", featurenames
    print "Means:", neutralscaler.mean_
    print "Std. devs.:", neutralscaler.scale_
    print "Std. dev. of average across lexica:", stdev_avg


def select_sentences(inputname, outputname,
                     minlength, maxlength,
                     targetvalences, delta=0.01):
    """Select sentences of within length range at target valence levels."""
    import csv
    from operator import itemgetter

    valenceranges = [(x - delta, x + delta, x) for x in targetvalences]
    sentences = []
    targetcounts = {}

    with open(inputname, 'rU') as infile:
        for counter, row in enumerate(csv.reader(infile)):
            sentlen = len(row[1].split())
            if (sentlen >= minlength and sentlen <= maxlength):
                targetval = inrange(float(row[2]), valenceranges)
                if targetval is not None:
                    sentences.append((row[1], sentlen, targetval, float(row[2])))
                    targetcounts[targetval] = targetcounts.get(targetval, 0) + 1
            if counter % 500000 == 0:
                print "Processing sentence %d" % counter
    with open(outputname, 'wb') as outfile:
        outwriter = csv.writer(outfile)
        outwriter.writerows(sorted(sentences, key=itemgetter(3)))
    print "\nFound %d sentences matching criteria." % len(sentences)
    for target, count in sorted(targetcounts.items()):
        print "Target valence %4.2f - %d sentences" % (target, count)


def inrange(val, valranges):
    """Check if value is in one of the acceptable ranges."""
    for rangemin, rangemax, target in valranges:
        if val >= rangemin and val <= rangemax:
            return target
    return None


def filter_sentences(inputname, outputname, N=5):
    """Let user pick N sentences for each target valence.

    inputname should be a file created by select_sentences:
    each row contains a sentence, plus its length, target and actual valences
    """
    import csv
    from operator import itemgetter
    from itertools import groupby

    # Read in the entire dataset
    sentences = []
    with open(inputname, 'rU') as infile:
        for row in csv.reader(infile):
            sentences.append((row[0], int(row[1]), float(row[2]), float(row[3])))

    sents_tokeep = []
    print "When asked whether to keep a sentence, type 'y' for yes, hit return for no."

    for key, group in groupby(sentences, key=lambda x: x[2]):
        thisgroup = sorted(list(group), key=lambda x: abs(x[3] - x[2]))
        nrkept = 0
        print "\n*** Pick %d of the next %d sentences ***" % (N, len(thisgroup))
        for sent in thisgroup:
            # print sent[2], sent[3], abs(sent[3] - sent[2])
            print '\n' + sent[0][0].upper() + sent[0][1:]
            decision = raw_input('Keep? ')
            if len(decision) > 0 and decision.lower()[0] == 'y':
                sents_tokeep.append(sent)
                nrkept += 1
                if nrkept == N:
                    break  # exit for loop
                else:
                    print "Nr. left to keep at this level: %d" % (N - nrkept,)
        if nrkept != N:
            print "Warning: you did not accept enough sentences"

    with open(outputname, 'wb') as outfile:
        outwriter = csv.writer(outfile)
        outwriter.writerows(sorted(sents_tokeep, key=itemgetter(3)))
    print "All done!"


def pair_sentences(inputname, outputname, pairings, shufflepairs=True):
    """Produce sentence pairings as requested.

    inputname should be a file created by filter_sentences.
    Do an exhaustive pairing of the sentences at each level specified
    (i.e.: if each level has 5 sentences, produce 25 pairings).

    output will contain (with sentences in random order):
    sentence 1, sentence 2, which is more positive (1 or 2), valence 1, valence 2

    shuffle determines whether output rows are randomly shuffled.
    """
    import csv
    from itertools import groupby
    from random import randint, shuffle

    # Read in the sentences
    sentences = []
    with open(inputname, 'rU') as infile:
        for row in csv.reader(infile):
            sentences.append((row[0], int(row[1]), float(row[2]), float(row[3])))

    # Put in dictionary by key
    sent_dict = {}
    for key, group in groupby(sentences, key=lambda x: x[2]):
        sent_dict[key] = list(group)

    pairedsentences = []
    for target1, target2 in pairings:
        for sent1 in sent_dict[target1]:
            for sent2 in sent_dict[target2]:
                # Flip a coin to determine order
                if randint(0, 1) == 1:
                    pair = (sent1[0], sent2[0],
                            1 if float(sent1[2]) > float(sent2[2]) else 2,
                            sent1[2], sent2[2])
                else:
                    pair = (sent2[0], sent1[0],
                            1 if float(sent2[2]) > float(sent1[2]) else 2,
                            sent2[2], sent1[2])
                pairedsentences.append(pair)

    # Shuffle if necessary, then write out results
    if shufflepairs:
        shuffle(pairedsentences)
    with open(outputname, 'wb') as outfile:
        outwriter = csv.writer(outfile)
        outwriter.writerows(pairedsentences)


# ******************************** Lexicon functions **************************

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


def importValenceDicts(valencedir):
    """Read in the various lexica; should all be in the same folder."""
    import pickle
    # 1. Lexicoder
    with open(valencedir + 'LexicoderDictionary.pkl', 'rb') as LSDpickle:
        lexi = pickle.load(LSDpickle)
    # 2. SO-CAL
    with open(valencedir + 'SO-CAL.pkl', 'rb') as socalpickle:
        socal = pickle.load(socalpickle)
    # 3. Bing Liu
    with open(valencedir + 'BingLiuDictionary.pkl', 'rb') as liupickle:
        liu = pickle.load(liupickle)
    # 4. WordStat
    with open(valencedir + 'WordStatDictionary.pkl', 'rb') as WSpickle:
        WS = pickle.load(WSpickle)
    # 5. labMT: will need to center on 0, and delete values between -1 and +1
    with open(valencedir + 'labMT.pkl', 'rb') as labMTpickle:
        labMT = pickle.load(labMTpickle)
    # 6. NRC emotions
    with open(valencedir + 'NRCdictionary.pkl', 'rb') as NRCpickle:
        NRC = pickle.load(NRCpickle)
    # 7. SentiWordNet
    with open(valencedir + 'SWNdictionary.pkl', 'rb') as SWNpickle:
        SWN = pickle.load(SWNpickle)
    return lexi, socal, liu, WS, labMT, NRC, SWN[0]


def recenter_lexicon(adict, dval):
    """Add dval to every entry in adict."""
    return {key: val + dval for key, val in adict.iteritems()}


def lexiconmatch_multi(word, lexiconlist):
    """Run lexiconmatch on a list of lexica."""
    return [lexiconmatch(word, l) for l in lexiconlist]


def lexiconmatch(word, lexicon):
    """Return 0 if word not in lexicon; lexicon valence otherwise."""
    return lexicon[word] if word in lexicon else 0


def lexiconmatch_multiwild(word, lexiconlist):
    """Run lexiconmatch_wildcard on a list of lexica."""
    return [lexiconmatch_wildcard(word, l) for l in lexiconlist]


def lexiconmatch_wildcard(word, lexicon):
    """Return 0 if word not in lexicon; lexicon valence otherwise.

    Note: accepts * as a wildcard for '0 or more letters'.
    """
    if word in lexicon:
        return lexicon[word]
    else:
        word += '*'
        while len(word) > 2:
            if word in lexicon:
                return lexicon[word]
            else:
                word = word[:-2] + '*'
    return 0


def LIWCmatch(word, lexicon, nrcats):
    """Calculate category memberships for a word in a LIWC-style lexicon."""
    valencelist = [0,] * nrcats
    wordscaptured = 0
    lexmatch = lexiconmatch_wildcard(word, lexicon[0])
    if lexmatch != 0:
        wordscaptured += 1
        for x in lexmatch:
            valencelist[lexicon[1][x]] = 1
    return [wordscaptured,] + valencelist
