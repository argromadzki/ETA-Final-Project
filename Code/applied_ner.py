# encoding: utf-8

# by Maurits van der Veen
# last modified 2018-02-08

# Code adapted from geocoder.py


def extractNEs(theText, entity_types=('PERSON',),
               useStanford=True, useMIT=True, MITloc='/Users/maurits/STAIR/Code/',
               windowsize=(0,0), stripdups=False, report=False):
    """Extract named entities from a text. Use Stanford NER and/or MIT IE extractors.

    May add additional NER options later (e.g. Leetaru).

    Merge the results from multiple NERs

    If stripdups == True, return the set union of the results, to strip duplicates
    Otherwise simply return all values returned.

    If windowsize != (0,0) return words on either side. This requires re-finding person in text.

    Entity types accepted by both: PERSON, LOCATION, ORGANIZATION, MISC (not tested the latter for NER)

    Note 1. These classifiers may split up person names (first name, last name) -> find a way to handle
    Note 2. Play around with different classifiers for each (see -loadClassifier command for Stanford NER below)
            Eventually, may want to train our own
    """

    import sys
    sys.path.append(MITloc + 'MITIE-master/mitielib/')

    if useStanford:
        import ner
    if useMIT:
        import mitie

    NEs = []

    # Use MIT IE tokenizer (could also use punctuation.py)
    if useMIT:
        tokens = mitie.tokenize(theText)
    else:
        tokens = theText.split()   # this is NOT ideal! use tokenizer from punctuation.py!
    nrtokens = len(tokens)

    # First, use MIT IE, which tends to do a better & more complete job
    if useMIT:
        tagger = mitie.named_entity_extractor(MITloc + 'MITIE-models/english/ner_model.dat')
        entities = tagger.extract_entities(tokens)
        for e in entities:
            if e[1] in entity_types:
                entity_info = ([x for x in e[0]], " ".join(tokens[i] for i in e[0]))
                if windowsize != (0,0):  # find window of tokens on either side
                    locstart = e[0][0]
                    locend = e[0][-1]
                    windowcontext = (' '.join(tokens[i] \
                                              for i in xrange(max(0, locstart-windowsize[0]), locstart)),
                                     ' '.join(tokens[i] \
                                              for i in xrange(min(nrtokens, locend + 1),
                                                              min(nrtokens, locend + 1 + windowsize[1]))))
                else:
                    windowcontext = ('', '')
                NEs.append(entity_info + windowcontext)

    # Sort entities by token index
    NEs = sorted(NEs, key=lambda x: x[0][0])
    firstindices = [x[0][0] for x in NEs]

    # Next, use Stanford NER; must make sure a server is running.
    # From command line in Terminal, issue command:

    # java -mx1500m -cp Applications/stanford-ner-2016-10-31/stanford-ner.jar edu.stanford.nlp.ie.NERServer
    #   -loadClassifier Applications/stanford-ner-2016-10-31/classifiers/english.all.3class.distsim.crf.ser.gz
    #   -port 8080 -outputFormat inlineXML

    # Then keep that window (and thus the server) open

    if useStanford:  # doesn't usually add much, but use just in case
        tagger = ner.SocketNER(host='localhost', port=8080)
        entities = tagger.get_entities(theText)

        for entity_type in entity_types:

            NEs1 = entities.get(entity_type, [])

            # If special characters in entity ame, may need to do some encoding/decoding
            # NEs1 = [l.encode('utf-8') for l in list(set(NEs1[entity_type,]))]

            # May need to refine code to handle all possible contingencies for multi-word NEs
            lastfind = -1
            for NE in NEs1:
                for counter, token in enumerate(tokens):
                    # skip past previous finds <- assumes NEs within type appear in order!!
                    if counter <= lastfind:
                        continue
                    if token == NE.split()[0]:  # split to handle multi-word entities
                        if counter not in firstindices:  # new find
                            firstindices.append(counter)
                            endNE = counter + len(NE.split())
                            entity_info = (range(counter, endNE), NE)
                            if windowsize != (0, 0):  # find window of tokens on either side
                                windowcontext = (' '.join(tokens[i] \
                                                          for i in xrange(max(0, counter - windowsize[0]), counter)),
                                                 ' '.join(tokens[i] \
                                                          for i in xrange(min(nrtokens, endNE),
                                                                          min(nrtokens, endNE + windowsize[1]))))
                            else:
                                windowcontext = ('', '')
                            NEs.append(entity_info + windowcontext)

    # Clean up and remove duplicate entries
    newNEs = []
    NElocs = []
    prevNE = []
    NEstrings = []
    for NE in sorted(NEs, key=lambda x: x[0][0]):
        # print NE[0]
        # print NE[0][0] not in NElocs, not subset(NE[0], prevNE), (not stripdups or NE[1] not in NEstrings)
        if NE[0][0] not in NElocs and \
                not subset(NE[0], prevNE) and \
                (not stripdups or NE[1] not in NEstrings):
            NElocs.append(NE[0][0])
            # print prevNE, '***', NE[0]
            prevNE = NE[0]
            NEstrings.append(NE[1])
            newNEs.append(NE)

    if report:  # Print results
        print "Results, listed by location in overall text (identified NE is marked by double asterisks)\n"
        for NE in newNEs:
            print NE[0][0], NE[2], '**' + NE[1] + '**', NE[3]
    return newNEs


def subset(list1, list2):
    """See if all entries in list1 are also in list2."""
    return all(x in list2 for x in list1)


def delete_substrings(alist):
    """Delete all entries in alist that are substrings of another entry.

    Note: comparison is caps-sensitive.
    """
    todelete = []
    for item in alist:
        if any(item in otheritem and len(otheritem) > len(item) for otheritem in alist):
            todelete.append(item)
    return [x for x in alist if x not in todelete]


def filterNEs(NElist, keywords, capsmatter=False):
    """Return only named entities whose context contains one of the keywords."""
    filtered = []
    for NE in NElist:
        context = NE[2] + ' ' + NE[3]
        if not capsmatter:
            context = context.lower()
        contextlist = context.split()
        if any([x in contextlist for x in keywords]):
            filtered.append(NE)
    return filtered


def find_filteredNEs(inputfile, outputfile, keywords,
                     entity_types=('PERSON',), windowsize=(5,5),
                     useStanford=True, useMIT=True, MITloc='',
                     report=False, header=True, textcol=1,
                     append=False, partcol=13):
    """Find named entities meeting some specific keyword criteria.

    Used for finding celebrities in texts, for example.

    That is a slow process, so allow re-start, based on last id & part reached
    (if no part, just set partcol to 0)
    """
    import csv
    import os

    # If we need to continue where we left off, find out where that is
    if append:
        if not os.path.isfile(outputfile):
            started = True
        else:
            with open(outputfile, 'rU') as outf:
                for celeb_article in csv.reader(outf):
                    if len(celeb_article) < partcol + 1:  # incomplete line
                        break
                    lastarticle_id = celeb_article[0]
                    lastarticle_part = celeb_article[partcol]
            started = False
    else:
        started = True

    with open(inputfile, 'rU') as inf, \
            open(outputfile, 'ab' if (append and os.path.isfile(outputfile)) else 'wb') as outf:
        inreader = csv.reader(inf)
        outwriter = csv.writer(outf)

        for counter, article in enumerate(inreader):
            if header and started:
                header = False
                outwriter.writerow(article + ['targets',])
            else:
                if append and not started:
                    if article[0] == lastarticle_id and article[partcol] == lastarticle_part:
                        started = True  # start after this one
                    continue
                nes = extractNEs(article[textcol], entity_types=entity_types,
                                 useStanford=useStanford, useMIT=useMIT, MITloc=MITloc,
                                 windowsize=windowsize, stripdups=False, report=False)
                filterednes = filterNEs(nes, keywords)
                if len(filterednes) > 0:
                    # See if longer version of name elsewhere among names, and skip duplicates
                    allnames = [x[1] for x in nes]
                    targetnames = []
                    for ne in filterednes:
                        curname = ne[1]
                        for name in allnames:
                            if curname in name and len(curname) < len(name):
                                curname = name
                        if curname not in targetnames:
                            targetnames.append(curname)
                    namelist = ', '.join(targetnames)
                    if report:
                        print "article id %s: found %s" % (article[0], namelist)
                else:
                    namelist = ''
                outwriter.writerow(article + [namelist,])
            if (counter +1) % 50 == 0:
                print "Processed %d texts" % (counter + 1,)


def tally_NEs(inputfile, targetcol=2, minthres=3):
    """Tally entities found and report only those that are found in at least minthres texts."""
    import csv

    targetdict = {}

    # collect targets
    with open(inputfile, 'rU') as inf:
        for counter, article in enumerate(csv.reader(inf)):
            targets = article[targetcol].split(', ')
            for target in targets:
                if len(target) >  0:  # skip empties
                    if target in targetdict:
                        targetdict[target] += 1
                    else:
                        targetdict[target] = 1

    # filter by minthres
    todelete = []
    for key, value in targetdict.iteritems():
        if value < minthres:
            todelete.append(key)
    for key in todelete:
        del targetdict[key]

    return targetdict


def tally_NEs_multi(inputfiles, colname='targets', minsources=2, minthres=3):
    """Tally entities found across multiple corpora, and keep only those that meet thresholds.

    Minthres = threshold to be met for at least one source.
    Minsources = min. number of sources to include entity.

    Allow for the target column to vary across sources.
    """
    import csv

    combodict = {}

    # collect targets
    for inputfile in inputfiles:
        targetdict = {}
        with open(inputfile, 'rU') as inf:
            inreader = csv.reader(inf)
            headerline = next(inreader)
            targetcol = -1
            for counter, columntitle in enumerate(headerline):
                if columntitle == colname:
                    targetcol = counter
            if targetcol == -1:  # did not find target; instead take last column
                targetcol = len(headerline) -1

            for counter, article in enumerate(inreader):  # read the rest of the file
                targets = article[targetcol].split(', ')
                for target in targets:
                    if len(target) >  0:  # skip empties
                        if target in targetdict:
                            targetdict[target] += 1
                        else:
                            targetdict[target] = 1

        # Now merge with results from other sources
        for target, count in targetdict.iteritems():
            if target in combodict:
                combodict[target].append(count)
            else:
                combodict[target] = [count,]

    # Collected all; filter by thresholds
    todelete = []
    for key, values in combodict.iteritems():
        if len(values) < minsources or max(values) < minthres:
            todelete.append(key)
    for key in todelete:
        del combodict[key]

    # Return combined dictionary
    return combodict


def add_NEdict(combodict, newdict):
    """Add a (pre-filtered) dictionary to the combined dictionary."""
    for key, count in newdict.iteritems():
        if key in combodict:
            combodict[key].append(count)
        else:
            combodict[key] = [count, ]
    return combodict


def write_NEs(outputfile, combodict):
    # Sort combined dictionary in alphabetical order by last word; save as csv
    import csv

    sortedtargets = sorted(combodict.items(), key=lambda x: x[0].split()[-1])
    with open(outputfile, 'wb') as outf:
        outwriter = csv.writer(outf)
        outwriter.writerows(sortedtargets)

