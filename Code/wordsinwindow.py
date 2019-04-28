# encoding: utf-8

# wordsinwindow.py
# by Maurits van der Veen
# last modified 2017-07-25

# Functions to find single- or multi-word strings, possibly with wildcards,
# in a text, and return:
# - the number of occurrences of the strings
# - the text within and without a specified window around the strings


# TODO: make handling less convoluted:
# - return indices of initial find (startindex-endindex), instead of finding, then finding again to get locs,
#   then seeing how many words are at each loc


def wordsin2windows(atext, keywords1, keywords2, window, option,
                    capsmatter=False, skipperiods=True,
                    includekeys=True, returnval='text'):
    """Find text within window(s) using 2 keyword lists.

    Option = 0: intersection (1 and 2)
           = 1: exclusion (1 but not 2)
           = 2: xor (1 or 2 but not both)
    Return text within window(s) as well as text without window(s).
    Return either text or word lists.
    """

    atext_orig = atext
    if not capsmatter:
        atext = atext.lower()
        keywords1 = [term.lower() for term in keywords1]
        keywords2 = [term.lower() for term in keywords2]

    wordlist = atext.split()
    periodindices = []
    if skipperiods:
        for counter, word in enumerate(wordlist):
            if word == '.':
                periodindices.append(counter)

    indices1 = findindices(wordlist, keywords1, skipperiods=skipperiods)
    indices2 = findindices(wordlist, keywords2, skipperiods=skipperiods)

    targetinds1 = findtargets(wordlist, keywords1, indices1, window,
                              includekeys, periodindices)
    targetinds2 = findtargets(wordlist, keywords2, indices2, window,
                              includekeys, periodindices)

    if option == 0:
        targetinds = [ind for ind in targetinds1 if ind in targetinds2]
    elif option == 1:
        targetinds = [ind for ind in targetinds1 if ind not in targetinds2]
    else:  # option == 2
        targetinds = sorted([ind for ind in targetinds1 if ind not in targetinds2] + \
                            [ind for ind in targetinds2 if ind not in targetinds1])
    return getwords(atext_orig.split(), targetinds, returnval=='text', (len(indices1), len(indices2)))


def match_specs(atext, keywords1, keywords2,
                avoiddistance, includewindow,
                capsmatter=False, jointype='merge', joinstring=' ... ',
                skipperiods=True, includekeys=True):
    """Find text within window(s) using 2 keyword lists & distance specs.

    Generally similar functionality to wordsin2windows, but handles avoid phrases
    differently (don't subtract window locations). Instead, can specify acceptable
    distance for key- and avoid-phrases.

    In addition, can specify window in word terms as well as in sentence terms.

    Currently, includekeys is ignored, but see findtargets for how to use it if False
    """
    import numpy as np

    # Handle capitalization as appropriate
    atext_orig = atext
    if not capsmatter:
        atext = atext.lower()
        keywords1 = [term.lower() for term in keywords1]
        keywords2 = [term.lower() for term in keywords2]

    # Collect locations of periods (sentence separators)
    wordlist = atext.split()
    periodindices = []
    if skipperiods:
        for counter, word in enumerate(wordlist):
            if word == '.':
                periodindices.append(counter)

    # Collect locations that match search- and avoid-specs
    indices1 = findindices(wordlist, keywords1, skipperiods=skipperiods)
    indices2 = findindices(wordlist, keywords2, skipperiods=skipperiods)

    # See which indices are within acceptable distance of avoid-locations
    if len(indices2) > 0:
        avoidsents = avoiddistance[0] == 'sentences'
        indiceskept = []
        for index1 in indices1:
            rejectindex = False
            for index2 in indices2:
                if index2 < index1:
                    distance = sentencedistance(index2, index1, periodindices) if avoidsents \
                        else worddistance(index2, index1, periodindices)
                    if  distance <= avoiddistance[1]:
                        rejectindex = True
                        break  # break out of inner for loop
                else:
                    distance = sentencedistance(index1, index2, periodindices) if avoidsents \
                        else worddistance(index1, index2, periodindices)
                    if  distance <= avoiddistance[2]:
                        rejectindex = True
                        break
            if not rejectindex:
                indiceskept.append(index1)
    else:
        indiceskept = indices1

    # collect words/sentences surrounding the remaining indices
    nrwords = len(wordlist)
    wordarray = np.array(wordlist)
    contexts = []
    for loc in indiceskept:
        if includewindow[0] == 'sentences':
            # find the period representing the beginning of the first sentence to include on the left,
            leftedge = 0 if includewindow[1] == -1 else \
                       findperiod(loc, -1 - includewindow[1], periodindices, nrwords)
            rightedge = nrwords if includewindow[2] == -1 else \
                    findperiod(loc, includewindow[2], periodindices, nrwords)
            indices = range(leftedge, rightedge)
        else:  # includewindow[0] == 'words'
            indices = findtargets(wordlist, keywords1, [loc,],
                                  (nrwords if includewindow[1] == -1 else includewindow[1],
                                   nrwords if includewindow[2] == -1 else includewindow[2]),
                                  includekeys=True, periodindices=periodindices)
        # Append context.
        contexts.append(indices)

    # Handle potential overlap as specified
    subtexts = []
    if jointype == 'full' or len(contexts) <= 1:  # use all sets of indices, even if overlap
        for context in contexts:
            subtexts.append(' '.join(wordarray[context]))
    else:  # jointype = 'merge' -> combine overlapping ranges
        contexts.sort()  # sort in order from left to right
        # sometimes we end up with empty contexts somehow
        while len(contexts) > 1 and len(contexts[0]) == 0:
            contexts = contexts[1:]
        curcontext = contexts[0]
        if len(curcontext) > 0:  # at least 1 non-empty contexts
            if len(contexts) > 1:
                for context in contexts[1:]:
                    if context[0] <= curcontext[-1]:
                        curcontext = range(curcontext[0], max(context[-1], curcontext[-1]))
                    else:
                        subtexts.append(' '.join(wordarray[curcontext]))
                        curcontext = context
        # flush last context
        subtexts.append(' '.join(wordarray[curcontext]))
    return joinstring.join(subtexts), len(indiceskept)


def findperiod(loc, sentencedist, periodindices, nrwords):
    """Find the beginning or ending of a sentence sentencedist away from loc.

    Assumes periodindices is sorted.
    """
    # find first period after loc
    firstafter = len(periodindices)
    for counter, period in enumerate(periodindices):
        if period > loc:
            firstafter = counter
            break
    targetperiod = firstafter + sentencedist
    if sentencedist < 0:  # look left
        return 0 if targetperiod < 0 \
            else periodindices[targetperiod] + 1  # no need for period itself
    else:  # look right
        return nrwords if targetperiod >= len(periodindices) \
            else periodindices[targetperiod]


def sentencedistance(key1, key2, periodindices):
    """Figure out how many sentences apart index1 and index2 are in a text.

    key1 must be <= key2.
    """
    nrsentences = len(periodindices)
    periodnr = 0
    startsentence = 0

    while periodnr < nrsentences and key1 > periodindices[periodnr]:
        periodnr += 1
        startsentence += 1
    endsentence = startsentence
    while periodnr < nrsentences and key2 > periodindices[periodnr]:
        periodnr += 1
        endsentence += 1
    return endsentence - startsentence


def worddistance(key1, key2, periodindices, skipperiods):
    """Figure out how many words apart index1 and index2 are.

    key1 must be <= key2.
    """
    if not skipperiods:
        return key2 - key1

    worddist = key2 - key1
    for period in periodindices:
        if period >= key1 and period <= key2:
            worddist -= 1
    return worddist


def wordsinwindow_indexfilter(atext, keywords1, targetinds2, window, option=0,
                              capsmatter=False, skipperiods=True,
                              includekeys=True, returnval='text'):
    """Find text within window using keyword list (1) and list of indices to filter on (2).

    Option = 0: intersection (1 and 2)
           = 1: exclusion (1 but not 2)
           = 2: xor (1 or 2 but not both)
    Return text within window(s) as well as text without window(s).
    Return either text or word lists.

    Can be used to get intersections of word lists without having to re-run
    the index-finding part of the function
    """

    atext_orig = atext
    if not capsmatter:
        atext = atext.lower()
        keywords1 = [term.lower() for term in keywords1]

    wordlist = atext.split()
    periodindices = []
    if skipperiods:
        for counter, word in enumerate(wordlist):
            if word == '.':
                periodindices.append(counter)

    indices1 = findindices(wordlist, keywords1, skipperiods=skipperiods)
    targetinds1 = findtargets(wordlist, keywords1, indices1, window,
                              includekeys, periodindices)

    if option == 0:
        targetinds = [ind for ind in targetinds1 if ind in targetinds2]
    elif option == 1:
        targetinds = [ind for ind in targetinds1 if ind not in targetinds2]
    else:  # option == 2
        targetinds = sorted([ind for ind in targetinds1 if ind not in targetinds2] + \
                            [ind for ind in targetinds2 if ind not in targetinds1])
    return getwords(atext_orig.split(), targetinds, returnval=='text', len(indices1))


def wordsinwindow(atext, keywords, window,
                  capsmatter=False, skipperiods=True,
                  includekeys=True, returnval='text'):
    """Filter text by returning only words in window on either side of keyword.

    Return text within window(s) as well as text without window(s).
    Return either text or word lists.
    """

    atext_orig = atext
    if not capsmatter:
        atext = atext.lower()
        keywords = [term.lower() for term in keywords]

    wordlist = atext.split()
    periodindices = []
    if skipperiods:
        for counter, word in enumerate(wordlist):
            if word == '.':
                periodindices.append(counter)

    indices = findindices(wordlist, keywords, skipperiods=skipperiods)
    targetinds = findtargets(wordlist, keywords, indices, window,
                             includekeys, periodindices)
    if returnval == 'indices':
        return len(indices), targetinds
    else:
        return getwords(atext_orig.split(), targetinds, returnval=='text', len(indices))


def indicesinwindow(atext, keywords, window, indices,
                    capsmatter=False, skipperiods=True,
                    includekeys=True, returntext=True):
    """Filter text by returning only words in window on either side of keyword.

    Return text within window(s) as well as text without window(s).
    Return either text or word lists.
    """

    atext_orig = atext
    if not capsmatter:
        atext = atext.lower()
        keywords = [term.lower() for term in keywords]

    wordlist = atext.split()
    periodindices = []
    if skipperiods:
        for counter, word in enumerate(wordlist):
            if word == '.':
                periodindices.append(counter)

    targetinds = findtargets(wordlist, keywords, indices, window,
                             includekeys, periodindices)
    return getwords(atext_orig.split(), targetinds, returntext, len(indices))


def getwords(wordlist, targetinds, returntext, nrfound):
    """Return the words at indices in targetinds."""
    import numpy as np

    mask = np.ones(len(wordlist), dtype=bool)
    mask[targetinds] = False
    wordarray = np.array(wordlist)
    newlist = wordarray[targetinds]
    therest = wordarray[mask]
    if returntext:
        return ' '.join(newlist), ' '.join(therest), nrfound
    else:
        return newlist, therest, nrfound


def findtargets(wordlist, keywords, indices, window,
                includekeys=True, periodindices=()):
    """Return word indices matching the window specification."""

    indexlens = indexlengths(wordlist, keywords, indices)
    targetinds = expandindices(len(wordlist), indices, indexlens, window, periodindices)
    if not includekeys:  # strip out key words themselves
        indexranges = [range(x, x + y) for x, y in zip(indices, indexlens)]
        indicestostrip = set([x for indexrange in indexranges for x in indexrange])
        targetinds = [x for x in targetinds if x not in indicestostrip]
    return targetinds


def findindex(strindex, indexlist):
    """Given a key index and a list of indices,
    return the position of the last index smaller than the key.
    """
    previndex = 0
    for indexnr, index in enumerate(indexlist):
        if previndex <= strindex and strindex < index:
            return indexnr
        previndex = index
    return len(indexlist)


def adjustindices(indexlist, adjustlocs):
    """Given a list of indices and a list of locations skipped in generating
    that list, adjust the indices to account for the skipped locations.
    """
    for adjustloc in adjustlocs:
        indexlist = [x + 1 if x >= adjustloc else x for x in indexlist]
    return indexlist


def textbegins(text, start):
    return text[:len(start)] == start


def indexlengths(wordlist, keys, indices, wild='*', capsmatter=False):
    """Return the length (in words) of each key at the indices found.

    Assume each index does represent at least one key match.
    """
    import re

    # If no multi-word phrases, simply return 1s
    if not any(' ' in key for key in keys):
        return [1 for ind in indices]

    # remove capitalization
    if not capsmatter:
        wordlist = [w.lower() for w in wordlist]
        keys = [key.lower() for key in keys]

    # convert wildcard keys to regex version
    # accept alphanumerics and/or hyphens for wildcard
    rekeys = [re.sub('\\' + wild, '[\w-]*', key) for key in keys]
    # keys (with wildcard) must be bounded by word boundaries
    rekeys = ['\\b' + key + '\\b' for key in rekeys]

    keylens = []
    for index in indices:
        firstwordlen = len(wordlist[index])
        resttext = ' '.join(wordlist[index:])
        for key in rekeys:
            firstmatch = re.search(key, resttext)
            if firstmatch is not None and firstmatch.start() < firstwordlen:
                # found match
                keylens.append(len(key.split()))
    return keylens


def indexlengths_old(wordlist, keys, indices):
    """Return the length (in words) of each key at the indices found.

    Currently finds phrases with spaces or hyphens on either end.
    TODO: generalize to find with parentheses and quotations on either end too.
    """

    lengths = []
    keys_nowild = [key[:-1] if key[-1] == '*' else key for key in keys]
    keys_nospaces = [key for key in keys_nowild if ' ' not in key]
    keys_spaces = [key for key in keys_nowild if ' ' in key]

    # See if no multi-word phrases
    if len(keys_spaces) == 0:
        return [1 for ind in indices]

    # See which single-word keys match
    multindices = []
    for index in indices:
        resttext = ' '.join(wordlist[index:])
        posthyphen = 0
        hyphenfound = resttext.find('-')
        spacefound = resttext.find(' ')
        if hyphenfound != -1 and hyphenfound < spacefound:
            posthyphen = hyphenfound + 1
        singlematch = False
        for key in keys_nospaces:
            if textbegins(resttext, key) or textbegins(resttext[posthyphen:], key):
                singlematch = True
                break
        if not singlematch:
           multindices.append(index)
    if len(multindices) == 0:
        return [1 for ind in indices]

    # See which multi-word keys match
    multindexlens = {}
    for index in multindices:
        resttext = ' '.join(wordlist[index:])
        posthyphen = 0
        hyphenfound = resttext.find('-')
        spacefound = resttext.find(' ')
        if hyphenfound != -1 and hyphenfound < spacefound:
            posthyphen = hyphenfound + 1
        bestval = 99
        for key in keys_spaces:
            if textbegins(resttext, key) or textbegins(resttext[posthyphen:], key):
                matchval = len(key.split())
                if matchval < bestval:
                    bestval = matchval
        periodindex = 1
        while bestval == 99:  # must be a period in there
            tried = False
            for key in keys_spaces:
                keywords = key.split()
                if len(keywords) > periodindex:
                    tried = True
                    keyperiod = ' '.join(keywords[:periodindex]) + ' . ' + ' '.join(keywords[periodindex:])
                    if textbegins(resttext, keyperiod) or textbegins(resttext[posthyphen:], keyperiod):
                        matchval = len(keywords) + 1
                        if matchval < bestval:
                            bestval = matchval
            periodindex += 1
            if not tried:
                print "Error - no match found for index -> ignoring this location."
                print resttext[:100]
                bestval = 0
                break
        multindexlens[index] = bestval
    return [multindexlens.get(index, 1) for index in indices]


def findindices(wordlist, keys, wild='*', skipperiods=True, capsmatter=True):
    """Get indices of any keywords/phrases or stems thereof in the word list."""

    if len(keys) > 0:
        foundlocs, spaces, periodindices = \
            findlocs(wordlist, keys, wild=wild,
                     skipperiods=skipperiods, capsmatter=capsmatter)

        # find word indices for each match; eliminate duplicates by making a set
        indices = list(set([findindex(x, spaces) for x in foundlocs]))
        # adjust word indices for periods skipped
        if skipperiods:
            indices = adjustindices(indices, periodindices)
        return sorted(indices)
    return []


def keypresent(text, keys, wild='*', skipperiods=True, capsmatter=True):
    """See if any keywords/phrases or stems thereof are present in a text."""
    import re

    # remove capitalization
    if not capsmatter:
        text = text.lower()
        keys = [key.lower() for key in keys]
    # remove periods, so they're not counted as words
    # (also remove additional space -- don't do this if don't want to match across sentence break)
    if skipperiods:
        text = text.replace('. ', '')
        text = text.replace('.', '')

    # convert wildcard character to regex version
    rekeys = [re.sub('\\' + wild, '[\w-]*', key) for key in keys]
    rekeys = ['\\b' + key + '\\b' for key in rekeys]

    for keyw in rekeys:
        if re.search(keyw, text) != None:
            return True
    return False


def findlocs(wordlist, keys, wild='*', skipperiods=True, capsmatter=True):
    """Find locations of any keywords/phrases or stems thereof in a split text.

    Ignore periods if applicable (this might make us find multi-word strings
    across sentence boundaries; may need to think about desirability of doing so)
    """
    import re

    # remove capitalization
    if not capsmatter:
        wordlist = [w.lower() for w in wordlist]
        keys = [key.lower() for key in keys]
    # remove periods, so they're not counted as words
    periodindices = []
    if skipperiods:
        for counter, word in enumerate(wordlist):
            if word == '.':
                periodindices.append(counter)
        wordlist = [w for w in wordlist if w != '.']
    # reconstitute text, minus periods (if applicable)
    textstring = ' '.join(wordlist)
    spaces = [m.start() for m in re.finditer(' ', textstring)]

    # convert wildcard character to regex version
    # accept alphanumerics and/or hyphens for wildcard
    rekeys = [re.sub('\\' + wild, '[\w-]*', key) for key in keys]
    # keys (with wildcard) must be bounded by word boundaries
    rekeys = ['\\b' + key + '\\b' for key in rekeys]

    # Actual index search
    foundlocs = [m.start() for keyw in rekeys \
                 for m in re.finditer(keyw, textstring)]
    return foundlocs, spaces, periodindices


def plusminuswindow(locL, locR, window, periodindices):
    """return all indices in a window around n."""

    if locL == locR:  # error case
        return []
    try:  # assume it's a 2-entry tuple/list
        leftmost = locL - window[0]
        rightmost = locR + window[1]
    except TypeError:
        if is_int(window):  # same window-size in both directions
            leftmost = locL - window
            rightmost = locR + window
        else:
            print "Error: window must be integer or iterable"
            return range(locL, locR)
    if len(periodindices) == 0:
        return range(leftmost, rightmost)
    # Adjust for period locations
    for adjindex in periodindices:
        if leftmost <= adjindex and adjindex < locL:
            leftmost -= 1
        if locR < adjindex and adjindex < rightmost:
            rightmost += 1
    return range(leftmost, rightmost)


def expandindices(n, indices, indexlens, window, periodindices):
    """Expand list of indices by including window in either direction."""
    # expand every index by window
    allexpanded = [plusminuswindow(x, x + y, window, periodindices) \
                   for x, y  in zip(indices, indexlens)]
    # flatten sublists into single list
    allflattened = [x for sublist in allexpanded for x in sublist if x >= 0 and x < n]
    # eliminate duplicates by making set, then make list again and sort
    return sorted(list(set(allflattened)))


def is_int(aParam):
    """See if a parameter is representable as an integer."""
    try:
        int(aParam)
    except ValueError:
        return False
    else:
        return True


# Old code not using regular expressions; a bit too cumbersome

def findindices_old(wordlist, keywords, stems=False):
    """Get indices of any keywords/phrases or stems thereof in the word list."""
    indices = []
    if any(' ' in w for w in keywords):  # dealing with multi-word terms
        firstwords = [w.split()[0] for w in keywords]
        nextwords = [' '.join(w.split()[1:]) for w in keywords]
        nrwords = [len(w.split()) for w in keywords]
        for counter, word in enumerate(wordlist):
            for keynr, firstword in enumerate(firstwords):
                if firstword == word:
                    if nrwords[keynr] > 1: # check if remaining words in key term are there too
                        if stems:
                            if nextwords[keynr] == ' '.join(wordlist[counter + 1:])[:len(nextwords[keynr])]:
                                indices.append(counter)
                                break
                        elif nextwords[keynr] == ' '.join(wordlist[counter + 1: counter + nrwords[keynr]]):
                            indices.append(counter)
                            break
                    else:
                        indices.append(counter)
                        break  # exit this for loop
                elif stems and nrwords[keynr] == 1 and firstword == word[:len(firstword)]:
                    indices.append(counter)
                    break
    else:
        for counter, word in enumerate(wordlist):
            if stems:
                if any([keyword == word[:len(keyword)] for keyword in keywords]):
                    indices.append(counter)
            else:
                if any([keyword == word for keyword in keywords]):
                    indices.append(counter)
    return indices
