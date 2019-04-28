# coding=utf-8

# translate.py
# Program to translate texts from British dialects of English (and related ones)
# to American English.

# Used to make texts more directly comparable across dialects, and to ensure
# that stemming, sentiment analysis, and other functions work the same way.

# Also permits the direct replacement of words or phrases.
# This is useful, for example, to make sure transliterations all become the same
# (e.g. Beijing & Peking should be the same; this we can specify in our replacement file).

# by A. Maurits van der Veen
# Last modified: 2017-05-23




# Note: The British-to-American dictionary 'UK-to-US.csv' uses the list at
# http://www.tysto.com/uk-us-spelling-list.html, purged of all
# -is, -lys, and -our forms that are handled separately
# (this reduction is done in reduce_BAdict).
# As a result, we only need to look at 541 of the original 1800-plus entries.


# ***************************************************************************

def translate_text (infile, outfile, textcol=1, keepcols=(),
                    replacefile='', dialectfile='',
                    translationfunction=None):
    """Translate text as specified, substituting individual words/phrases.

    Two types of word substitutions:
    - make corpus-specific substitutions as specified in replacefile.
    - translate across dialects, using dialectfile & translationfunction

    Note: output file contains only id, translated text, and other columns specified in keepcols
    """
    import csv

    # Load corpus-specific & dialect-specific substitution dictionaries
    repDict, repKeys, dialectDict, dialectKeys = \
            load_replace_patterns(replacefile, dialectfile)

    # Process each text in infile
    with open(infile,'rU') as textFile, open(outfile, 'wb') as preppedFile:
        allTexts = csv.reader(textFile)
        preppedTexts = csv.writer(preppedFile)

        for count, row in enumerate(allTexts):
            preppedText = translate_1text(row[textcol], repDict, repKeys,
                                          dialectDict, dialectKeys)
            preppedTexts.writerow([row[0], preppedText] + [row[x] for x in keepcols])
            if count % 25000 == 0:
                print count
    return


def load_replace_patterns(replacefile='', dialectfile=''):
    """Load corpus-specific & dialect-specific substitution dictionaries.

    For dialect-specific substitution insist on word boundaries
    For general substitution we do too, but perhaps should make it optional.
    """
    from os.path import isfile
    import re

    repDict = getReplaceDict(replacefile) if isfile(replacefile) else {}
    # repKeys = re.compile('|'.join(repDict.keys()), flags=re.IGNORECASE)
    repKeys = re.compile('\\b(' + '|'.join(repDict.keys()) + ')\\b',
                             flags=re.IGNORECASE)

    dialectDict = getReplaceDict(dialectfile) if isfile(dialectfile) else {}
    dialectKeys = re.compile('\\b(' + '|'.join(dialectDict.keys()) + ')\\b',
                             flags=re.IGNORECASE)
    return repDict, repKeys, dialectDict, dialectKeys


def translate_1text_only(text, replacefile='', dialectfile='',
                         translationfunction=None):
    """Translate 1 text, with optional dialect/keyword replacements."""

    # Load corpus-specific & dialect-specific substitution dictionaries
    repDict, repKeys, dialectDict, dialectKeys = \
            load_replace_patterns(replacefile, dialectfile)
    return translate_1text(text, repDict, repKeys, dialectDict, dialectKeys,
                           translationfunction)


def translate_1text(text, repDict, repKeys, dialectDict, dialectKeys,
                    translationfunction=None):
    """Translate 1 text, with optional dialect/keyword replacements."""
    if translationfunction is not None:
        text = translationfunction(text)
    if len(dialectDict) > 0:
        text = replaceTerms(text, dialectDict, dialectKeys)
    if len(repDict) > 0:
        text = replaceTerms(text, repDict, repKeys)
    return text


# *************** Translate from British to American English ************

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
    TODO: check check against http://www.wordbyletter.com/suffixe.php for -ise/-ize words
    """

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

    origword = matchobj.group(0)
    word = origword.lower()

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


# *************** Auxiliary functions (manage translation dictionaries) ************

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


def writeReplaceDict(adict, outfilename):
    """Write a dictionary conversion table to a csv file."""
    import csv
    with open(outfilename, 'wt') as dictFile:
        dictData = csv.writer(dictFile)
        dictData.writerows(sorted(adict.items()))


def reduce_BAdict(filename):
    """Reduce the size of a translation dictionary.

    Auxiliary function: given a dictionary of translations to perform,
    remove -ise/ize, -lyse/lyze and -our/or from dictionary.
    """
    thedict = getReplaceDict(filename)
    newdict = {}
    for Bword, Aword in thedict.iteritems():
        if verbBtoA(Bword) ==  Bword and our_to_or(Bword) == Bword:
            newdict[Bword] = Aword
    writeReplaceDict(newdict,
                     '.'.join(filename.split('.')[:-1]) + '_reduced.csv')





