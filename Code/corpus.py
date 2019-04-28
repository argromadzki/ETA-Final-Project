# encoding: utf-8

# corpus.py
# by Maurits van der Veen
# last modified 2017-07-25

# Functions to manage a text corpus:
# - read in from file
# - split into lines
# - split into words

__author__ = 'maurits'


# *************************** notes, thoughts, etc. **************************

# Extension: stream from disk (see gensim), if too big to read in at once

# TODO: organize into sections better and more clearly


# *************************** input/output functions *************************

def split_on_class(filename):
    """Split a corpus into pieces by class."""
    import csv
    outstem = '.'.join(filename.split('.')[:-1]) + '_'
    with open(filename, 'rU') as infile:
        for row in csv.reader(infile):
            with open(outstem + str(int(float(row[2]))) + '.csv', 'ab') as outfile:
                csv.writer(outfile).writerow(row[:2])


def split_on_phrase(inf1, phrase, caps=False):
    """Split a corpus file into two based on text contents."""
    import csv
    import sys
    csv.field_size_limit(1000000000)

    if not caps:
        phrase = phrase.lower()
    withcount, withoutcount = 0, 0
    ids, texts = readidtextfile(inf1)
    outf_stem = '.'.join(inf1.split('.')[:-1])
    with open(outf_stem + '_with_' + phrase + '.csv', 'wb') as outf1, \
            open(outf_stem + '_without_' + phrase + '.csv', 'wb') as outf2:
        out_with = csv.writer(outf1)
        out_without = csv.writer(outf2)
        for id, text in zip(ids, texts):
            if (caps and phrase in text) or (not caps and phrase in text.lower()):
                out_with.writerow((id, text))
                withcount += 1
            else:
                out_without.writerow((id, text))
                withoutcount += 1
    print "Articles with %s: %d; without: %d" % (phrase, withcount, withoutcount)


def split_on_col(infilestem, targetcol, contents, capsmatter=True, header=True):
    """Split a corpus file based on column contents."""
    import csv
    csv.field_size_limit(1000000000)

    if not capsmatter:
        contents = [x.lower() for x in contents]
    counts = [0,] * (len(contents) + 1)

    with open(infilestem + '.csv') as inf:
        inreader = csv.reader(inf)
        if header:
            headerline = next(inreader)

        # Create output files; write header as appropriate
        for outsuffix in contents:
            with open(infilestem + '_' + outsuffix + '.csv', 'wb') as outf:
                outwriter = csv.writer(outf)
                if header:
                    outwriter.writerow(headerline)

        # Run through file, looking for target phrases
        for counter, row in enumerate(inreader):
            if len(row) <= targetcol:
                counts[-1] += 1
            else:
                targettext = row[targetcol]
                if not capsmatter:
                    targettext = targettext.lower()
                foundphrase = False
                for targetcount, targetphrase in enumerate(contents):
                    if targetphrase in targettext:
                        foundphrase = True
                        counts[targetcount] += 1
                        with open(infilestem + '_' + targetphrase + '.csv', 'ab') as outf:
                            csv.writer(outf).writerow(row)
                if not foundphrase:
                    counts[-1] += 1
            if counter % 50000 == 0:
                print "Processing row %d" % counter

    print "Processed %d rows, of which %d did not contain any target phrase" % \
            (counter + 1, counts[-1])
    for tcount, targetphrase in enumerate(contents):
        print "Contained %s: %d" % (targetphrase, counts[tcount])
    return


def split_on_list(terms, termlist):
    """Split a list of terms into two lists: those in termlist and those that are not."""
    inlist, notinlist = [], []
    termset = set(termlist)
    for term in terms:
        if term in termset:
            inlist.append(term)
        else:
            notinlist.append(term)
    return inlist, notinlist


def strip_string(infile, outfile, str):
    """Remove the specified string from every text in the file.

    TODO: offer options re: beginning/end of word, caps-sensitive, etc.
    """
    import csv
    import re
    with open(infile, 'rU') as inf, open(outfile, 'wb') as outf:
        outwriter = csv.writer(outf)
        for row in csv.reader(inf):
            outwriter.writerow((row[0], re.sub(str, '', row[1],
                                               flags=re.IGNORECASE)))


def strip_asterisks(infile, outfile, words, textcols):
    """Remove asterisks in the file around the specified words.

    """
    import csv
    import re

    rekey = r'([*](' + '|'.join(words) + r')[*])'

    with open(infile, 'rU') as inf, open(outfile, 'wb') as outf:
        outwriter = csv.writer(outf)
        for row in csv.reader(inf):
            outwriter.writerow([x if col not in textcols else \
                                    re.sub(rekey, '\\2', x, flags=re.IGNORECASE) \
                                for col, x in enumerate(row)])


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
    return minval, maxval


def combine_textfiles(inflist, outf, id_offset=100000):
    """Combine multiple text files (csv format: <id>,<text>)."""
    import csv
    import sys

    csv.field_size_limit(1000000000)
    with open(outf, 'wb') as outfile:
        outwriter = csv.writer(outfile)
        for nr, inf in enumerate(inflist):
            counter = 0
            with open(inf, 'rU') as infile:
                for row in csv.reader(infile):
                    counter += 1
                    id = int(row[0])
                    if id >= id_offset:
                        print "Error: id exceeds offset"
                    newid = id + nr * id_offset
                    outwriter.writerow([newid, row[1]])
            print "File %s contained %d texts." % (inf, counter)


def combine_textfiles_checkoverlap(inflist, outf):
    """Combine multiple text files (csv format: <id>,<text>)."""
    import csv
    import sys

    csv.field_size_limit(1000000000)
    idlist = []
    with open(outf, 'wb') as outfile:
        outwriter = csv.writer(outfile)
        for nr, inf in enumerate(inflist):
            counter = 0
            with open(inf, 'rU') as infile:
                for row in csv.reader(infile):
                    counter += 1
                    id = int(row[0])
                    if id not in idlist:
                        outwriter.writerow(row)
                        idlist.append(id)
            print "File %s contained %d texts." % (inf, counter)


def merge_texts(infile, merge_n):
    """Combine n consecutive texts in a csv file.

    Generate new ids; save component ids."""
    import csv
    import sys

    csv.field_size_limit(1000000000)
    outfile = infile.split('.')[0] + 'X.csv'
    idsfile = infile.split('.')[0] + '_mergedids.csv'
    with open(infile, 'rb') as inf, open(outfile, 'wb') as outf, \
                                    open(idsfile, 'wb') as idsf:
        outwriter = csv.writer(outf)
        idwriter = csv.writer(idsf)
        ids = []
        id = 0
        text = ''
        for counter, row in enumerate(csv.reader(inf)):
            ids.append(int(row[0]))
            text += row[1] + ' '
            if (merge_n > 100 and len(text) > merge_n) or \
                    (merge_n <= 100 and (counter + 1) % merge_n == 0):
                outwriter.writerow((id, text))
                idwriter.writerow([id,] + ids)
                ids = []
                id += 1
                text = ''


def read_texts(infilespec, outfilename, filename_as_id=True):
    """Read one or more text files into csv.

    Files must have the extension .txt to be read."""
    import csv
    import glob
    import os
    import re
    import sys
    from unidecode import unidecode

    csv.field_size_limit(1000000000)
    infilelist = glob.glob(infilespec)
    with open(outfilename,
              'ab' if os.path.isfile(outfilename) else 'wb') as outfile:
        output = csv.writer(outfile)
        count = 0
        for filename in infilelist:
            if os.path.isdir(filename):
                print 'Entering subdirectory: ' + filename
                read_texts(filename + '/*', outfilename)
            elif filename.split('.')[-1] == 'txt': # textfile to process
                with open(filename, 'rU') as infile:
                    text = re.sub("\n", " ", infile.read())
                    text = unidecode(text.decode('ascii', 'ignore'))
                    id = filename.split('/')[-1].split('.')[:-1] if filename_as_id else count
                    output.writerow((count, text))
                    fname = filename if '/' not in filename else \
                                filename.split('/')[-1]
                    print "File %s into line %d" % (fname, count)
                    count += 1
    return count


def split_texts(infile, split_n):
    """Split csv file entries into pieces (on a space); generate new ids."""
    import csv
    import sys

    csv.field_size_limit(1000000000)
    # outfile = infile.split('.')[0] + '_split' +  str(split_n) + '.csv'
    outfile = infile.split('.')[0] + 'X.csv'
    with open(infile, 'rb') as inf, open(outfile, 'wb') as outf:
        outwriter = csv.writer(outf)
        for counter, row in enumerate(csv.reader(inf)):
            text = row[1].strip()
            idbase = int(row[0]) * 100
            nrofparts = split_n if split_n <= 100 else 1 + len(text)/split_n
            lenofparts = int(len(text)/float(nrofparts))
            for cut in xrange(nrofparts):  # cut at space
                cutpoint = min(lenofparts, len(text) -1)
                while text[cutpoint] != ' ':
                    cutpoint += 1
                    if cutpoint == len(text):
                        break
                textpart = text[:cutpoint]
                outwriter.writerow((idbase + cut, textpart))
                if cutpoint == len(text):
                    break
                text = text[cutpoint+1:]


def readtextfile(filename):
    """Read in text data from file, as separate lines."""
    with open(filename, 'rU') as infile:
        return infile.readlines()


def readidtextfile(filename, item=1):
    """Extract id & text data from file in csv format (id, text)."""
    import csv
    import sys

    csv.field_size_limit(1000000000)
    ids = []
    texts = []
    with open(filename, 'rU') as infile:
        for row in csv.reader(infile):
            if 'id' not in row[0].lower():
                ids.append(int(row[0].strip()))
                texts.append(row[item])
    return ids, texts


def renumber_idtextfile(filename, newfilename, idmapname):
    """Renumber the ids, while keeping a mapping in a separate file."""
    import csv
    import sys

    csv.field_size_limit(1000000000)
    with open(filename, 'rU') as infile, open(newfilename, 'wb') as outfile, \
            open(idmapname, 'wb') as idfile:
        outwriter = csv.writer(outfile)
        idwriter = csv.writer(idfile)
        for counter, row in enumerate(csv.reader(infile)):
            outwriter.writerow((counter + 1, row[1]))
            idwriter.writerow((int(row[0]), counter + 1))
        print 'New id range: 1 - ', counter
    return


def incrementids(filename, newfilename, increment):
    """Renumber the ids."""
    import csv
    import sys

    csv.field_size_limit(1000000000)
    with open(filename, 'rU') as infile, open(newfilename, 'wb') as outfile:
        outwriter = csv.writer(outfile)
        for row in csv.reader(infile):
            outwriter.writerow((int(row[0]) + increment, row[1]))
    return


def readidtextfile_subset(filename, allids):
    """Read text for a subset of ids."""
    import csv
    import sys

    csv.field_size_limit(1000000000)
    alltexts = []
    texts = {}
    with open(filename, 'rU') as infile:
        for row in csv.reader(infile):
            if 'id' not in row[0].lower():
                texts[int(row[0].strip())] = row[1]
    for id in allids:
        alltexts.append(texts.get(id, ''))
    return allids, texts


def extract1col(filename, col=0):
    """Extract 1 column from specified file.

    Default = 0, which will generally be the id.
    Supersedes older readidtextfile_ids and readidtextfile_texts.
    """
    import csv

    csv.field_size_limit(1000000000)
    data = []
    with open(filename, 'rU') as infile:
        for row in csv.reader(infile):
            data.append(row[col].strip())
    return data


def readwords(wordfile):
    """Extract word list from file with 1 word per line."""
    wordlist = []
    with open(wordfile, 'rU') as words:
        for w in words:
            wordlist.append(w.strip())
    return wordlist


def writewords(filename, wordlist):
    """Write words to file, 1 per line."""
    with open(filename, 'wt') as outfile:
        outfile.writelines([w + '\n' for w in wordlist])


def insert_string(infile, outfile, str, level='sentence'):
    """Insert a string at the front of every article/sentence."""
    import csv
    import sys

    csv.field_size_limit(1000000000)
    with open(infile, 'rU') as inf, open(outfile, 'wb') as outf:
        outwriter = csv.writer(outf)
        for row in csv.reader(inf):
            if 'id' not in row[0].lower():  # skip over header row
                if level == 'article':
                    outwriter.writerow((row[0], str + ' ' + row[1]))
                elif level == 'sentence':
                    newarticle = ' . '.join([str + ' ' + x \
                                             for x in row[1].split(' . ')])
                    outwriter.writerow((row[0], newarticle))
                else:
                    print "unknown level value"
                    break
    return


def subsetbyID(infile, outfile, idlist, idcol=0, header=False, include=True):
    """Save a subset of a corpus based on list of ids."""
    import csv

    idlist = set(idlist)
    nrextracted = len(idlist)
    print "Looking for %d texts" % (nrextracted,)
    extracted = []
    with open(infile, 'rU') as inf, open(outfile, 'wb') as outf:
        inreader = csv.reader(inf)
        outwriter = csv.writer(outf)
        if header:
            outwriter.writerow(next(inreader))
        for count, article in enumerate(inreader):
            articleid = article[idcol]
            if (include and articleid in idlist) or \
                    (not include and articleid not in idlist):
                outwriter.writerow(article)
                extracted.append(articleid)
            if (count + 1) % 100000 == 0:
                print "Processed %d texts; found %d" % (count + 1, len(extracted))

        # Check we got everything
        extracted = set(extracted)
        notextracted = idlist - extracted
        if len(extracted) != nrextracted:
            print "Warning: not all ids in idlist found"
            print "Not extracted: %d out of %d possible" % (nrextracted - len(extracted), nrextracted)
            # print notextracted


def splitbyID(itemfile, idfile, item=1):
    """Split ids & items into training & test sets."""
    import numpy as np

    ids, items = readidtextfile(itemfile, item)
    # read test IDs into a list
    with open(idfile, 'rU') as selectIDs:
        testIDs = [int(IDline.strip()) for IDline in selectIDs.readlines()]

    idmask = np.array([True if x in testIDs else False for x in ids])
    idarray = np.array(ids)
    ids1 = idarray[~idmask]
    ids2 = idarray[idmask]  # redundant for content, but not for ordering
    itemarray = np.array(items)
    items1 = itemarray[~idmask]
    items2 = itemarray[idmask]
    return ids1, ids2, items1, items2


def splitonmask(items, idmask):
    """Split items on mask."""
    import numpy as np

    itemarray = np.array(items)
    items1 = itemarray[~idmask]
    items2 = itemarray[idmask]
    return items1, items2


def splituncoded(ids, items, labels):
    """Split uncoded entries from the coded ones."""
    idmask = np.array([True if l == -1 else False for l in labels])
    ids1, ids2 = splitonmask(ids, idmask)
    items1, items2 = splitonmask(items, idmask)
    return ids1, ids2, items1, items2


def tally_vals(filename, header=True, targetcol=2, partcol=13,
               separator=';', printresults=True):
    """Tally different values in the specified column.

    To ignore part-column, set it to -1.
    """
    from operator import itemgetter
    import csv
    csv.field_size_limit(1000000000)

    targetdict = {}
    with open(filename, 'rU') as infile:
        for counter, article in enumerate(csv.reader(infile)):
            if header:
                header = False
            else:
                val = article[targetcol]
                if separator != '':
                    val = val.split(separator)[0]
                if val not in targetdict:
                    targetdict[val] = 1
                else:
                    # skip multiple parts, if called for
                    if partcol == -1 or article[partcol] == '0':
                        targetdict[val] += 1
                if counter % 50000 == 0:
                    print "Processing article %d" % counter
    print "\nNumber of different values: %d; total number of texts: %d" % \
        (len(targetdict), sum(zip(*targetdict.iteritems())[1]))
    if printresults:
        for key, val in sorted(targetdict.iteritems(), key=itemgetter(1), reverse=True):  # sort by frequency
            print "- %s: %d texts" % (key, val)
    return targetdict


def idslost(file1, file2):
    """Simple function to find out which ids are in the first but not the second file.

    Probably more efficient to use sets, but this is simpler.
    """
    import csv

    idsleft, idslost = [], []
    with open(file2, 'rU') as f2:
        for row in csv.reader(f2):
            idsleft.append(row[0])
    idsleft = set(idsleft)
    print "Unique ids in file 2 (possibly including header: %d)" % len(idsleft)
    with open(file1, 'rU') as f1:
        for row in csv.reader(f1):
            if row[0] not in idsleft:
                idslost.append(row[0])
    return idslost


def extract_cols(inputfile, outputfile, keepcols=(0, 1)):
    """Extract specified columns only."""
    import csv
    csv.field_size_limit(1000000000)

    with open(inputfile, 'rU') as inf, open(outputfile, 'wb') as outf:
        outwriter = csv.writer(outf)
        for counter, row in enumerate(csv.reader(inf)):
            outwriter.writerow([row[x] for x in keepcols])
            if (counter +1 ) % 500000 == 0:
                print "Processed %d texts" % (counter + 1,)
    return counter


def extract_cols_byflag(inputfile, outputfile, selectionfile,
                        selectioncol=1, keepcols=(0, 1),
                        header=True, selectionheader=True):
    """Extract specified columns, based on flags.

    Don't pay attention to IDs -- simply go by order in the file.
    """
    import csv
    csv.field_size_limit(1000000000)

    anycols = False
    try:
        selectioncol = int(selectioncol)
    except:
        anycols = True

    includerows = []
    included = 0
    # print "Reading in selection criterion..."
    with open(selectionfile, 'rU') as featuref:
        featuredata = csv.reader(featuref)
        if selectionheader:
            skipheader = next(featuredata)
        for counter, row in enumerate(featuredata):
            if (anycols and len(row) <= max(selectioncol)) or (not anycols and len(row) <= selectioncol):
                print "Error: line %d has does not have enough columns" % (counter,)
            if anycols:  # assume have list of selectioncols
                includerows.append(any([int(row[x]) > 0 for x in selectioncol]))
            else:
                includerows.append(int(row[selectioncol]) > 0)
    with open(inputfile, 'rU') as inf, open(outputfile, 'wb') as outf:
        inreader = csv.reader(inf)
        outwriter = csv.writer(outf)
        if header:
            inheader = next(inreader)
            outwriter.writerow([inheader[x] for x in keepcols])

        for includerow, row in zip(includerows, inreader):
            if includerow:
                outwriter.writerow([row[x] for x in keepcols])
                included += 1
    print "Saved metadata for %d texts." % (included,)
    return included


def extract_column_value(inputcorpus, outputcorpus, header=True,
                          parallelin='', parallelout='', parallelheader=False,
                          valuecol=1, criterion='matchstr', critvalue=0,
                          idsonly=False, include=True):
    """Extract all texts (articles) by applying a criterion the data in a specified column.

    Criterion can be match, minthres, or maxthres.
    For minthres, values are accepted if they are greater than or equal to the threshold specified;
    for maxthres, values are accepted only if they are less than or equal.

    A parallel corpus can be similarly pared down; articles in that corpus must have the same ids.

    See extract_sources_dates for a similar companion function.
    """
    import csv
    csv.field_size_limit(1000000000)

    extractcount = 0
    articles = []

    print "\nFinding articles in column %d meeting criterion '%s'" % (valuecol, criterion)
    print "Criterion value:", critvalue

    with open(inputcorpus, 'rU') as inf, \
            open(outputcorpus, 'wb') as outf:
        infile = csv.reader(inf)
        if header:
            headerline = next(infile)  # just skip

        for counter, article in enumerate(infile):
            articleid = int(article[0])
            # See if an earlier part of this article met the criterion
            if articleid not in articles:
                # See if criterion is met here
                colvalue = article[valuecol]
                if (criterion == 'matchstr' and colvalue == critvalue) or \
                        (criterion == 'matchval' and \
                            (int(colvalue) == critvalue or float(colvalue) == critvalue)) or \
                        (criterion == 'minthres' and float(colvalue) >= critvalue) or \
                        (criterion == 'maxthres' and float(colvalue) <= critvalue):
                    extractcount += 1
                    articles.append(articleid)
        totalarticles = counter + 1
    articles = set(articles)
    nrextracted = len(articles)

    print "Extracting articles from main file..."
    with open(inputcorpus, 'rU') as inf, \
            open(outputcorpus, 'wb') as outf:
        infile = csv.reader(inf)
        outwriter = csv.writer(outf)
        if header:
            headerline = next(infile)
            outwriter.writerow(headerline)
        for article in infile:
            if (include and int(article[0]) in articles) or (not include and int(article[0]) not in articles):
                if idsonly:
                    outwriter.writerow(article[:1])
                else:
                    outwriter.writerow(article)

    if parallelin != '' and parallelout != '':
        print "Extracting articles from parallel file..."
        extracted = []
        with open(parallelin, 'rU') as inf2, open(parallelout, 'wb') as outf2:
            infile2 = csv.reader(inf2)
            outwriter2 = csv.writer(outf2)
            if parallelheader:
                headerline = next(infile2)
                outwriter2.writerow(headerline)
            for article2 in infile2:
                articleid = int(article2[0])
                if (include and articleid in articles) or (not include and articleid not in articles):
                    extracted.append(articleid)
                    outwriter2.writerow(article2)
        extracted = set(extracted)
        notextracted = articles - extracted
        if len(extracted) != nrextracted:
            print "Warning: parallel file didn't match primary file"
            print "Not extracted: %d out of %d possible" % (nrextracted - len(extracted), nrextracted)
            # print notextracted

    # Give summary data
    print "\nExtracted %d articles out of %d total" % (nrextracted, totalarticles)
    return extractcount


def extract_sources_dates(inputcorpus, outputcorpus, header=True,
                          parallelin='', parallelout='', parallelheader=False,
                          sourcelist=[], sourceterms=[], capsmatter=False,
                          startdate=[], enddate=[],
                          idsonly=False, include=True,
                          sourcecol=2, yearcol=4, monthcol=6, daycol=7,
                          include_bad_dates=False):
    """Extract all texts (articles) by source(s) and date(s).

    Corpus does not need to be sorted by either source or date.
    If a parallel corpus is extracted as well, articles in that corpus must be in the same order
        (could double-check ids if not certain about this)

    See extract_column_value for a similar companion function.
    """
    import csv
    csv.field_size_limit(1000000000)

    startval = 0 if startdate == [] else \
        startdate[0] * 10000 + startdate[1] * 100 + startdate[2]
    endval = 20990000 if enddate == [] else \
        enddate[0] * 10000 + enddate[1] * 100 + enddate[2]

    extractcount = 0
    articles = []

    with open(inputcorpus, 'rU') as inf, \
            open(outputcorpus, 'wb') as outf:
        infile = csv.reader(inf)
        outwriter = csv.writer(outf)

        if header:
            headerline = next(infile)
            outwriter.writerow(headerline)

        for counter, article in enumerate(infile):

            # See if date is within range first
            try:
                year = int(article[yearcol])
            except:
                print "Not a year, line %d" % counter
                print article[:8]
                if include_bad_dates:
                    year = 1900
                else:
                    continue
            try:
                month = int(article[monthcol])
            except:
                print "Not a month, line %d" % counter
                print article[:8]
                if include_bad_dates:
                    month = 1
                else:
                    continue
            try:
                day = int(article[daycol])
            except:
                print "Not a date, line %d" % counter
                print article[:8]
                if include_bad_dates:
                    day = 1
                else:
                    continue

            dateval = year * 10000 + month * 100 + day
            datefits = dateval >= startval and dateval <= endval

            # Next see if source is OK
            sourceOK = False
            sourcename = article[sourcecol] if capsmatter else article[sourcecol].lower()
            if len(sourcelist) == 0:
                if len(sourceterms) == 0 or \
                        any([x in sourcename for x in sourceterms]):
                    sourceOK = True
            elif (include and sourcename in sourcelist) or \
                    (not include and sourcename not in sourcelist):
                sourceOK = True

            if datefits and sourceOK:
                extractcount += 1
                articles.append(counter)
                if idsonly:
                    outwriter.writerow(article[:1])
                else:
                    outwriter.writerow(article)
            if counter % 50000 == 0:
                print "Processing article", counter
    totalarticles = counter + (0 if header else 1)

    if parallelin != '' and parallelout != '':
        extractcount2 = 0
        articles = set(articles)
        with open(parallelin, 'rU') as inf2, open(parallelout, 'wb') as outf2:
            infile2 = csv.reader(inf2)
            outwriter2 = csv.writer(outf2)
            if parallelheader:
                headerline = next(infile2)
                outwriter2.writerow(headerline)
            for counter, article2 in enumerate(infile2):
                if counter in articles:
                    extractcount2 += 1
                    outwriter2.writerow(article2)
                if counter % 50000 == 0:
                    print "Processing parallel article", counter
        if extractcount2 != extractcount:
            print "Error: parallel file didn't match primary file"

    # Give summary data
    print "\nExtracted %d articles out of %d total" % (extractcount, totalarticles)
    return extractcount


def extract_bycontent(inputname, outputname, searchspec,
                      parallelin='', parallelout='',
                      textcol=1, capsmatter=False,
                      header=False, parallelheader=True):
    """Extract texts from larger corpus based on content.

    Best run on cleaned text. Optionally extract, in parallel,
    data from non-cleaned text.
    """
    import csv
    import os
    from wordsinwindow import findindices

    extractcount = 0
    articleids = []

    # Loop through file first to extract ids that match
    # Don't write yet, because some ids may have multiple parts, and would like all

    print "Finding articles that match the search criterion..."
    with open(inputname, 'rU') as inf:
        infile = csv.reader(inf)
        if header:
            headerline = next(infile)
        for counter, article in enumerate(infile):
            text = article[textcol]
            wordlist = text.split()

            locs = findindices(wordlist, searchspec[1],
                               capsmatter=capsmatter, skipperiods=True)
            # if keycount > 0:
            #     print ' '.join([wordlist[x] for x in locs])
            if len(searchspec[2]) > 0:  # deal with avoidspecs
                locs2 = findindices(wordlist, searchspec[2],
                                    capsmatter=capsmatter, skipperiods=True, )
                # if len(locs2) > 0:
                #     print 'avoid ' + ' '.join([wordlist[x] for x in locs2])
                locs = [x for x in locs if x not in locs2]
            keycount = len(locs)

            if keycount > 0:
                articleids.append(int(article[0]))
            if counter % 50000 == 0:
                print "Processing article", counter
    articles = set(articleids)

    print "\nWriting out selected articles"
    if not os.path.exists(os.path.dirname(outputname)):
        os.makedirs(os.path.dirname(outputname))
    with open(inputname, 'rU') as inf, open(outputname, 'wb') as outf:
        infile = csv.reader(inf)
        outwriter = csv.writer(outf)
        if header:
            headerline = next(infile)
            outwriter.writerow(headerline)
        for counter, article in enumerate(infile):
            if int(article[0]) in articles:
                extractcount += 1
                outwriter.writerow(article)
            if counter % 50000 == 0:
                print "Processing article", counter
    totalarticles = counter + (0 if header else 1)

    if parallelin != '' and parallelout != '':
        print "\nWriting out selected articles from parallel file"
        extractcount2 = 0
        with open(parallelin, 'rU') as inf2, open(parallelout, 'wb') as outf2:
            infile2 = csv.reader(inf2)
            outwriter2 = csv.writer(outf2)
            if parallelheader:
                headerline = next(infile2)
                outwriter2.writerow(headerline)
            for counter, article2 in enumerate(infile2):
                if int(article2[0]) in articles:
                    extractcount2 += 1
                    outwriter2.writerow(article2)
                if counter % 50000 == 0:
                    print "Processing parallel article", counter
        if extractcount2 != extractcount:
            print "Error: parallel file didn't match primary file"

    # Give summary data
    print "\nExtracted %d articles (in %d parts out of %d total)" % \
          (len(articles), extractcount, totalarticles)
    return extractcount


def filter_bycontent(inputname, outputname,
                     searchphrases, avoidphrases, avoiddistance, textwindow,
                     textcol=1, capsmatter=False, writeempty=False,
                     jointype='merge', joinstring=' ... ',
                     header=False, addheader=False):
    """Extract texts from larger corpus based on content.

    More powerful than extract_bycontent, but does not allow parallel filtering.
    Best run on cleaned text.

    In output file, text will be in column 1; other data cols follow
    (this may be different from the input file!)
    """
    import csv
    from wordsinwindow import match_specs

    csv.field_size_limit(1000000000)

    print "Finding texts that match the search criterion..."
    matchcount, textcount = 0, 0

    if not capsmatter:
        searchphrases = [phrase.lower() for phrase in searchphrases]
        avoidphrases = [phrase.lower() for phrase in avoidphrases]

    with open(inputname, 'rU') as inf, open(outputname, 'wb') as outf:
        infile = csv.reader(inf)
        outwriter = csv.writer(outf)

        if header or addheader:
            if header:
                headerline = next(infile)
            else:  # assume simple csv structure
                headerline = ['id', 'text']
            outwriter.writerow([headerline[0], 'filtered text'] + \
                               headerline[1:textcol] + headerline[textcol + 1:])

        curid = -1
        curtext = ''
        curdata = ''
        for counter, article in enumerate(infile):
            if article[0] != curid:
                if len(curtext) > 0:
                    filteredtext, nrmatches = \
                        match_specs(curtext, searchphrases, avoidphrases,
                                    avoiddistance, textwindow,
                                    jointype=jointype, joinstring=joinstring)
                    if len(filteredtext) > 0:
                        textcount += 1
                        matchcount += nrmatches
                    # write out result -- note: may be long!!
                    # if this causes problems, could split it up again
                    if writeempty or len(filteredtext) > 0:
                        outwriter.writerow([curid, filteredtext] + curdata)
                curid = article[0]
                curtext = article[textcol]
                curdata = article[1:textcol] + article[textcol+1:]
            else:
                curtext = curtext + ' ' + article[textcol]
            if (counter + 1) % 50000 == 0:
                print "Processing text", counter + 1
        # flush last text
        filteredtext, nrmatches = match_specs(curtext, searchphrases, avoidphrases,
                                    avoiddistance, textwindow,
                                    jointype=jointype, joinstring=joinstring)
        if len(filteredtext) > 0:
            textcount += 1
            matchcount += nrmatches
        # write out result -- note: may be long!!
        # if this causes problems, could split it up again
        if writeempty or len(filteredtext) > 0:
            outwriter.writerow([curid, filteredtext] + curdata)

    # Give summary data
    print "\nRetained %d pieces of text from %d articles" % (matchcount, textcount)
    return textcount


def reify_keys(keylist):
    """Convert a search specification with a possible wild card to one for the
    built-in regular expression module: put '\b' at the start and end, and add
    a re-format wildcard specification as appropriate.
    """
    import re

    keylist = [re.sub(r'\*', '[\w-]*', key) for key in keylist]
    return ['\\b' + key + '\\b' for key in keylist]


def filter_bycontent_specific(curtext, searchphrases, avoidphrases,
                              avoiddistance, textwindow, capsmatter=False):
    """Do filtering for a single text."""

    from wordsinwindow import match_specs

    resultsfound, textsfound = [], []
    textcount, matchcount = 0, 0

    for searchthis, avoidthis in zip(searchphrases, avoidphrases):
        filteredtext, nrmatches = match_specs(curtext, searchthis, avoidthis,
                                              avoiddistance, textwindow, capsmatter=capsmatter)
        # store result -- note: may be long!!
        # if this causes problems, could split it up again
        if len(filteredtext) > 0:
            resultsfound.append(nrmatches)
            textsfound.append(filteredtext)
        else:
            resultsfound.append(0)
    return sum(resultsfound), resultsfound, textsfound


def filter_bycontent_overall(curtext, rekeys):
    """Identify articles that contain any of the key phrase(s)."""
    import re
    features = [1 if any([re.search(pattern, curtext) != None \
                         for pattern in keylist]) else 0 \
                for keylist in rekeys]
    return sum(features), features


def filter_bycontent_multi(inputname, outputname,
                           searchphrases, avoidphrases, avoiddistance, textwindow,
                           textcol=1, capsmatter=False, joinstring=" *** ",
                           header=False, addheader=False, startat=0):
    """Extract texts from larger corpus based on content; multiple search phrases.

    A generalization of filter_bycontent to allow for a series of search & avoid-phrases.
    Columns are labeled by the first item in each set of search phrases

    In output file, text will be in column 1, followed by flags for searches met;
    other data cols follow at the end
    """

    import csv
    csv.field_size_limit(1000000000)

    print "Finding texts that match the search criteria..."
    matchcount, textcount = 0, 0

    if startat > 0:
        w_or_a = 'ab'
        started = False
    else:
        w_or_a = 'wb'
        started = True

    with open(inputname, 'rU') as inf, open(outputname, w_or_a) as outf:
        infile = csv.reader(inf)
        outwriter = csv.writer(outf)

        if header or addheader and started:
            if header:
                headerline = next(infile)
            else:  # assume simple csv structure
                headerline = ['id', 'text']
            columnlabels = zip(*searchphrases)[0]
            outwriter.writerow([headerline[0], 'filtered text', 'nrmatches'] + \
                               list(columnlabels) + \
                               headerline[1:textcol] + headerline[textcol + 1:])

        # See if we need filtering within a text
        if textwindow[1] == -1 and textwindow[2] == -1 and \
                all([len(x) == 0 for x in avoidphrases]):
            internalfilter = False
            # Convert search keys
            rekeys = [reify_keys(featurespec[1]) for featurespec in searchphrases]
        else:
            searchphrases = [x[1] for x in searchphrases]
            internalfilter = True

        # Run down each article in the file
        # Note that we test for internalfilter each time ->
        #  could take that test outside the loop and just have 2 separate loops to make marginally more efficient
        curid = -1
        curtext = ''
        curdata = ''
        for counter, article in enumerate(infile):
            if not started and counter < startat:
                continue
            started = True
            if article[0] != curid:
                if len(curtext) > 0:
                    if internalfilter:
                        nrfound = filter_bycontent_specific(curid, curdata, curtext, outwriter,
                                                  searchphrases, avoidphrases,
                                                  avoiddistance, textwindow,
                                                  capsmatter=capsmatter, joinstring=joinstring)
                    else:
                        nrfound = filter_bycontent_overall(curid, curdata, curtext, outwriter,
                                                  rekeys)
                    if nrfound > 0:
                        textcount += 1
                    matchcount += nrfound
                curid = article[0]
                curtext = article[textcol]
                curdata = article[1:textcol] + article[textcol + 1:]
            else:
                curtext = curtext + ' ' + article[textcol]
            if counter % 1000 == 0:
                print "Processing text", counter

        # flush last text
        if len(curtext) > 0:
            if internalfilter:
                nrfound = filter_bycontent_specific(curid, curdata, curtext, outwriter,
                                          searchphrases, avoidphrases,
                                          avoiddistance, textwindow)
            else:
                nrfound = filter_bycontent_overall(curid, curdata, curtext, outwriter,
                                          rekeys)
            if nrfound > 0:
                textcount += 1
            matchcount += nrfound

    # Give summary data
    print "\nRetained %d pieces of text from %d articles" % (matchcount, textcount)
    return textcount


def filter_bycontent_parallel(inputnames, outputnames,
                              searchphrases, avoidphrases, avoiddistance, textwindow,
                              textcol=1, capsmatter=False, joinstring=" *** ",
                              header=False, addheader=False, nrjobs=1):
    """Like filter_bycontent_multi, but providing for parallel processing

    Assumes one job per input file.
    """

    import csv
    from functools import partial
    import gc
    import multiprocessing as mp

    csv.field_size_limit(1000000000)

    # Read each input file into memory (be wary of memory constraints!)
    # Could do this whole thing as list comprehension, but for loop is a bit more intelligible
    nrtasks = len(inputnames)
    jobs = []
    for inputname, outputname in zip(inputnames, outputnames):
        print "Loading data for %s" % inputname.split('/')[-1]
        indata = []
        with open(inputname, 'rU') as inf:
            infile = csv.reader(inf)
            for row in infile:
                indata.append(row)
        jobs.append([outputname, indata])

    nrrounds = 1 + nrtasks / nrjobs  # integer division truncates down
    if nrtasks % nrjobs == 0:
        nrrounds -= 1

    gc.collect()  # minimize memory load prior to spawning processes

    for round in nrrounds:
        if nrjobs == 1:
            filter_bycontent_parallX(jobs[round],
                searchphrases = searchphrases, avoidphrases = avoidphrases,
                avoiddistance = avoiddistance, textwindow = textwindow,
                textcol = textcol, capsmatter = capsmatter, joinstring = joinstring,
                header = header, addheader = addheader)

        else:
            # Partial function to call in parallel: all parameters constant except for the data
            partial_fbc_parallX = partial(filter_bycontent_parallX,
                                          searchphrases=searchphrases, avoidphrases=avoidphrases,
                                          avoiddistance=avoiddistance, textwindow=textwindow,
                                          textcol=textcol, capsmatter=capsmatter, joinstring=joinstring,
                                          header=header, addheader=addheader)

            procPool = mp.Pool(processes = len(jobs))
            results = procPool.map(partial_fbc_parallX, jobs[round * nrjobs : (round + 1) * nrjobs])
            procPool.close()
            procPool.join()
            procPool.terminate()
            # Combine the results
            resultsX = (sum(x) for x in zip(*results))
            print "\nOverall: retained %d pieces of text from %d articles" % tuple(resultsX)


def filter_bycontent_parallX(jobdata,
                             searchphrases=(), avoidphrases=(),
                             avoiddistance=(), textwindow=(),
                             textcol=1, capsmatter=False, joinstring=' *** ',
                             header=True, addheader=False):
    import csv

    outputname, corpusdata = jobdata
    outputname_end = outputname.split('/')[-1]
    print "Writing texts to %s that match the search criteria" % outputname_end
    matchcount, textcount = 0, 0

    outdata = []
    currow = 0
    if header or addheader:
        if header:
            headerline = corpusdata[currow]
            currow += 1
        else:  # assume simple csv structure
            headerline = ['id', 'text']
        columnlabels = zip(*searchphrases)[0]
        outdata.append([headerline[0], 'filtered text', 'nrmatches'] + \
                       list(columnlabels) + \
                       headerline[1:textcol] + headerline[textcol + 1:])

    # See if we need filtering within a text
    if textwindow[1] == -1 and textwindow[2] == -1 and \
            all([len(x) == 0 for x in avoidphrases]):
        internalfilter = False
        # Convert search keys
        rekeys = [reify_keys(featurespec[1]) for featurespec in searchphrases]
    else:
        searchphrases = [x[1] for x in searchphrases]
        internalfilter = True

    # Run down each article in the corpus
    # Note that we test for internalfilter each time ->
    #  could take that test outside the loop and just have 2 separate loops to make marginally more efficient
    curid = -1
    curtext = ''
    curdata = ''
    for counter, article in enumerate(corpusdata[currow:]):
        if article[0] != curid:
            if len(curtext) > 0:
                if internalfilter:
                    nrfound, resultsfound, textsfound = \
                        filter_bycontent_specific(curtext,
                                              searchphrases, avoidphrases,
                                              avoiddistance, textwindow,
                                              capsmatter=capsmatter)
                    if len(textsfound) > 0:
                        outdata.append([curid, joinstring.join(textsfound), nrfound] + \
                                       resultsfound + curdata)
                else:
                    nrfound, features = filter_bycontent_overall(curtext, rekeys)
                    if nrfound > 0:
                        outdata.append([curid, curtext, nrfound] + features + curdata)
                if nrfound > 0:
                    textcount += 1
                    matchcount += nrfound
            curid = article[0]
            curtext = article[textcol]
            curdata = article[1:textcol] + article[textcol + 1:]
        else:
            curtext = curtext + ' ' + article[textcol]
        if counter % 1000 == 0:
            print "Processing text %d for %s" % (counter, outputname_end)

    # flush last text
    if len(curtext) > 0:
        if internalfilter:
            nrfound, resultsfound, textsfound = \
                filter_bycontent_specific(curtext,
                                          searchphrases, avoidphrases,
                                          avoiddistance, textwindow,
                                          capsmatter=capsmatter)
            if len(textsfound) > 0:
                outdata.append([curid, joinstring.join(textsfound), nrfound] + \
                               resultsfound + curdata)
        else:
            nrfound, features = filter_bycontent_overall(curtext, rekeys)
            if nrfound > 0:
                outdata.append([curid, curtext, nrfound] + features + curdata)
        if nrfound > 0:
            textcount += 1
            matchcount += nrfound

    # Give summary data & write out results
    with open(outputname, 'wb') as outf:
        csv.writer(outf).writerows(outdata)
    print "\nRetained %d pieces of text from %d articles" % (matchcount, textcount)

    return (matchcount, textcount)


def getcsv(filename, commentchar='#'):
    """Simple function to read csv file into memory.

    Skip lines that are commented out.
    """
    import csv

    csv.field_size_limit(1000000000)

    data = []
    with open(filename, 'rU') as inf:
        for row in csv.reader(inf):
            if len(row[0].strip()) > 0 and row[0][0] != commentchar:
                data.append([x.strip() for x in row if len(x.strip()) > 0])
    # Strip possible byte order mark in first item
    data[0][0] = data[0][0].decode("utf-8-sig").encode("utf-8")
    # Check to make sure first line is not a comment after that stripping
    return data if data[0][0][0] != commentchar else data[1:]


def separate_first(data, keepheaders=False, stripempties=True):
    """Take a list of lists; replace each item by a tuple: first item, all (or just subsequent) ones.

    Optionally strip those that have just a single item.
    """
    return [(x[0], x if keepheaders else x[1:]) for x in data if (not stripempties or len(x) > 1)]


def subsetNrows(corpusfile, outputfile, N=100, startobs=0, header=False):
    """Extract N rows from a csv file, from specified starting observation."""
    import csv

    csv.field_size_limit(1000000000)
    with open(corpusfile, 'rU') as articlefile, open(outputfile, 'wb') as outfile:
        outwriter = csv.writer(outfile)
        started = False
        subsetcount = 0
        for counter, row in enumerate(csv.reader(articlefile)):
            if header:
                header = False
                startobs += 1
                outwriter.writerow(row)
            else:
                if counter == startobs:
                    started = True
                if started:
                    outwriter.writerow(row)
                    subsetcount += 1
                    if subsetcount == N:
                        break
    print "Extracted %d rows" % (counter + 1 - startobs)


def subset_fraction(corpusfile, outputfile, fraction=0.1, seed=42,
                    idsonly=False, header=False):
    """Extract fraction of observations from a csv file."""
    import csv
    import random

    csv.field_size_limit(1000000000)
    random.seed(seed)

    with open(corpusfile, 'rU') as articlefile, open(outputfile, 'wb') as outfile:
        outwriter = csv.writer(outfile)
        started = False
        subsetcount = 0
        for counter, row in enumerate(csv.reader(articlefile)):
            if header:
                header = False
                if idsonly:
                    outwriter.writerow([row[0], ])
                else:
                    outwriter.writerow(row)
            elif random.random() < fraction:
                if idsonly:
                    outwriter.writerow([row[0], ])
                else:
                    outwriter.writerow(row)
                subsetcount += 1
    print 'Extracted %d rows' % (subsetcount)


# *************************** from sentiment.py ************************

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


def featureinfo(filename):
    """Print out column number - feature name pairs."""
    import csv

    with open(filename, 'rU') as inf:
        headerrow = next(csv.reader(inf))
    for index, varname in enumerate(headerrow):
        print "%3d - %s" % (index, varname)


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


def deletefeatures(targetname, backupname, cols2delete):
    """Keep specified column indices only. Add optional flag feature."""
    import csv
    import os

    # Save backup copy, by renaming
    os.rename(targetname, backupname)

    # Now open backup as source, and overwrite original inputfile as output
    with open(backupname, 'rU') as inf, open(targetname, 'wb') as outf:
        inputf = csv.reader(inf)
        outputf = csv.writer(outf)

        # Read in column headers, and write new headers
        # Calculate cols2keep just once, to make subsequent iterations faster
        headers = next(inputf)
        cols2keep = [x for x in range(len(headers)) if x not in cols2delete]
        outputf.writerow([headers[x] for x in cols2keep])

        for obs in inputf:
            outputf.writerow([obs[x] for x in cols2keep])
    return


def idrange(filename, header=True):
    """Find smallest and largest id numbers in corpus.

    Assumes integer ids.
    """
    import csv
    minval = -1
    maxval = -1
    with open(filename, 'rU') as infile:
        for counter, article in enumerate(csv.reader(infile)):
            if header:
                header = False
            else:
                try:
                    idnr = int(article[0])
                except:
                    print "error in %s, line %d" % (filename, counter)
                    print header
                    print article[:3]
                    idnr = 0
                if minval == -1 or idnr < minval:
                    minval = idnr
                if maxval == -1 or idnr > maxval:
                    maxval = idnr
    # print "\nSmallest and largest id numbers: %d, %d" % \
    # (minval, maxval)
    return minval, maxval


def combine_files(inputdata, outputname, varnames=(), adjustvarnames=True):
    """Combine multiple data files with same ids, keeping specified columns.

    Keeps all data in memory -- don't try with very large files!

    For each input file, specify filename, data columns to keep,
    and how to combine multiple parts if applicable
    """
    import csv
    from os.path import basename

    # process each input file separately, merging multiple parts per item if necessary
    collected = []
    for inputspec in inputdata:
        # elegantly handle an optional 4th inputspec
        inputfile, inputcols, mergespec = inputspec[:3]
        if len(inputspec) == 4 and inputspec[3] == 'noheader':
            header = False
        else:
            header = True

        # *** this section should be extracted as a function!
        newcollected = []
        with open(inputfile, 'rU') as inputf:
            curid = -999
            firstrow = True
            for row in csv.reader(inputf):
                if len(inputcols) == 0:
                    cols2keep = row[1:]
                else:
                    cols2keep = [row[col] for col in inputcols]
                if curid != row[0]:  # new article
                    if firstrow:
                        if not header:
                            # generate dummy header line
                            newcollected.append(['id',] + ['var' + str(x) for x in range(len(cols2keep))])
                            idval = row[0]
                        else:
                            idval = 'id'  # make sure id column has 'id' as its label
                        firstrow = False
                    else:
                        idval = row[0]
                    newcollected.append([idval,] + cols2keep)
                else:   # subsequent part of same article
                    lastadded = newcollected[-1]
                    if mergespec == 'max':
                        replaceval = [max(x,y) for x, y in zip(lastadded[1:], cols2keep)]
                    elif mergespec == 'sum':
                        replaceval = [x + y for x, y in zip(lastadded[1:], cols2keep)]
                    else:  # mergespec == 'none'
                        replaceval = lastadded[1:]  # i.e. do nothing
                    newcollected[-1] = [lastadded[0],] + replaceval
                curid = row[0]
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
        for counter, (inputfile, inputcols, mergespec) in enumerate(inputdata):
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


# *************************** from texts to sentences ************************


def texts2lines(ids_texts):
    """Split texts into lines, but keep track (by id) of which text."""
    linelist = []
    idlist = []
    for id, atext in ids_texts:
        for aline in atext.split('.'):
            linelist.append(aline)
            idlist.append(id)
    return linelist, idlist


def texts2lines_file(filestem, textcol=1):
    """As texts2lines, but streaming in/out (for large corpora)."""
    import csv

    linefile = filestem + '_lines.txt'
    idfile = filestem + '_lineids.txt'
    with open(filestem + '.csv', 'rU') as intexts, \
            open(linefile, 'wt') as outlines, \
            open(idfile, 'wt') as outids:
        for row in csv.reader(intexts):
            id = row[0]
            atext = row[textcol]
            if 'id' not in id.lower():
                for aline in atext.split('.'):
                    outlines.write(aline + '\n')
                    outids.write(id + '\n')
    return linefile, idfile


def texts2lines_fromfile(filestem, inputsuffix='_lower', textcol=1,
                         linesuffix='_lines', idsuffix='_lineids'):
    """As texts2lines_file, but with more control over filenames."""
    import csv

    linefile = filestem + linesuffix + '.txt'
    idfile = filestem + idsuffix + '.txt'
    with open(filestem + inputsuffix + '.csv', 'rU') as intexts, \
            open(linefile, 'wt') as outlines, \
            open(idfile, 'wt') as outids:
        for row in csv.reader(intexts):
            id = row[0]
            atext = row[textcol]
            if 'id' not in id.lower():
                for aline in atext.split('.'):
                    outlines.write(aline + '\n')
                    outids.write(id + '\n')
    return linefile, idfile


# ***************************** from texts to words **************************

def getwords_fromfile(filename, textsonly=True, ignorecase=True):
    """Return list of unique words in file."""
    return getwords_fromcorpus(readtextfile(filename) if textsonly \
                                   else readidtextfile_texts(filename),
                               ignorecase)


def getwords_fromcorpus(alltexts, ignorecase=True):
    """Return list of unique words in corpus."""
    return getwords(alltexts, mincount=1, ignorecase=ignorecase,
                    excludelist=['.', ',', 'a', 'an', 'the', 'and'])


def getwords(texts, mincount, ignorecase, excludelist):
    """Return a list of distinct tokens (words) from input texts.

    Filter out rarely occurring words with mincount.
    """
    # import csv
    import pickle
    from collections import Counter

    words = Counter()
    for count, text in enumerate(texts):
        for sent in text.split('.'):
            for word in sent.split():
                if ignorecase:
                    words.update([word.lower(),])
                else:
                    words.update([word,])
        if count % 500 == 0:
            print count

    # filter out low-frequency words
    words = filterdict(words, mincount)
    # pickle frequency dictionary for possible later use
    with open('worddict.pkl', 'wb') as outFile:
        pickle.dump(words, outFile)
    # return list of words only
    return [w for w in words.keys() if w not in excludelist]


def corpus_top(wordlist, corpusdict, mincount=50):
    """Get frequency stats on most common words in wordlist.

    Note: assumes all entries in wordlist are in corpusdict.
    """
    from operator import itemgetter
    wordstats = sorted([(word, corpusdict[word]) for word in wordlist],
                       key=itemgetter(1), reverse=True)
    for x in xrange(nrwords):
        if wordstats[x][1] < mincount:
            ind = x
            break
        else:
            print wordstats[x]
    return wordstats[:ind]


# **************************** from texts to n-grams *************************

def getgrams(texts, mincount, mingram):
    """Return a list of common 1-4 grams in input file.

    Expanded version of getwords.
    Filter out low-frequency words (threshold mincount) & phrases (mingram).
    """
    # import csv
    import pickle
    from collections import Counter

    words, bigrams, trigrams, quadgrams = Counter(), Counter(), \
                                          Counter(), Counter()
    # If passed a filename, the next 3 lines replace the one that follows
    # (also remember to import csv!)
    # with open(infile,'rU') as textFile:
    #     texts = csv.reader(textFile)
    #     for count, (id, text) in enumerate(texts):
    for count, text in enumerate(texts):
        # reset n-gram list for each new text
        prevquad = ['xdummyx', 'xdummyx', 'xdummyx']
        for sent in text.split('.'):
            for word in sent.split():
                words.update([word,])
                if prevquad[0] != 'xdummyx':
                    bigram = prevquad[0] + '_' + word
                    bigrams.update([bigram,])
                if prevquad[1] != 'xdummyx':
                    trigram = prevquad[1] + '_' + bigram
                    trigrams.update([trigram,])
                if prevquad[2] != 'xdummyx':
                    quadgram = prevquad[2] + '_' + trigram
                    quadgrams.update([quadgram,])
                prevquad[2] = prevquad[1]
                prevquad[1] = prevquad[0]
                prevquad[0] = word
        if count % 500 == 0:
            print count

    # filter out low-frequency words
    words = filterdict(words, mincount)
    bigrams = filterdict(bigrams, mingram)
    trigrams = filterdict(trigrams, mingram)
    quadgrams = filterdict(quadgrams, mingram)

    # save in pickle format & return
    wordlist = list(words.keys())
    bilist = list(bigrams.keys())
    trilist = list(trigrams.keys())
    quadlist = list(quadgrams.keys())
    with open('gramlist.pkl', 'wb') as outFile:
            pickle.dump((wordlist, bilist, trilist, quadlist), outFile)
    return wordlist + bilist + trilist + quadlist


# ************************** manage corpus with metadata ************************

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


def keepfeaturesX(inputname, outputname, featurelist, flag=False, header=True):
    """Keep specified column indices only

    Additional operations:
    - make length numeric
    - extract page #s
    - add optional flag
    """
    import csv

    errorcount = [0, 0, 0, 0]
    with open(inputname, 'rU') as inf, open(outputname, 'wb') as outf:
        output = csv.writer(outf)
        for counter, article in enumerate(csv.reader(inf)):
            newarticle = [article[x] for x in featurelist]
            if header:
                newarticle += ['length', 'section', 'subsection', 'page', 'column']
                if flag is not False:
                    newarticle.append('flag')
                header = False
            else:
                # Get length info (ignore officially provided length, which may exclude title)
                newarticle.append(len(article[10].split()) + len(article[12].split()))

                # Get section, part, page, and column information
                pageinfo, errorcodes = extractpagenumber(article[16], article[18])
                errorcount = [x + y for x, y in zip(errorcount, errorcodes)]
                newarticle += pageinfo

                # Add optional flag
                if flag is not False:
                    newarticle.append(flag)

            output.writerow(newarticle)
            if counter % 50000 == 0:
                print "Processing article %d" % counter

        print "Done. Error counts: problem - %d, duplicate info - %d, ignored letters - %d, no data - %d." % \
              tuple(errorcount)


def fixlength(inputfile, outputfile, lengthcol=10, header=True):
    """Length column may have 'words' in it. Remove.

    Note that it would be even better to calculate our own length, since
    this length info probably omits the title's length.
    """
    import csv
    with open(inputfile, 'rU') as inf, open(outputfile, 'wb') as outf:
        inreader = csv.reader(inf)
        outwriter = csv.writer(outf)
        if header:
            inheader = next(inreader)
            outwriter.writerow(inheader)
        for row in inreader:
            lendata = row[lengthcol]
            row[lengthcol] = 0 if len(lendata) == 0 else lendata.split()[0]
            outwriter.writerow(row)
    return


def stripmeta(textfile, header=True, textcells=[10,12,14], add_id=False):
    """Strip csv file for text corpus of metadata & create id,text format."""
    import csv
    import sys

    csv.field_size_limit(1000000000)
    outfile = '.'.join(textfile.split('.')[:-1]) + '_texts.csv'
    with open(textfile, 'rU') as inf, open(outfile, 'wb') as outf:
        output = csv.writer(outf)
        for counter, article in enumerate(csv.reader(inf)):
            if header:
                header = False
            else:
                id = counter if add_id else article[0]
                text = ' . '.join([article[x] for x in textcells])
                output.writerow([id, text])


def extractpagenumber(sectioninfo, extrainfo):
    """Extract page number from the section (& other) data.

    Return 4 values: section, part, page, and column.
    Convert letters (usually section or column) to numbers.
    """

    # Key page indicators & associated constants
    typenames = ['section ', 'part ', 'pg. ', ' pg ', 'page ', 'col. ', ' col ' 'column ']
    pageindex = [0, 1, 2, 2, 2, 3, 3, 3]
    pageinfo = [0, 0, 0, 0]
    errorcodes = [0, 0, 0, 0]  # problem, duplicate, ignored data, no data
    letters = '_abcdefghijklmnopqrstuvwxyz'
    numbers = '0123456789'
    WPweekly_strings = ['d.c.', 'dc.', 'dc', 'va.', 'va', 'md.', 'md']
    section_strings = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']

    # Get section, part, page & column info, if available
    # Not most efficient, but cleanest code to do these separately
    sectiontext = sectioninfo.lower().strip()
    sectionwords = sectiontext.split()

    # If last word contains (internal) punctuation, replace
    if len(sectionwords) > 0 and any(x in sectionwords[-1] for x in ',.;:-'):
        sectionwords = sectionwords[:-1] + \
                       del_internal_punct(strippunct(sectionwords[-1], leading=True)).split()

    # Check section text for key indicators & process
    if any(x in sectiontext for x in typenames):
        # Try each indicator in turn
        for pagepart, ind in zip(typenames, pageindex):
            if pagepart in sectiontext:
                # Extract data following the indicator
                loc = sectiontext.find(pagepart) + len(pagepart)
                letterval, numval = convert_alphanumeric(sectiontext[loc:].split()[0])
                if numval == -99:
                    # print "Problem value for %s: %s" % \
                    #   (pagepart, sectiontext[loc:].split()[0])
                    errorcodes[0] = 1
                # Assign data to correct part
                if letterval > 0:
                    if numval > 0:  # have letters as well as numbers
                        # Assign numbers to current item (ind) in typenames
                        if pageinfo[ind] != 0:
                            # print "Duplicate info: ignored %s" % pagepart
                            # print sectioninfo
                            errorcodes[1] = 1
                        else:
                            pageinfo[ind] = numval
                        # Assign letters to section
                        if pageinfo[0] == 0:
                            pageinfo[0] = letterval
                        else:  # complicated page designation -> just discard
                            # print "Ignored letter-number combo: ", sectioninfo
                            errorcodes[2] = 1
                    else:  # have letters only
                        if pageinfo[ind] != 0:
                            # print "Duplicate info: ignored %s" % pagepart
                            # print sectioninfo
                            errorcodes[1] = 1
                        else:
                            pageinfo[ind] = letterval
                elif numval > 0:
                    if pageinfo[ind] != 0:
                        # print "Duplicate info: ignored %s" % pagepart
                        # print sectioninfo
                        errorcodes[1] = 1
                    else:
                        pageinfo[ind] = numval

    # No explicit markers -> look for page info at the end
    elif len(sectionwords) > 1 and \
                (sectionwords[-2][-1] in ';:,.' or
                 sectionwords[-2] in WPweekly_strings or
                 sectionwords[-2] in section_strings):
        if len(sectionwords[-2]) == 1 and sectionwords[-2][0] in 'abcdefghijkl':
            pageinfo[0] = letters.find(sectionwords[-2][0])
        elif sectionwords[-2] in WPweekly_strings:
            pageinfo[0] = 50  # could specify which of the weeklies here
        pagestr = sectionwords[-1]
        # Strip to info after last period, semicolon, or comma (could use re here)
        while '.' in pagestr:
            pagestr = pagestr[pagestr.find('.') + 1:]
        while ';' in pagestr:
            pagestr = pagestr[pagestr.find(';') + 1:]
        while ',' in pagestr:
            pagestr = pagestr[pagestr.find(',') + 1:]
        val = strippunct(pagestr, leading=True)
        if len(val) > 1 and val[0] in 'clxvi' and val[-1] in 'clxvi':
            pageinfo[2] = str(deromanize(val))  # assume roman nr.
        else:
            letterval, numval = convert_alphanumeric(val)
            if numval == -99:
                # print "Problem value for %s: %s" % \
                #   (pagepart, sectiontext[loc:].split()[0])
                errorcodes[0] = 1
            elif numval == 0:
                pageinfo[2] = letterval
            else:  # numval non-zero, non-error
                pageinfo[2] = numval
                if pageinfo[0] > 0 and letterval > 0:
                    # complicated page designation -> just discard
                    # print "Ignored letter-number combo: ", sectioninfo
                    errorcodes[2] = 1

    # No implicit markers -> see if last word could be a page marker
    elif len(sectionwords) > 0 and \
            any(x in numbers or x in letters for x in sectionwords[-1]) and \
            sectionpage(del_internal_punct(strippunct(sectionwords[-1], leading=True)).split()[-1]):
        letterval, numval = convert_alphanumeric(sectionwords[-1])
        pageinfo[0] = letterval
        pageinfo[2] = numval

    else:  # No explicit or implicit markers; see if elsewhere

        # See if there is a word following a semicolon that could be a page marker
        loc = sectiontext.find('; ')
        while loc >= 0:
            nextword = sectiontext[loc + 2:].split()[0]
            if len(nextword) > 0 and sectionpage(nextword):
                letterval, numval = convert_alphanumeric(nextword)
                pageinfo[0] = letterval
                pageinfo[2] = numval
                return pageinfo, errorcodes
            sectiontext = sectiontext.replace(';', ' ', 1)
            loc = sectiontext.find('; ')

        # Exhausted section info -> turn to extra info
        possiblepage = extrainfo.lower().split()
        if len(possiblepage) == 1:
            pageword = del_internal_punct(strippunct(possiblepage[0], leading=True))
            if ' ' in pageword:
                pageword = pageword.split()[-1]
            if sectionpage(pageword):
                letterval, numval = convert_alphanumeric(pageword)
                pageinfo[0] = letterval
                pageinfo[2] = numval
            else:
                errorcodes[3] = 1
                pageinfo[2] = -101
        else:
            errorcodes[3] = 1
            pageinfo[2] = -101
    return pageinfo, errorcodes


def numeric(val):
    """See if all chars in a string are numeric"""
    return all(x in '0123456789' for x in val)


def sectionpage(aword):
    """See if string is of format 123, A123 or 123A"""
    letters = 'abcdefghijklmnopqrstuvwxyz'
    isalphanumeric = len(aword) > 1 and \
                     ((aword[0] in letters and numeric(aword[1:])) or
                      (aword[-1] in letters and numeric(aword[:-1])))
    return numeric(aword) or isalphanumeric


def deromanize(val):
    """Convert from Roman number"""
    sum = 0
    while len(val) > 0 and val[0] == 'm':
        sum += 1000
        val = val[1:]
    while len(val) > 0 and val[0] == 'd':
        sum += 500
        val = val[1:]
    while len(val) > 0 and val[0] == 'c':
        sum += 100
        val = val[1:]
    while len(val) > 0 and val[0] == 'l':
        sum += 50
        val = val[1:]
    while len(val) > 0 and val[0] == 'x':
        sum += 10
        val = val[1:]
    while len(val) > 0 and val[0] == 'v':
        sum += 5
        val = val[1:]
    while len(val) > 0 and val[0] == 'i':
        sum += 1
        val = val[1:]
    return sum


def convert_alphanumeric(val):
    """Convert a numeric value with a possible letter prefix or suffix (A1, 2C, etc.)

    Strip out internal punctuation.
    """
    letters = '_abcdefghijklmnopqrstuvwxyz'
    nrletters = 26
    letterval = 0

    # Pre-process our target string
    val = del_internal_punct(strippunct(val, leading=True))  # Strip punctuation
    # If there is now a space, only consider first word
    if ' ' in val:
        val = val[:val.find(' ')]
    if len(val) == 0:
        return 0, 0

    # Process the target string; allow letters either at beginning or end but not both
    # Letter(s) at beginning
    leadingletters = False
    while len(val) > 0 and val[0] in letters:
        letterval = letterval * nrletters + letters.find(val[0])
        val = val[1:]
        leadingletters = True
    if not leadingletters:
        # Trailing letter(s)
        while len(val) > 0 and val[-1] in letters:
            letterval = letterval * nrletters + letters.find(val[-1])
            val = val[:-1]
    if len(val) == 0:
        return letterval, 0
    elif all([x in '0123456789' for x in val]):
        return letterval, int(val)
    else:
        return letterval, -99


def strippunct(val, leading=False, trailing=True):
    """Remove leading and/or trailing punctuation."""
    if leading:
        while len(val) > 0 and val[0] in '.;,!:/-("':
            val = val[1:]
    if trailing:
        while len(val) > 0 and val[-1] in '.;,!:/-)"':
            val = val[:-1]
    return val


def del_internal_punct(aword):
    """Remove internal punctuation in a word.

    Replace by space; then concatenate if section-page or page-section pattern.
    """
    import re
    aword = re.sub(r'(\w{,4})[\.,-:;](\w)', r'\1 \2', aword)
    # append letter-number and number-letter patterns; no more than 2 letters
    aword = re.sub(r'(\b[a-z]{1,2}) ([0-9]+)\b', r'\1\2', aword)
    aword = re.sub(r'(\b[0-9]+) ([a-z]{1,2})\b', r'\1\2', aword)
    return aword


def calcsectionpage(pageinfo, sectionoffset):
    """Convert page indication of format A1 to numeric, using offset for letters"""
    return 'abcdefghijklmnopqrstuvwxyz'.find(pageinfo[0]) * sectionoffset + \
           (0 if len(pageinfo) == 1 else int(pageinfo[1:]))


def removesources(inputname, outputname, sourcelist, header=True):
    """Remove specified sources."""
    import csv

    removedcount = 0
    with open(inputname, 'rU') as inf, open(outputname, 'wb') as outf:
        output = csv.writer(outf)
        for article in csv.reader(inf):
            if header:
                output.writerow(article)
                header = False
            else:
                if article[2] not in sourcelist:
                    output.writerow(article)
                else:
                    removedcount += 1
    print "Removed %d articles" % removedcount



# ********************** information about corpus contents ********************

def counts_dates_bysource(filename, header=True,
                          sourcecol=2, yearcol=4, monthcol=6, daycol=7,
                          alldates=False):
    """Find article count, earliest and latest date for all sources in corpus.

    Corpus does not need to be sorted.
    """
    import csv
    csv.field_size_limit(1000000000)

    sourcedates = {}
    sourcecounts = {}
    sourcedatelist = {}
    alldatelist = []

    with open(filename, 'rU') as infile:
        for article in csv.reader(infile):
            if header:
                header = False
            else:
                year = int(article[yearcol])
                month = int(article[monthcol])
                day = int(article[daycol])
                dateval = year * 10000 + month * 100 + day
                source = article[sourcecol]
                if source in sourcedates:
                    sourcecounts[source] += 1
                    if sourcedates[source][0] > dateval:
                        sourcedates[source][0] = dateval
                    if sourcedates[source][1] < dateval:
                        sourcedates[source][1] = dateval
                    if alldates and dateval not in sourcedatelist[source]:
                        sourcedatelist[source].append(dateval)
                else:
                    sourcecounts[source] = 1
                    sourcedates[source] = [dateval, dateval]
                    if alldates:
                        sourcedatelist[source] = [dateval,]
                if alldates and dateval not in alldatelist:
                    alldatelist.append(dateval)

    print "Counts & date ranges by source:"
    for source, dates in sourcedates.iteritems():
        minval = dates[0]
        minyear = minval / 10000
        minmonth = (minval % 10000) / 100
        minday = minval % 100
        maxval = dates[1]
        maxyear = maxval / 10000
        maxmonth = (maxval % 10000) / 100
        maxday = maxval % 100
        print "%s (%d): %d/%d/%d - %d/%d/%d" % \
              (source, sourcecounts[source],
               minyear, minmonth, minday, maxyear, maxmonth, maxday)

    if alldates:
        print "\nCounts & date ranges by source:"
        for source, dates in sourcedatelist.iteritems():
            print "\n", source
            sourcestring = ""
            curyear = 1900
            for date in sorted(dates):
                year = date / 10000
                if curyear != year:
                    if len(sourcestring) > 0:
                        print sourcestring[:-2]
                        sourcestring = ""
                    curyear = year
                month = (date % 10000) / 100
                day = date % 100
                sourcestring += "%d/%d/%d, " % (year, month, day)
            if len(sourcestring) > 0:
                print sourcestring[:-2]

        print "\nDays of week for dates in file:"
        for date in sorted(alldatelist):
            year = date / 10000
            month = (date % 10000) / 100
            day = date % 100
            dayofweek = weekDay(year, month, day)
            print "%d/%d/%d: %s" % (year, month, day, dayofweek)


def weekDay(year, month, day):
    """calculate day of week for any date after Jan. 1, 1900"""

    # First entry in monthOffset is a dummy (months start counting at 1)
    monthOffset = [-1, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    weekdays   = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                  'Friday', 'Saturday', 'Sunday']
    beforeLeap = 1 if month <= 2 else 0
    yearsElapsed = (year - 1900)
    leapyearCandidates = yearsElapsed - beforeLeap
    # Centuries are not leapyears, except centuries divisible by 400
    # We start in 1900, so after 100 years we get our first century leap
    leapdaysHappened = leapyearCandidates / 4 - leapyearCandidates / 100 + \
                       (leapyearCandidates + 300) / 400
    # Jan. 1, 1900 was a Monday (so index 0 into weekdays)
    # Simply add years, leapdays, and days so far this year (excl. leapday)
    dayOfWeek = (yearsElapsed * 365 + leapdaysHappened + monthOffset[month] + day - 1) % 7
    return weekdays[dayOfWeek]


def count_sentences(infile, textcol=2, header=False):
    """Count # sentences and # words in the corpus."""
    import csv
    csv.field_size_limit(1000000000)

    nrwords, nrsents = [], []
    headeradjustment = 1 if header else 0
    partcounter, longarticlecounter, zerolen_articles, zerolen_parts = 0, 0, 0, 0
    counter = -1
    prevprevid = -2
    previd = -1

    with open(infile, 'rU') as inf:
        if textcol == -1:  # textfile, not csv
            indata = inf
        else:  # csv file
            indata = csv.reader(inf)

        for counter, article in enumerate(indata):
            if header:
                header = False
            else:
                if textcol == -1:
                    articletext = article
                    articleid = counter
                else:
                    articletext = article[textcol]
                    articleid = article[0]

                # Don't count special characters -> 1-letter words should be alphanumeric
                # Also simply skip this entry if it is zero-word length
                articlewords = len([x for x in articletext.split() \
                                    if (len(x) > 1 or x.lower() in 'abcdefghijklmnopqrstuvxyz0123456789')])
                if articlewords > 0:
                    nrwords.append(articlewords)
                    nrsents.append(len(articletext.split('.')))
                else:
                    zerolen_articles += 1
                if articleid == previd:
                    if articlewords == 0:
                        zerolen_parts += 1
                    partcounter += 1
                    if articleid != prevprevid:  # 2nd part, not a higher number
                        longarticlecounter += 1
                    prevprevid = previd

                previd = articleid
                if counter % 100000 == 0:
                    print "Processing article %d" % counter
    nrarticles = counter + 1 - headeradjustment - partcounter - (zerolen_articles - zerolen_parts)
    totalwords = sum(nrwords)
    totalsents = sum(nrsents)
    print "\nArticles: %d, (%d extra parts, spread over %d articles), sentences: %d (mean %f), words: %d (mean %f)" % \
          (nrarticles, partcounter, longarticlecounter,
           totalsents, totalsents/float(nrarticles),
           totalwords, totalwords/float(nrarticles))
    print "Articles (or article parts) of length 0 ignored in this calculation: %d" % (zerolen_articles)


def articlecounts(corpus, targetyear=-1, targetmonth=-1, targetday=-1, header=True,
                  yearcol=4, monthcol=6, daycol=7):
    """Count articles by month, day for specified year.

    Option to pick a specific month and/or day within the year too.
    """
    import csv
    csv.field_size_limit(1000000000)

    articles = {}
    with open(corpus, 'rU') as infile:
        for row in csv.reader(infile):
            if header:
                header = False
            else:
                # Check if right year
                try:
                    year = int(row[yearcol])
                except ValueError:
                    print "Error: column %d does not contain a valid year: %s" % (yearcol, row[yearcol])
                    print row
                if targetyear == -1 or targetyear == year:
                    month = int(row[monthcol])
                    day = int(row[daycol])
                    # Check if right month & day
                    if (targetmonth == -1 or targetmonth == month) and \
                            (targetday == -1 or targetday == day):
                        if month in articles:
                            if day in articles[month]:
                                articles[month][day] += 1
                            else:
                                articles[month][day] = 1
                        else:
                            articles[month] = {day: 1}

    # Print results
    if targetyear != -1:
        print "\nYear: %d" % targetyear
    for month, days in sorted(articles.iteritems()):
        for day, val in sorted(days.iteritems()):
            print "Month %d: day %d: %d" % (month, day, val)


def articleyears(corpus, targetyear=-1, header=True):
    """Count articles by year, optionally for a specified year."""
    import csv
    header = True
    years = {}
    with open(corpus, 'rU') as infile:
        for row in csv.reader(infile):
            if header:
                header = False
            else:
                # Check if right year
                year = int(row[4])
                if targetyear == -1 or targetyear == year:
                    if year in years:
                        years[year] += 1
                    else:
                        years[year] = 1
                        # Print results
    for year, count in sorted(years.iteritems()):
        print "Year %d: %d" % (year, count)


# ****************************** combining corpora ****************************

def combine_corpora_multi(corpuslist, outputcorpus, header=True, minoffset=False):
    """Combine two or more corpora, making sure not to have overlapping ids.

    Normal offset is simply to add an order of magnitude
    This is nice because it preserves the starting id number
    (simply truncate the combined number)
    However, if we do this repeatedly, id numbers can blow up quickly,
    so the minoffset option makes a smaller adjustment
    """
    import csv

    # First get the largest id number in any of the corpora, to calculate offset
    maxid = max([idrange(corpus)[1] for corpus in corpuslist])
    offset = maxid + 10 if minoffset else int('1' + '0' * len(str(maxid)))
    print "\nAdding %d to each consecutive corpus, to set ids apart." % offset

    # Keep track of how many rows in each file
    counts = [0, ] * len(corpuslist)
    for counter, corpus in enumerate(corpuslist):
        corpheader = header
        with open(corpus, 'rU') as infile, \
                open(outputcorpus, 'ab') as outfile:
            outwriter = csv.writer(outfile)
            artcount = 0
            curid = -1
            for article in csv.reader(infile):
                if corpheader:
                    if counter == 0:  # for first corpus only
                        outwriter.writerow(article)
                    corpheader = False
                else:
                    rawid = int(article[0])
                    if rawid != curid:
                        artcount += 1
                        curid = rawid
                    id = counter * offset + rawid
                    outwriter.writerow([id, ] + article[1:])
            print "%d texts in %s" % (artcount, corpus)
            counts[counter] = artcount
    print "Total number of texts in combined corpus: %d" % sum(counts)
    return offset


def append_subcorpora(maincorpus, subcorpusfolder, subcorpussuffix='',
                      header=True, backupsuffix='_nopreprocessed',
                      minoffset=False):
    """Append subcorpora to the main corpus, saving the original main corpus i"""
    import csv
    import os

    csv.field_size_limit(1000000000)

    # Save a backup copy of the main corpus
    bkupcorpus = '.'.join(maincorpus.split('.')[:-1]) + backupsuffix + '.csv'
    os.rename(maincorpus, bkupcorpus)
    print "Saved a backup copy of main corpus as %s" % (os.path.split(bkupcorpus)[1])
    # Get a list of filenames within subcorpusfolder
    corpuslist = [bkupcorpus,] + findcsvfiles(subcorpusfolder, subcorpussuffix)
    print "Appending the following sub-corpora:"
    prefix_pathlength = len(subcorpusfolder)
    print [x[prefix_pathlength:] for x in corpuslist[1:]]
    # Call combine_corpora_multi to do the actual combining
    offset = combine_corpora_multi(corpuslist, maincorpus, header, minoffset)
    return offset


def findcsvfiles(afolder, suffix=''):
    """Return list of csv files inside a folder (recursive).

    Optionally take only those csv files with matching suffix.
    """
    import glob
    import os
    csvfiles = []
    filelist = glob.glob(afolder + '/*')
    for filename in filelist:
        if os.path.isdir(filename):
            csvfiles += findcsvfiles(filename, suffix)
        else:
            fname = os.path.split(filename)[1]
            if fname.split('.')[-1] == 'csv':
                if suffix == '':
                    csvfiles.append(filename)
                elif suffix == None:  # no suffixes accepted
                    if len(fname.split('_')) == 1:
                        csvfiles.append(filename)
                elif suffix + '.csv' == '_' + fname.split('_')[-1]:
                    csvfiles.append(filename)
    return csvfiles


def undo_corpusoffset(datafiles, newsuffix= '_X', header=True, offset=1000000):
    """Undo id offset."""
    import csv

    for infile in datafiles:
        instem = '.'.join(infile.split('.')[:-1])
        with open(infile, 'rU') as inf, open(instem + newsuffix + '.csv', 'wb') as outf:
            inreader = csv.reader(inf)
            outwriter = csv.writer(outf)
            if header:
                headerrow = next(inreader)
                outwriter.writerow(headerrow)
            for counter, row in enumerate(inreader):
                outwriter.writerow([int(row[0]) % offset,] + row[1:])


# ***************************** auxiliary functions **************************

def makelower(infile, outfile, header=False, textcol=1):
    """Make all text lower-case."""
    import csv
    csv.field_size_limit(1000000000)

    # outfilestem = '.'.join(textfile.split('.')[:-1])
    # outfile = outfilestem[:-1] + 'Y.csv' if outfilestem[-1] == 'X' else \
    #     outfilestem + '_lower.csv'
    with open(infile, 'rU') as inf, open(outfile, 'wb') as outf:
        output = csv.writer(outf)
        for article in csv.reader(inf):
            if header:
                header = False
            else:
                output.writerow([article[0], article[textcol].lower()])


def makelower_all(textfile, header=False):
    """Make all text lower-case."""
    import csv
    csv.field_size_limit(1000000000)

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


def filterdict(adict, mincount):
    """Remove entries in dictionary with a count less than mincount."""
    keystodelete = []
    for key in adict:
        if adict[key] < mincount:
            keystodelete.append(key)
    for key in keystodelete:
        del adict[key]
    return adict


def flag_presence(texts, words, matchcase=False):
    """Check for presence of at least 1 of words in texts."""
    if not matchcase:
        words = [w.lower() for w in words]
    print "Word list:", words
    return [1 if any([word in (text if matchcase else text.lower()) \
                      for word in words]) else 0 \
            for text in texts]


def presence_flags(texts, wordlists, matchcase=False):
    """Generate features representing presence of words in wordlists."""
    return zip(*[flag_presence(texts, wordlist, matchcase) for wordlist in wordlists])


# ***************************** external invocation **************************

if __name__ == "__main__":
    import sys
    from distutils.util import strtobool

    # See if we were called with the right number of arguments
    nrArgs = len(sys.argv) -1  # Arguments to python include program name
    if nrArgs == 0: # take hardcoded params (listed above function)
        pass
    elif nrArgs < 1 or nrArgs > 3:
        print "Error in nr. of arguments"
    elif nrArgs == 3:
        wordlist = getwords_fromfile(sys.argv[1], strtobool(sys.argv[2]), strtobool(sys.argv[3]))
        print "Words in corpus: %d" % len(wordlist)
        writewords('corpus_words.txt', wordlist)
    elif nrArgs == 2:
        pass
    else: # nrArgs == 1:
        # texts2lines_file(sys.argv[1])
        stripmeta(sys.argv[1])