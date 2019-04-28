# encoding: utf-8

# process_raw.py
# by Maurits van der Veen
# last modified 2017-11-21

# Python script to parse LexisNexis Academic, NexisUni, Factiva, or ProQuest output
# of plain-text document search results from a wide range of newspapers
# in different languages.

# Arguments: input file path, including wildcards
#            output filename
#            (optional) source database (default '')
#            (optional) number to add to all document numbers (default 0)

# Output a csv file with the specified filename, plus a _duplicates file
# containing duplicates


# ******************************* notes & thoughts ***************************

# Tries to make as few assumptions as possible regarding presence, order,
# and length (sometimes multi-line) of different types of meta-data;
# does not catch all issues, so it's important to look through & clean output.

# Handles very large texts by splitting into parts about 32600 chars in size.
# Filters out duplicate articles (published on same day, with title and text
# beginning the same (see details in each processing file).

# TODO: duplicate detection, source numbering, and month number
# TODO: should be spliced out, since they're not database-specific

# Currently handles at least the following newspaper sources:
# TODO: update list (is grossly outdated)

# DE: Die Welt / Welt am Sonntag, Die Zeit, Der Tagesspiegel, taz (die tageszeitung), Der Spiegel,
#     Bild am Sonntag
# DK: Politiken
# ES: Pais, Periodico de Catalunya
# FR: Le Monde, Le Figaro, Le Parisien
# IE: Irish Times
# IT: Corriere della Sera, La Stampa
# NL: NRC, Telegraaf, Volkskrant, Trouw, AD/Algemeen Dagblad
# UK: Guardian, Observer, Mail (incl. Sunday), Mirror (incl. Sunday),
#     Telegraph (incl. Sunday), Times (incl. Sunday), Financial Times,
#     Independent (incl. Sunday), i (by Independent), Star (incl. Sunday),
#     Daily Record, Sunday Mail, Evening Standard, Business (incl. Sunday),
#     News of the World, Express (incl. Sunday), People
# US: New York Times, Washington Post, USA Today, ...
# CDN: ...
# AUS: Sydney Morning Herald, ...
# NZ: ...


# ******************************* main functions ***************************

def process(infilespec, outfilename, database='', startDocNr=0,
            initfiles=False, excludeblogs=True, excludeweb=True,
            print_articlenr=False):
    """Process files from multiple databases."""
    import glob
    import os
    global skippedsources

    # Zero out outputfiles, so we don't append to existing files
    if initfiles:
        print "Erasing old %s and associated duplicates file." % outfilename
        init_file(outfilename)
        init_file(outfilename.replace('.', '_duplicates.'))

    skippedsources = []  # this resets skippedsources too often; maybe reset it outside?
    docNr = startDocNr
    infilelist = glob.glob(infilespec)
    for filename in infilelist:
        fname = os.path.split(filename)[1]
        if os.path.isdir(filename):
            lookingfor = '' if database == '' \
                else (', looking for %s downloads' % database)
            print 'Entering subdirectory %s%s' % (fname, lookingfor)
            if fname == 'LN' or database == 'LN':
                docNr = process(filename + '/*', outfilename, 'LN', docNr,
                                excludeblogs=excludeblogs, excludeweb=excludeweb,
                                print_articlenr=print_articlenr)
            elif fname == 'NexisUni' or database == 'NexisUni':
                docNr = process(filename + '/*', outfilename, 'NexisUni', docNr,
                                excludeblogs=excludeblogs, excludeweb=excludeweb,
                                print_articlenr=print_articlenr)
            elif fname == 'Factiva' or database == 'Factiva':
                docNr = process(filename + '/*', outfilename,
                                'Factiva', docNr,
                                excludeblogs=excludeblogs, excludeweb=excludeweb,
                                print_articlenr=print_articlenr)
            elif fname == 'ProQuest' or database == 'ProQuest':
                docNr = process(filename + '/*', outfilename,
                                'ProQuest', docNr,
                                excludeblogs=excludeblogs, excludeweb=excludeweb,
                                print_articlenr=print_articlenr)
            else:  # database remains unknown
                docNr = process(filename + '/*', outfilename, database, docNr,
                                excludeblogs=excludeblogs, excludeweb=excludeweb,
                                print_articlenr=print_articlenr)
        else:  # text file; process only if database specified
            if database in ['LN', 'NexisUni', 'Factiva', 'ProQuest']:
                olddocNr = docNr
                docNr = process_file(filename, outfilename, database, docNr,
                                     excludeblogs=excludeblogs, excludeweb=excludeweb,
                                     print_articlenr=print_articlenr)
                print "Processed %d articles in %s" % (docNr - olddocNr, fname)
            else:
                print "Ignoring %s (not a directory)" % fname
    print "Running tally: %d articles" % docNr
    return docNr


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


def processX(corpusfolder, subcorpora, outfilename, startDocNr=0):
    """Process files from list of databases.

    Database name must equal folder name.
    Right now works only for 'LN' and 'Factiva'.
    To be deprecated; process handles these cases just as well.
    """
    import processFactiva
    import glob
    import os

    docNr = startDocNr

    for database in subcorpora:
        infilelist = glob.glob(corpusfolder + database + '/*')
        for filename in infilelist:
            fname = filename if '/' not in filename else \
                        filename.split('/')[-1]
            if os.path.isdir(filename):
                print 'Entering subdirectory: ' + fname
                docNr = process(filename + '/*', outfilename, 'LN', docNr) \
                    if database == 'LN' else \
                    process(filename + '/*', outfilename,
                                   'Factiva', docNr)
            else: # assume this is a textfile to process
                olddocNr = docNr
                docNr = process_file(filename, outfilename, 'LN', docNr) \
                    if 'LN' in database else \
                    process_file(filename, outfilename, 'Factiva', docNr)
                print "Processed %d articles in %s" % (docNr - olddocNr, fname)
        print "Running tally: %d articles" % docNr
    return docNr


def process_tree(infilespec, outfilename, database='', startDocNr=0):
    """Process raw article output on a folder-by-folder level.

    To be deprecated; process handles this AFAIK."""
    import glob
    import os

    docNr = startDocNr
    infilelist = glob.glob(infilespec)
    for filename in infilelist:
        fname = filename if '/' not in filename else \
                    filename.split('/')[-1]
        if os.path.isdir(filename):
            print 'Entering subdirectory: ' + fname
            docNr = process_tree(filename + '/*', filename + '.csv', database, docNr)
        else: # assume this is a textfile to process
            olddocNr = docNr
            docNr = process_file(filename, outfilename, database, docNr)
            print "Processed %d articles in %s" % (docNr - olddocNr, fname)
    print "Running tally: %d articles" % docNr
    return docNr


def process_file(infilename, outfileName, database, startDocNr=0,
                 excludeblogs=True, excludeweb=True, print_articlenr=False):
    """Prepare to process 1 raw article file into csv format."""
    import csv
    import os

    # Get output files ready; append if file already exists
    # outfileName = infileName.replace(infileName.split('.')[-1],'csv')
    dupfileName = outfileName.replace('.','_duplicates.')
    if os.path.isfile(outfileName):
        counter = 0
        with open(outfileName, 'rU') as inf:
            for row in csv.reader(inf):
                counter += 1
                break
        header = True if counter == 0 else False
        outfile = open(outfileName,'ab')
        dupfile = open(dupfileName,'ab')
    else:
        header = True
        outfile = open(outfileName,'wb')
        dupfile = open(dupfileName,'wb')
    outwriter = csv.writer(outfile)
    dupwriter = csv.writer(dupfile)

    # Open inputfile and read the entire file into memory at once.
    # Given limitations on articles downloadable at once,
    # this is not a problem for either LexisNexis or Factiva.
    with open(infilename, 'rU') as infile:
        text = infile.read()
    # Sometimes we may need to remove byte order mark if present
    # text = text.replace('\xef\xbb\xbf\r\n','')

    if database == 'LN':
        return processLN_text(text, outwriter, dupwriter,
                              header, startDocNr,
                              excludeblogs, excludeweb, print_articlenr)
    elif database == 'NexisUni':
        return processNexisUni_text(text, outwriter, dupwriter,
                              header, startDocNr,
                              excludeblogs, excludeweb)
    elif database == 'Factiva':
        return processFactiva_text(text, outwriter, dupwriter,
                                   header, startDocNr,
                                   excludeblogs, excludeweb)
    elif database == 'ProQuest':
        return processProQuest_text(text, outwriter, dupwriter,
                                    header, startDocNr)
    else:
        print "Unknown database; not processing %s" % infilename
        return startDocNr


# ***************************** LexisNexis processing ************************

def processLN_text(text, outwriter, dupwriter, header, startDocNr,
                   excludeblogs, excludeweb, print_articlenr=False, print_info=False):
    """Process the text from a LexisNexis output file.

    Assumes main components of an article are separated by a blank line.
    """
    import csv
    import re
    from unidecode import unidecode
    global skippedsources

    # Replace article breaks by a character string that doesn't
    # occur anywhere else
    # LexisNexis articles end with "Copyright xxxx publisher-name".
    # Sometimes, copyright is rendered as the copyright character.
    textX = re.sub(r"\b[Cc]opyright( \(c\))?\s+[0-9]{4}.*?\n",
                    "---article separator AMvdV---", text)
    textX = re.sub("©\s+[0-9]{4}.*?\n",
                    "---article separator AMvdV---", textX)
    # Very rarely an extraneous Copyright line appears between the
    # first line (<nr> of <nr> DOCUMENTS) and the source name.
    # Since we split on the Copyright line, we need to deal with this now.
    textX = \
        re.sub(r"([0-9]+ of [0-9]+ DOCUMENTS\s+)---article separator AMvdV---\s+",
               "\\1", textX)

    # Remove any special characters that may mess up reading & writing files
    try:
        textX = unidecode(textX.decode('utf-8-sig'))
    except:
        try:
            textX = unidecode(textX.decode('utf-8'))
        except:
            textX = unidecode(textX.decode('ascii', 'ignore'))

    # Get list of meta-data headers available.
    # These are identified by full-caps, and beginning at start of a line
    # Take only the commonly occurring ones (those present in at least 33% of
    # the articles) and filter out the source name (often capitalized) itself

    # Note: if we have sources in more than 2 languages, key metadata may
    # have different names in different languages -> if so, need to merge
    # appropriately. See treatment of Type / Pub-Type below as example.

    # To keep columns similar across output files, hard-code a standard
    # set of meta-data headers; handle multiple languages
    meta_basics_engl = ['BYLINE','LENGTH','SECTION','LANGUAGE','LOAD-DATE','TYPE',
                        'PUB-TYPE','PUBLICATION-TYPE','JOURNAL-CODE','GRAPHIC']
    meta_basics = meta_basics_engl
    meta = list(set(re.findall(r'\n([A-ZÄ][A-ZÄ][A-ZÄ-]*?):', text)))
    meta = \
        [m for m in meta if float(text.count(m + ':')) / len(textX) > .33 \
         and m not in meta_basics]
    meta_list = meta_basics + meta

    # NRC has an odd pair of meta-data: SAMENVATTING without a colon, followed
    # by a blank line, coupled with VOLLEDIGE TEKST: with a colon but also
    # containing a space. Just hard-code for these.
    # For the NRC, also correct opening quotes (rendered as two consecutive
    # commas). (Later, may generalize to do this for all Dutch papers)
    if 'NRC Handelsblad' in textX and 'SAMENVATTING' in textX:
        textX = re.sub(',,' , '"', textX)
        meta_list = meta_list + ['SAMENVATTING',]

    meta_tuple=('DocNr', 'SearchResultNr', 'Publication', 'PubNr', 'Year',
                'Month', 'MonthNr', 'Day', 'Day of week', 'Sequence', 'Title',
                'Edition', 'Text', 'Part')
    for item in meta_list:
        if item <> 'TYPE' and item <> 'PUB-TYPE': # retain only PUBLICATION-TYPE
            meta_tuple += (item,)
    # Write the list of headers to the output file
    if header:
        outwriter.writerow(meta_tuple)
        dupwriter.writerow(meta_tuple)

    # Commas in text rarely cause problems in csv files.
    # If they do, set commaString to a space, or (to remember where commas
    # were) to ' @comma' (note space before @, to avoid concatenating words)
    commaString = ','

    # Split the textfile into a list of articles & initialize variables
    split_text = textX.split('---article separator AMvdV---')
    docsinfile = len(split_text)
    nrtooshort, nrduplicates, nrexcluded = 0, 0, 0

    # print "%d articles in %s" % (len(split_text), infilename)
    docID = startDocNr
    prevDay = 0
    prevMonth = 0
    prevYear = 0
    todayTitles = ''
    todayArticles = ''

    months_engl = ['january', 'february', 'march', 'april', 'may', 'june',
                   'july', 'august', 'september', 'october', 'november',
                   'december']
    months_engl_abbr = [x[:3] for x in months_engl]
    months_fran = ['janvier', 'fevrier', 'mars', 'avril', 'mai', 'juin',
                   'juillet', 'aout', 'septembre', 'octobre', 'novembre',
                   'decembre']

    # Loop over each article in the list
    for a_doc in split_text:
        # Split into an array of (non-empty) lines
        a_docX = a_doc.split('\n')
        # Make sure it's not an empty document
        if len(a_docX) < 5 or 'DOCUMENTS' not in a_doc:
            nrtooshort += 1
            continue

        # Initialize variables not all papers may have info for
        edition = ''
        docDayofweek = ''

        # Skip to the line that has the document number in it
        # (each LN article begins with a line ending in 'n of N DOCUMENTS')
        curLine = 0
        while 'DOCUMENTS' not in a_docX[curLine]:
            curLine += 1
        num = re.search('[0-9]+', a_docX[curLine])
        searchResNr = int(num.group())
        docID += 1
        if print_articlenr:
            print searchResNr, docID  # use to pinpoint error source

        # Now get source information, which may be more than 1 line long,
        # but is off-set on both sides by blank lines. For the Guardian,
        # the source name line also has edition information on it
        # (as of Jan. 1, 2004). This code will produce the dummy
        # edition info "The" for articles prior to that date
        curLine += 1
        while len(a_docX[curLine].lstrip()) == 0:
            curLine += 1

        # Extract sourceName from the first source info line
        sourceName = a_docX[curLine].lstrip().lower()
        # strip information in parentheses at end (space & location info, etc.)
        sourceName = sourceName.split('(')[0].rstrip()

        # Known erroneous source names in LexisNexis:
        #       The The Sunday Times, SUNDAY SUTELEAPH <- fixed here
        #       Washington Posts <- fixed in downloaded files
        if sourceName[:4] == "the ":
            sourceName = sourceName[4:]
            # Test for "The The Sunday Times"
            if sourceName[:4] == "the ":
                sourceName = sourceName[4:]
        # Test for "SUNDAY SUTELEAPH"
        if sourceName == "sunday suteleaph":
            sourceName = "sunday telegraph"
        sourceName = sourceName.split(' - ')[0].rstrip()

        # Simplify/correct source names (also for sources that changed name over time)
        if sourceName == 'atlanta journal and constitution':
            sourceName = 'atlanta journal-constitution'
        if sourceName == 'washington posts':
            sourceName = 'washington post'
        if 'financial post' in sourceName:  # also get longer versions of name
            sourceName = 'national post'
        if 'sunday herald sun' in sourceName:  # also get SHS magazine
            sourceName = 'herald sun'
        # Both the Scottish "Daily Record and Sunday Mail" and Australia's "Courier Mail"
        # have "Sunday Mail" as their Sunday paper name => don't change, disambiguate later if needed
        # if sourceName == 'sunday mail':
        #     sourceName = 'courier mail'
        # if 'sunday telegraph' in sourceName:  # also get ST magazine
            sourceName = 'daily telegraph'  # beware: name equivalence between UK & Aus versions!!
        if sourceName == 'mercury news' or sourceName == 'san mateo county times':
            sourceName = 'san jose mercury news'

        # Exclude sources not meeting our criteria
        if skipsource(sourceName, excludeblogs, excludeweb):
            nrexcluded += 1
            if sourceName not in skippedsources:
                print "Skipping article from %s" % sourceName
                skippedsources.append(sourceName)
            continue  # move on to next article

        # strip source name (or words therein) from the meta list
        # (might be in there due to interviews)
        if sourceName in meta_list:
            meta_list.remove(sourceName)
        elif any(x in meta_list for x in sourceName.split()):
            for x in sourceName.split():
                if x in meta_list:
                    meta_list.remove(x)

        # Identify television sources
        istv = any([x in sourceName for x in ('abc news', 'cbs news', 'fox news', 'nbc news',
                                              'cnn', 'pbs news', 'lehrer')])

        # Deal with Guardian edition info
        if 'guardian' in sourceName:
            sourceName = 'guardian'
            editionX = a_docX[curLine].split(' - ')
            if len(editionX) > 1:
                edition = editionX[1].lstrip()
            # editionX = a_docX[curLine].replace(
            #               'The Guardian (London) - ','')
            # editionX = editionX.lstrip()
            # edition = editionX.split(' ')[0]

        # Skip past additional lines of source info
        # (source name is always on first line)
        while curLine < len(a_docX) and not len(a_docX[curLine].lstrip()) == 0:
            curLine += 1
        # Skip past the blank lines that follow the source information
        while curLine < len(a_docX) and len(a_docX[curLine].lstrip()) == 0:
            curLine += 1
        if len(a_docX) <= curLine + 1:
            continue  # have reached end of doc without useful info -> skip

        alter_metas = False
        # Update list of basic meta info, depending on language
        if any(x in sourceName for x in ['zeit', 'welt', 'spiegel', 'zeitung', 'sonntag']):
            # Sometimes German-language sources have English metadata names
            meta_basicsX = meta_basics_engl
            alter_metas = True
            meta_basics = ['AUTOR', 'LANGE', 'RUBRIK', 'SPRACHE', 'UPDATE', 'TYPE',
                            'PUB-TYPE', 'PUBLICATION-TYPE', 'ZEITUNGS-CODE', 'GRAFIK']
            meta_list = meta_basics + [x for x in meta_list[10:] if x not in meta_basics]
            # taz has HIGHLIGHT as a summary, which simply reproduces the subtitle
            # capture it, so we don't have that text in there twice when combining title & text
            if 'tageszeitung' in sourceName and 'HIGHLIGHT' not in meta_list:
                meta_list.append('HIGHLIGHT')

        # CNN also often has 'highlight' info -- filter out to avoid causing erroneous isduplicate flag
        if 'cnn' in sourceName and 'HIGHLIGHT' not in meta_list:
            meta_list.append('HIGHLIGHT')
        # All tv transcripts tend to have a guests field (todo: see if sometimes it is also 'GUEST'
        if istv:
            if 'GUESTS' not in meta_list:
                meta_list.append('GUESTS')

        elif 'monde' in sourceName or 'figaro' in sourceName or 'parisien' in sourceName:
            meta_basics = ['AUTEUR', 'LONGUEUR', 'RUBRIQUE', 'LANGUE', 'DATE-CHARGEMENT', 'TYPE',
                            'PUB-TYPE', 'PUBLICATION-TYPE', 'JOURNAL-CODE', 'GRAPHIC']
            meta_list = meta_basics + [x for x in meta_list[10:] if x not in meta_basics]

        # Now get the date
        docDate = a_docX[curLine].lstrip()
        date_pending = False

        if 'McClatchy' in docDate or 'This content' in docDate or 'Distributed by' in docDate:
            # distributed-by notice -> skip
            while curLine < len(a_docX) and not len(a_docX[curLine].lstrip()) == 0:
                curLine += 1
            # Skip past the blank lines that follow the source information
            while curLine < len(a_docX) and len(a_docX[curLine].lstrip()) == 0:
                curLine += 1
            if len(a_docX) == curLine + 1:
                continue  # have reached end of doc without useful info -> skip
            docDate = a_docX[curLine].lstrip()

        if sourceName in ['straits times', 'bond buyer']:
            temptitle = []
            docDate_parts = docDate.split()
            if len(docDate_parts) < 3 or docDate_parts[2][:2] not in '1920':
                 # title first, then date
                while curLine < len(a_docX) and not len(a_docX[curLine].lstrip()) == 0:
                    temptitle += a_docX[curLine].strip() + ' '
                    curLine += 1
                if len(temptitle) > 0:
                    temptitle = title[:-1]
                # Skip past the blank lines that follow the title
                while curLine < len(a_docX) and len(a_docX[curLine].lstrip()) == 0:
                    curLine += 1
                if len(a_docX) == curLine + 1:
                    continue  # have reached end of doc without useful info -> skip
                docDate = a_docX[curLine].lstrip()

        if any(x in sourceName for x in ['politiken', 'figaro', 'parisien',
                                         'welt', 'zeit', 'spiegel', 'zeitung']):
            # Date format "<day-of-week> <day-nr>. <month-string> <year>
            docDate = docDate.replace('.','')  # remove periods (for example 1. for date)
            docDate_split = docDate.split(' ')
            if len(docDate_split) > 3:
                docDayofweek = docDate_split[0]
                docDate_split = docDate_split[1:]
            docYear = docDate_split[2]
            docMonth = docDate_split[1]
            docDay = docDate_split[0]

        elif any(x in sourceName for x in \
                   ['della sera', 'stampa', 'nrc', 'volkskrant', 'telegraaf',
                    'dagblad', 'trouw', 'zeit', 'catalunya', 'sonntag', 'mundo']) \
                or ('monde' in sourceName and not (',' in docDate)):
            # Date format "<day-nr> <month-string> <year> <day-of-week>"
            # except in some results, where it is
            # "<month-string> <day-nr>, <year>"
            docDate = docDate.replace(',', '')  # remove commas
            docDate_split = docDate.split(' ')
            docYear = docDate_split[2]
            docMonth = docDate_split[1]
            docDay = docDate_split[0]
            if docDay[-1] == '.':
                docDay = docDay[:-1]
            # See if alternate Dutch/Italian/Spanish format, in which case change variables
            if any(x in sourceName for x in ['nrc', 'volkskrant', 'telegraaf',
                                             'trouw', 'stampa', 'mundo']) \
                        and is_int(docMonth):
                docMonth = docDate_split[0]
                docDay = docDate_split[1]
            docDayofweek = docDate_split[3] if len(docDate_split) > 3 else ''
            # Corriere, Stampa, NRC, Mundo may have edition info on line following date
            if any(x in sourceName for x in ['della sera', 'stampa', 'nrc', 'mundo']):
                curLine += 1
                edition = a_docX[curLine].strip()
                if len(edition) == 0:  # was a blank line -> reset curLine
                    curLine -= 1

        elif sourceName == 'gazette' and \
                (len(docDate.split()) < 3 or not is_year(docDate.split()[2][:4]) or
                 not is_month(docDate.split()[0])):
            # Sometimes Montreal Gazette skips date; get from load-date instead
            date_pending = True
            curLine -= 1

        else: # Default English-language paper
            # Date format "<month-string> <day-nr>, <year> <day-of-week>"
            # (Observer, SundayTimes do not have day-of-week)
            # Some papers may have edition information
            # on the same line or on the 1-2 lines following the date
            docDate = docDate.replace(',', ' ')  # replace commas by spaces
            docDate = docDate.replace('  ', ' ') # should not be necessary

            docDate_split = docDate.split(' ')
            if len(docDate_split) < 3: # not one of our sources -> just skip
                # print "docDate:", docDate
                continue

            # Make sure we don't accidentally have weekday first
            # or the date is in yyyy-mm-dd format
            if docDate_split[0].lower() not in months_engl + months_engl_abbr + months_fran:
                if docDate_split[0].find('-') != -1:  # contains hyphen -> assume yyyy-mm-dd
                    docDateinfo = docDate_split[0].split('-')
                    docYear = docDateinfo[0]
                    docMonth = months_engl[int(docDateinfo[1])-1]
                    docDay = docDateinfo[2]
                elif len(docDate_split) < 4:
                    continue
                else:
                    docDayofweek = docDate_split[0]
                    docYear = docDate_split[3][:4]
                    docMonth = docDate_split[1]
                    docDay = docDate_split[2]
            else:
                docYear = docDate_split[2][:4]
                docMonth = docDate_split[0]
                docDay = docDate_split[1]
            # Get day of week if not yet assigned
            if docDayofweek == '' and len(docDate_split) > 3:
                docDayofweek = docDate_split[3]
            if docDayofweek == '' and ('sunday' in sourceName or
                                       sourceName in ['observer', 'news of the world']):
                docDayofweek = 'Sunday'

            if len(docDate_split) > 4:  # edition info on same line as date
                if sourceName == 'irish times':  # no date of week either, just edition
                    edition = ' '.join(docDate_split[3:])
                    docDayofweek = ''
                else:
                    edition = ' '.join(docDate_split[4:])

        # See if edition info on the next line
        # Note: This condition & action should probably be merged with the next!
        if sourceName in ['new york times', 'washington post',
                          'richmond times dispatch', 'philadelphia daily news'] \
                and edition == '':
            curLine += 1
            edition = a_docX[curLine].lstrip()
            if 'Correction' in edition:
                # Actual edition info on next line
                curLine += 1
                edition = a_docX[curLine].lstrip()
            if not ('Late Edition' in edition \
                    or 'New York Times on the Web' in edition \
                    or 'International Herald Tribune' in edition):
                edition = ''

        # This next test may become too generic; check to make sure
        # papers with similar names don't accidentally get caught!!
        elif any([x in sourceName for x in ('mail', 'telegraph', 'times',
                'sun', 'independent', 'daily star', 'express', 'mirror',
                'parisien', 'della sera', 'pais')]) \
                or sourceName in ('daily record', 'scotsman',
                                  'evening standard', 'people',
                                  'guardian', 'news of the world',
                                  'usa today', 'gazette',
                                  'national post'):
            # Edition information and locality information
            # on 0 or more lines after the date
            edinfo = a_docX[curLine+1].lstrip()
            while len(edinfo) > 0:
                edition += ' ' + edinfo
                curLine += 1
                edinfo = a_docX[curLine+1].lstrip()
            edition = edition.lstrip()

        # Check for television programs: program title goes in "Edition" field
        elif istv:
            # Show information follows blank line
            edinfo = a_docX[curLine+1].lstrip()
            if len(edinfo) == 0:  # blank line
                curLine += 1
                edinfo = a_docX[curLine + 1].lstrip()
            # Now check for 'SHOW' info
            if edinfo.split(':')[0].strip() == 'SHOW':  # Show name in edition field
                while len(edinfo) > 0:
                    edition += ' ' + edinfo
                    curLine += 1
                    edinfo = a_docX[curLine+1].lstrip()
            edition = edition.lstrip()

        # Initialize list of meta items
        meta_dict={k : '' for k in meta_list}

        # Get article title. Assume it starts after the article header
        # info (document count, publication, edition, date) and is
        # followed by article text or meta-data, but separated from those
        #  by a blank line.
        title = ''
        curLine = curLine +1
        while curLine < len(a_docX) and len(a_docX[curLine].lstrip()) == 0:
            curLine = curLine +1
        while curLine < len(a_docX) and not len(a_docX[curLine].lstrip()) == 0:
            titleLine = a_docX[curLine].lstrip()
            title = title + titleLine.replace(',', commaString) + ' '
            curLine = curLine +1

        # Make sure this was not a piece of meta-data rather than a title
        # (some sources have articles without titles)
        if any([x in sourceName for x in \
                ('new york times', 'nrc', 'spiegel', 'parisien', 'pais')]) or istv:
            ismeta = re.findall('^([A-ZÄ][A-ZÄ-]*?):', title)
            if len(ismeta) > 0:  # This article has no title
                if alter_metas and ismeta[0] in meta_basicsX:  # replace meta marker
                    new_meta = meta_list[meta_basicsX.index(ismeta[0])]
                    title = new_meta + title[len(ismeta[0]):]
                    ismeta = [new_meta, ]
                if ismeta[0] in meta_list:
                    meta_dict[ismeta[0]] = \
                        title.replace(ismeta[0] + ': ', '')
                title = ''

        # Check if it is NYT article with byline on same line as title
        # Try to make sure it is not a title beginning with the word by,
        # by testing for ';' in the line, which should separate byline
        # from title
        elif sourceName == 'new york times' and title[:3] == 'BY ' \
                and ';' in title:
            titlesplit = title.split(';')
            title = titlesplit[1]
            meta_dict['BYLINE'] = titlesplit[0][3:]

        if sourceName in ['straits times', 'bond buyer'] and len(temptitle) > 0:
            # already had title; this was author
            meta_dict['BYLINE'] = title
            title = temptitle

        # Now get the various pieces of meta-data, along with main text
        # Assume each meta-data item is separated by a blank line
        # from the next or from the main text
        text = ['',]
        textnr = 0

        while curLine < len(a_docX):
            line = a_docX[curLine].lstrip()
            if len(line) > 0:
                line = line.replace(',',commaString)

                # If this is a byline, make it recognizable
                wordcaps = all([x[0] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' for x in line.split()])
                if line[:3] == 'By ' and wordcaps:
                    line = 'BYLINE: ' + line[3:]
                # Deal with meta-data first
                ismeta = re.findall('^([A-ZÄ][A-ZÄ-]*?):', line)
                if 'nrc' in sourceName and line == 'SAMENVATTING':
                    ismeta = ['SAMENVATTING',]
                    curLine = curLine + 2  # skip blank line
                    line = a_docX[curLine].lstrip().replace(',', commaString)
                # Make sure this is meta-data we wish to keep.
                # If not, it is either uncommon meta-data or normal text.
                # Avoid skipping over the latter, so just include.
                if len(ismeta) > 0 and alter_metas and ismeta[0] in meta_basicsX:  # replace meta marker
                    new_meta = meta_list[meta_basicsX.index(ismeta[0])]
                    line = new_meta + line[len(ismeta[0]):]
                    ismeta = [new_meta,]

                elif len(ismeta) > 0 and ismeta[0] in meta_list:  # meta marker to store
                        meta_dict[ismeta[0]] = \
                            line.replace(ismeta[0] + ':', '') + ' '
                        # move to next line, in case of multi-line meta-data
                        curLine = curLine +1
                        if curLine == len(a_docX): # end of article
                            break  # exit while loop
                        line = a_docX[curLine].lstrip()
                        while len(line) > 0 or ('nrc' in sourceName and
                                                ismeta[0] == 'SAMENVATTING'):
                            if not ('nrc' in sourceName
                                    and ismeta[0] == 'SAMENVATTING'
                                    and 'VOLLEDIGE TEKST' in line):
                                # Not doing SAMENVATTING, or not done with it yet
                                meta_dict[ismeta[0]] = \
                                    meta_dict[ismeta[0]] + \
                                    line.replace(',', commaString) + ' '
                            else: # Done with SAMENVATTING; change meta setting
                                ismeta[0] = 'not_a_meta'
                            curLine += 1
                            if curLine == len(a_docX): # end of article
                                break  # exit while loop
                            line = a_docX[curLine].lstrip()

                # Now deal with what is most likely document text,
                # but possibly some other stuff
                else:
                    if text[0] == '' and sourceName == 'irish times' and \
                            meta_dict['BYLINE'] == '' and \
                            line.split()[0] == line.split()[0].upper():
                        # Irish Times byline without following space
                        authorname, restofline = splitITname(line)
                        meta_dict['BYLINE'] = authorname

                    # Skip '***' separator lines (Zeit) and
                    #     Politiken author emails
                    elif (line != line.upper() and
                          not (line == line.split()[0] and
                               line.endswith('@pol.dk'))):

                        text[textnr] += line + ' '
                        if len(text[textnr]) > 32600:
                            textnr += 1
                            text.append('')
            curLine = curLine +1

        # Fill in text from graphic description if text is empty
        if text[0] == '' and sourceName == 'der tagesspiegel' \
                and not meta_dict['GRAFIK'] == '':
            text[0] = meta_dict['GRAFIK']
            meta_dict['GRAFIK'] = '<text accompanying graphic moved to main text>'

        # Calculate some additional data from the info gathered so far,
        # to prep for writing out
        country = 'US'  # TODO: determine this by sourceName; easy enough
        pubName = sourceName

        if country == 'UK':
            pubs = ['guardian', 'observer', 'mail', 'mirror',
                    'daily telegraph', 'sunday telegraph', 'times',
                    'sunday times', 'record', 'star', 'financial times',
                    'independent', 'i', 'evening standard', 'express', 'sun',
                    'people', 'news of the world', 'business']
            if 'mirror' in pubName:
                pubName = 'mirror'
            elif 'daily mail' in pubName or pubName == 'mail on sunday':
                pubName = 'mail'
            elif 'daily record' in pubName or 'sunday mail' in pubName:
                pubName = 'record'
            elif 'star' in pubName:
                pubName = 'star'
            elif 'i-' in pubName:
                pubName = 'i'
            elif 'independent' in pubName:
                pubName = 'independent'
            elif pubName == 'telegraph':
                pubName = 'daily telegraph'
            elif 'express' in pubName:
                pubName = 'express'
            elif 'business' in pubName:
                pubName = 'business'
            if pubName in pubs:
                pubNr = 1 + pubs.index(pubName)
            else:
                pubNr = 0

        elif country == 'US':
            # Assign publication code to the publication
            US_sourcelist = ['new york times', 'washington post', 'usa today',
                             'wall street journal', 'daily news',
                             'atlanta journal-constitution',
                             'san jose mercury news', 'denver post',
                             'new york post', 'boston globe',
                             'philadelphia inquirer', 'tampa bay times',
                             'star-tribune']
            try:
                pubNr = US_sourcelist.index(sourceName.lower())
            except ValueError:
                pubNr = -1

        else:
            pubNr = -1

        # Get date, if still pending
        if date_pending:
            if len(meta_dict['LOAD-DATE']) > 0:
                dateparts = meta_dict['LOAD-DATE'].split()
                docYear = dateparts[2]
                docMonth = dateparts[0]
                docDay = int(dateparts[1][:-1]) -1   # don't include comma,
                                                     # and subtract 1 because loaded next day
            else:
                docYear = '0000'
                docMonth = 'dummy'
                docDay = '0'

        # Get month nr
        if 'catalyunya' in sourceName or 'mundo' in sourceName:
            months = ['enero', 'febrero', 'marzo', 'abril', 'mayo',
                      'junio', 'julio', 'agosto', 'septiembre',
                      'octubre', 'noviembre', 'diciembre']
        elif "della sera" in sourceName or 'stampa' in sourceName:
            months = ['gennaio', 'febbraio', 'marzo', 'aprile', 'maggio',
                      'giugno', 'luglio', 'agosto', 'settembre', 'ottobre',
                      'novembre', 'dicembre']
        elif any(paper in sourceName for paper in \
                 ('volkskrant', 'telegraaf', 'nrc', 'algemeen dagblad', 'trouw')):
            months = ['januari', 'februari', 'maart', 'april', 'mei',
                      'juni', 'juli', 'augustus', 'september',
                      'oktober', 'november', 'december']
        elif any(x in sourceName for x in ['zeit', 'welt', 'spiegel', 'zeitung', 'sonntag']):
            months = ['januar', 'februar', 'marz', 'april', 'mai',
                      'juni', 'juli', 'august', 'september', 'oktober',
                      'november', 'dezember']
        elif 'monde' in sourceName or 'figaro' or 'parisien' in sourceName:
            months = months_fran
        else:
            months = months_engl
        if docMonth.lower() in months:
            monthNr = 1 + months.index(docMonth.lower())
        elif docMonth.lower() in months_engl:
            monthNr = 1 + months_engl.index(docMonth.lower())
        elif docMonth.lower() in months_engl_abbr:
            monthNr = 1 + months_engl_abbr.index(docMonth.lower())
        elif docMonth.lower() == 'maerz':  # German umlaut-less spelling
            monthNr = 3
        else:
            monthNr = 0
        if monthNr != 0 and docDayofweek == '':
            docDayofweek = weekDay(int(docYear), monthNr, int(docDay))

        sequenceNr = 0  # Dummy value so far -> change as applicable

        # Combine the TYPE, PUB-TYPE, PUBLICATION-TYPE metas
        thetype = meta_dict['TYPE'] + meta_dict['PUB-TYPE'] + \
                  meta_dict['PUBLICATION-TYPE']

        # For some Dutch newspapers (the NRC, the Volkskrant, and Trouw, at least),
        # some articles contain, instead of the text, the line
        # "De uitgever heeft voor dit artikel geen publicatierecht".
        # Skip these
        if len(text[0]) < 100 and "voor dit artikel geen publicatierecht" in text[0]:
            continue

        # Check if this is a duplicate, to determine where to write it to
        # If the title is long and non-generic (i.e. not "corrections", for example)
        # or if the article begisn the same way, count it as duplicate
        if prevDay == docDay and prevMonth == docMonth and \
                prevYear == docYear:
            titleComp = min(50,len(title))
            isDuplicate = (titleComp > 32 and title[:titleComp] in todayTitles) \
                            or text[0][:200] in todayArticles
            # Add to list for today
            todayTitles = todayTitles + title + ' '
            todayArticles = todayArticles + text[0][:200] + ' '
        else:  # different day -> reset vars
            isDuplicate = False
            prevDay = docDay
            prevMonth = docMonth
            prevYear = docYear
            todayTitles = title + ' '
            todayArticles = text[0] + ' '

        # Finally write results to the appropriate csv file
        # Consider re-encoding character strings
        basicdata = [x if not isinstance(x, str) else \
                     x.decode('ascii', 'ignore') for x in \
                        [docID, searchResNr, pubName, pubNr, docYear,
                         docMonth, monthNr, docDay, docDayofweek,
                         sequenceNr, title, edition]]
        extradata = []


        for item in meta_list:
            if item == 'TYPE':
                extradata.append(thetype)
            elif item <> 'PUB-TYPE' and item <> 'PUBLICATION-TYPE' \
                    and item <> 'TYPE-PUBLICATION':
                extradata.append(meta_dict[item])
        textrows = [basicdata + [x, nr] + extradata \
                    for nr, x in enumerate(text)]
        if isDuplicate:
            nrduplicates += 1
            dupwriter.writerows(textrows)
            docID -= 1
        else:
            outwriter.writerows(textrows)

    if print_info:
        print "%d docs in text file: %d duplicates, %d too short, %d source-excluded" % \
              (docsinfile, nrduplicates, nrtooshort, nrexcluded)

    return docID


# ****************************** NexisUni processing **************************

def processNexisUni_text(text, outwriter, dupwriter, header, startDocNr,
                         excludeblogs, excludeweb):
    """Process the text from a NexisUni output file.

    Note: NexisUni may want to produce rtf; our input needs to be plain text.
    Open in TextEdit first, and choose command-shift-T to switch to plain.
    """
    import csv
    import re
    from unidecode import unidecode
    global skipped_sources

    # Remove any special characters that may mess up reading & writing files
    try:
        text = unidecode(text.decode('utf-8-sig'))
    except:
        try:
            text = unidecode(text.decode('utf-8'))
        except:
            text = unidecode(text.decode('ascii', 'ignore'))

    # Commas in text rarely cause problems in csv files.
    # If they do, set commaString to a space, or (to remember where commas
    # were) to ' @comma' (note space before @, to avoid concatenating words)
    commaString = ','

    # NexisUni puts search result summaries at the front of a file,
    # before getting to the actual full-text.

    # Get the number of documents in the file.
    # A line near the top should say something like 'Documents (100)'.
    textlines = text.split('\n')
    totallines = len(textlines)
    curline = 0
    while curline < totallines and textlines[curline][:11] != 'Documents (':
        curline += 1
        if curline == totallines:
            print "Error: never found a line indicating nr. of documents. Exiting."
            return
    endparen = textlines[curline].find(')')
    nrdocs = textlines[curline][11:endparen]
    print "Number of docs in file: %s" % nrdocs

    # Find the start of the summary info for the last document
    nrdigits = len(nrdocs)
    while curline < totallines and textlines[curline][:nrdigits + 3] != nrdocs + '.  ':
        curline += 1
        if curline == totallines:
            print "Error: did not find last summary heading. Exiting."
            return

    # Find the end of the last summary information
    # Last line should begin 'Timeline: ', but sometimes it does not begin on a new line
    while curline < totallines and 'Timeline: ' not in textlines[curline]:
        curline += 1
        if curline == totallines:
            print "Error: did not find end of last summary heading. Exiting."
            return

    # This should be followed by a single blank line, followed by the title of the first article.
    # The end of each article is marked by 'End of Document' on a line by itself
    fulltext = '\n'.join(textlines[curline + 2:])
    articles = fulltext.split('\nEnd of Document\n')
    if len(articles) != int(nrdocs):
        if len(articles[-1].split('\n')) > 5:
            print "Error: wrong number of articles. Exiting."
            return
        elif len(articles) != int(nrdocs) +1:  # try ignore last 'article'
            print "Last article too short, but cannot ignore. Exiting."
            return
        else:
            articles.pop()

    # Start processing actual articles

    docID = startDocNr
    prevDay = 0
    prevMonth = 0
    prevYear = 0
    todayTitles = ''
    todayArticles = ''

    # First make list of items to store.
    header_tuple=('DocNr', 'SearchResultNr', 'Publication', 'PubNr', 'Year',
                  'Month', 'MonthNr', 'Day', 'Day of week', 'Sequence', 'Title',
                  'Edition', 'Text', 'Part', 'Author', 'Length', 'Section',
                  'Language', 'Load-Date', 'Type', 'Photos', 'Graphic')
    # Write the list of headers to the output file
    if header:
        outwriter.writerow(header_tuple)
        dupwriter.writerow(header_tuple)

    # We have a list of articles. Run through them
    for counter, article in enumerate(articles):

        seqnr = counter
        docID += 1
        articledata = {}  # empty dictionary for metadata
        articlelines = article.split('\n')
        totallines = len(articlelines)
        curline = 0

        # Article should start with title, source, date/edition, then blank line(s)
        # Allow for title to possibly run across more than 1 line
        while curline < totallines and articlelines[curline].strip() != '':
            curline += 1
            if curline == totallines:
                print "Error: did not find blank line marking end of title/source/date info. Skipping article."
                continue
        title = ' '.join(articlelines[:curline -2]).strip()

        source = articlelines[curline -2].strip().lower()
        # shorten & simplify publication name
        if source[:4] == 'the ':
            source = source[4:]

        # Exclude sources not meeting our criteria
        if skipsource(source, excludeblogs, excludeweb):
            if source not in skippedsources:
                print "Skipping article from %s" % source
                skippedsources.append(source)
            continue  # move on to next article

        # Note: Date might have different formats depending on language
        # (See all the possibilities in processLN_text above)
        date_ed_info = articlelines[curline -1].strip()
        datesplit = date_ed_info.split()
        edition = '' if len(datesplit) <= 4 else datesplit[4:]
        date_pending = False
        docYear = datesplit[2]
        if docYear[-1] == ',':  # Sometimes the year is followed by a comma
            docYear = docYear[:-1]
        docMonth = datesplit[0]
        docDay = datesplit[1][:-1]
        docDayofweek = datesplit[3][:-1]

        # Next get copyright line
        while curline < totallines and articlelines[curline][:10] != 'Copyright ':
            curline += 1
            if curline == totallines:
                print "Error: Expected to find copyright line. Skipping article."
                continue

        # Get metadata info
        # One or more of the metadata might be missing
        # Assume body is marked by the word 'Body'
        curline += 1
        metastartline = curline
        while curline < totallines and articlelines[curline].strip() != 'Body':
            curline += 1
            if curline == totallines:
                print "Error: Expected to find 'Body' marker. Skipping article."
                continue

        curmeta = None
        curmetadata = ''
        for metaline in range(metastartline, curline):
            if len(articlelines[metaline].strip()) == 0:
                if curmeta != None:
                    articledata[curmeta] = curmetadata
                curmeta = None
                curmetadata = ''
            elif articlelines[metaline].split()[0][-1] != ':':  # continuation of previous line
                curmetadata += ' ' + articlelines[metaline].rstrip()
            else:  # line begins 'ddddd:' where ddddd is some metadata flag
                if curmeta != None:
                    articledata[curmeta] = curmetadata
                curmeta = articlelines[metaline].split()[0][:-1]
                metalen = len(curmeta)
                curmetadata = articlelines[metaline][metalen + 2:]
        # flush last metadata
        if curmeta != None:
            articledata[curmeta] = curmetadata

        # Body text begins 'Body' followed by a couple of blank lines.
        # Skip past blank lines first
        curline += 1
        while curline < totallines and articlelines[curline].strip() == '':
            curline += 1
            if curline == totallines:
                print "Error: did not body of article. Skipping article."
                continue
        # Body ends with a blank line followed by either metadata or a URL
        # To be as robust as possible, set it up to accept blank lines within body
        bodystartline = curline
        seenblank = False
        while curline < totallines and \
                not (seenblank and (articlelines[curline][:4] == 'http' or \
                                    (len(articlelines[curline].split()) > 0 and \
                                     articlelines[curline].split()[0][-1] == ':'))):
            curline += 1
            if curline == totallines:  # end of article -> assume end of body
                continue
            if len(articlelines[curline].strip()) == 0:
                seenblank = True

        # If text begins with an abstract, append that to title -- this makes duplicates easier to catch
        if articlelines[bodystartline].strip() == 'ABSTRACT':
            fulltextline = -1
            for textline in range(bodystartline, curline):
                if articlelines[textline].strip() == 'FULL TEXT':
                    fulltextline = textline
                    break
            if fulltextline != -1:
                title += ' '.join(articlelines[bodystartline + 1:fulltextline])
                bodystartline = fulltextline + 1
            else:
                bodystartline += 1  # simply skip over 'ABSTRACT' label

        # If body text is extra long, chunk it up
        bodytext = ['', ]
        textnr = 0
        for textline in range(bodystartline, curline):
            line = articlelines[textline].strip()
            if line != 'ABSTRACT' and line != 'FULL TEXT':
                bodytext[textnr] += line + ' '
            if len(bodytext[textnr]) > 32000:
                textnr += 1
                bodytext.append('')

        # Now process remaining metadata
        curmeta = None
        curmetadata = ''
        for metaline in range(curline, totallines):
            if len(articlelines[metaline].strip()) == 0:
                if curmeta != None:
                    articledata[curmeta] = curmetadata
                curmeta = None
                curmetadata = ''
            elif articlelines[metaline].split()[0][-1] == ':':
                if curmeta != None:
                    articledata[curmeta] = curmetadata
                curmeta = articlelines[metaline].split()[0][:-1]
                if curmeta in ('PHOTO', 'PHOTOS'):
                    curmeta = 'Photos'
                metalen = len(curmeta)
                curmetadata = articlelines[metaline][metalen + 2:]
            elif articlelines[metaline][:4] == 'http':  # URL
                curmeta = 'URL'
                curmetadata = articlelines[metaline].rstrip()
            else:  # continuation of previous line
                curmetadata += ' ' + articlelines[metaline].rstrip()
        # flush last metadata
        if curmeta != None:
            articledata[curmeta] = curmetadata

        # Now process the document & metadata

        # Get length info if not encountered as metadata
        if 'Length' not in articledata:
            articledata['Length'] = sum([len(x.split()) for x in bodytext])

        # Check if this is a duplicate, to determine where to write it to
        # Note: works only if articles in downloaded text are sorted by date!
        if prevDay == docDay and prevMonth == docMonth and prevYear == docYear:
            # See if duplicate; look at text, since titles sometimes do repeat
            # If want to test on title too, use 'title in todayTitles' or a more complex version:
            # titleComp = min(50, len(title))
            # isDuplicate = (titleComp > 32 and title[:titleComp] in todayTitles) \
            #               or bodytext[0][:100] in todayArticles
            isDuplicate = bodytext[0][:100] in todayArticles
            # Add to list for today
            todayTitles = todayTitles + title + ' '
            todayArticles = todayArticles + bodytext[0][:200] + ' '
        else:
            isDuplicate = False
            prevDay = docDay
            prevMonth = docMonth
            prevYear = docYear
            todayTitles = title + ' '
            todayArticles = bodytext[0][:200] + ' '

        # Calculate some additional data from the info gathered so far,
        # to prep for writing out
        country = 'US'  # TODO: determine ctry by source; easy enough
        pubName = source

        # Adjust UK source names (works only if country is set to 'UK' above)
        if country == 'UK':
            pubs = ['guardian', 'observer', 'mail', 'mirror',
                    'daily telegraph', 'sunday telegraph', 'times',
                    'sunday times', 'record', 'star', 'financial times',
                    'independent', 'i', 'evening standard', 'express', 'sun',
                    'people', 'news of the world', 'business']
            if 'mirror' in pubName:
                pubName = 'mirror'
            elif 'daily mail' in pubName or pubName == 'mail on sunday':
                pubName = 'mail'
            elif 'daily record' in pubName or 'sunday mail' in pubName:
                pubName = 'record'
            elif 'star' in pubName:
                pubName = 'star'
            elif 'i-' in pubName:
                pubName = 'i'
            elif 'independent' in pubName:
                pubName = 'independent'
            elif pubName == 'telegraph':
                pubName = 'daily telegraph'
            elif 'express' in pubName:
                pubName = 'express'
            elif 'business' in pubName:
                pubName = 'business'
            # Assign UK publication code to the publication
            if pubName in pubs:
                pubNr = 1 + pubs.index(pubName)
            else:
                pubNr = -1

        # Assign US publication code to the publication
        elif country == 'US':
            US_sourcelist = ['new york times', 'washington post', 'usa today',
                             'wall street journal', 'daily news',
                             'atlanta journal-constitution',
                             'san jose mercury news', 'denver post',
                             'new york post', 'boston globe',
                             'philadelphia inquirer', 'tampa bay times',
                             'star-tribune']
            try:
                pubNr = US_sourcelist.index(source)
            except ValueError:
                pubNr = -1

        # All other countries: just assign -1 as pubNr
        else:
            pubNr = -1

        # Get date, if still pending
        if date_pending:
            if 'Load-Date' in articledata:
                dateparts = articledata['Load-Date'].split()
                docYear = dateparts[2]
                docMonth = dateparts[0]
                docDay = int(dateparts[1][:-1]) # don't include comma,
                # Note: if most articles loaded next day, need to subtract 1 to get article date!
            else:
                docYear = '0000'
                docMonth = 'dummy'
                docDay = '0'

        # Get month nr
        months_engl = ['january', 'february', 'march', 'april', 'may', 'june',
                       'july', 'august', 'september', 'october', 'november',
                       'december']
        months_engl_abbr = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                            'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        if source == "el periodico de catalunya":
            months = ['enero', 'febrero', 'marzo', 'abril', 'mayo',
                      'junio', 'julio', 'agosto', 'septiembre',
                      'octubre', 'noviembre', 'diciembre']
        elif any(paper in source for paper in \
                 ('volkskrant', 'telegraaf', 'nrc', 'algemeen dagblad', 'trouw')):
            months = ['januari', 'februari', 'maart', 'april', 'mei',
                      'juni', 'juli', 'augustus', 'september',
                      'oktober', 'november', 'december']
        elif any(x in source for x in ['zeit', 'welt', 'spiegel', 'zeitung']):
            months = ['januar', 'februar', 'marz', 'april', 'mai',
                      'juni', 'juli', 'august', 'september', 'oktober',
                      'november', 'dezember']
        elif 'monde' in source or 'figaro' in source or 'parisien' in source:
            months = ['janvier', 'fevrier', 'mars', 'avril', 'mai', 'juin',
                      'juillet', 'aout', 'septembre', 'octobre', 'novembre',
                      'decembre']
        else:
            months = months_engl
        if docMonth.lower() in months:
            monthNr = 1 + months.index(docMonth.lower())
        # some foreign language sources use English months in LN
        elif docMonth.lower() in months_engl:
            monthNr = 1 + months_engl.index(docMonth.lower())
        elif docMonth.lower() in months_engl_abbr:
            monthNr = 1 + months_engl_abbr.index(docMonth.lower())
        elif docMonth.lower() == 'maerz':  # German umlaut-less spelling
            monthNr = 3
        else:
            monthNr = 0
        if monthNr != 0 and docDayofweek == '':
            docDayofweek = weekDay(int(docYear), monthNr, int(docDay))

        # Combine the TYPE, PUB-TYPE, PUBLICATION-TYPE metas
        if 'Type' in articledata:
            thetype = articledata['Type']
        elif 'Pub-Type' in articledata:
            articledata['Type'] = articledata['Pub-Type']
        elif 'Publication-Type' in articledata:
            articledata['Type'] = articledata['Publication-Type']
        elif 'Type-Publication' in articledata:
            articledata['Type'] = articledata['Type-Publication']

        # For some Dutch newspapers (the NRC, the Volkskrant, and Trouw, at least),
        # some articles contain, instead of the text, the line
        # "De uitgever heeft voor dit artikel geen publicatierecht".
        # Skip these
        if len(text[0]) < 100 and "voor dit artikel geen publicatierecht" in text[0]:
            continue

        sequenceNr = 0  # At some point, might want to keep track of sequencing within a day

        # Finally write results to the appropriate csv file
        # Consider re-encoding character strings
        basicdata = [x if not isinstance(x, str) else \
                         x.decode('ascii', 'ignore') for x in \
                     [docID, seqnr, pubName, pubNr, docYear,
                      docMonth, monthNr, docDay, docDayofweek,
                      sequenceNr, title, edition]]

        header_tuple = ('DocNr', 'SearchResultNr', 'Publication', 'PubNr', 'Year',
                        'Month', 'MonthNr', 'Day', 'Day of week', 'Sequence', 'Title',
                        'Edition', 'Text', 'Part', 'Author', 'Length', 'Section',
                        'Language', 'Load-Date', 'Type', 'Photos', 'Graphic')

        # Add additional metadata
        extradata = []
        for metaname in header_tuple[14:]:
            if metaname in articledata:
                extradata.append(articledata[metaname])
            else:  # no value; supply dummy
                extradata.append(' ')
        # Could check if any other metadata in articledata, but for now just ignore
        textrows = [basicdata + [x, nr] + extradata \
                    for nr, x in enumerate(bodytext)]
        if isDuplicate:
            dupwriter.writerows(textrows)
            docID -= 1
        else:
            outwriter.writerows(textrows)
    return docID


# ****************************** Factiva processing **************************

def processFactiva_text(text, outwriter, dupwriter, header, startDocNr,
                        excludeblogs, excludeweb):
    """Process the text from a Factiva output file.

    Note: Factiva will often produce rtf; our input needs to be plain text.
    Open in TextEdit first, and choose command-shift-T to switch to plain.
    """
    import csv
    import re
    from unidecode import unidecode
    global skipped_sources

    # Factiva articles end with a blank line and then "Document <sourceabbrev><long id>"
    # on a line by itself, sometimes followed by another blank line.
    # Put a dummy marker to split on after the ID
    textX = re.sub(r'(Document)\s+(\w{25})\s+\n',r'\1 \2\nEndOfDocHere', text)

    # Strip text of other special characters
    textX = unidecode(textX.decode('utf-8-sig'))
	

    # Factiva download files begin with a blank line, a line with "Factiva",
    # a blank line, a line with "Dow Jones", and another blank line.
    # Together, this is 22-24 characters. Strip these.
    if 'Factiva' in textX[:22] and 'Dow Jones' in textX[:22]:
        textX = textX[22:]

    # Make list of meta-data to store.
    # These do not have identifying markers in Factiva, so go by position in document.
    header_tuple=('DocNr', 'FactivaID', 'Publication', 'PubNr', 'Year',
                  'Month', 'MonthNr', 'Day', 'Day of week', 'Sequence',
                  'Title', 'Edition', 'Text', 'Part', 'Author', 'Length',
                  'Section', 'Page', 'PubCode', 'City')
    # Write the list of headers to the output file
    if header:
        outwriter.writerow(header_tuple)
        dupwriter.writerow(header_tuple)

    # Commas in text rarely cause problems in csv files.
    # If they do, set commaString to a space, or (to remember where commas
    # were) to ' @comma' (note space before @, to avoid concatenating words)
    commaString = ','

    # Split the textfile into a list of articles & initialize variables
    split_text = textX.split('EndOfDocHere')

    docID = startDocNr
    prevDay = 0
    prevMonth = 0
    prevYear = 0
    todayTitles = ''
    todayArticles = ''

    # Now loop over each document in the output file
    for a_doc in split_text:

        # Split into an array of (non-empty) lines
        a_docX = a_doc.split('\n')
        # Skip empty documents and the search summary at the end
        if len(a_docX) > 10 and ('Copyright' in a_doc or '(c)' in a_doc) \
                and a_docX[0] != 'Search Summary':
            try:
                FactivaID = a_docX[-2].split()[1]
            except IndexError:
                print "No Factiva id found; assigning dummy"
                FactivaID = -1
            docID += 1
            # print FactivaID, docID  # use to pinpoint error source

            # Skip to the first non-blank line, which has section/title info
            curLine = 0
            while len(a_docX[curLine].strip()) == 0:
                curLine += 1

            # Get the main meta-data, which ends at the copyright line
            metalines = []
            while not ('(c)' in a_docX[curLine] or 'Copyright' in a_docX[curLine]):
                metalines.append(a_docX[curLine].strip())
                curLine += 1
                if curLine == len(a_docX):
                    break
            # Very occasionally we encounter a copyright line too early
            if curLine < len(a_docX) - 2 and \
                    not (len(a_docX[curLine + 1].strip()) == 0 or \
                         len(a_docX[curLine + 2].strip()) == 0):
                curLine += 1
                while not ('(c)' in a_docX[curLine] or 'Copyright' in a_docX[curLine]):
                    metalines.append(a_docX[curLine].strip())
                    curLine += 1
                    if curLine == len(a_docX):
                        break

            # Now curLine is at the copyright line,
            # and metalines contains the data until then

            # Metadata ordering, by line:
            # - section (optional, may be followed by (first part of) title on same line)
            # - title (may have multiple lines)
            # - author (optional, sometimes preceded by 'By '
            # - word count
            # - date
            # - newspaper name
            # - newspaper code
            # - edition info (optional)
            # - page number (optional)
            # - language

            # This code is clunky -> rewrite to make more elegant
				
            if len(metalines) > 6 and wordcountline(metalines[-7]):
                # full complement of data
                length = metalines[-7].split()[0]
                docYear, docMonth, docDay = dateinfo(metalines[-6])
                source = metalines[-5].lower()
                pubCode = metalines[-4]
                edition = metalines[-3]
                page = metalines[-2]
                linesprocessed = 7
            elif len(metalines) > 5 and wordcountline(metalines[-6]):
                # no edition info
                length = metalines[-6].split()[0]
                docYear, docMonth, docDay = dateinfo(metalines[-5])
                source = metalines[-4].lower()
                pubCode = metalines[-3]
                edition = ''
                page = metalines[-2]
                linesprocessed = 6
            elif len(metalines) > 4 and wordcountline(metalines[-5]):
                # no edition or page info
                length = metalines[-5].split()[0]
                docYear, docMonth, docDay = dateinfo(metalines[-4])
                source = metalines[-3].lower()
                pubCode = metalines[-2]
                edition = ''
                page = ''
                linesprocessed = 5
            
                continue
            else:
                print "Did not find word count data where expected"
                print metalines
                continue  # skip to next article

            # shorten & simplify publication name
            if source[:4] == 'the ':
                source = source[4:]
            if source in ['star tribune (mpls.-st. paul) newspaper of the twin cities',
                          'star-tribune newspaper of the twin cities mpls.-st. paul']:
                source = 'star-tribune'
            elif source == 'st. petersburg times':
                source = 'tampa bay times'

            # Exclude sources not meeting our criteria
            if skipsource(source, excludeblogs, excludeweb):
                if source not in skippedsources:
                    print "Skipping article from %s" % source
                    skippedsources.append(source)
                continue  # move on to next article

            # Check if line before word count is author, as marked by:
            # - line begins 'by '
            # - line contains ' by ' (slightly ambiguous, but rarely wrong)
            # - line is all caps and not the first line in metalines
            # - line is not all caps, but preceding line is
            metalines = metalines[:-linesprocessed]
            if metalines[-1].lower()[:3] == 'by ' or \
                    metalines[-1].lower()[:9] == 'words by ' or \
                    (len(metalines) > 1 and
                        (metalines[-1].upper() == metalines[-1] or
                         (metalines[-1].upper() != metalines[-1] and
                          metalines[-2].upper() == metalines[-2]))):
                author = metalines[-1]
                if author.lower()[:3] == 'by ':
                    author = author[3:]
                metalines = metalines[:-1]
            else:
                author = ''

            # Now we have section (optional) and title left
            section = ''
            title = ''
            if len(metalines) > 0:
                section = metalines[0]

                # For some papers, (start of) title follows section info on same line.
                if ' --- ' in section:
                    sectionparts = section.split(' --- ')
                    section = sectionparts[0]
                    title = ' --- '.join(sectionparts[1:]) + ' '
                elif ' -- ' in section:
                    sectionparts = section.split(' -- ')
                    section = sectionparts[0]
                    title = ' -- '.join(sectionparts[1:]) + ' '
                elif ': ' in section:
                    sectionparts = section.split(': ')
                    section = sectionparts[0]
                    title = ': '.join(sectionparts[1:]) + ' '

                if len(section.split()) > 3:
                    # too many words for section name -> assume part of title
                    title = section + ' ' + title
                    section = ''

            if len(metalines) > 1:
                title += ' '.join(metalines[1:])
            if title == '':
                title = section
                section = ''

            # NZ Herald has several additional byline marker options
            # (report(s)/write(s) first1 last1 (and first2 last2), for example)
            # Elaborate on this code more if that seems useful
            if source == 'new zealand herald':
                titleparts = title.split()
                if len(titleparts) > 2 and titleparts[-1] == 'reports':
                    author = ' '.join(titleparts[-3:-1])
                    title = ' '.join(titleparts[:-3])
                elif len(titleparts) > 2 and \
                        titleparts[-1].upper() == titleparts[-1] and \
                        titleparts[-2].upper() == titleparts[-2] and \
                        titleparts[-3].upper() != titleparts[-3]:
                    author = ' '.join(titleparts[-2:])
                    title = ' '.join(titleparts[:-2])
                # NZ Herald also has no section info
                if len(section) > 0:
                    title = section + ' ' + title
                    section = ''

            # Copyright date ends standard metadata
            # Next is a blank line
            curLine += 2
            # After that, sometimes, there is the reporting city,
            # followed by another blank line
            thisLine = a_docX[curLine].strip()
            nextLine = a_docX[curLine + 1].strip()
            if len(thisLine.split()) == 1 and len(nextLine) == 0:
                city = thisLine
                curLine += 2
            else:
                city = ''

            # Now curLine is at the first line of text
            text = ['',]
            textnr = 0
            while curLine < len(a_docX):
                line = a_docX[curLine].lstrip()
                text[textnr] += line + ' '
                if len(text[textnr]) > 32600:
                    textnr += 1
                    text.append('')
                curLine = curLine +1

            # Check if this is a duplicate, to determine where to write it to
            # Note: works only if articles are sorted by date!
            if prevDay == docDay and prevMonth == docMonth and prevYear == docYear:
                # See if duplicate; look at text, since titles sometimes do repeat
                # If want to test on title too:   title in todayTitles
                isDuplicate = text[0][:100] in todayArticles
                # Add to list for today
                todayTitles = todayTitles + title + ' '
                todayArticles = todayArticles + text[0][:200] + ' '
            else:
                isDuplicate = False
                prevDay = docDay
                prevMonth = docMonth
                prevYear = docYear
                todayTitles = title + ' '
                todayArticles = text[0] + ' '

            # Calculate some additional data from the info gathered so far, to prep for writing out
            months = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November',
                      'December']
            if docMonth in months:
                monthNr = 1 + months.index(docMonth)
                docDayofweek = weekDay(docYear, monthNr, docDay)
            else:
                monthNr = 0
                docDayofweek = ''
            # At some point may want to track articles on a given day
            sequenceNr = 0 # Dummy value -> change as applicable (can do in Excel)

            # Assign publication code to the publication
            US_sourcelist = ['new york times', 'washington post', 'usa today',
                             'wall street journal', 'daily news',
                             'atlanta journal and constitution',
                             'san jose mercury news', 'denver post',
                             'new york post', 'boston globe',
                             'philadelphia inquirer', 'tampa bay times',
                             'star-tribune']
            try:
                pubNr = US_sourcelist.index(source.lower())
            except ValueError:
                pubNr = -1

            # Finally write results to the appropriate csv file
            # Consider re-encoding character strings
            basicdata = [x if not isinstance(x, str) else \
                          x.decode('ascii', 'ignore') for x in \
                              [docID, FactivaID, source, pubNr, docYear,
                               docMonth, monthNr, docDay, docDayofweek,
                               sequenceNr, title, edition]]
            extradata = [author, length, section, page, pubCode,
                         city]
            textrows = [basicdata + [x, nr] + extradata \
                            for nr, x in enumerate(text)]
            if isDuplicate:
                dupwriter.writerows(textrows)
                docID -= 1
            else:
                outwriter.writerows(textrows)
    # Return last document number
    return docID


# ***************************** ProQuest processing ************************

def processProQuest_text(text, outwriter, dupwriter, header, startDocNr):
    """Process the text from a ProQuest output file (e.g. US Newsstream)"""
    import csv
    import re
    from unidecode import unidecode
    global skipped_sources

    maxlen = 32600  # text len above which we'll split into parts

    # Proquest articles begin with a line of 60 underlines:
    # ____________________________________________________________
    # then a blank line, and then "Document <nnn> of <nnn>

    # Sometimes that "Document <nnn> of <nnn> line is missing, so don't rely on it

    # Put a dummy marker to split on after the ID
    # textX = re.sub(r'_+\s+Document ([0-9]+) of [0-9]+\s+\n', r'EndOfDocHere\n\1\n\n', text)
    # At the end, there is another line of 60 underlines, a blank line,
    # and then 3 lines of basic info; these need to be stripped
    # textX = re.sub(r'_+\s+.+\n.+\n.+', r'', textX)
    textX = re.sub(r'_{60}\s+', r'EndOfDocHere\n\n', text)

    # Strip text of other special characters
    textX = unidecode(textX.decode('utf-8-sig'))

    # Make list of meta-data to store.
    # These do not have identifying markers in Factiva, so go by position in document.
    header_tuple = ('DocNr', 'ProQuestID', 'Publication', 'PubNr', 'Year',
                    'Month', 'MonthNr', 'Day', 'Day of week', 'Sequence',
                    'Title', 'Edition', 'Text', 'Part', 'Author', 'Length',
                    'Section', 'Language', 'Page', 'PubCode', 'City')
    # Write the list of headers to the output file
    if header:
        outwriter.writerow(header_tuple)
        dupwriter.writerow(header_tuple)

    # Commas in text rarely cause problems in csv files.
    # If they do, set commaString to a space, or (to remember where commas
    # were) to ' @comma' (note space before @, to avoid concatenating words)
    commaString = ','

    # Split the textfile into a list of articles & initialize variables
    split_text = textX.split('EndOfDocHere')[:-1]  # skip last text snippet; not an article

    docID = startDocNr

    # initialize variables for identifying duplicates
    prevDay = 0
    prevMonth = 0
    prevYear = 0
    todayTitles = ''
    todayArticles = ''

    # initialize variables for filling in missing data
    prevdateinfo = ['nomonth', '0 ', '0000']
    prevsource = 'unknown source'
    prevlanguage = 'unknown language'
    prevCode = 'no ISSN'
    prevcity = 'no publication city'

    # Now loop over each document in the output file
    for a_doc in split_text:

        # Split into an array of (non-empty) lines
        doc_items = a_doc.split('\n\n')
        # Skip 'documents' too short to be normal articles
        if len(doc_items) > 3:

            # Within ProQuest, every item after the title appears to be flagged by a field title
            # with the exception of the Links: item, which often immediately precedes text
            # Also, need to skip past any excess blank space first
            curitem = 0
            while len(doc_items[curitem].strip()) == 0:
                curitem += 1
            # For debugging purposes:
            # print doc_items[curitem]

            # Now we're at the 'Document x of xxx' line
            title = doc_items[curitem + 1]

            # Convert metadata headers into dictionary keys
            doc_items_dict = {}
            for item in doc_items[curitem + 2:]:
                itemsplit = item.split(':')
                if itemsplit[0] not in doc_items_dict:  # don't overwrite earlier entries
                    doc_items_dict[itemsplit[0]] = (':'.join(itemsplit[1:])).lstrip()
            # 'Links' item is sometimes directly followed by full text
            if 'Full text' not in doc_items_dict and 'Links' in doc_items_dict:
                doc_items_dict['Full text'] = ' '.join(doc_items_dict['Links'].split('\n')[1:])

            # Process metadata dictionary
            if 'http' not in doc_items_dict and 'https' not in doc_items_dict:
                # not a real search result: probably just header / content information
                continue  # skip to next item
            # 0. ProQuest ID -- extract from article's permanent link
            if 'https' in doc_items_dict:
                proquestID = doc_items_dict['https'].split('/')[-1].split('?')[0]
            else:
                proquestID = doc_items_dict['http'].split('/')[-1].split('?')[0]

            # 1. Date
            if 'Publication date' not in doc_items_dict:
                dateinfo = prevdateinfo
            else:
                dateinfo = doc_items_dict['Publication date'].split()
            docMonth = dateinfo[0]
            docDay = int(dateinfo[1][:-1])
            docYear = int(dateinfo[2])
            prevdateinfo = dateinfo
            # Alternatives: docYear = doc_items_dict['Year'] and also ...['Publication year']

            # 2. Newspaper info
            if 'Publication title' not in doc_items_dict:
                source = prevsource
            else:
                source = doc_items_dict['Publication title'].lower()
            prevsource = source

            if 'Language of publication' not in doc_items_dict:
                language = prevlanguage
            else:
                language = doc_items_dict['Language of publication']
            prevlanguage = language

            if 'ISSN' not in doc_items_dict:
                pubCode = prevCode
            else:
                pubCode = doc_items_dict['ISSN']
            prevCode = pubCode

            if 'Place of publication' not in doc_items_dict:
                city = prevcity
            else:
                city = doc_items_dict['Place of publication']
            prevcity = city

            # 2a. Simplify/correct source names (also for sources that changed name over time)
            if "wall street journal" in source:  # delete any edition info
                source = "wall street journal"
            elif "boston globe" in source:
                source = "boston globe"
            elif source == "star tribune":
                source = 'star-tribune'
            elif source == 'st. petersburg times':
                source = 'tampa bay times'

            # 3. Article specifics
            author = doc_items_dict['Author'] if 'Author' in doc_items_dict else ''
            # Already have title, but process to find edition
            if 'Title' in doc_items_dict:
                fulltitle = doc_items_dict['Title']
                if '[' in fulltitle and 'Edition' in fulltitle:
                    edition = doc_items_dict['Title'].split('[')[1][:-1]
                else:
                    edition = ''
            else:  # should be redundant, but just in case
                edition = ''
            # Note: right now we don't do anything with fulltitle, just using the already-created title
            # Should maybe replace one by the other, especially if the latter is lacking

            text = doc_items_dict['Full text'].replace('\n', ' ').strip()
            section = doc_items_dict['Section'] if 'Section' in doc_items_dict else ''
            page = doc_items_dict['Pages'] if 'Pages' in doc_items_dict else ''
            length = len(title.split()) + len(text.split())

            docID += 1
            # print proquestID, docID  # use to pinpoint error source

            # Check if this is a duplicate, to determine where to write it to
            # Note: works only if articles are sorted by date!
            textstart = text[:100]
            if prevDay == docDay and prevMonth == docMonth and prevYear == docYear:
                # See if duplicate; look at text, since titles sometimes do repeat
                # If want to test on title too:   title in todayTitles
                isDuplicate = textstart in todayArticles
                # Add to list for today
                todayTitles = todayTitles + title + ' '
                todayArticles = todayArticles + textstart + ' '
            else:
                isDuplicate = False
                prevDay = docDay
                prevMonth = docMonth
                prevYear = docYear
                todayTitles = title + ' '
                todayArticles = textstart + ' '

            # Calculate some additional data from the info gathered so far, to prep for writing out
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            if docMonth in months:
                monthNr = 1 + months.index(docMonth)
                docDayofweek = weekDay(docYear, monthNr, docDay)
            else:
                monthNr = 0
                docDayofweek = ''
            # At some point may want to track articles on a given day
            sequenceNr = 0  # Dummy value -> change as applicable (can do in Excel)

            # Assign publication code to the publication
            US_sourcelist = ['new york times', 'washington post', 'usa today',
                             'wall street journal', 'daily news',
                             'atlanta journal and constitution',
                             'san jose mercury news', 'denver post',
                             'new york post', 'boston globe',
                             'philadelphia inquirer', 'tampa bay times',
                             'star-tribune', 'philadelphia daily news',
                             'richmond times dispatch', 'arizona republic',
                             'las vegas review - journal']
            try:
                pubNr = US_sourcelist.index(source.lower())
            except ValueError:
                pubNr = -1

            # Finally write results to the appropriate csv file
            # Consider re-encoding character strings
            basicdata = [x if not isinstance(x, str) else \
                             x.decode('ascii', 'ignore') for x in \
                         [docID, proquestID, source, pubNr, docYear,
                          docMonth, monthNr, docDay, docDayofweek,
                          sequenceNr, title, edition]]
            extradata = [author, length, section, language, page, pubCode,
                         city]
            if len(text) >  maxlen:
                textparts = splittext(text, maxlen)
                textrows = [basicdata + [x, nr] + extradata \
                            for nr, x in enumerate(textparts)]
            else:
                textrows = [basicdata + [text, 0] + extradata, ]
            if isDuplicate:
                dupwriter.writerows(textrows)
                docID -= 1
            else:
                outwriter.writerows(textrows)
    # Return last document number
    return docID


def splittext (text, maxlen):
    """Split text into parts, taking care to break on space"""
    parts = []
    while len(text) > maxlen:
        newpart = text[:32600]
        remainder = text[32600:]
        firstspace = remainder.find(' ')
        if firstspace >= 0:
            newpart += remainder[:firstspace]
            text = remainder[firstspace + 1:]
        elif len(remainder) < 100:
            newpart = text
            text = ''
        else:
            prevchunk = text[32500:32600]
            prevspace = prevchunk.find(' ')
            if prevspace >= 0:
                newpart = text[:32500 + prevspace]
                text = text[32500 + prevspace + 1:]
        parts.append(newpart)
    parts.append(text)
    return parts


# ****************************** auxiliary functions *************************

# Global variables for use in skipsource

skippedsources = []
sourcesUS = ['wall street journal', 'boston globe',
                     'tampa bay times', 'star-tribune',
                     'new york times', 'washington post',
                     'usa today', 'atlanta journal-constitution',
                     'philadelphia inquirer', 'denver post',
                     'daily news', 'new york post',
                     'san jose mercury news']
sourcesCAN = ['toronto star', 'globe and mail', 'national post',
                      'gazette', 'vancouver sun']
sourcesAUS = ['sydney morning herald', 'west australian',
                      'daily telegraph', 'courier mail', 'herald sun']
sourcesNZ = ['new zealand herald', 'otago daily times',
                     'waikato times', 'dominion post', 'press']


def skipsource(sourcename, excludeblogs=True, excludeweb=True,
               excludelaw=True, sourcefilter=False):
    """See if this is a source to be skipped.

    Sources to excluded are hard-coded global variables at the moment.
    Consider making a parameter to be passed.
    """
    return (excludeblogs and 'blogs' in sourcename) or \
            (excludeweb and '.co' in sourcename) or \
            (excludelaw and 'law' in sourcename) or \
            (sourcefilter and (sourcename not in sourcesUS))


def wordcountline(aline):
    """See if aline represents word count data."""
    words = aline.split()
    return len(words) == 2 and words[1] == 'words'


def dateinfo(datestring):
    """Parse date string; return year, month, day."""
    dateparts = datestring.split()
    return int(dateparts[2]), dateparts[1], int(dateparts[0])


def stripempties(infilename, outfilename):
    """Remove empty articles from the Dutch papers.

    These contain as text "De uitgever heeft voor dit artikel geen publicatierecht"
    """
    import csv
    skipped = 0
    with open(infilename, 'rU') as infile, open(outfilename, 'wb') as outfile:
        outwriter = csv.writer(outfile)
        for counter, row in enumerate(csv.reader(infile)):
            if "voor dit artikel geen publicatierecht" in row[12]:
                skipped += 1
            else:
                outwriter.writerow(row)
        print "Processed %d articles, of which %d were skipped (geen publicatierecht)." % (counter, skipped)


def newids(infile, sourcenum=0, yearcol=1, monthcol=2, daycol=3, partcol=4, header=True):
    """Replace the id numbers in the filename.

    Note: no error-checking & does not guard against duplicate numbers.
    """
    import csv
    outfile = '.'.join(infile.split('.')[:-1]) + '_newids.csv'
    with open(infile, 'rU') as inf, open(outfile, 'wb') as outf:
        outwriter = csv.writer(outf)
        for row in csv.reader(inf):
            if header:
                outwriter.writerow(row)
                header = False
            else:
                year = int(row[yearcol])
                year_adj = year - 1940
                newid = sourcenum * 10000000 + year_adj * 100000 + \
                        int(row[monthcol]) * 1000 + int(row[daycol]) * 10 + \
                        int(row[partcol])
                outwriter.writerow([newid,] + row[1:])


def isdateinfo(astring):
    """See if a string contains date & time info.

    Assume info has format: month day, year; weekday time timezone
    """
    words = astring.split()
    return len(words) > 1 and is_month(words[0]) and is_int(words[1][:-1]) and words[-1] == 'Time'


def is_int(aParam):
    """See if a parameter is representable as an integer."""
    try:
        int(aParam)
    except ValueError:
        return False
    else:
        return True
    

def is_year(str):
    """See if a string could represent a publication year.

    Accept a value between 1900 and 2020.
    """
    try:
        x = int(str)
    except ValueError:
        return False
    else:
        return (x > 1799 and x < 2101)


def is_month(str):
    """See if a string could represent an English-language month name."""
    return str in ('January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November',
                   'December')


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


def splitITname(line):
    """Split Irish Times author name from rest of line's text."""

    commaString = ','
    lineWords = line.split()
    curIndex = 0
    isText = False
    author = ''
    text = ''
    # print "Starting with %d words " % len(lineWords)
    while curIndex < len(lineWords):
        curWord = lineWords[curIndex]
        if isText:
            text = text + curWord + ' '
        elif curWord == 'and' or curWord == commaString.strip():
            # separation between 2 authors
            author = author + 'and '
        elif curWord == curWord.upper(): # all caps: name
            author = author + curWord + ' '
        elif curWord[0:3] == 'and' and curWord[3:] == curWord[3:].upper():
            # 'and' and name not separated by space
            author = author + 'and ' + curWord[3:] + ' '
        elif len(curWord) >= 3 and curWord[0:3] == curWord[0:3].upper() \
                and curWord[len(curWord)-1].islower():
            # Text begins; figure out where name ends
            wordIndex = 0
            while curWord[wordIndex].isupper():
                wordIndex = wordIndex +1
            author = author + curWord[:wordIndex-1]
            text = curWord[wordIndex-1:] + ' '
            isText = True
        else:
            # word begins lower-case or is very short; just assume it's text
            text = curWord + ' '
            isText = True
        curIndex = curIndex +1
    return author, text


def check_processed(filename, header=True):
    """See whether we produced good results, looking for acceptable dates."""
    import csv

    headeradjust = 0 if header else 1
    errorcount = 0
    with open(filename, 'rU') as infile:
        for counter, article in enumerate(csv.reader(infile)):
            if header:
                header = False
            else:
                try:
                    year = int(article[4])
                except ValueError:
                    print "Year error in article %d" % (counter + headeradjust)
                    print article
                    errorcount += 1
                    continue
                try:
                    monthnr = int(article[6])
                except ValueError:
                    print "Month error in article %d" % (counter + headeradjust)
                    print article
                    errorcount += 1
                    continue
                try:
                    daynr = int(article[7])
                except ValueError:
                    print "Day error in article %d" % (counter + headeradjust)
                    print article
                    errorcount += 1
                    continue
    if errorcount == 0:
        print "All dates appear to have the appropriate format."
    else:
        print "Found %d errors" % errorcount
    return


def add_datestring(infilename, outfilename, datestring, startnum, endnum, bufferlines=3):
    """Add date string to LN download missing dates."""

    with open(infilename, 'rU') as inf, open(outfilename, 'wb') as outf:

        linecounter = 99
        for line in inf:
            linewords = line.split()
            if len(linewords) == 4 and linewords[1] == 'of' and linewords[3] == 'DOCUMENTS' \
                    and int(linewords[0]) >= startnum and int(linewords[0]) <= endnum:
                linecounter = 0
            if linecounter == bufferlines:
                outf.write('\n')
                outf.write(datestring + '\n')
            outf.write(line)
            linecounter += 1



# ********************************* below here is draft stuff; ignore ****************

# def dummy():
#     """Duplicated code of the above, to use for ProQuest output.
#
#     Not sure where I left off in adapting for ProQuest.
#     Need to compare this code to the above.
#     """
# else:  # ProQuest
#     # PQ articles are separated by a string of 60 underscores.
#     textX = re.sub("____________________________________________________________\n",
#                     "---article separator AMvdV---", text)
#
#     # Remove any special characters that may mess up reading & writing files
#     textX = unidecode(textX.decode('utf-8-sig'))
#
#     # Get list of meta-data headers available.
#     # These are identified by full-caps, and beginning at start of a line
#     # Take only the commonly occurring ones (those present in at least 50% of
#     # the articles) and filter out the source name (often capitalized) itself
#
#     # Note: if we have sources in more than 2 languages, key metadata may
#     # have different names in different languages -> if so, need to merge
#     # appropriately. See treatment of Type / Pub-Type below as example.
#
#     # To keep columns similar across output files, hard-code a standard
#     # set of meta-data headers
#     meta_basics = ['BYLINE','LENGTH','SECTION','LANGUAGE','LOAD-DATE','TYPE',
#                    'PUB-TYPE','PUBLICATION-TYPE','JOURNAL-CODE','GRAPHIC']
#     meta = list(set(re.findall(r'\n([A-ZÄ][A-ZÄ][A-ZÄ-]*?):', text)))
#     meta = \
#         [m for m in meta if float(text.count(m + ':')) / len(textX) > .5 \
#          and m not in meta_basics]
#     meta_list = meta_basics + meta
#
#     # NRC has an odd pair of meta-data: SAMENVATTING without a colon, followed
#     # by a blank line, coupled with VOLLEDIGE TEKST: with a colon but also
#     # containing a space. Just hard-code for these.
#     # For the NRC, also correct opening quotes (rendered as two consecutive
#     # commas). (Later, may generalize to do this for all Dutch papers)
#     if 'NRC Handelsblad' in textX and 'SAMENVATTING' in textX:
#         textX = re.sub(',,' , '"', textX)
#         meta_list = meta_list + ['SAMENVATTING',]
#     meta_tuple=('DocNr', 'SearchResultNr', 'Publication', 'PubNr', 'Year',
#                 'Month', 'MonthNr', 'Day', 'Day of week', 'Sequence', 'Title',
#                 'Edition', 'Text', 'Overflow', 'MoreText', 'Truncated')
#     for item in meta_list:
#         if item <> 'TYPE' and item <> 'PUB-TYPE': # retain only PUBLICATION-TYPE
#             meta_tuple += (item,)
#     # Write the list of headers to the output file
#     if header:
#         outwriter.writerow(meta_tuple)
#         dupwriter.writerow(meta_tuple)
#
#     # Split the textfile into a list of articles & initialize variables
#     split_text = textX.split('---article separator AMvdV---')
#     docID = startDocNr
#     prevDay = 0
#     prevMonth = 0
#     prevYear = 0
#     todayTitles = ''
#     todayArticles = ''
#
#     # Loop over each article in the list
#     for a_doc in split_text:
#         # Split into an array of (non-empty) lines
#         a_docX = a_doc.split('\n')
#         # Make sure it's not an empty document
#         if len(a_docX) >= 5 and 'DOCUMENTS' in a_doc:
#
#             # Initialize variables not all papers may have info for
#             edition = ''
#             docDayofweek = ''
#
#             # Skip to the line that has the document number in it
#             # (each LN article begins with a line ending in 'n of N DOCUMENTS')
#             curLine = 0
#             while 'DOCUMENTS' not in a_docX[curLine]:
#                 curLine += 1
#             num = re.search('[0-9]+', a_docX[curLine])
#             searchResNr = int(num.group())
#             docID += 1
#             # print searchResNr, docID  # use to pinpoint error source
#
#             # Now get source information, which may be more than 1 line long,
#             # but is off-set on both sides by blank lines. For the Guardian,
#             # the source name line also has edition information on it
#             # (as of Jan. 1, 2004). This code will produce the dummy
#             # edition info "The" for articles prior to that date
#             curLine += 1
#             while len(a_docX[curLine].lstrip()) == 0:
#                 curLine += 1
#
#             # Extract sourceName from the first source info line
#             sourceName = a_docX[curLine].lstrip().lower()
#
#             # Exclude blogs, web articles if so desired
#             excludeblogs = True
#             excludeweb = True
#             if excludeblogs and 'blogs' in sourceName or \
#                     excludeweb and '.co' in sourceName:
#                 continue  # move on to next article
#
#             # Known erroneous source names in LexisNexis:
#             #       The The Sunday Times, SUNDAY SUTELEAPH <- fixed here
#             #       Washington Posts <- fixed in downloaded files
#             if sourceName[:3] == "the":
#                 sourceName = sourceName[4:]
#                 # Test for "The The Sunday Times"
#                 if sourceName[:3] == "the":
#                     sourceName = sourceName[4:]
#             # strip excess space & location info at end
#             sourceName = sourceName.split('(')[0].rstrip()
#             # Test for "SUNDAY SUTELEAPH"
#             if sourceName == "sunday suteleaph":
#                 sourceName = "sunday telegraph"
#             sourceName = sourceName.split(' - ')[0].rstrip()
#             # strip source name from the meta list
#             # (might be in there due to interviews)
#             if sourceName in meta_list:
#                 meta_list.remove(sourceName)
#
#             # Deal with Guardian edition info
#             if 'guardian' in sourceName:
#                 sourceName = 'guardian'
#                 editionX = a_docX[curLine].split(' - ')
#                 if len(editionX) > 1:
#                     edition = editionX[1].lstrip()
#                 # editionX = a_docX[curLine].replace(
#                 #               'The Guardian (London) - ','')
#                 # editionX = editionX.lstrip()
#                 # edition = editionX.split(' ')[0]
#
#             # Skip past additional lines of source info
#             # (title is always on first line)
#             while not len(a_docX[curLine].lstrip()) == 0:
#                 curLine += 1
#             # Skip past the blank lines that follow the source information
#             while len(a_docX[curLine].lstrip()) == 0:
#                 curLine += 1
#
#             # Now get the date
#             docDate = a_docX[curLine].lstrip()
#             date_pending = False
#             if 'politiken' in sourceName:
#                 # Date format "<day-of-week> <day-nr>. <month-string> <year>
#                 docDate = docDate.replace('.','')
#                 docDate_split = docDate.split(' ')
#                 docYear = docDate_split[3]
#                 docMonth = docDate_split[2]
#                 docDay = docDate_split[1]
#
#             elif sourceName == 'le monde' \
#                     or sourceName == 'corriere della sera' \
#                     or sourceName == 'nrc' or sourceName == 'zeit' \
#                     or sourceName == 'el periodico de catalunya':
#                 # Date format "<day-nr> <month-string> <year> <day-of-week>"
#                 # except in NRC results before July 26, 2009, where it is
#                 # "<month-string> <day-nr>, <year>"
#                 docDate = docDate.replace(',','')
#                 docDate_split = docDate.split(' ')
#                 docYear = docDate_split[2]
#                 docMonth = docDate_split[1]
#                 docDay = docDate_split[0]
#                 docDayofweek = docDate_split[3]
#                 # See if before the NRC format-change, in which case change
#                 # the variables
#                 if sourceName == 'nrc' and \
#                         (docYear < 2009 or
#                          (docYear == 2009 and is_int(docMonth))):
#                     docMonth = docDate_split[0]
#                     docDay = docDate_split[1]
#                 # Corriere may have edition info on line following date
#                 if sourceName == 'corriere della sera':
#                     curLine += 1
#                     edition = a_docX[curLine].lstrip()
#                 if not 'edizione' in edition.lower():
#                     edition = ''
#
#             elif sourceName == 'gazette' and \
#                     (len(docDate) < 3 or not is_int(docDate.split(' ')[2])):
#                 # Sometimes Montreal Gazette skips title; get from load-date instead
#                 date_pending = True
#
#             else: # Default English-language paper
#                 # Date format "<month-string> <day-nr>, <year> <day-of-week>"
#                 # (Observer, SundayTimes do not have day-of-week)
#                 # NYT, WaPo, Mail, Mirror, and Telegraph may have edition information
#                 # on the 1-2 lines following the date
#                 # FT has edition information on the same line as the date
#                 docDate = docDate.replace(',','')
#                 docDate_split = docDate.split(' ')
#                 docYear = docDate_split[2][:4]
#                 docMonth = docDate_split[0]
#                 docDay = docDate_split[1]
#                 if 'sunday' in sourceName or \
#                         sourceName in ['observer', 'news of the world']:
#                     docDayofweek = 'Sunday'
#                 elif len(docDate_split) > 3:
#                     docDayofweek = docDate_split[3]
#                     if len(docDate_split) > 4:
#                         edition = ' '.join(docDate_split[4:])
#
#                 if sourceName in ['new york times', 'washington post'] \
#                         and edition == '':
#                     # Edition info on next line
#                     curLine += 1
#                     edition = a_docX[curLine].lstrip()
#                     if 'Correction' in edition:
#                         # Actual edition info on next line
#                         curLine += 1
#                         edition = a_docX[curLine].lstrip()
#                     if not ('Late Edition' in edition \
#                             or 'New York Times on the Web' in edition \
#                             or 'International Herald Tribune' in edition):
#                         edition = ''
#                 if 'mail' in sourceName or 'telegraph' in sourceName \
#                         or 'times' in sourceName or 'sun' in sourceName \
#                         or 'independent' in sourceName \
#                         or 'daily star' in sourceName \
#                         or 'express' in sourceName or 'mirror' in sourceName \
#                         or sourceName == 'daily record' \
#                         or sourceName == 'scotsman' \
#                         or sourceName == 'evening standard' \
#                         or sourceName == 'people' \
#                         or sourceName == 'guardian' \
#                         or sourceName == 'news of the world':
#                     # Edition information and locality information
#                     # on 0 or more lines after the date
#                     edinfo = a_docX[curLine+1].lstrip()
#                     while len(edinfo) > 0:
#                         edition += ' ' + edinfo
#                         curLine += 1
#                         edinfo = a_docX[curLine+1].lstrip()
#                     edition = edition.lstrip()
#
#             # Initialize list of meta items
#             meta_dict={k : '' for k in meta_list}
#
#             # Get article title. Assume it starts after the article header
#             # info (document count, publication, edition, date) and is
#             # followed by article text or meta-data, but separated from those
#             #  by a blank line.
#             title = ''
#             curLine = curLine +1
#             while len(a_docX[curLine].lstrip()) == 0:
#                 curLine = curLine +1
#             while not len(a_docX[curLine].lstrip()) == 0:
#                 titleLine = a_docX[curLine].lstrip()
#                 title = title + titleLine.replace(',', commaString) + ' '
#                 curLine = curLine +1
#             # Make sure this was not a piece of meta-data rather than a title
#             # (NRC has some articles without titles)
#             if sourceName == 'NRC':
#                 ismeta = re.findall('^([A-ZÄ][A-ZÄ-]*?):', title)
#                 if len(ismeta) > 0:  # This article has no title
#                     if ismeta[0] in meta_list:
#                         meta_dict[ismeta[0]] = \
#                             title.replace(ismeta[0] + ': ', '')
#                     title = ''
#             # Check if it is NYT article with byline on same line as title
#             # Try to make sure it is not a title beginning with the word by,
#             # by testing for ';' in the line, which should separate byline
#             # from title
#             elif sourceName == 'new york times' and title[:3] == 'BY ' \
#                     and ';' in title:
#                 titlesplit = title.split(';')
#                 title = titlesplit[1]
#                 meta_dict['BYLINE'] = titlesplit[0][3:]
#
#             # Now get the various pieces of meta-data, along with main text
#             # Assume each meta-data item is separated by a blank line
#             # from the next or from the main text
#             text = ''
#             text2 = ''
#             text2used = False
#             text2truncated = False
#
#             while curLine < len(a_docX):
#                 line = a_docX[curLine].lstrip()
#                 if len(line) > 0:
#                     line = line.replace(',',commaString)
#
#                     # Deal with meta-data first
#                     ismeta = re.findall('^([A-ZÄ][A-ZÄ-]*?):',line)
#                     if sourceName == 'nrc' and line == 'SAMENVATTING':
#                         ismeta = ['SAMENVATTING',]
#                         curLine = curLine + 2  # skip blank line
#                         line = \
#                             a_docX[curLine].lstrip().replace(',', commaString)
#                     # Make sure this is meta-data we wish to keep.
#                     # If not, it is either uncommon meta-data or normal text.
#                     # Avoid skipping over the latter, so just include.
#                     if len(ismeta) > 0 and ismeta[0] in meta_list:
#                         meta_dict[ismeta[0]] = \
#                             line.replace(ismeta[0]+':','') + ' '
#                         # move to next line, in case of multi-line meta-data
#                         curLine = curLine +1
#                         if curLine == len(a_docX): # end of article
#                             break  # exit while loop
#                         line = a_docX[curLine].lstrip()
#                         while len(line) > 0 or (sourceName == 'nrc' and
#                                                 ismeta[0] == 'SAMENVATTING'):
#                             if not (sourceName == 'nrc'
#                                     and ismeta[0] == 'SAMENVATTING'
#                                     and 'VOLLEDIGE TEKST' in line):
#                                 # Not doing SAMENVATTING, or not done with it yet
#                                 meta_dict[ismeta[0]] = \
#                                     meta_dict[ismeta[0]] + \
#                                     line.replace(',', commaString) + ' '
#                             else: # Done with SAMENVATTING; change meta setting
#                                 ismeta[0] = 'not_a_meta'
#                             curLine += 1
#                             if curLine == len(a_docX): # end of article
#                                 break  # exit while loop
#                             line = a_docX[curLine].lstrip()
#
#                     # Now deal with what is most likely document text,
#                     # but possibly some other stuff
#                     else:
#                         if text == '' and sourceName == 'irish times' and \
#                                 meta_dict['BYLINE'] == '' and \
#                                 line.split()[0] == line.split()[0].upper():
#                             # Irish Times byline without following space
#                             authorname, restofline = splitITname(line)
#                             meta_dict['BYLINE'] = authorname
#
#                         # Skip '***' separator lines (Zeit) and
#                         #     Politiken author emails
#                         elif not text2truncated and \
#                                 (line != line.upper() and
#                                  not (line == line.split()[0] and
#                                       line.endswith('@pol.dk'))):
#                             if text2used:
#                                 text2 = text2 + line + ' '
#                                 if len(text2) > 32600:
#                                     text2truncated = True
#                             else:
#                                 text = text + line + ' '
#                                 if len(text) > 32600:
#                                     text2used = True
#                 curLine = curLine +1
#
#             # Check if this is a duplicate, to determine where to write it to
#             if prevDay == docDay and prevMonth == docMonth and \
#                     prevYear == docYear:
#                 titleComp = min(40,len(title))
#                 isDuplicate = title[:titleComp] in todayTitles \
#                                 or text[:100] in todayArticles
#                 # Add to list for today
#                 todayTitles = todayTitles + title + ' '
#                 todayArticles = todayArticles + text + ' '
#             else:  # different day -> reset vars
#                 isDuplicate = False
#                 prevDay = docDay
#                 prevMonth = docMonth
#                 prevYear = docYear
#                 todayTitles = title + ' '
#                 todayArticles = text + ' '
#
#             # Calculate some additional data from the info gathered so far,
#             # to prep for writing out (PubNr is specific to UK periodicals)
#             pubs = ['guardian', 'observer', 'mail', 'mirror',
#                     'daily telegraph', 'sunday telegraph', 'times',
#                     'sunday times', 'record', 'star', 'financial times',
#                     'independent', 'i', 'evening standard', 'express', 'sun',
#                     'people', 'news of the world', 'business']
#             pubName = sourceName
#             if 'mirror' in pubName:
#                 pubName = 'mirror'
#             elif pubName == 'daily mail' or pubName == 'mail on sunday':
#                 pubName = 'mail'
#             elif 'daily record' in pubName or 'sunday mail' in pubName:
#                 pubName = 'record'
#             elif 'star' in pubName:
#                 pubName = 'star'
#             elif 'i-' in pubName:
#                 pubName = 'i'
#             elif 'independent' in pubName:
#                 pubName = 'independent'
#             elif pubName == 'telegraph':
#                 pubName = 'daily telegraph'
#             elif 'express' in pubName:
#                 pubName = 'express'
#             elif 'business' in pubName:
#                 pubName = 'business'
#             if pubName in pubs:
#                 pubNr = 1 + pubs.index(pubName)
#             else:
#                 pubNr = 0
#
#             # Get date, if still pending
#             if date_pending and len(meta_dict['LOAD-DATE']) > 0:
#                 dateparts = meta_dict['LOAD-DATE'].split(' ')
#                 docYear = dateparts[2]
#                 docMonth = dateparts[0]
#                 docDay = dateparts[1][:-1]  # don't include comma
#
#             # Get month nr
#             months = ['January', 'February', 'March', 'April', 'May', 'June',
#                       'July', 'August', 'September', 'October', 'November',
#                       'December']
#             if sourceName == "el periodico de catalunya":
#                 months = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo',
#                           'Junio', 'Julio', 'Agosto', 'Septiembre',
#                           'Octubre', 'Noviembre', 'Diciembre']
#
#             if docMonth in months:
#                 monthNr = 1 + months.index(docMonth)
#             else:
#                 monthNr = 0
#             sequenceNr = 0  # Dummy value so far -> change as applicable
#             # Combine the TYPE, PUB-TYPE, PUBLICATION-TYPE metas
#             thetype = meta_dict['TYPE'] + meta_dict['PUB-TYPE'] + \
#                       meta_dict['PUBLICATION-TYPE']
#
#             # Finally write results to the appropriate csv file
#             # Consider re-encoding character strings
#             meta_tuple=tuple([x if not isinstance(x, str) else \
#                                   x.decode('ascii','ignore') for x in \
#                               [docID, searchResNr, pubName, pubNr, docYear,
#                                docMonth, monthNr, docDay, docDayofweek,
#                                sequenceNr, title, edition, text, text2used,
#                                text2, text2truncated]])
#             for item in meta_list:
#                 if item == 'TYPE':
#                     meta_tuple += (thetype,)
#                 elif item <> 'PUB-TYPE' and item <> 'PUBLICATION-TYPE':
#                     meta_tuple += (meta_dict[item],)
#             if isDuplicate:
#                 dupwriter.writerow(meta_tuple)
#                 docID -= 1
#             else:
#                 outwriter.writerow(meta_tuple)
#     return docID


