#!/usr/bin/env python

"""Term frequency extractor for the PAN19 hyperpartisan news detection task"""
# Version: 2018-11-23

# Parameters:
# --inputDataset=<directory>
#   Directory that contains the articles XML file with the articles for which a prediction should be made.
# --outputFile=<file>
#   File to which the term frequency vectors will be written. Will be overwritten if it exists.

# Output is one article per line:
# <article id>,<term1>:<count>,<term2>:<count>,...

import os
import getopt
import sys
import xml.sax
import lxml.sax
import lxml.etree
import re
import pandas as pd
from collections import defaultdict

########## OPTIONS HANDLING ##########
def parse_options():
    """Parses the command line options."""
    try:
        long_options = ["inputDataset=", "outputFile="]
        opts, _ = getopt.getopt(sys.argv[1:], "d:o:", long_options)
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)

    inputDataset = "undefined"
    outputFile = "undefined"

    for opt, arg in opts:
        if opt in ("-d", "--inputDataset"):
            inputDataset = arg
        elif opt in ("-o", "--outputFile"):
            outputFile = arg
        else:
            assert False, "Unknown option."
    if inputDataset == "undefined":
        sys.exit("Input dataset, the directory that contains the articles XML file, is undefined. Use option -d or --inputDataset.")
    elif not os.path.exists(inputDataset):
        sys.exit("The input dataset folder does not exist (%s)." % inputDataset)

    if outputFile == "undefined":
        sys.exit("Output file, the file to which the vectors should be written, is undefined. Use option -o or --outputFile.")

    return (inputDataset, outputFile)

########## ARTICLE HANDLING ##########
def handleArticle(article, term_frequencies):
    termfrequencies = defaultdict(int)

    # get text from article
    text = lxml.etree.tostring(article, method="text", encoding="unicode")
    textcleaned = re.sub('[^a-z ]', '', text.lower())

    # counting tokens
    for token in textcleaned.split():
        termfrequencies[token] += 1

    article_id = article.get("id")
    term_frequencies[article_id] = termfrequencies

########## SAX FOR STREAM PARSING ##########
class HyperpartisanNewsTFExtractor(xml.sax.ContentHandler):
    def __init__(self, term_frequencies):
        xml.sax.ContentHandler.__init__(self)
        self.term_frequencies = term_frequencies
        self.lxmlhandler = "undefined"

    def startElement(self, name, attrs):
        if name != "articles":
            if name == "article":
                self.lxmlhandler = lxml.sax.ElementTreeContentHandler()

            self.lxmlhandler.startElement(name, attrs)

    def characters(self, data):
        if self.lxmlhandler != "undefined":
            self.lxmlhandler.characters(data)

    def endElement(self, name):
        if self.lxmlhandler != "undefined":
            self.lxmlhandler.endElement(name)
            if name == "article":
                # pass to handleArticle function
                handleArticle(self.lxmlhandler.etree.getroot(), self.term_frequencies)
                self.lxmlhandler = "undefined"

########## MAIN ##########
def main(inputDataset, outputFile):
    """Main method of this module."""
    term_frequencies = {}

    for file in os.listdir(inputDataset):
        if file.endswith(".xml"):
            with open(os.path.join(inputDataset, file), 'r', encoding='utf-8') as inputRunFile:  # Specify encoding for the input file
                parser = xml.sax.make_parser()
                parser.setContentHandler(HyperpartisanNewsTFExtractor(term_frequencies))
                parser.parse(inputRunFile)  # Directly parse the inputRunFile without creating an InputSource

    # Convert term frequencies to DataFrame
    df = pd.DataFrame.from_dict(term_frequencies, orient='index').fillna(0).astype(int)
    df.index.name = 'article_id'
    df.reset_index(inplace=True)

    # Save DataFrame to CSV
    df.to_csv(outputFile, index=False)

    print("The vectors have been written to the output file.")

if __name__ == '__main__':
    main(*parse_options())