#!/usr/bin/python
# encoding=utf8

import csv
from HTMLParser import HTMLParser
from collections import defaultdict
import sys
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Change the default encoding
reload(sys)
sys.setdefaultencoding('utf8')

# Uncomment the following line to print logs
# printCountInfo = True
printCountInfo = False

def getCountOfWordInSentence(sWord, sentence):
    count = 0
    sentence = sentence.split()
    for word in sentence:
        if sWord == word:
            count = count + 1
    return count

def cleanSentence(sentence):
    sentence = re.sub('[^A-Za-z0-9]+', ' ', sentence).lower()
    sentence = re.sub(' +', ' ', sentence)
    sentence = sentence.strip()
    return sentence


def countTagOccurenceInQuestion(questionId, questionName, questionContent, tags, countTagsDictionary):
    questionName = cleanSentence(questionName)
    questionContent = cleanSentence(questionContent)
    tags = cleanSentence(tags)

    individualTags = tags.split(' ')

    if printCountInfo == True:
        print questionName + "\n" + questionContent + "\n"

    numberOfTags = len(individualTags)
    numberOfTagsFoundInNameOrContent = 0

    for tag in individualTags:
        countInName = getCountOfWordInSentence(tag, questionName)
        countInContent = getCountOfWordInSentence(tag, questionContent)
        if countInName > 0 or countInContent > 0:
        # if countInName > 0:
            numberOfTagsFoundInNameOrContent += 1

        if printCountInfo == True:
            print tag + " occurrence count:"
            print "\t In question name:" + str(countInName)
            # print "\t In question content:" + str(countInContent)

    if float(numberOfTagsFoundInNameOrContent) / numberOfTags > 1.0:
        print 'wtf!'

    if printCountInfo == True:
        print 'Finished processing'
        print '---------------------------------'
    # Make an entry in dictionary for this questionId
    countTagsDictionary[questionId] = (numberOfTagsFoundInNameOrContent, numberOfTags)

def consolidateResults(countTagsDictionary, fileName):
    percRanges = [(0 , 0), (.1, .25), (.26, .50), (.51, .75), (.76, 1.0)]
    countArr = []
    totalQuestions = 0
    rangeDictionary = defaultdict(int)

    for key in countTagsDictionary:
        totalQuestions += 1
        (numOfTagsFoundInQuestion, TotalTagsInQuestion) = countTagsDictionary[key]
        perc = float(numOfTagsFoundInQuestion) / TotalTagsInQuestion
        for (lower, upper) in percRanges:
            if perc >= lower and perc <= upper:
                rangeDictionary[(lower, upper)] += 1
                break

    print "\n\nFilename: " + fileName
    print "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    print "Total number of questions:" + str(totalQuestions)
    for key in percRanges:
        print str(key) + ":" + str(float(rangeDictionary[key]) / totalQuestions)

def plot_histogram(countTagsDictionaries):

    f, axarr = plt.subplots(2, 3)
    c = 0
    for topic, countTagsDictionary in countTagsDictionaries.iteritems():
        # nQuestions = len(countTagsDictionary)
        tag_fractions = []
        for k, v in countTagsDictionary.iteritems():
            v1, v2 = v
            tag_fractions.append(float(v1) / v2)

        hist, bin_edges = np.histogram(tag_fractions, 10, range = (0.0, 1.0))
        print hist
        print bin_edges

        row = c / 3
        col = c % 3
        plt.axes(axarr[row, col])
        plt.hist(tag_fractions, bins = 10, range = (0.0, 1.0))
        # plt.title('Number of Questions v/s Fractional Tag Count in Question Title for {0}'.format(topic))
        plt.tight_layout()
        plt.title(topic)
        # plt.ylabel('Number of Questions')
        # plt.xlabel('Fractional Tag Count in Bins ' + str(bin_edges))
        # plt.show()

        c += 1

    f.savefig('/home/siddharth/workspace/vis2.jpg')
    plt.close(f)

def parseFile(topic, fileName):
    countTagsDictionary = dict()
    with open(fileName) as f:
        reader = csv.reader(f)
        for row in reader:
            countTagOccurenceInQuestion(row[0], row[1], row[2], row[3], countTagsDictionary)
    # consolidateResults(countTagsDictionary, fileName)

    return countTagsDictionary

def parseFiles():
    inputFileNames = {'Biology': 'biology_processed.csv', 'Cooking': 'cooking_processed.csv', 'Crypto': 'crypto_processed.csv', 'DIY': 'diy_processed.csv', 'Robotics': 'robotics_processed.csv', 'Travel': 'travel_processed.csv'}
    # inputFileName = ['biology_processed.csv']

    countTagsDictionaries = {}
    for topic, file in inputFileNames.iteritems():
        countTagsDictionary = parseFile(topic, '../ProcessedData/' + file)
        countTagsDictionaries[topic] = countTagsDictionary

    plot_histogram(countTagsDictionaries)

parseFiles()
