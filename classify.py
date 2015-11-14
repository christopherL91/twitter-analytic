#!/usr/bin/env python3

from termcolor import colored
import json
import os
import nltk
import argparse
from sys import exit
import pickle
from TwitterAPI import TwitterAPI
from normalizr import Normalizr

CONSUMER_KEY = os.environ['CONSUMER_KEY']
CONSUMER_SECRET = os.environ['CONSUMER_SECRET']
ACCESS_TOKEN_KEY = os.environ['ACCESS_TOKEN_KEY']
ACCESS_TOKEN_SECRET = os.environ['ACCESS_TOKEN_SECRET']

def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
        all_words.extend(words)
    return all_words

def get_word_features(wordlist):
    return nltk.FreqDist(wordlist).keys()

def create_extracter(word_features):
    def extract_features(document):
        document_words = set(document)
        features = {}
        for word in word_features:
            features['contains(%s)' % word] = (word in document_words)
        return features
    return extract_features

def install(pos_path, neg_path):
    tweets = []

    print(colored('Reading positive tweets', 'blue'))
    with open(pos_path) as fp:
        for line in fp:
            d = json.loads(line)
            words_filtered = [e.lower() for e in d['text'].split() if e.isalnum()]
            tweets.append((words_filtered, 'positive'))

    print(colored('Reading negative tweets', 'blue'))
    with open(neg_path) as fp:
        for line in fp:
            d = json.loads(line)
            words_filtered = [e.lower() for e in d['text'].split() if e.isalnum()]
            tweets.append((words_filtered, 'negative'))

    print(colored('Dumping memory to file...', 'blue'))
    with open('dump.p', 'wb') as fp:
        pickle.dump(tweets, fp)

def main(tweets, classifier, hashtag):

    word_features = get_word_features(get_words_in_tweets(tweets))
    f = create_extracter(word_features)

    training_set = nltk.classify.apply_features(f, tweets)

    if not classifier:
        classifier = nltk.NaiveBayesClassifier.train(training_set)
        with open('classifier.p', 'wb') as fp:
            pickle.dump(classifier, fp)
        print(colored('Done generating classifier','red'))

    api = TwitterAPI(CONSUMER_KEY,
                 CONSUMER_SECRET,
                 ACCESS_TOKEN_KEY,
                 ACCESS_TOKEN_SECRET)

    print(colored('Will search for tweets with hashtag: #' + hashtag, 'blue'))
    r = api.request('statuses/filter', {'track': '#' + hashtag, 'language': 'en'})

    normalizr = Normalizr(language='en')
    try:
        for item in r:
            if 'text' in item:
                tweet = normalizr.normalize(item['text'].lower())
                res = classifier.classify(f(tweet.split()))
                output = colored(tweet, 'green') + ': ' + res
                print(output)
    except KeyboardInterrupt:
        pass

    #   Use the collected data and plot

if __name__ == "__main__":
    print(colored('Welcome to Twitter analytics','blue'))

    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--pos_tweets',
            help = 'JSON of positive tweets',
            type = str)
    parser.add_argument('-n','--neg_tweets',
            help = 'JSON of negative tweets',
            type = str)
    parser.add_argument('-s','--search',
            help = 'Classifier',
            type = str)
    args = parser.parse_args()

    if args.pos_tweets and args.neg_tweets:
        install(args.pos_tweets, args.neg_tweets)
    elif not args.pos_tweets and not args.neg_tweets:
        tweets = []
        classifier = None
        hashtag = 'yolo'
        if args.search:
            hashtag = args.search

        try:
            with open('dump.p', 'rb') as fp:
                tweets = pickle.load(fp)
        except IOError as e:
                print(colored('No tweet file found. Please generate one','red'))
        try:
            with open('classifier.p', 'rb') as fp:
                classifier = pickle.load(fp)
        except IOError as e:
            print(colored('No classifier file found, generating...','red'))
        main(tweets, classifier, hashtag)
    else:
        exit('No sufficient fields')
