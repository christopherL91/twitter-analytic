#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from termcolor import colored
import json
from normalizr import Normalizr
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import nltk
import argparse
from sys import exit
import pickle
from TwitterAPI import TwitterAPI, TwitterRequestError, TwitterConnectionError

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

def plot(titletext, data, start, end):
    positive = data['positive']
    negative = data['negative']
    total = positive + negative

    labels = ['Positive','Negative']
    sizes = [positive, negative]
    colors = ['green','red']
    
    plt.figure(figsize=(10,10))
    plt.pie(sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90)
    
    title = 'Topic: {:s}\n Start: {:s} End: {:s}  \n Total number of tweets: {:d}'.format(titletext, start, end, total)
    plt.title(title, y=1, bbox={'facecolor': '0.8', 'pad': 5})

    plt.axis('equal')
    timestr = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig('plots/' + timestr + '.svg',format='svg') # write plot to disk
    plt.show()

def main(tweets, classifier, topic):
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

    print(colored('Will search for tweets with topic: ' + topic, 'blue'))
    data = {'positive': 0, 'negative': 0}
    normalizr = Normalizr(language='en')
    start = datetime.now().isoformat()
    while True:
        try:
            r = api.request('statuses/filter', {'track': topic, 'language': 'en'})
            for item in r:
                if 'text' in item:
                    text = item['text'].lower()
                    #tweet = normalizr.normalize(text)
                    res = classifier.classify(f(text.split()))
                    data[res] += 1 # update result set
                    output = colored(text, 'green') + ': ' + res
                    print(output)
                elif 'disconnect' in r:
                    event = item['disconnect']
                    if event['code'] in [2,5,6,7]:
                        raise Exception(event['reason'])
                    else:
                        break
        except TwitterRequestError as e:
            if e.status_code < 500:
                raise
            else:
                pass
        except TwitterConnectionError:
            pass
        except KeyboardInterrupt:
            print(colored('\nFinnished importing data... Will not continue generating plot', 'red'))
            break
    end = datetime.now().isoformat()
    plot(topic, data, start, end)

if __name__ == "__main__":
    print(colored('Welcome to Twitter analytics','blue'))

    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--pos_tweets',
            help = 'JSON of positive tweets',
            type = str)
    parser.add_argument('-n','--neg_tweets',
            help = 'JSON of negative tweets',
            type = str)
    parser.add_argument('-t','--topic',
            help = 'Classifier',
            type = str,
            default = 'yolo')
    args = parser.parse_args()

    if args.pos_tweets and args.neg_tweets:
        install(args.pos_tweets, args.neg_tweets)
    elif not args.pos_tweets and not args.neg_tweets:
        tweets = []
        classifier = None

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
        main(tweets, classifier, args.topic)
    else:
        exit('Missing required fields')
