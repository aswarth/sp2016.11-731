#!/usr/bin/env python
import argparse # optparse is deprecated
from itertools import islice # slicing for iterators
from nltk.stem.porter import *
#from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
import string
import os
import numpy as np
import copy
import math

SVM_HOME = '/Users/Aswarth/libsvm-3.18/'
SVM_TOOLS_HOME = os.path.join(SVM_HOME, "tools")
SVM_TRAIN = os.path.join(SVM_HOME, "svm-train")
SVM_SCALE = os.path.join(SVM_HOME, "svm-scale")
SVM_PREDICT = os.path.join(SVM_HOME, "svm-predict")
SVM_SUBSET_SELECT = os.path.join(SVM_TOOLS_HOME, "subset.py")
SVM_FEAT_SELECT = os.path.join(SVM_TOOLS_HOME, "fselect.py")

stemmer = PorterStemmer()
#stemmer = SnowballStemmer("english")
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
stopwords = {}

def compute_levenshtein_distance(hyp, ref, alpha):
    word_one_length = len(hyp)
    word_two_length = len(ref)
    if word_one_length == 0:
        return word_two_length
    if word_two_length == 0:
        return word_one_length
    prev = [i for i in xrange(word_two_length + 1)]
    current = [0 for i in xrange(word_two_length + 1)]
    for i in xrange(1, word_one_length + 1):
        current[0] = i
        for j in xrange(1, word_two_length + 1):
            if hyp[i-1] == ref[j-1]:
                current[j] = prev[j-1]
            else:
                current[j] = min(prev[j-1], min(prev[j], current[j-1])) + 1
        for k in xrange(0, word_two_length + 1):
            prev[k] = current[k]
    
    R = 1.0 - current[word_two_length]/(1.0 * word_two_length)
    P = 1.0 - current[word_two_length]/(1.0 * word_one_length) 
    
    numerator = P * R
    denominator = alpha * R + (1 - alpha) * P
    if denominator > 0:
        return numerator/denominator
    return 0

def preprocess(words):
    global stemmer
    sentence = ' '.join(words)
    words = word_tokenize(sentence.encode('ascii', errors='ignore'))
    for i in xrange(len(words)):
        if words[i] in string.punctuation:
            words[i] = 'PUNC'
    return map(lambda x: stemmer.stem(x), words)

# DRY
def word_matches(h, ref, alpha):
    rset = set(ref)
    hset = set(h)
    precision_count = sum(1.0 for w in h if w in rset)
    recall_count = sum(1.0 for w in ref if w in hset)
    P = precision_count/len(h)
    R = recall_count/len(ref)
    if P == 0 and R == 0:
        return 0
    return 1.0/(alpha/P + (1 - alpha)/R)

def remove_stop_words(rset):
    dup_set = copy.deepcopy(rset)
    for word in rset:
        if word in stopwords:
            dup_set.remove(word)
    return dup_set

def content_word_matches(h, ref, alpha):
    rset = set(ref)
    hset = set(h)
    rset = remove_stop_words(rset)
    hset = remove_stop_words(hset)
    precision_count = sum(1.0 for w in h if w in rset)
    recall_count = sum(1.0 for w in ref if w in hset)
    P = precision_count/len(h)
    R = recall_count/len(ref)
    if P == 0 and R == 0:
        return 0
    return 1.0/(alpha/P + (1 - alpha)/R)

def load_stop_words():
    global stopwords
    fp = open('stopwords.txt', 'r')
    for line in fp:
        stopwords[line.strip()] = True
    fp.close()

def write_to_file(output_file, features):
    fp = open(output_file, 'w')
    for feature in features:
        fp.write(feature + '\n')

def svm_train_and_predict(train_features, test_features):
    write_to_file('svm_train.data.ngram5', train_features)
    write_to_file('svm_test.data.ngram5', test_features)

def get_counts(hyp, n):
    word_freqs = {}
    size = len(hyp) - n + 1
    for i in xrange(size):
        key = ' '.join(hyp[i:i+n])
        word_freqs[key] = word_freqs.get(key, 0) + 1
    
    return word_freqs

def modified_ngram_precision(h, length, ref):
    count = 0.0
    for word in h:
        if word in ref:
            count += min(h[word], ref[word])
    return count/length

def brevity_penalty(h, ref):
    c = len(h)
    r = abs(len(ref) - c)
    if c > r:
        return 1
    else:
        return math.exp(1 - r / c)

def bleu_features(N, weights, h1, h2, ref):
    local_features = []
    h1_ngram_precision = []
    h2_ngram_precision = []
    for i in xrange(1, N):
        ref_max_counts = get_counts(ref, i)
        h1_ngram_precision.append(modified_ngram_precision(get_counts(h1, i), len(h1), ref_max_counts))
        h2_ngram_precision.append(modified_ngram_precision(get_counts(h2, i), len(h2), ref_max_counts))
        
    local_features += h1_ngram_precision
    local_features += h2_ngram_precision
        
    sh1 = math.fsum(w * math.log(precision) for w, precision in zip(weights, h1_ngram_precision) if precision)
    sh2 = math.fsum(w * math.log(precision) for w, precision in zip(weights, h2_ngram_precision) if precision)

    bp1 = brevity_penalty(h1, ref)
    bp2 = brevity_penalty(h2, ref)
        
    local_features.append(bp1 * math.exp(sh1))
    local_features.append(bp2 * math.exp(sh2))
    return local_features

def main():
    parser = argparse.ArgumentParser(description='Evaluate translation hypotheses.')
    parser.add_argument('-i', '--input', default='data/train-test.hyp1-hyp2-ref',
            help='input file (default data/train-test.hyp1-hyp2-ref)')
    parser.add_argument('-g', '--gold', default='data/train.gold',
            help='gold file (default data/train.gold)')
    parser.add_argument('-n', '--num_sentences', default=None, type=int,
            help='Number of hypothesis pairs to evaluate')
    parser.add_argument('-a', '--alpha', default=0.25, type=float,
            help='Trade-off parameter between Precision and Recall')

    opts = parser.parse_args()
    alpha = opts.alpha
    labels = map(lambda x:int(x.strip()), open(opts.gold, 'r').readlines()) 
    load_stop_words()
    
    def sentences():
        with open(opts.input) as f:
            for pair in f:
                yield [sentence.strip().split() for sentence in pair.split(' ||| ')]
 
    # note: the -n option does not work in the original code
    instance_count = 0
    features = []
    training_size = 26208
    N = 5
    weights = [1.0/N for i in xrange(N)]
    for h1, h2, ref in islice(sentences(), opts.num_sentences):
        print 'Processing Sentence {}'.format(instance_count + 1)
        local_features = []
        
        h1 = map(lambda x: string.lower(x), h1)
        h2 = map(lambda x: string.lower(x), h2)
        ref = map(lambda x: string.lower(x), ref)

        h1_match = word_matches(h1, ref, alpha)
        h2_match = word_matches(h2, ref, alpha)
        hyp_match = word_matches(h1, h2, alpha)
        
        local_features.append(h1_match)
        local_features.append(h2_match)
        local_features.append(h1_match - h2_match)
        local_features.append(hyp_match)
        
        local_features += bleu_features(N, weights, h1, h2, ref)

        ref = preprocess(ref)
        h1 = preprocess(h1)
        h2 = preprocess(h2)
        
        h1_match = word_matches(h1, ref, alpha)
        h2_match = word_matches(h2, ref, alpha)
        hyp_match = word_matches(h1, h2, alpha)
        
        local_features.append(h1_match)
        local_features.append(h2_match)
        local_features.append(h1_match - h2_match)
        local_features.append(hyp_match)
        
        h1_match = content_word_matches(h1, ref, alpha)
        h2_match = content_word_matches(h2, ref, alpha)
        hyp_match = content_word_matches(h1, h2, alpha)
        
        local_features.append(h1_match)
        local_features.append(h2_match)
        local_features.append(h1_match - h2_match)
        local_features.append(hyp_match)
        
        #'''
        h1_match = compute_levenshtein_distance(h1, ref, alpha)
        h2_match = compute_levenshtein_distance(h2, ref, alpha)
        hyp_match = compute_levenshtein_distance(h1, h2, alpha)
        
        local_features.append(h1_match)
        local_features.append(h2_match)
        local_features.append(h1_match - h2_match)
        local_features.append(hyp_match)
        
        local_features += bleu_features(N, weights, h1, h2, ref)
        
        feature_vector = ''
        for i in xrange(1, len(local_features) + 1):
            feature_vector += '{}:{}'.format(i, local_features[i - 1]) + ' '
        feature_vector = feature_vector.strip()
        if instance_count < training_size:
            feature_vector = '{} {}'.format(labels[instance_count], feature_vector)
        else:
            feature_vector = '{} {}'.format(2, feature_vector)   
       
        features.append(feature_vector)
        instance_count += 1
    
    train_features = features[0:training_size]
    test_features = features[training_size:]
 
    svm_train_and_predict(train_features, test_features)
 
# convention to allow import of this file as a module
if __name__ == '__main__':
    main()
