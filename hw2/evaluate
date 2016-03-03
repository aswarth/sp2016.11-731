#!/usr/bin/env python
import argparse # optparse is deprecated
from itertools import islice # slicing for iterators
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize
import string

stemmer = PorterStemmer()
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

def preprocess(words):
    global stemmer
    words = map(lambda x: string.lower(x), words)
    sentence = ' '.join(words)
    words = word_tokenize(sentence.encode('ascii', errors='ignore'))
    for i in xrange(len(words)):
    	if words[i] in string.punctuation:
		words[i] = 'PUNC'
    return map(lambda x: stemmer.stem(x), words)

def get_no_chunks(hyp, ref):
	word_one_length = len(hyp)
	word_two_length = len(ref)
	table = [[0 for j in xrange(word_two_length + 1)] for i in xrange(word_one_length + 1)]
	for i in xrange(1, wordOneLength + 1):
		for j in xrange(1, wordTwoLength + 1):
			if hyp[i-1] == ref[j-1]:
				current[j] = prev[j-1]
			else:
				current[j] = min(prev[j-1], min(prev[j], current[j-1])) + 1
		for k in xrange(0, wordTwoLength + 1):
			prev[k] = current[k]
    

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
    # or sum(w in ref for w in f) # cast bool -> int
    # or sum(map(ref.__contains__, h)) # ugly!
 

def compute_distance(hyp, ref, alpha):
	wordOneLength = len(hyp)
	wordTwoLength = len(ref)
	if wordOneLength == 0:
		return wordTwoLength
	if wordTwoLength == 0:
		return wordOneLength
	prev = [i for i in xrange(wordTwoLength + 1)]
	current = [0 for i in xrange(wordTwoLength + 1)]
	for i in xrange(1, wordOneLength + 1):
		current[0] = i
		for j in xrange(1, wordTwoLength + 1):
			if hyp[i-1] == ref[j-1]:
				current[j] = prev[j-1]
			else:
				current[j] = min(prev[j-1], min(prev[j], current[j-1])) + 1
		for k in xrange(0, wordTwoLength + 1):
			prev[k] = current[k]
	R = 1.0 - current[wordTwoLength]/(1.0 * wordTwoLength)
	P = 1.0 - current[wordTwoLength]/(1.0 * wordOneLength) 
    	denominator = P + alpha * R
	if denominator == 0:
		return 0
    	return (P * R)/ denominator

def main():
    parser = argparse.ArgumentParser(description='Evaluate translation hypotheses.')
    # PEP8: use ' and not " for strings
    parser.add_argument('-i', '--input', default='data/train-test.hyp1-hyp2-ref',
            help='input file (default data/train-test.hyp1-hyp2-ref)')
    parser.add_argument('-n', '--num_sentences', default=None, type=int,
            help='Number of hypothesis pairs to evaluate')
    parser.add_argument('-a', '--alpha', default=0.25, type=float,
            help='Trade-off parameter between Precision and Recall')
    # note that if x == [1, 2, 3], then x[:None] == x[:] == x (copy); no need for sys.maxint
    opts = parser.parse_args()
    alpha = opts.alpha 
    # we create a generator and avoid loading all sentences into a list
    def sentences():
        with open(opts.input) as f:
            for pair in f:
                yield [sentence.strip().split() for sentence in pair.split(' ||| ')]
 
    # note: the -n option does not work in the original code
    
    for h1, h2, ref in islice(sentences(), opts.num_sentences):
        #rset = set(ref)
        ref = preprocess(ref)
	h1 = preprocess(h1)
	h2 = preprocess(h2)
	h1_match = word_matches(h1, ref, alpha)
        h2_match = word_matches(h2, ref, alpha)
        #h1_match = compute_distance(h1, ref, alpha)
	#h2_match = compute_distance(h2, ref, alpha)
	print(-1 if h1_match > h2_match else # \begin{cases}
                (0 if h1_match == h2_match
                    else 1)) # \end{cases}
 
# convention to allow import of this file as a module
if __name__ == '__main__':
    main()
