#!/usr/bin/env python
import argparse
import heapq
import models
import sys
import time

from collections import defaultdict
from collections import namedtuple

parser = argparse.ArgumentParser(description='Simple phrase based decoder.')
parser.add_argument('-i', '--input', dest='input', default='data/input', help='File containing sentences to translate (default=data/input)')
parser.add_argument('-t', '--translation-model', dest='tm', default='data/tm', help='File containing translation model (default=data/tm)')
parser.add_argument('-d', '--distortion-limit', dest='d', default=4, type=int, help='Distortion limit (default=4)')
parser.add_argument('-e', '--eta', dest='eta', default=-1, type=float, help='Distortion parameter (default=-1)')
parser.add_argument('-b', '--beam-search-size', dest='beam_size', default=10, type=int, help='Beam search size (default=100)')
parser.add_argument('-n', '--num_sentences', dest='num_sents', default=sys.maxint, type=int, help='Number of sentences to decode (default=no limit)')
parser.add_argument('-l', '--language-model', dest='lm', default='data/lm', help='File containing ARPA-format language model (default=data/lm)')
parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', default=False,  help='Verbose mode (default=off)')
opts = parser.parse_args()

start = time.time()
tm = models.TM(opts.tm, sys.maxint)
lm = models.LM(opts.lm)
d = opts.d
eta = opts.eta
beam_size = opts.beam_size

#sys.stderr.write('Decoding %s...\n' % (opts.input,))
input_sents = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

def get_all_possible_phrases(n):
    all_possible_phrases = []
    for i in range(n):
        for j in range(i + 1 , n + 1):
            if f[i:j] in tm:
                for phrase in tm[f[i:j]]:
                    p = (i, j, phrase.english, phrase.logprob)
                    all_possible_phrases.append(p)
    return all_possible_phrases

def beam(Q_i):
    return sorted(Q_i, key=lambda x: x[3], reverse = True)[ : min(beam_size, len(Q_i))]

def ph(all_possible_phrases, q, d):
    P = []
    for phrase in all_possible_phrases:
        i = phrase[0]
        j = phrase[1]
        english = phrase[2]
        logprob = phrase[3]

        #must not overlap
        overlap = False
        bit = q[1]
        for k in range(i, j):
            if bit[k] == 1:
                overlap = True
                break
        if overlap:
            continue

        #distortion limit must not be violated
        r = q[2]
        if abs(r + 1 - i) > d:
            continue

        #valid phrase p = (start, end, english phrase, translation loprob)
        p = (i, j, english, logprob)
        P.append(p)
    return P

def lm_log_prob(sentence, lm_state, end):
    logprob = 0.0
    for word in sentence.split():
        (lm_state, word_logprob) = lm.score(lm_state, word)
        logprob += word_logprob
    logprob += lm.end(lm_state) if end else 0.0
    return lm_state, logprob

def next(q, p, n):
    phrase_prime = (q[0] + " " + p[2]).lstrip(" ")
    bit_prime = list(q[1])
    for k in range(p[0], p[1]):
        bit_prime[k] = 1
    r_prime = p[1]
    end = True if p[1] == n else False
    lm_state, logprob = lm_log_prob(p[2], q[4], end)
    alpha_prime = q[3] + p[3] + logprob + eta * abs(q[2] + 1 - p[0])
    q_prime = (phrase_prime, bit_prime, r_prime, alpha_prime, lm_state)

    return q_prime

def add(Q_j, q_prime, q, p):
    flag = False
    for q_double_prime in Q_j:
        if q_double_prime[0] == q_prime[0] and q_double_prime[1] == q_prime[1] and q_double_prime[2] == q_prime[2]:
            if q_prime[3] > q_double_prime[3]:
                Q_j.remove(q_double_prime)
                Q_j.append(q_prime)
                return
    Q_j.append(q_prime)

for f in input_sents:
    Q = defaultdict(list)
    n = len(f)
    d = min(d, n / 2)

    #initialization
    #state q = (phrase, bit, r, alpha, lm_state)
    q_0 = ("", [0] * n, 0, 0, lm.begin())
    Q[0].append(q_0)

    #sys.stderr.write("%s\n" % ' '.join(f))
    all_possible_phrases = get_all_possible_phrases(n)

    #decoding
    for i in range(n + 1):
        #sys.stderr.write("%d / %d\n" % (i, len(f)))
        for q in beam(Q[i]):
            for p in ph(all_possible_phrases, q, d):
                q_prime = next(q, p, n)
                add(Q[sum(q_prime[1])], q_prime, q, p)
        if len(Q[i]) == 0:
            break
        #for key, value in Q.iteritems():
        #    sys.stderr.write("Q_%s size: %d\n" % (str(key), len(Q[key])))
        #print sorted(Q[i], key=lambda x: x[3], reverse = True)[0:5]
        #sys.stderr.write("%s\n" % ("=" * 50))
    if len(Q[i]) == 0:
        best_sentence = sorted(Q[i-1], key=lambda x: x[3], reverse = True)[0][0]
    else:
        best_sentence = sorted(Q[i], key=lambda x: x[3], reverse = True)[0][0]
    print best_sentence

end = time.time()
#sys.stderr.write("%f seconds\n" % (end - start))
