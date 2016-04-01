#!/usr/bin/env python
import argparse
import sys
import models
import heapq
import copy
from collections import namedtuple
from _collections import defaultdict
import numpy as np

parser = argparse.ArgumentParser(description='Simple phrase based decoder.')
parser.add_argument('-i', '--input', dest='input', default='data/input', help='File containing sentences to translate (default=data/input)')
parser.add_argument('-t', '--translation-model', dest='tm', default='data/tm', help='File containing translation model (default=data/tm)')
parser.add_argument('-s', '--stack-size', dest='s', default=10, type=int, help='Maximum stack size (default=1)')
parser.add_argument('-n', '--num_sentences', dest='num_sents', default=sys.maxint, type=int, help='Number of sentences to decode (default=no limit)')
parser.add_argument('-l', '--language-model', dest='lm', default='data/lm', help='File containing ARPA-format language model (default=data/lm)')
parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', default=False,  help='Verbose mode (default=off)')
opts = parser.parse_args()

tm = models.TM(opts.tm, sys.maxint)
lm = models.LM(opts.lm)
sys.stderr.write('Decoding %s...\n' % (opts.input,))
input_sents = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

hypothesis = namedtuple('hypothesis', 'logprob, lm_state, predecessor, phrase, index')
mini = -sys.maxint
count = 0

current_index = 0

def get_distortion_prob(h, phrase, i):
        global current_index
        if h.predecessor == None:
            if phrase == None:
                current_index = 0
                return 0
            current_index = len(phrase.english.split())
            return abs(i)
        distortion = get_distortion_prob(h.predecessor, h.predecessor.phrase, h.predecessor.index)
        value = abs(current_index - i)
        current_index += len(phrase.english.split())
        return distortion + value 

def get_standard_normal(prob):
    return (1.0/2.5) * np.exp((-prob * prob)/2)

for f in input_sents:
    count += 1
    sys.stderr.write('Decoding Sentence %s\n' % (count))
    # The following code implements a DP monotone decoding
    # algorithm (one that doesn't permute the target phrases).
    # Hence all hypotheses in stacks[i] represent translations of 
    # the first i words of the input sentence.
    # HINT: Generalize this so that stacks[i] contains translations
    # of any i words (remember to keep track of which words those
    # are, and to estimate future costs)
    '''
    translation_scores = [[mini for j in xrange(len(f))] for i in xrange(len(f))]
    min_computational_costs = [[mini for j in xrange(len(f))] for i in xrange(len(f))]
    for i in xrange(len(f)):
        for j in xrange(i+1, len(f) + 1):
            if f[i:j] in tm:
                for phrase in tm[f[i:j]]:
                        translation_scores[i][j-1] = max(translation_scores[i][j-1], phrase.logprob)
    #print translation_scores 
    for i in xrange(len(f)):
        min_computational_costs[i][i] = translation_scores[i][i]
        
    for i in xrange(len(f)):
        k = 0
        j = i + 1
        while k < len(f) and j < len(f):
            min_computational_costs[k][j] = translation_scores[k][j]
            for m in xrange(k, j):                        
                if j >= len(f):
                    continue
                min_computational_costs[k][j] = max(min_computational_costs[k][j], min_computational_costs[k][m] + min_computational_costs[m+1][j])    
            k += 1
            j += 1
    '''       
    #print min_computational_costs
    #continue
    #break 
    
    initial_hypothesis = hypothesis(0.0, lm.begin(), None, None, 0)

    stacks = [{} for _ in f] + [{}]
    stacks[0][lm.begin()] = initial_hypothesis
    for i, stack in enumerate(stacks[:-1]):
        # extend the top s hypotheses in the current stack
        for h in heapq.nlargest(opts.s, stack.itervalues(), key=lambda h: h.logprob): # prune
            for j in xrange(i+1,len(f)+1):
                if f[i:j] in tm:
                    for phrase in tm[f[i:j]]:
                        #logprob = min_computational_costs[i][j-1]
                        logprob = h.logprob + phrase.logprob
                        lm_state = h.lm_state
                        for word in phrase.english.split():
                            (lm_state, word_logprob) = lm.score(lm_state, word)
                            logprob += word_logprob
                        logprob += lm.end(lm_state) if j == len(f) else 0.0
                        logprob += get_standard_normal(get_distortion_prob(h, phrase, i))     
                        new_hypothesis = hypothesis(logprob, lm_state, h, phrase, i)
                        if lm_state not in stacks[j] or stacks[j][lm_state].logprob < logprob: # second case is recombination
                            stacks[j][lm_state] = new_hypothesis 
                        temp = ''
                        for k in xrange(10):
                            queue = []
                            temp = h
                            flag = 0
                            for m in xrange(k): 
                                queue.append(copy.deepcopy(temp))
                                temp = temp.predecessor
                                if temp == None:
                                    flag = 1
                                    break
                            if flag == 1 or not temp.predecessor:
                                break
                            logprob = temp.predecessor.logprob + phrase.logprob
                            lm_state = temp.predecessor.lm_state
                            for word in phrase.english.split(): 
                                (lm_state, word_logprob) = lm.score(lm_state, word)
                                logprob += word_logprob
                            new_hypothesis_copy = hypothesis(logprob, lm_state, temp.predecessor, phrase, i)
                            logprob += temp.phrase.logprob
                            for word in temp.phrase.english.split(): 
                                (lm_state, word_logprob) = lm.score(lm_state, word)
                                logprob += word_logprob
                            hyp = hypothesis(logprob, lm_state, new_hypothesis_copy, temp.phrase, i)
                            lm_state = hyp.lm_state
                            while queue:
                                temp = queue.pop()
                                logprob += temp.phrase.logprob
                                for word in temp.phrase.english.split(): 
                                    (lm_state, word_logprob) = lm.score(lm_state, word)
                                    logprob += word_logprob
                                hyp = hypothesis(logprob, lm_state, hyp, temp.phrase, i)
                                lm_state = hyp.lm_state
                                
                            logprob += lm.end(lm_state) if j == len(f) else 0.0
                            logprob += get_standard_normal(get_distortion_prob(hyp, temp.phrase, i))
                            if lm_state not in stacks[j] or stacks[j][lm_state].logprob < logprob: # second case is recombination
                                stacks[j][lm_state] = hyp
    # find best translation by looking at the best scoring hypothesis
    # on the last stack
    winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)
    #print winner.logprob
    
    def extract_english_recursive(h):
        return '' if h.predecessor is None else '%s%s ' % (extract_english_recursive(h.predecessor), h.phrase.english)
    print extract_english_recursive(winner)

    if opts.verbose:
        def extract_tm_logprob(h):
            return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
        tm_logprob = extract_tm_logprob(winner)
        sys.stderr.write('LM = %f, TM = %f, Total = %f\n' % 
            (winner.logprob - tm_logprob, tm_logprob, winner.logprob))
