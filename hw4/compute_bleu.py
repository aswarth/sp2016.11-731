import math
from collections import Counter
import numpy as np

def bleu_stats(hypothesis, reference):
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))
    for n in xrange(1, 5):
        s_ngrams = Counter([tuple(hypothesis[i:i+n]) for i in xrange(len(hypothesis)+1-n)])
        r_ngrams = Counter([tuple(reference[i:i+n]) for i in xrange(len(reference)+1-n)])
        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(hypothesis)+1-n, 0]))
    return stats

def bleu(stats):
    if len(filter(lambda x: x==0, stats)) > 0:
        return 0
    (c, r) = stats[:2]
    log_bleu_prec = sum([math.log(float(x)/y) for x,y in zip(stats[2::2],stats[3::2])]) / 4.
    return math.exp(min([0, 1-float(r)/c]) + log_bleu_prec)

def get_bleu(hyp):
    stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    ref = map(lambda x:x.strip().split(), open('data/dev.tgt').readlines())
    for h, r in zip(hyp, ref):
        stats += np.array(bleu_stats(h, r))
    return "%.2f" % (100*bleu(stats))
