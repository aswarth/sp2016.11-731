import argparse # optparse is deprecated
from itertools import islice # slicing for iterators

def main():
    parser = argparse.ArgumentParser(description='Evaluate translation hypotheses.')
    # PEP8: use ' and not " for strings
    parser.add_argument('-i', '--input', default='data/train-test.hyp1-hyp2-ref',
            help='input file (default data/train-test.hyp1-hyp2-ref)')
    parser.add_argument('-g', '--gold', default='data/train.gold',
            help='gold file (default data/train.gold)')
    parser.add_argument('-n', '--num_sentences', default=26208, type=int,
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
    labels = map(lambda x:int(x.strip()), open(opts.gold, 'r').readlines()) 
    instance = 0
    equal_count = 0.0
    actual_zero = 0.0
    for h1, h2, ref in islice(sentences(), opts.num_sentences):
    	if labels[instance] == 0 and ' '.join(h1) == ' '.join(h2):
		equal_count += 1
	if labels[instance] == 0:
		actual_zero += 1
	if labels[instance] == 0:
		print ' '.join(ref)
		print ' '.join(h1)
		print ' '.join(h2)
		print '-------------------------------------------------------------------------------------------'
	instance += 1
    print 'Ratio {}/{} = {}'.format(equal_count, actual_zero, equal_count/actual_zero)

if __name__ == '__main__':
    main()
