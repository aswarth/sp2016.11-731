#!/usr/bin/env python
import optparse
import sys
import time
from collections import defaultdict
from IBM1 import IBM1

class IBM2:
	def __init__(self, opts, ibm1):
		self.bitext = self.read_corpus(opts)
		self.translation_probability, self.alignment_probability, self.count_e_f, self.count_e, self.count_m_l_i_j, self.count_m_l_i = self.initialize_parameter(ibm1)
		self.iteration_number = opts.iteration_number

	def read_corpus(self, opts):
		sys.stderr.write(">> Reading corpus...\n")
		bitext = [[sentence.strip().split() for sentence in pair.split(' ||| ')] for pair in open(opts.bitext)][:opts.num_sents]
		if opts.reverse:
			for (n, (f, e)) in enumerate(bitext):
				bitext[n][0], bitext[n][1] = bitext[n][1], bitext[n][0]
				
		return bitext

	def initialize_parameter(self, ibm1):
		sys.stderr.write(">> Initializing model parameters...\n")
		translation_probability = ibm1.translation_probability
		alignment_probability = defaultdict(float)
		count_e = defaultdict(float)
		count_e_f = defaultdict(float)
		count_m_l_i = defaultdict(float)
		count_m_l_i_j = defaultdict(float)

		for (n, (f, e)) in enumerate(self.bitext):
			(m, l) = (len(f), len(e))
			count_m_l_i[(m, l)] = defaultdict(float)
			count_m_l_i_j[(m, l)] = defaultdict(float)
			for e_j in e:
				count_e[e_j] = defaultdict(float)
				count_e_f[e_j] = defaultdict(float)
				for f_i in f:
					count_e_f[e_j][f_i] = 0
			for (i, f_i) in enumerate(f):
				count_m_l_i[(m, l)][i] = 0
				count_m_l_i_j[(m, l)][i] = defaultdict(float)
				for (j, e_j) in enumerate(e):
					count_m_l_i_j[(m, l)][i][j] = defaultdict(float)
					alignment_probability[(len(f), len(e), i, j)] = 1 / float(len(e))

		return translation_probability, alignment_probability, count_e_f, count_e, count_m_l_i_j, count_m_l_i

	def em_algorithm(self):
		sys.stderr.write(">> EM algorithm...\n")
		for iteration in range(self.iteration_number):
			sys.stderr.write(" Iteration: %i" % (iteration + 1))

			for e_j in self.count_e_f.keys():
				self.count_e[e_j] = 0
				for f_i in self.count_e_f[e_j].keys():
					self.count_e_f[e_j][f_i] = 0
			for (m, l) in self.count_m_l_i_j.keys():
				for i in self.count_m_l_i_j[(m, l)].keys():
					self.count_m_l_i[(m, l)][i] = 0
					for j in self.count_m_l_i_j[(m, l)][i].keys():
						self.count_m_l_i_j[(m, l)][i][j] = 0
			
			sys.stderr.write(" Calculating delta...\n")
			delta = defaultdict(float)
			for (n, (f, e)) in enumerate(self.bitext):
				(m, l) = (len(f), len(e))
				for (i, f_i) in enumerate(f):
					summation = 0
					for (j, e_j) in enumerate(e):
						summation += self.alignment_probability[(len(f), len(e), i, j)] * self.translation_probability[(f_i, e_j)]
					for (j, e_j) in enumerate(e):
						delta = self.alignment_probability[(len(f), len(e), i, j)] * self.translation_probability[(f_i, e_j)] / summation
						self.count_e[e_j] += delta
						self.count_e_f[e_j][f_i] += delta
						self.count_m_l_i[(m, l)][i] += delta
						self.count_m_l_i_j[(m, l)][i][j] += delta

			sys.stderr.write(" Updating translation probability...\n")
			summation_dict = {}
			for (f_i, e_j) in self.translation_probability.keys():
				if e_j not in summation_dict:
					summation_dict[e_j] = sum(self.count_e_f[e_j].values())
				self.translation_probability[(f_i, e_j)] = self.count_e_f[e_j][f_i] / summation_dict[e_j]
				#self.translation_probability[(f_i, e_j)] = self.count_e_f[e_j][f_i] / self.count_e[e_j]

			sys.stderr.write(" Updating alignment probability...\n")
			summation_dict = {}
			for (m, l, i, j) in self.alignment_probability.keys():
				if (m, l, i) not in summation_dict:
					summation_dict[(m, l, i)] = sum(self.count_m_l_i_j[(m, l)][i].values())
				self.alignment_probability[(m, l, i, j)] = self.count_m_l_i_j[(m, l)][i][j] / summation_dict[(m, l, i)]

	def predict_alignment(self):
		sys.stderr.write(">> Predicting alignment...\n")
		for (f, e) in self.bitext:
			m = len(f)
			l = len(e)
			for (i, f_i) in enumerate(f):
				maximum=0
				pos=0
				for (j, e_j) in enumerate(e):
					if self.alignment_probability[(m, l, i, j)] * self.translation_probability[(f_i, e_j)] > maximum:
						maximum = self.alignment_probability[(m, l, i, j)] * self.translation_probability[(f_i, e_j)]
						pos = j
				sys.stdout.write("%i-%i " % (i, pos))
			sys.stdout.write("\n")


if __name__ == "__main__":
	optparser = optparse.OptionParser()
	optparser.add_option("-b", "--bitext", dest="bitext", default="data/dev-test-train.de-en", help="Parallel corpus (default data/dev-test-train.de-en)")
	optparser.add_option("-i", "--iteration_number", dest="iteration_number", default=5, type="int", help="Number of iterations for EM algorithm (default=5)")
	optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
	optparser.add_option("-r", "--reverse", dest="reverse", default=False, help="Reverse Translation Probability")

	(opts, _) = optparser.parse_args()

	t0=time.clock()

	ibm1 = IBM1(opts)
	ibm1.em_algorithm()
	ibm2 = IBM2(opts, ibm1)
	ibm2.em_algorithm()
	ibm2.predict_alignment()

	sys.stderr.write(">> run time: %f sec \n" % (time.clock() - t0))
