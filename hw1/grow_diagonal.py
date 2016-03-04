#!/usr/bin/env python
import sys
import time

def read_alignments(file_name, reverse):
	alignments = []
	with open(file_name) as lines:
		for line in lines:
			alignment_set = set()
			line = line.strip().split(" ")
			for alignment_pair in line:
				x = int(alignment_pair.split("-")[0])
				y = int(alignment_pair.split("-")[1])
				if reverse == False:
					alignment_set.add((x, y))
				else:
					alignment_set.add((y, x))
			alignments.append(alignment_set)
	return alignments

def is_neighbor(candidate, final_alignment):
	if abs(candidate[0] - final_alignment[0]) < 2 and abs(candidate[1] - final_alignment[1]) < 2:
		return True
	else:
		return False

def grow_alignments(alignments, reverse_alignments):
	final_aligments = []
	for alignment_set, reverse_alignment_set in zip(alignments, reverse_alignments):
		intersection = alignment_set & reverse_alignment_set
		union = alignment_set | reverse_alignment_set
		final_alignment_set = intersection
		final_alignment_x_set = {final_aligment[0] for final_aligment in final_alignment_set}
		final_alignment_y_set = {final_aligment[1] for final_aligment in final_alignment_set}
		while True:
			to_add = set()
			for final_alignment in final_alignment_set:
				for candidate in union:
					if (candidate[0] not in final_alignment_x_set or candidate[1] not in final_alignment_y_set) and is_neighbor(candidate, final_alignment):
						to_add.add(candidate)
						final_alignment_x_set.add(candidate[0])
						final_alignment_y_set.add(candidate[1])
			if len(to_add)==0:
				break
			final_alignment_set |= to_add
		final_aligments.append(final_alignment_set)
	return final_aligments

def write_final_alignments(final_aligments):
	for final_alignment in final_aligments:
		output = [x for x in sorted(final_alignment, key=lambda x: x[0])]
		for x in output:
			sys.stdout.write("%i-%i " % (x[0], x[1]))
		sys.stdout.write("\n")

if __name__ == "__main__":
	t0=time.clock()
	alignment_file = sys.argv[1]
	reverse_alignment_file = sys.argv[2]
	sys.stderr.write(">> input files: %s and %s \n" % (alignment_file, reverse_alignment_file))

	alignments = read_alignments(alignment_file, False)
	reverse_alignments = read_alignments(reverse_alignment_file, True)
	sys.stderr.write(">> grow diagnol... \n")
	final_aligments = grow_alignments(alignments, reverse_alignments)
	write_final_alignments(final_aligments)

	sys.stderr.write(">> run time: %f sec \n" % (time.clock() - t0))


