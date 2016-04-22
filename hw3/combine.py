import sys

def main():
	files = sys.argv[1:]
	file_pointers = []
	for f in files:
		fp = open(f, 'r')
		file_pointers.append(fp)

	while True:
		maxi = -sys.maxint
		maxisentence = ''
		flag = 0
		for i in xrange(len(file_pointers)):
			line = file_pointers[i].readline().split('|')
			if line[0] == '':
				flag = 1
				break
			score = float(line[0])
			sentence = line[1]
			if score > maxi:
				maxi = score
				maxisentence = sentence
		if flag == 1:
			break
		print maxisentence.strip()

if __name__ == "__main__":
	main()
