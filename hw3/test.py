import sys

def main():
	scores = [[-sys.maxint for j in xrange(9)] for i in xrange(9)]
	
	scores[0][0] = -1
	scores[1][1] = -2
	scores[2][2] = -1.5
	scores[3][3] = -2.4
	scores[4][4] = -1.4
	scores[5][5] = -1.0
	scores[6][6] = -1.0
	scores[7][7] = -1.9
	scores[8][8] = -1.6
	scores[2][3] = -4.0
	scores[4][5] = -2.5
	scores[6][7] = -2.2
	scores[5][6] = -1.3
	scores[7][8] = -2.4
	scores[6][8] = -2.3
	scores[5][8] = -2.3
	scores[5][7] = -2.3
	
	max_computational_costs = [[0 for j in xrange(9)] for i in xrange(9)]

	for i in xrange(len(scores)):
		max_computational_costs[i][i] = scores[i][i]

	for i in xrange(len(scores) - 1):
		k = 0
		j = i + 1
		while k < len(scores) and j < len(scores):
			max_computational_costs[k][j] = scores[k][j]
			for m in xrange(k, j):                        
				if j >= len(scores):
					continue
				print '[{}, {}], [{}, {}], [{}, {}]'.format(k, m, m+1, j, k, j)
				max_computational_costs[k][j] = max(max_computational_costs[k][j], max_computational_costs[k][m] + max_computational_costs[m+1][j])    
			k += 1
			j += 1
		print ''
	#for i in xrange(len(max_computational_costs)):
	#	print max_computational_costs[i]
	for i in xrange(len(scores)):
		k = 0
		j = i
		while k < len(scores) and j < len(scores):
			print max_computational_costs[k][j],
			k += 1
			j += 1
		print ''

if __name__ == "__main__":
	main()
