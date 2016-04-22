import os

def main():
	i = 0.1
	while i < 1.0:
		os.system('./evaluate -a {} | ./check | ./grade'.format(i))
		i += 0.01
if __name__ == '__main__':
	main()
