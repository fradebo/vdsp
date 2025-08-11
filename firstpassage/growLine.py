import random
import numpy as np



L_target = 10
trials_tot = 2000

file = open("data.txt",'w')

for p_succ in np.arange(0.31,0.9,0.03):
	FP = 0

	for trial in range(trials_tot):
		L = 3
		steps = 0
		while(True):
			steps += 1
			if (random.uniform(0.0,1.0) < p_succ):
				L += 1
			else:
				L = max(3, L-1)
			if(L==L_target):
				FP += steps
				break
	file.write('{0} {1} \n'.format(p_succ, FP/trials_tot))
	print('{0} {1} '.format(p_succ, FP/trials_tot))
file.close()


