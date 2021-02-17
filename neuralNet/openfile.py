import numpy as np
import sys
from pandas import read_csv

data = np.array( read_csv( sys.argv[1] + sys.argv[2] ) ).tolist()

f = open(sys.argv[1] + "output.txt","w+")
f.write(sys.argv[1] + sys.argv[2] + "\n")
for line in data:
	print(line)
	f.write(line+"\n")
f.close() 	