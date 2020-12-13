import pandas as pd 
import matplotlib.pyplot as plt
import itertools
import os
import math
import numpy as np


File_path = '10am.csv'

data = pd.read_csv(File_path, error_bad_lines=False) 
	# Preview the first 5 lines of the loaded data 
data.head()
df = pd.DataFrame(data) 

luminance = pd.DataFrame(data, columns = [' luminance[cd/m2]']) 

patchArea = pd.DataFrame(data, columns = [' patchArea']) 


new = luminance.values/(np.pi*patchArea.values)
print((new))



luminance = np.sum(luminance)
print(luminance)


patchArea = np.sum(patchArea)
print(patchArea)


# print(np.sum(patchArea*luminance)/np.pi)
