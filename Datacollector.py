# here we get CSV and harvest the relevent data to plot it

import pandas as pd 
import matplotlib.pyplot as plt
import itertools
import os
import math

# repete for each file
File_path = '0.csv'
	# Read data from file 'filename.csv' 
	# (in the same directory that your python process is based)
	# Control delimiters, rows, column names with read_csv (see later) 
data = pd.read_csv(File_path, error_bad_lines=False) 
	# Preview the first 5 lines of the loaded data 
data.head()
df = pd.DataFrame(data) 

# Azimuth = input("Enter Azimuth")
# Elivation = input("Enter Elivation")

# # take on row out as list..
# azimuths = df['# azimuth[deg]'].to_list()


# # if not find a way to narrow it down..
# print(azimuths.index(float(Azimuth)))

# altitudes = df[' altitude[deg]'].to_list()

# print(altitudes.index(float(Elivation)))

# find the a and b in the matrix ...
fulldataset = pd.DataFrame(data, columns = ['# azimuth[deg]',' altitude[deg]', ' radiance[W/m2]']) 
# Just plot irradiances later on 
for filename in os.listdir('CSV_DATA'):

	File_path = 'CSV_DATA/' + filename
	print(File_path)
	# Read data from file 'filename.csv' 

	# (in the same directory that your python process is based)
	# Control delimiters, rows, column names with read_csv (see later) 

	data = pd.read_csv(File_path, error_bad_lines=False) 
	# Preview the first 5 lines of the loaded data 
	data.head()
	df = pd.DataFrame(data) 
	# add just the row from this dataset..
	# make this not get overwritten
	RAD_column = pd.DataFrame(data, columns = [' radiance[W/m2]']) 
	# name the columns correctly by removing the csv tag
	feildName, file_extension = os.path.splitext(filename)
	# feildName = os.path.splitext(filename)[0]
	fulldataset[feildName] = RAD_column
	# remove first 4 columns
# print(fulldataset)
fulldataset = fulldataset.drop('# azimuth[deg]', 1)
fulldataset = fulldataset.drop(' altitude[deg]', 1)
fulldataset = fulldataset.drop(' radiance[W/m2]', 1)
fulldataset = fulldataset.drop('.DS_Store', 1)
# print(fulldataset)


# create new columm from the titles value
# This is the column of x values 
# then plot against each y 
columnsNamesArr = fulldataset.columns.values
listOfColumnNames = list(columnsNamesArr)
# get the itrh column..
# check witch graphs you want to get out for each elivation..
# for a constant avimuth at different elivations
# This is from the user inputtted value

i = 818
forone = fulldataset[i:i+1]
oneLine = forone.values.tolist()


listOfColumnNames
oneLine = oneLine[0]

new_listofnames = [int(j) for j in listOfColumnNames] 
	# lists = sorted(itertools.zip(*[(listOfColumnNames, oneLine]))
the_toup_pack = [(new_listofnames[j], oneLine[j]) for j in range(len(oneLine))]

the_toup_pack.sort()

new_x, new_y = zip(*the_toup_pack)

# convert irradiance to radince w/cm^2
final_y = []

skydome_divisions = 827

solid_angle = (2*math.pi/skydome_divisions) 
for number in new_y:
	number = (number/(solid_angle*10000))
	final_y.append(number)




plt.plot(new_x, final_y)

plt.show()



# input i --> output az and el..

# get print out of the az and el ...
# get az and el out cretui gui with all the values.. 


