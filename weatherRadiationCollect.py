# Weather radiation collector
# here we get CSV and harvest the relevent data to plot it
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import os
import math

# repete for each file
filename = 'beersheva.csv'
File_path = 'CSV_DATA/' + filename
data = pd.read_csv(File_path, error_bad_lines=False)
# Preview the first 5 lines of the loaded data
data.head()
df = pd.DataFrame(data)
# find the a and b in the matrix ...find labeling issue
fulldataset = pd.DataFrame(data, columns=[
                           'Date', 'Time', 'globalIllum(W/m²)', 'directIllum(W/m²)', 'speadIllum(W/m²)'])

data = pd.read_csv(File_path, error_bad_lines=False)
data.head()
df = pd.DataFrame(data)
# Create searching algorithm...
#RAD_column = pd.DataFrame(data, columns = [' radiance[W/m2]'])
# name the columns correctly by removing the csv tag
feildName, file_extension = os.path.splitext(filename)
feildName = os.path.splitext(filename)[0]
# fulldataset[feildName] = RAD_column
# print(fulldataset)
# fulldataset = fulldataset.drop('# azimuth[deg]', 1)
# create new columm from the titles value
# This is the column of x values
# then plot against each y
columnsNamesArr = fulldataset.columns.values
listOfColumnNames = list(columnsNamesArr)
# get list of dates...

dateList = listOfColumnNames[0]
df.query('Date = ')
# way to queery with pandas???
# now match input 
# get the itrh column..
# check witch graphs you want to get out for each elivation..
# for a constant avimuth at different elivations
# This is from the user inputtted value
# find the data then time do get sub dataset...
forone = fulldataset[i:i+1]
oneLine = forone.values.tolist()



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
