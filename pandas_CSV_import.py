import pandas as pd

data = pd.read_csv (r'CSVTRAIL.csv')

# Now  collect each csv file and graph each situation over the hours...

df = pd.DataFrame(data, columns= ['radiance'])
print (df)