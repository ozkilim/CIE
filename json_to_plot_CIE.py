#Script to read the json and graph over the value for each radiance and azimuth
#think about physical meaning

#7first turn the documents into correct json?


import json

with open('JSON_HOURS/CIE_12.json') as json_file:
    data = json.load(json_file)
