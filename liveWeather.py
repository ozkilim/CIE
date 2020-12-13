import json
import requests

url = "https://api.ims.gov.il/v1/Envista/stations"

headers = {
    'Authorization': 'ApiToken XXXXXXXX'
}

response = requests.request("GET", url, headers=headers)
data = json.loads(response.text.encode('utf8'))
