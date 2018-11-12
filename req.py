import requests
url = 'http://127.0.0.1:5000/'
params ={'vector' : str([1.5]*11)}
response = requests.get(url, params)
response.json()
