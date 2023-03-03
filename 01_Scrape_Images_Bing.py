# Import Libraries to Scrape Images Using Bing
import requests
import shutil

from parameters import endpoint, subscription_key, saved_images

# list to save image URLs and image file names. 
images = []
image_name = []

# Query term(s) to search for. 
query = "car broken window"

# Construct a request
mkt = 'en-US'
params = {'q': query, 'mkt': mkt, 'count' : 150, 'offset': 2100}
headers = {'Ocp-Apim-Subscription-Key': subscription_key}

# Call the API
try:
    response = requests.get(endpoint, headers=headers, params=params)
    response.raise_for_status()
except Exception as ex:
    raise ex

# Convert response to json
response_JSON = response.json()

# add url of images to list
for i in range(0, len(response_JSON['value'])):
    images.append(response_JSON['value'][i]["contentUrl"])
    
# add name of each image to list. We will need this when we save the images.
for i in range(0, len(response_JSON['value'])):
    image_name.append(response_JSON['value'][i]["imageId"])

# Save images to Data folder
headers = {'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36'}
for image, name in zip(images, image_name):
    try:
        r = requests.get(image, stream=True, headers=headers)
        if r.status_code == 200:
            with open(f'{saved_images}{name}.png', 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw,f)
    except:
        pass