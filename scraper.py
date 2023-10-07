import requests
from bs4 import BeautifulSoup
import os
from time import sleep
import urllib.request
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Function to download image
def download_image(url, name):
    # Make the request
    r = requests.get(url)
    if 'image' not in r.headers.get('Content-Type', ''):
        print("Not an image.")
        return False

    # Check if the content has some size
    if len(r.content) <= 1:
        print("Empty file.")
        return False

    # Save the file
    with open(name, 'wb') as f:
        f.write(r.content)
        f.flush()
        f.seek(0, 2)
        file_size = f.tell()
        print(f"File size: {file_size} bytes")

    return True


def download_image_alternative(url, name):
    try:
        urllib.request.urlretrieve(url, name)

        file_size = os.path.getsize(name)
        print(f"File size: {file_size} bytes")
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


def download_and_convert_image(url, name):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        temp_name = "temp.webp"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response, open(temp_name, 'wb') as out_file:
            data = response.read()
            out_file.write(data)

        im = Image.open(temp_name).convert("RGBA")
        im.save(name, "PNG")

        file_size = os.path.getsize(name)
        print(f"File size: {file_size} bytes")

        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


def convert_url(url):
    parts = url.split('/')
    parts[3] = 'source'
    del parts[4]
    parts[-1] = parts[-1].replace('.webp', '.png')
    new_url = '/'.join(parts)

    return new_url

options = webdriver.ChromeOptions()
options.add_argument('--headless') 
driver = webdriver.Chrome(options=options)

driver.get('https://emojipedia.org/apple')


wait = WebDriverWait(driver, 10)  # Wait up to 20 seconds
wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, 'flex')))

soup = BeautifulSoup(driver.page_source, 'html.parser')

if not os.path.exists('emojis'):
    os.mkdir('emojis')

#print(soup)


for idx, emoji_div in enumerate(soup.find_all('a', {'class': 'Emoji_emoji__P7Lkz'})):

    emoji_name = emoji_div['href']
    emoji_name = emoji_name.split("/")[-1]
    print(emoji_name)

    img_url = emoji_div['data-src']

    if img_url == '':
        continue

    down_url = convert_url(img_url)

    print(down_url)
    download_image(down_url, f'emojis/{emoji_name}.png')


print('Done scraping emojis.')
