import os
import requests
from PIL import Image
from io import BytesIO
import random
import string

from config import unsplash_access_key, min_width, min_height, output_folder


def get_random_image_url():
    url = f"https://api.unsplash.com/photos/random?client_id={unsplash_access_key}&w={min_width}&h={min_height}"
    response = requests.get(url)
    data = response.json()
    return data['urls']['full']


def download_image(url, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))

        if img.width >= min_width and img.height >= min_height:
            # Генерируем случайное имя файла
            filename = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10)) + ".jpg"
            filepath = os.path.join(output_folder, filename)
            img.save(filepath)

            print(f"Image saved: {filepath}")
        else:
            print(f"Image skipped: {url} (size {img.width}x{img.height})")
    except Exception as e:
        print(f"Failed to download image from {url}: {e}")


def get_started(num_images=50):
    for _ in range(num_images):
        image_url = get_random_image_url()
        download_image(image_url, output_folder)


get_started()
