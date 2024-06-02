import os

from PIL import Image

from shape import split_image
from utils import list_images_in_folder

folder = 'downloaded_images'

for filename in list_images_in_folder(folder):
    path = os.path.join(folder, filename)

    img = Image.open(
        path
    )

    print(f'cant read {path}')
    split_image(img)
