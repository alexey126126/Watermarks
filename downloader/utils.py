import os
import random
import string

from PIL import Image


def count_images_in_folder(folder_path):
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}
    image_count = 0

    try:
        for filename in os.listdir(folder_path):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_count += 1
        return image_count
    except Exception as e:
        print(f"Failed to count images in folder: {e}")
        return 0


def list_images_in_folder(folder_path):
    # Список допустимых расширений для изображений
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}
    images = []

    try:
        for filename in os.listdir(folder_path):
            # Проверка расширения файла
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                images.append(filename)
        return images
    except Exception as e:
        print(f"Failed to list images in folder: {e}")
        return []


def load_image(image_path):
    """
    Загрузка изображения.
    """
    return Image.open(image_path)


def gen_rnd_text(size=6):
    return ''.join(
        random.choice(string.ascii_lowercase + string.ascii_uppercase) for i in range(size)
    )
